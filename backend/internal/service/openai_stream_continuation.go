package service

import (
	"bufio"
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/Wei-Shaw/sub2api/internal/pkg/logger"
	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
	"go.uber.org/zap"
)

// 续接沿用管理员 1–10 次、100–10000ms 的共享预算；建议等待不超过单次 60s、累计 120s。
const openAIContinuationMaxWait = time.Minute

func withOpenAIStreamContinuation(req *http.Request, resp *http.Response, send func(*http.Request) (*http.Response, error), state *upstreamErrorRetryState) *http.Response {
	if !isOpenAIContinuationStreamResponse(resp) || req.GetBody == nil ||
		HTTPUpstreamProfileFromContext(req.Context()) != HTTPUpstreamProfileOpenAI ||
		!strings.HasSuffix(strings.TrimRight(req.URL.Path, "/"), "/responses") ||
		!state.getPolicy().settings.Enabled {
		if resp != nil && resp.StatusCode == http.StatusOK &&
			HTTPUpstreamProfileFromContext(req.Context()) == HTTPUpstreamProfileOpenAI &&
			strings.HasSuffix(strings.TrimRight(req.URL.Path, "/"), "/responses") && state.getPolicy().settings.Enabled {
			logger.FromContext(req.Context()).Warn("gateway.openai_stream_continuation_skipped",
				zap.Bool("stream_response_supported", isOpenAIContinuationStreamResponse(resp)), zap.Bool("request_replayable", req.GetBody != nil))
		}
		return resp
	}
	copyBody, err := req.GetBody()
	if err != nil {
		logger.FromContext(req.Context()).Warn("gateway.openai_stream_continuation_skipped", zap.String("reason", "request_copy_failed"))
		return resp
	}
	body, err := io.ReadAll(io.LimitReader(copyBody, openAIContinuationByteLimit+1))
	_ = copyBody.Close()
	if err != nil || len(body) > openAIContinuationByteLimit || !gjson.ValidBytes(body) || !gjson.GetBytes(body, "stream").Bool() {
		logger.FromContext(req.Context()).Warn("gateway.openai_stream_continuation_skipped", zap.String("reason", "invalid_or_nonstream_request"), zap.Int("request_bytes", len(body)))
		return resp
	}
	if conversation := gjson.GetBytes(body, "conversation"); (conversation.Exists() && conversation.Type != gjson.Null) || gjson.GetBytes(body, "background").Bool() {
		// 服务端会话和后台任务会在请求之外修改状态，不能当作独立采样重放。
		logger.FromContext(req.Context()).Warn("gateway.openai_stream_continuation_skipped", zap.String("reason", "stateful_request"))
		return resp
	}
	ctx, cancel := context.WithCancel(req.Context())
	b := &openAIContinuationBody{
		ctx: ctx, cancel: cancel, req: req, original: body, send: send, state: state,
		body: resp.Body, reader: bufio.NewReader(resp.Body), headers: resp.Header.Clone(), sequence: -1,
	}
	b.mu.Lock()
	b.stopClient = context.AfterFunc(state.clientCtx, func() { _ = b.Close() })
	b.mu.Unlock()
	resp.Body = b
	resp.ContentLength = -1
	resp.Header.Del("Content-Length")
	logger.FromContext(req.Context()).Info("gateway.openai_stream_continuation_attached",
		zap.Bool("upstream_content_type_missing", strings.TrimSpace(resp.Header.Get("Content-Type")) == ""))
	return resp
}

func isOpenAIContinuationStreamResponse(resp *http.Response) bool {
	if resp == nil || resp.StatusCode != http.StatusOK || resp.Body == nil {
		return false
	}
	// ChatGPT OAuth 可返回无 Content-Type 的合法 SSE；原生消费者按请求 stream=true
	// 读流。续接沿用此语义，仅兼容缺失头，显式 JSON/其他媒体类型仍原样处理。
	return isEventStreamResponse(resp.Header) || strings.TrimSpace(resp.Header.Get("Content-Type")) == ""
}

// Read 由原有流解析器单线程驱动，Close 可以由客户端取消回调并发调用。
// 不启动预读 goroutine；文本事件实时释放，客户端工具事件等本次尝试终态。
type openAIContinuationBody struct {
	ctx        context.Context
	cancel     context.CancelFunc
	stopClient func() bool
	req        *http.Request
	original   []byte
	send       func(*http.Request) (*http.Response, error)
	state      *upstreamErrorRetryState
	mu         sync.Mutex
	body       io.ReadCloser
	closed     bool
	reader     *bufio.Reader
	headers    http.Header
	pending    []byte
	bareError  []byte
	checkpoint openAIStreamCheckpoint
	sequence   int64
	responseID string
	createdAt  int64
	created    bool
	attempt    int
	waited     time.Duration
	eof        bool
	readErr    error
	raw        bool
	usage      OpenAIUsage
	priorUsage OpenAIUsage
	staged     [][]byte
	stageBytes int
}

func (b *openAIContinuationBody) Close() error {
	b.mu.Lock()
	if b.closed {
		b.mu.Unlock()
		return nil
	}
	b.closed = true
	body := b.body
	stopClient := b.stopClient
	b.mu.Unlock()
	b.cancel()
	if stopClient != nil {
		stopClient()
	}
	return body.Close()
}

func (b *openAIContinuationBody) Read(p []byte) (int, error) {
	if len(p) == 0 {
		return 0, nil
	}
	for len(b.pending) == 0 {
		if err := b.ctx.Err(); err != nil {
			return 0, err
		}
		if b.eof {
			if b.readErr != nil {
				return 0, b.readErr
			}
			return 0, io.EOF
		}
		if b.raw {
			return b.reader.Read(p)
		}
		frame, err := b.readFrame()
		if err != nil && len(frame) == 0 {
			if errors.Is(err, io.EOF) || errors.Is(err, io.ErrUnexpectedEOF) {
				// A response can be accepted with HTTP 200 and still terminate
				// before its terminal SSE event. When the configured policy matches
				// this sanitized upstream read failure, reuse the same checkpoint
				// continuation path as an explicit capacity/error event.
				readFailure, _ := openAIStreamReadErrorPayload(err)
				if ok, retryErr := b.continueAfter(readFailure); retryErr != nil {
					return 0, retryErr
				} else if ok {
					continue
				}
			}
			if errors.Is(err, io.EOF) && len(b.bareError) > 0 {
				failure := b.bareError
				b.bareError = nil
				if ok, retryErr := b.continueAfter(failure); retryErr != nil {
					return 0, retryErr
				} else if ok {
					continue
				}
				if b.attempt > 0 {
					failure = b.terminal(failure, "error")
				}
				b.pending = append(b.flushTools(), b.encode(failure)...)
				b.eof = true
				break
			}
			if len(b.staged) > 0 {
				b.pending = b.flushTools()
				b.eof, b.readErr = true, err
				break
			}
			return 0, err
		}
		if b.raw {
			b.pending = append(b.flushTools(), frame...)
			break
		}
		if processErr := b.process(frame); processErr != nil {
			return 0, processErr
		}
	}
	n := copy(p, b.pending)
	b.pending = b.pending[n:]
	return n, nil
}

// SSE 以空行分帧；超大初始事件退回原样转发，已续接后超限则明确终止而不产生错误拼接。
func (b *openAIContinuationBody) readFrame() ([]byte, error) {
	var frame []byte
	for {
		line, err := b.reader.ReadSlice('\n')
		frame = append(frame, line...)
		if len(frame) > openAIContinuationByteLimit {
			if b.attempt > 0 {
				return nil, fmt.Errorf("OpenAI continuation event exceeds %d bytes", openAIContinuationByteLimit)
			}
			b.raw = true
			return frame, nil
		}
		if errors.Is(err, bufio.ErrBufferFull) {
			continue
		}
		if err != nil {
			return frame, err
		}
		if bytes.Equal(line, []byte("\n")) || bytes.Equal(line, []byte("\r\n")) {
			return frame, nil
		}
	}
}

func continuationEvent(frame []byte) ([]byte, string) {
	var data []byte
	var event string
	for _, line := range bytes.Split(frame, []byte("\n")) {
		line = bytes.TrimSuffix(line, []byte("\r"))
		if bytes.HasPrefix(line, []byte("event:")) {
			event = strings.TrimSpace(string(line[6:]))
		}
		if bytes.HasPrefix(line, []byte("data:")) {
			if len(data) > 0 {
				data = append(data, '\n')
			}
			data = append(data, bytes.TrimPrefix(line[5:], []byte(" "))...)
		}
	}
	return data, effectiveOpenAISSEEventType(data, event)
}

func (b *openAIContinuationBody) process(frame []byte) error {
	payload, kind := continuationEvent(frame)
	if !gjson.ValidBytes(payload) {
		b.pending = frame
		return nil
	}
	if seq := gjson.GetBytes(payload, "sequence_number").Int(); b.attempt == 0 && seq > b.sequence {
		b.sequence = seq
	}
	if b.responseID == "" {
		b.responseID = gjson.GetBytes(payload, "response.id").String()
		b.createdAt = gjson.GetBytes(payload, "response.created_at").Int()
	}
	if parsed, ok := extractOpenAIUsageFromJSONBytes(payload); ok {
		mergeOpenAIUsageNonZero(&b.usage, parsed)
	}
	itemType := gjson.GetBytes(payload, "item.type").String()
	clientTool := kind == "response.output_item.added" && (itemType == "function_call" || itemType == "custom_tool_call" ||
		(itemType == "tool_search_call" && gjson.GetBytes(payload, "item.execution").String() == "client"))
	if (kind == "response.output_item.added" && itemType != "message" && itemType != "reasoning" && !clientTool) || strings.Contains(kind, "_call.") {
		// 即使在客户端工具暂存期间或错误之后，也不能重放已在上游执行的工具。
		b.checkpoint.blocked = "server_tool_or_unsupported_output"
	}
	if kind == "error" && isOpenAIContinuationRetryableFailure(b.ctx, payload) {
		if b.canContinue(payload) {
			// 等 response.failed 的权威 usage；若上游直接 EOF，Read 用此明确错误续接。
			b.bareError = append([]byte(nil), payload...)
			return nil
		}
		b.logContinuationBlocked("capacity_policy_or_client_state", payload)
	}
	if kind == "response.failed" {
		b.bareError = nil
		if ok, err := b.continueAfter(payload); err != nil {
			return err
		} else if ok {
			return nil
		}
	}
	if len(b.bareError) > 0 && kind != "response.completed" && kind != "response.done" && kind != "response.incomplete" {
		return nil
	}
	if kind == "response.created" || kind == "response.in_progress" {
		if kind == "response.created" && !b.created {
			b.created = true
		} else if b.attempt > 0 {
			return nil
		}
	}
	terminal := kind == "response.completed" || kind == "response.done" || kind == "response.incomplete" || kind == "response.failed"
	if !terminal && kind != "error" && (len(b.staged) > 0 || clientTool) {
		// 文本照常实时交付；客户端工具调用必须等本次尝试成功，才能暴露给执行方。
		if b.stageBytes+len(frame) <= openAIContinuationByteLimit && len(b.staged) < 4096 {
			b.staged = append(b.staged, append([]byte(nil), frame...))
			b.stageBytes += len(frame)
			return nil
		}
		b.checkpoint.blocked = "tool_staging_limit"
		b.pending = append(b.flushTools(), b.outgoing(frame, payload)...)
		return nil
	}
	b.checkpoint.observe(payload, kind)
	if terminal {
		b.bareError = nil
		b.eof = true
		if b.attempt > 0 {
			payload = b.terminal(payload, kind)
		}
	} else if kind == "error" && b.attempt > 0 {
		payload = b.terminal(payload, kind)
	}
	b.pending = append(b.flushTools(), b.outgoing(frame, payload)...)
	return nil
}

func (b *openAIContinuationBody) outgoing(frame, payload []byte) []byte {
	if b.attempt == 0 {
		return frame
	}
	return b.encode(payload)
}

func (b *openAIContinuationBody) flushTools() []byte {
	var output []byte
	if len(b.staged) > 0 && b.checkpoint.blocked == "" {
		b.checkpoint.blocked = "client_tool_dispatched"
	}
	for _, frame := range b.staged {
		payload, _ := continuationEvent(frame)
		output = append(output, b.outgoing(frame, payload)...)
	}
	b.staged = nil
	b.stageBytes = 0
	return output
}

func continuationErrorBody(payload []byte) []byte {
	err := gjson.GetBytes(payload, "response.error")
	if !err.Exists() {
		err = gjson.GetBytes(payload, "error")
	}
	if err.IsObject() {
		return []byte(`{"error":` + err.Raw + `}`)
	}
	// 兼容顶层 code/message 的标准 Responses error 事件。
	body, _ := json.Marshal(map[string]any{"error": map[string]any{
		"code": gjson.GetBytes(payload, "code").String(), "message": gjson.GetBytes(payload, "message").String(),
	}})
	return body
}

// isOpenAIStreamReadFailure recognizes sanitized transport failures generated
// after an HTTP 200 SSE response has already been accepted.
func isOpenAIStreamReadFailure(payload []byte) bool {
	code := openAIStreamFailedEventErrorCode(payload)
	if code == "" {
		code = strings.ToLower(strings.TrimSpace(gjson.GetBytes(payload, "code").String()))
	}
	return code == OpenAIUpstreamHTTP2StreamErrorCode ||
		code == OpenAIUpstreamStreamReadErrorCode ||
		code == OpenAIUpstreamStreamTruncatedCode
}

// isOpenAIContinuationRetryableFailure limits stream-local replay to the
// established capacity errors, generic OpenAI 503 failures, and read failures
// that match the administrator policy.
func isOpenAIContinuationRetryableFailure(ctx context.Context, payload []byte) bool {
	if isOpenAIUpstreamCapacityShedEvent(payload) {
		return true
	}
	if openAIStreamFailedEventErrorCode(payload) != "server_error" {
		return configuredOpenAIStreamRetryFailure(ctx, payload, extractOpenAISSEErrorMessage(payload), nil) != nil
	}
	return isOpenAITransientProcessingError(
		http.StatusBadRequest,
		extractOpenAISSEErrorMessage(payload),
		payload,
	)
}

func (b *openAIContinuationBody) canContinue(payload []byte) bool {
	if b.checkpoint.blocked != "" || !isOpenAIContinuationRetryableFailure(b.ctx, payload) || b.ctx.Err() != nil {
		return false
	}
	// 这里只处理容量 503 和 OpenAI generic server_error 503；限流/账单/鉴权继续走原有策略；usage 由续接层累计，不丢弃。
	code := openAIStreamFailedEventErrorCode(payload)
	if code == "" {
		code = gjson.GetBytes(payload, "code").String()
	}
	if code == "slow_down" || IsUpstreamBillingError(http.StatusOK, payload) {
		return false
	}
	if !isOpenAIStreamReadFailure(payload) && code != "server_error" && openAIStreamFailedEventSemanticStatus(payload, extractOpenAISSEErrorMessage(payload)) != http.StatusServiceUnavailable {
		return false
	}
	status := http.StatusServiceUnavailable
	if isOpenAIStreamReadFailure(payload) {
		status = openAIStreamFailedEventSemanticStatus(payload, extractOpenAISSEErrorMessage(payload))
	}
	return b.state.getPolicy().matches(status, continuationErrorBody(payload))
}

func (b *openAIContinuationBody) continueAfter(payload []byte) (bool, error) {
	if !b.canContinue(payload) {
		if isOpenAIContinuationRetryableFailure(b.ctx, payload) {
			b.logContinuationBlocked("capacity_policy_or_client_state", payload)
		}
		return false, nil
	}
	output, closing, safe := b.checkpoint.prepare()
	if !safe {
		b.logContinuationBlocked("checkpoint_not_replayable", payload)
		return false, nil
	}
	body, safe := b.checkpoint.requestBody(b.original, output)
	if !safe {
		b.logContinuationBlocked("request_not_replayable", payload)
		return false, nil
	}
	delay, permitted := b.retryDelay(payload)
	if !permitted {
		b.logContinuationBlocked("retry_wait_limit", payload)
		return false, nil
	}
	status := http.StatusServiceUnavailable
	if isOpenAIStreamReadFailure(payload) {
		status = openAIStreamFailedEventSemanticStatus(payload, extractOpenAISSEErrorMessage(payload))
	}
	if _, claimed := b.state.claim(status, continuationErrorBody(payload)); !claimed {
		b.logContinuationBlocked("retry_budget_or_client_cancelled", payload)
		return false, nil
	}
	b.mu.Lock()
	old := b.body
	b.mu.Unlock()
	_ = old.Close()
	if err := b.state.wait(b.ctx, delay); err != nil {
		return false, err
	}
	b.waited += delay
	next := b.req.Clone(b.ctx)
	next.Body = io.NopCloser(bytes.NewReader(body))
	next.ContentLength = int64(len(body))
	next.GetBody = func() (io.ReadCloser, error) { return io.NopCloser(bytes.NewReader(body)), nil }
	next.Header.Del("Content-Length")
	if key := next.Header.Get("Idempotency-Key"); key != "" {
		// 续接输入已改变，每个新采样独立幂等；其 HTTP 层重试继续复用此键。
		digest := sha256.Sum256([]byte(fmt.Sprintf("%s/continuation/%d", key, b.attempt+1)))
		next.Header.Set("Idempotency-Key", fmt.Sprintf("%x", digest))
	}
	resp, err := doWithOpenAIContinuationTransportRetry(next, b.send)
	if err != nil || resp == nil {
		_ = next.Body.Close()
		if b.ctx.Err() != nil {
			return false, b.ctx.Err()
		}
		logger.FromContext(b.ctx).Warn("gateway.openai_stream_continuation_failed", zap.Error(err))
		return false, nil
	}
	if !isOpenAIContinuationStreamResponse(resp) {
		if resp.Body != nil {
			_ = resp.Body.Close()
		}
		logger.FromContext(b.ctx).Warn("gateway.openai_stream_continuation_rejected", zap.Int("upstream_status", resp.StatusCode))
		return false, nil
	}
	b.mu.Lock()
	if b.closed {
		b.mu.Unlock()
		_ = resp.Body.Close()
		return false, context.Canceled
	}
	b.body = resp.Body
	b.mu.Unlock()
	// 先关闭旧尝试的未完成文本项，再发布新尝试事件；只保留一个下游响应身份。
	for _, event := range closing {
		raw, _ := json.Marshal(event)
		b.pending = append(b.pending, b.encode(raw)...)
	}
	b.checkpoint.commit(output)
	b.staged = nil
	b.stageBytes = 0
	addContinuationUsage(&b.priorUsage, b.usage)
	b.usage = OpenAIUsage{}
	b.attempt++
	b.reader = bufio.NewReader(resp.Body)
	b.headers = resp.Header.Clone()
	b.eof = false
	logger.FromContext(b.ctx).Warn("gateway.openai_stream_continuation",
		zap.Int("continuation", b.attempt), zap.Int("checkpoint_items", len(output)), zap.Duration("retry_delay", delay),
		zap.String("upstream_request_id", resp.Header.Get("x-request-id")))
	return true, nil
}

// doWithOpenAIContinuationTransportRetry replays one continuation request when
// the transport fails before a response is received. The continuation body is
// checkpoint-derived and replayable, so an EOF here is safe to retry once;
// ordinary inference requests remain covered by the conservative policy in
// doWithConfiguredUpstreamRetry and are not replayed on transport errors.
func doWithOpenAIContinuationTransportRetry(req *http.Request, send func(*http.Request) (*http.Response, error)) (*http.Response, error) {
	resp, err := doWithConfiguredUpstreamRetry(req, send, false)
	if !isOpenAIContinuationTransportEOF(req, err) {
		return resp, err
	}
	body, cloneErr := req.GetBody()
	if cloneErr != nil {
		return resp, err
	}
	if req.Body != nil {
		_ = req.Body.Close()
	}
	retryReq := req.Clone(req.Context())
	retryReq.Body = body
	retryReq.GetBody = req.GetBody
	logger.FromContext(req.Context()).Info("gateway.openai_stream_continuation_transport_retry",
		zap.String("reason", "eof"))
	return doWithConfiguredUpstreamRetry(retryReq, send, false)
}

func isOpenAIContinuationTransportEOF(req *http.Request, err error) bool {
	if req == nil || err == nil || req.GetBody == nil || req.Context().Err() != nil {
		return false
	}
	if HTTPUpstreamProfileFromContext(req.Context()) != HTTPUpstreamProfileOpenAI {
		return false
	}
	return errors.Is(err, io.EOF) || errors.Is(err, io.ErrUnexpectedEOF)
}

func (b *openAIContinuationBody) logContinuationBlocked(reason string, payload []byte) {
	if b.checkpoint.blocked != "" {
		reason = b.checkpoint.blocked
	}
	logger.FromContext(b.ctx).Warn("gateway.openai_stream_continuation_blocked",
		zap.String("reason", reason), zap.String("event_type", gjson.GetBytes(payload, "type").String()),
		zap.Int("semantic_status", openAIStreamFailedEventSemanticStatus(payload, extractOpenAISSEErrorMessage(payload))),
		zap.Bool("slow_down", openAIStreamFailedEventErrorCode(payload) == "slow_down"),
		zap.Bool("client_cancelled", b.state.clientCtx.Err() != nil))
}

func (b *openAIContinuationBody) retryDelay(payload []byte) (time.Duration, bool) {
	delay := time.Duration(b.state.getPolicy().settings.DelayMS) * time.Millisecond
	value := ""
	for _, path := range []string{"response.error.headers", "error.headers", "headers"} {
		gjson.GetBytes(payload, path).ForEach(func(key, v gjson.Result) bool {
			if strings.EqualFold(key.String(), "Retry-After") && v.Type == gjson.String {
				value = v.String()
				return false
			}
			return true
		})
		if value != "" {
			break
		}
	}
	if value == "" {
		value = b.headers.Get("Retry-After")
	}
	value = strings.TrimSpace(value)
	if value != "" && len(value) <= 128 && !strings.ContainsAny(value, "\r\n") {
		digits := true
		for _, c := range value {
			if c < '0' || c > '9' {
				digits = false
				break
			}
		}
		if digits {
			seconds, err := strconv.ParseUint(value, 10, 64)
			if err != nil || seconds > uint64(openAIContinuationMaxWait/time.Second) {
				return 0, false
			}
			delay = time.Duration(seconds) * time.Second
		} else if at, err := http.ParseTime(value); err == nil {
			delay = time.Until(at)
			if delay < 0 {
				delay = 0
			}
		}
	}
	return delay, delay <= openAIContinuationMaxWait && b.waited+delay <= 2*openAIContinuationMaxWait
}

func (b *openAIContinuationBody) encode(payload []byte) []byte {
	b.sequence++
	updated, _ := sjson.SetBytes(payload, "sequence_number", b.sequence)
	if gjson.GetBytes(updated, "response.id").Exists() && b.responseID != "" {
		updated, _ = sjson.SetBytes(updated, "response.id", b.responseID)
	}
	if gjson.GetBytes(updated, "response.created_at").Exists() && b.createdAt > 0 {
		updated, _ = sjson.SetBytes(updated, "response.created_at", b.createdAt)
	}
	if gjson.GetBytes(updated, "response_id").Exists() && b.responseID != "" {
		updated, _ = sjson.SetBytes(updated, "response_id", b.responseID)
	}
	if index := gjson.GetBytes(updated, "output_index"); index.Exists() {
		updated, _ = sjson.SetBytes(updated, "output_index", index.Int()+int64(len(b.checkpoint.retained)))
	}
	kind := gjson.GetBytes(updated, "type").String()
	return []byte("event: " + kind + "\ndata: " + string(updated) + "\n\n")
}

func addContinuationUsage(dst *OpenAIUsage, src OpenAIUsage) {
	// 防止兼容上游的恶意/畸形 token 数触发整型溢出；实际计费数据保持单调。
	add := func(a, n int) int {
		if n <= 0 {
			return a
		}
		limit := int(^uint(0) >> 1)
		if a > limit-n {
			return limit
		}
		return a + n
	}
	dst.InputTokens = add(dst.InputTokens, src.InputTokens)
	dst.OutputTokens = add(dst.OutputTokens, src.OutputTokens)
	dst.CacheReadInputTokens = add(dst.CacheReadInputTokens, src.CacheReadInputTokens)
	dst.CacheCreationInputTokens = add(dst.CacheCreationInputTokens, src.CacheCreationInputTokens)
	dst.ImageInputTokens = add(dst.ImageInputTokens, src.ImageInputTokens)
	dst.ImageCacheReadTokens = add(dst.ImageCacheReadTokens, src.ImageCacheReadTokens)
	dst.ImageOutputTokens = add(dst.ImageOutputTokens, src.ImageOutputTokens)
}

func (b *openAIContinuationBody) terminal(payload []byte, kind string) []byte {
	if kind != "error" {
		output := append([]json.RawMessage(nil), b.checkpoint.retained...)
		for _, item := range gjson.GetBytes(payload, "response.output").Array() {
			output = append(output, json.RawMessage(item.Raw))
		}
		payload, _ = sjson.SetBytes(payload, "response.output", output)
	}
	usage := b.priorUsage
	addContinuationUsage(&usage, b.usage)
	total := OpenAIUsage{InputTokens: usage.InputTokens}
	addContinuationUsage(&total, OpenAIUsage{InputTokens: usage.OutputTokens})
	values := map[string]int{
		"input_tokens": usage.InputTokens, "output_tokens": usage.OutputTokens,
		"total_tokens":                            total.InputTokens,
		"input_tokens_details.cached_tokens":      usage.CacheReadInputTokens,
		"input_tokens_details.cache_write_tokens": usage.CacheCreationInputTokens,
		"input_tokens_details.image_tokens":       usage.ImageInputTokens,
		"output_tokens_details.image_tokens":      usage.ImageOutputTokens,
		"cache_creation_input_tokens":             usage.CacheCreationInputTokens,
	}
	usagePaths := []string{"response.usage."}
	if kind == "error" {
		usagePaths = []string{"usage."}
	} else if gjson.GetBytes(payload, "usage").Exists() {
		usagePaths = append(usagePaths, "usage.")
	}
	for _, prefix := range usagePaths {
		for path, value := range values {
			payload, _ = sjson.SetBytes(payload, prefix+path, value)
		}
	}
	return payload
}
