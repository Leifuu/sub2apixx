package service

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/Wei-Shaw/sub2api/internal/config"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
	"github.com/tidwall/gjson"
)

const continuationOverload = `{"type":"response.failed","sequence_number":8,"response":{"id":"resp_first","error":{"code":"server_is_overloaded","message":"Our servers are currently overloaded","headers":{"Retry-After":"0"}}}}`

const continuationGenericServerError = `{"type":"response.failed","sequence_number":8,"response":{"id":"resp_first","error":{"code":"server_error","message":"An error occurred while processing your request. Please try again later.","headers":{"Retry-After":"0"}}}}`

func continuationResponse(stream string) *http.Response {
	return &http.Response{StatusCode: 200, Header: http.Header{"Content-Type": {"text/event-stream"}}, Body: io.NopCloser(strings.NewReader(stream))}
}

type continuationUnexpectedEOFBody struct {
	reader *strings.Reader
	ended  bool
}

func (b *continuationUnexpectedEOFBody) Read(p []byte) (int, error) {
	if b.ended {
		return 0, io.ErrUnexpectedEOF
	}
	n, err := b.reader.Read(p)
	if err == io.EOF {
		b.ended = true
		return 0, io.ErrUnexpectedEOF
	}
	return n, err
}

func (b *continuationUnexpectedEOFBody) Close() error { return nil }

func continuationSSE(events ...string) string {
	var out strings.Builder
	for _, event := range events {
		fmt.Fprintf(&out, "event: %s\ndata: %s\n\n", gjson.Get(event, "type").String(), event)
	}
	return out.String()
}

func continuationRequest(t *testing.T, ctx context.Context) *http.Request {
	t.Helper()
	ctx = WithHTTPUpstreamProfile(ctx, HTTPUpstreamProfileOpenAI)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, "https://upstream.example/v1/responses", strings.NewReader(`{"model":"test","stream":true,"input":[{"role":"user","content":"hello"}],"store":false}`))
	require.NoError(t, err)
	req.Header.Set("Authorization", "Bearer test")
	return req
}

func TestOpenAIStreamContinuationOAuthWithoutContentType(t *testing.T) {
	for _, test := range []struct {
		name                string
		firstType, nextType string
		wantAttempts        int
	}{
		{"both missing", "", "", 2},
		{"initial missing", "", "text/event-stream", 2},
		{"continuation missing", "text/event-stream", "", 2},
		{"explicit JSON preserved", "application/json", "", 1},
	} {
		t.Run(test.name, func(t *testing.T) {
			ctx := upstreamRetryTestContext(t, context.Background(), "503", 1)
			req := continuationRequest(t, ctx)
			// 生产 OAuth 账号714的事件顺序：无Content-Type、metadata后明确过载。
			req.URL.Scheme, req.URL.Host, req.URL.Path = "https", "chatgpt.com", "/backend-api/codex/responses"
			failure := continuationSSE(
				`{"type":"response.created","sequence_number":0,"response":{"id":"resp_oauth"}}`,
				`{"type":"response.in_progress","sequence_number":1,"response":{"id":"resp_oauth"}}`,
				`{"type":"response.metadata","sequence_number":2}`,
				`{"type":"error","sequence_number":3,"error":{"code":"server_is_overloaded","type":"service_unavailable_error","message":"Our servers are currently overloaded. Please try again later."}}`,
				`{"type":"response.failed","sequence_number":4,"response":{"id":"resp_oauth","error":{"code":"server_is_overloaded","type":"service_unavailable_error","message":"Our servers are currently overloaded. Please try again later."}}}`,
			)
			attempts := 0
			resp, err := DoWithConfiguredUpstreamRetry(req, func(r *http.Request) (*http.Response, error) {
				attempts++
				defer r.Body.Close()
				stream, contentType := failure, test.firstType
				if attempts == 2 {
					body, readErr := io.ReadAll(r.Body)
					require.NoError(t, readErr)
					require.Equal(t, "hello", gjson.GetBytes(body, "input.0.content").String())
					contentType = test.nextType
					stream = continuationSSE(
						`{"type":"response.created","sequence_number":0,"response":{"id":"resp_recovered"}}`,
						`{"type":"response.completed","sequence_number":1,"response":{"id":"resp_recovered","status":"completed","output":[],"usage":{"input_tokens":3,"output_tokens":1}}}`,
					)
				}
				response := continuationResponse(stream)
				response.Header.Del("Content-Type")
				if contentType != "" {
					response.Header.Set("Content-Type", contentType)
				}
				return response, nil
			})
			require.NoError(t, err)
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			require.NoError(t, err)
			require.Equal(t, test.wantAttempts, attempts)
			if test.wantAttempts == 1 {
				require.Equal(t, failure, string(body))
				return
			}
			require.NotContains(t, string(body), "server_is_overloaded")
			require.NotContains(t, string(body), "response.failed")
			require.NotContains(t, string(body), "resp_recovered")
			require.Contains(t, string(body), "response.completed")
			require.Equal(t, 1, strings.Count(string(body), "event: response.created"))
		})
	}
}

func TestOpenAIStreamContinuationAfterPartialText(t *testing.T) {
	ctx := upstreamRetryTestContext(t, context.Background(), "503", 2)
	req := continuationRequest(t, ctx)
	attempts := 0
	resp, err := DoWithConfiguredUpstreamRetry(req, func(r *http.Request) (*http.Response, error) {
		attempts++
		body, readErr := io.ReadAll(r.Body)
		require.NoError(t, readErr)
		_ = r.Body.Close()
		require.Equal(t, req.Header.Get("Authorization"), r.Header.Get("Authorization"))
		stream := continuationSSE(
			`{"type":"response.created","sequence_number":0,"response":{"id":"resp_first","created_at":100}}`,
			`{"type":"response.output_item.added","sequence_number":1,"output_index":0,"item":{"id":"msg_first","type":"message","role":"assistant","status":"in_progress","content":[]}}`,
			`{"type":"response.content_part.added","sequence_number":2,"output_index":0,"item_id":"msg_first","content_index":0,"part":{"type":"output_text","text":"","annotations":[]}}`,
			`{"type":"response.output_text.delta","sequence_number":3,"output_index":0,"item_id":"msg_first","content_index":0,"delta":"你好"}`,
			`{"type":"response.failed","sequence_number":4,"response":{"id":"resp_first","error":{"code":"server_is_overloaded","message":"Our servers are currently overloaded","headers":{"Retry-After":"0"}},"usage":{"input_tokens":10,"output_tokens":2}}}`,
		)
		if attempts == 2 {
			require.Equal(t, "hello", gjson.GetBytes(body, "input.0.content").String())
			require.Equal(t, "assistant", gjson.GetBytes(body, "input.1.role").String())
			require.Equal(t, "你好", gjson.GetBytes(body, "input.1.content.0.text").String())
			stream = continuationSSE(
				`{"type":"response.created","sequence_number":0,"response":{"id":"resp_second","created_at":200}}`,
				`{"type":"response.output_item.added","sequence_number":1,"output_index":0,"item":{"id":"msg_second","type":"message","role":"assistant","content":[]}}`,
				`{"type":"response.output_text.delta","sequence_number":2,"output_index":0,"item_id":"msg_second","content_index":0,"delta":"，继续"}`,
				`{"type":"response.output_item.done","sequence_number":3,"output_index":0,"item":{"id":"msg_second","type":"message","role":"assistant","content":[{"type":"output_text","text":"，继续","annotations":[]}]}}`,
				`{"type":"response.completed","sequence_number":4,"response":{"id":"resp_second","created_at":200,"status":"completed","output":[{"id":"msg_second","type":"message","role":"assistant","content":[{"type":"output_text","text":"，继续","annotations":[]}]}],"usage":{"input_tokens":20,"output_tokens":3}}}`,
			)
		}
		return &http.Response{StatusCode: 200, Header: http.Header{"Content-Type": {"text/event-stream"}}, Body: io.NopCloser(strings.NewReader(stream))}, nil
	})
	require.NoError(t, err)
	defer resp.Body.Close()
	body, err := io.ReadAll(resp.Body)
	require.NoError(t, err)
	require.Equal(t, 2, attempts, "流中途过载应在网关内部续接")
	require.NotContains(t, string(body), "response.failed")
	require.NotContains(t, string(body), "resp_second")
	require.Equal(t, 1, strings.Count(string(body), "event: response.created"))
	var lastSequence int64 = -1
	var terminal gjson.Result
	for _, line := range strings.Split(string(body), "\n") {
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		event := gjson.Parse(strings.TrimPrefix(line, "data: "))
		require.Greater(t, event.Get("sequence_number").Int(), lastSequence)
		lastSequence = event.Get("sequence_number").Int()
		if event.Get("type").String() == "response.completed" {
			terminal = event
		}
	}
	require.Equal(t, "resp_first", terminal.Get("response.id").String())
	require.Equal(t, int64(2), terminal.Get("response.output.#").Int())
	require.Equal(t, "你好", terminal.Get("response.output.0.content.0.text").String())
	require.Equal(t, int64(30), terminal.Get("response.usage.input_tokens").Int())
	require.Equal(t, int64(5), terminal.Get("response.usage.output_tokens").Int())
	var valid any
	require.NoError(t, json.Unmarshal([]byte(terminal.Raw), &valid))
}

func TestOpenAIStreamContinuationRetriesGenericServerError(t *testing.T) {
	ctx := upstreamRetryTestContext(t, context.Background(), "503", 1)
	attempts := 0
	resp, err := DoWithConfiguredUpstreamRetry(continuationRequest(t, ctx), func(r *http.Request) (*http.Response, error) {
		attempts++
		_ = r.Body.Close()
		if attempts == 1 {
			return continuationResponse(continuationSSE(
				`{"type":"response.created","sequence_number":0,"response":{"id":"resp_first"}}`,
				`{"type":"response.in_progress","sequence_number":1,"response":{"id":"resp_first"}}`,
				continuationGenericServerError,
			)), nil
		}
		return continuationResponse(continuationSSE(
			`{"type":"response.created","sequence_number":0,"response":{"id":"resp_recovered"}}`,
			`{"type":"response.completed","sequence_number":1,"response":{"id":"resp_recovered","status":"completed","output":[]}}`,
		)), nil
	})
	require.NoError(t, err)
	defer resp.Body.Close()
	body, err := io.ReadAll(resp.Body)
	require.NoError(t, err)
	require.Equal(t, 2, attempts)
	require.NotContains(t, string(body), "response.failed")
	require.Contains(t, string(body), "response.completed")
}

func TestOpenAIStreamContinuationRetriesConfiguredUpstreamResponseEOF(t *testing.T) {
	ctx := upstreamRetryTestContext(t, context.Background(), "Upstream response", 1)
	attempts := 0
	resp, err := DoWithConfiguredUpstreamRetry(continuationRequest(t, ctx), func(r *http.Request) (*http.Response, error) {
		attempts++
		body, readErr := io.ReadAll(r.Body)
		require.NoError(t, readErr)
		_ = r.Body.Close()
		if attempts == 1 {
			return &http.Response{
				StatusCode: 200,
				Header:     http.Header{"Content-Type": {"text/event-stream"}},
				Body: &continuationUnexpectedEOFBody{reader: strings.NewReader(continuationSSE(
					`{"type":"response.created","sequence_number":0,"response":{"id":"resp_first"}}`,
					`{"type":"response.output_item.added","sequence_number":1,"output_index":0,"item":{"id":"msg_first","type":"message","role":"assistant","content":[]}}`,
					`{"type":"response.output_text.delta","sequence_number":2,"output_index":0,"item_id":"msg_first","content_index":0,"delta":"部分"}`,
				))},
			}, nil
		}
		require.Contains(t, string(body), "部分")
		return continuationResponse(continuationSSE(
			`{"type":"response.created","sequence_number":0,"response":{"id":"resp_recovered"}}`,
			`{"type":"response.output_text.delta","sequence_number":1,"output_index":0,"item_id":"msg_second","content_index":0,"delta":"续接"}`,
			`{"type":"response.completed","sequence_number":2,"response":{"id":"resp_recovered","status":"completed","output":[]}}`,
		)), nil
	})
	require.NoError(t, err)
	defer resp.Body.Close()
	body, err := io.ReadAll(resp.Body)
	require.NoError(t, err)
	require.Equal(t, 2, attempts)
	require.Contains(t, string(body), "部分")
	require.Contains(t, string(body), "续接")
	require.NotContains(t, string(body), "upstream_stream_read_error")
	require.Equal(t, 1, strings.Count(string(body), "event: response.created"))
}

func TestOpenAIStreamContinuationLeavesUnmatchedEOF(t *testing.T) {
	ctx := upstreamRetryTestContext(t, context.Background(), "503", 1)
	attempts := 0
	resp, err := DoWithConfiguredUpstreamRetry(continuationRequest(t, ctx), func(r *http.Request) (*http.Response, error) {
		attempts++
		_ = r.Body.Close()
		return &http.Response{
			StatusCode: 200,
			Header:     http.Header{"Content-Type": {"text/event-stream"}},
			Body: &continuationUnexpectedEOFBody{reader: strings.NewReader(continuationSSE(
				`{"type":"response.created","sequence_number":0,"response":{"id":"resp_first"}}`,
			))},
		}, nil
	})
	require.NoError(t, err)
	defer resp.Body.Close()
	body, readErr := io.ReadAll(resp.Body)
	require.ErrorIs(t, readErr, io.ErrUnexpectedEOF)
	require.Equal(t, 1, attempts)
	require.Contains(t, string(body), "response.created")
}

func TestOpenAIStreamContinuationDeliversBeforeRetryAndCancelsWait(t *testing.T) {
	parent, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	ctx := upstreamRetryTestContext(t, parent, "503", 2)
	pipeReader, pipeWriter := io.Pipe()
	var attempts atomic.Int32
	resp, err := DoWithConfiguredUpstreamRetry(continuationRequest(t, ctx), func(r *http.Request) (*http.Response, error) {
		attempts.Add(1)
		_ = r.Body.Close()
		return &http.Response{StatusCode: 200, Header: http.Header{"Content-Type": {"text/event-stream"}}, Body: pipeReader}, nil
	})
	require.NoError(t, err)
	defer resp.Body.Close()
	go func() {
		_, _ = io.WriteString(pipeWriter, continuationSSE(
			`{"type":"response.created","sequence_number":0,"response":{"id":"resp_first"}}`,
			`{"type":"response.output_item.added","sequence_number":1,"output_index":0,"item":{"type":"message","id":"msg","role":"assistant","content":[]}}`,
			`{"type":"response.output_text.delta","sequence_number":2,"output_index":0,"content_index":0,"item_id":"msg","delta":"实时内容"}`,
		))
	}()
	reader := bufio.NewReader(resp.Body)
	var firstOutput strings.Builder
	for !strings.Contains(firstOutput.String(), "实时内容") {
		line, readErr := reader.ReadString('\n')
		require.NoError(t, readErr)
		firstOutput.WriteString(line)
	}
	require.Equal(t, int32(1), attempts.Load(), "首次流还未结束，文本已交付")
	closed := make(chan struct{})
	// 失败建议 1 秒，取消应立即打断等待，不能发起第二次请求。
	go func() {
		failure := strings.Replace(continuationOverload, `"Retry-After":"0"`, `"Retry-After":"1"`, 1)
		_, _ = io.WriteString(pipeWriter, continuationSSE(failure))
		_ = pipeWriter.Close()
		close(closed)
	}()
	done := make(chan error, 1)
	go func() { _, readErr := io.ReadAll(reader); done <- readErr }()
	<-closed
	cancel()
	select {
	case readErr := <-done:
		require.ErrorIs(t, readErr, context.Canceled)
	case <-time.After(time.Second):
		t.Fatal("取消未停止续接等待")
	}
	require.Equal(t, int32(1), attempts.Load())
}

func TestOpenAIStreamContinuationBudgetAndToolBoundary(t *testing.T) {
	for _, test := range []struct {
		name   string
		prefix string
		want   int
	}{
		{"budget", "", 3},
		{"server tool already executed", continuationSSE(`{"type":"response.output_item.added","sequence_number":1,"output_index":0,"item":{"id":"search_item","type":"web_search_call","status":"in_progress"}}`), 1},
	} {
		t.Run(test.name, func(t *testing.T) {
			ctx := upstreamRetryTestContext(t, context.Background(), "503", 2)
			attempts := 0
			resp, err := DoWithConfiguredUpstreamRetry(continuationRequest(t, ctx), func(r *http.Request) (*http.Response, error) {
				attempts++
				_ = r.Body.Close()
				return continuationResponse(test.prefix + continuationSSE(continuationOverload)), nil
			})
			require.NoError(t, err)
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			require.NoError(t, err)
			require.Equal(t, test.want, attempts)
			require.Equal(t, 1, strings.Count(string(body), "event: response.failed"))
		})
	}
}

func TestOpenAIStreamContinuationBareErrorEOFAndEncryptedHistory(t *testing.T) {
	ctx := upstreamRetryTestContext(t, context.Background(), "503", 1)
	initial := continuationSSE(
		`{"type":"response.created","sequence_number":0,"response":{"id":"resp_first","created_at":100}}`,
		`{"type":"response.output_item.added","sequence_number":1,"output_index":0,"item":{"id":"reason_1","type":"reasoning","summary":[]}}`,
		`{"type":"response.output_item.done","sequence_number":2,"output_index":0,"item":{"id":"reason_1","type":"reasoning","summary":[],"encrypted_content":"opaque_verified_content"}}`,
		`{"type":"error","sequence_number":3,"error":{"code":"server_is_overloaded","message":"Our servers are currently overloaded","headers":{"retry-after":"0"}}}`,
	)
	attempts := 0
	resp, err := DoWithConfiguredUpstreamRetry(continuationRequest(t, ctx), func(r *http.Request) (*http.Response, error) {
		attempts++
		defer r.Body.Close()
		if attempts == 1 {
			return continuationResponse(initial), nil
		}
		body, err := io.ReadAll(r.Body)
		require.NoError(t, err)
		require.Equal(t, "opaque_verified_content", gjson.GetBytes(body, "input.1.encrypted_content").String())
		return continuationResponse(continuationSSE(`{"type":"response.completed","sequence_number":0,"response":{"id":"resp_new","status":"completed","output":[],"usage":{"input_tokens":3,"output_tokens":1}}}`)), nil
	})
	require.NoError(t, err)
	defer resp.Body.Close()
	body, err := io.ReadAll(resp.Body)
	require.NoError(t, err)
	require.Equal(t, 2, attempts)
	require.NotContains(t, string(body), "event: error")
	require.Contains(t, string(body), "response.completed")
}

func TestOpenAIStreamContinuationRetriesTransportEOF(t *testing.T) {
	ctx := upstreamRetryTestContext(t, context.Background(), "503", 2)
	attempts := 0
	resp, err := DoWithConfiguredUpstreamRetry(continuationRequest(t, ctx), func(r *http.Request) (*http.Response, error) {
		attempts++
		body, readErr := io.ReadAll(r.Body)
		require.NoError(t, readErr)
		_ = r.Body.Close()
		if attempts == 1 {
			require.Contains(t, string(body), `"stream":true`)
			return continuationResponse(continuationSSE(continuationOverload)), nil
		}
		if attempts == 2 {
			return nil, fmt.Errorf("Post %s: %w", r.URL, io.EOF)
		}
		return continuationResponse(continuationSSE(
			`{"type":"response.completed","sequence_number":0,"response":{"id":"resp_recovered","status":"completed","output":[]}}`,
		)), nil
	})
	require.NoError(t, err)
	defer resp.Body.Close()
	body, err := io.ReadAll(resp.Body)
	require.NoError(t, err)
	require.Equal(t, 3, attempts)
	require.Contains(t, string(body), "response.completed")
	require.NotContains(t, string(body), "response.failed")
}

func TestOpenAIStreamContinuationPreservesUnmatchedAndBoundedBodies(t *testing.T) {
	for _, test := range []struct {
		name, stream string
		rules        string
	}{
		{"unmatched", continuationSSE(continuationOverload), "502"},
		{"rate limit", continuationSSE(strings.Replace(continuationOverload, "server_is_overloaded", "slow_down", 1)), "503"},
		{"missing output index", continuationSSE(`{"type":"response.output_text.delta","delta":"已交付"}`, continuationOverload), "503"},
		{"long advice", continuationSSE(strings.Replace(continuationOverload, `"Retry-After":"0"`, `"Retry-After":"3600"`, 1)), "503"},
		{"huge event", "data: {\"type\":\"response.output_text.delta\",\"delta\":\"" + strings.Repeat("x", openAIContinuationByteLimit+1) + "\"}\n\n" + continuationSSE(continuationOverload), "503"},
	} {
		t.Run(test.name, func(t *testing.T) {
			ctx := upstreamRetryTestContext(t, context.Background(), test.rules, 2)
			attempts := 0
			resp, err := DoWithConfiguredUpstreamRetry(continuationRequest(t, ctx), func(r *http.Request) (*http.Response, error) {
				attempts++
				_ = r.Body.Close()
				return continuationResponse(test.stream), nil
			})
			require.NoError(t, err)
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			require.NoError(t, err)
			require.Equal(t, 1, attempts)
			require.Equal(t, test.stream, string(body))
		})
	}
}

func TestOpenAIStreamContinuationClientToolsAreDeliveredOnce(t *testing.T) {
	ctx := upstreamRetryTestContext(t, context.Background(), "503", 2)
	attempts := 0
	resp, err := DoWithConfiguredUpstreamRetry(continuationRequest(t, ctx), func(r *http.Request) (*http.Response, error) {
		attempts++
		body, err := io.ReadAll(r.Body)
		require.NoError(t, err)
		_ = r.Body.Close()
		require.NotContains(t, string(body), "call_discarded", "客户端未执行的工具不能加入续接历史")
		call := "call_discarded"
		terminal := continuationOverload
		if attempts == 2 {
			call = "call_delivered"
			terminal = `{"type":"response.completed","sequence_number":4,"response":{"id":"resp_new","output":[],"usage":{"input_tokens":1,"output_tokens":1}}}`
		}
		return continuationResponse(continuationSSE(
			fmt.Sprintf(`{"type":"response.output_item.added","sequence_number":1,"output_index":0,"item":{"id":"%s","type":"function_call","call_id":"%s","name":"pay","arguments":""}}`, call, call),
			fmt.Sprintf(`{"type":"response.function_call_arguments.delta","sequence_number":2,"output_index":0,"item_id":"%s","delta":"{}"}`, call),
			fmt.Sprintf(`{"type":"response.output_item.done","sequence_number":3,"output_index":0,"item":{"id":"%s","type":"function_call","call_id":"%s","name":"pay","arguments":"{}"}}`, call, call),
			terminal,
		)), nil
	})
	require.NoError(t, err)
	defer resp.Body.Close()
	body, err := io.ReadAll(resp.Body)
	require.NoError(t, err)
	require.Equal(t, 2, attempts)
	require.NotContains(t, string(body), "call_discarded")
	require.Equal(t, 1, strings.Count(string(body), "event: response.output_item.done"))
	require.Contains(t, string(body), "call_delivered")
}

func TestOpenAIStreamContinuationToolsPreserveOrderingAndBlockAfterDispatch(t *testing.T) {
	tool := `{"type":"response.output_item.added","sequence_number":1,"output_index":0,"item":{"id":"call_item","type":"function_call","call_id":"call_1","name":"pay","arguments":""}}`
	for _, tail := range []string{
		continuationSSE(`{"type":"response.completed","sequence_number":2,"response":{"output":[]}}`),
		"",
		continuationSSE(`{"type":"error","sequence_number":2,"error":{"code":"invalid_request","message":"bad"}}`, continuationOverload),
		continuationSSE(`{"type":"response.output_item.added","sequence_number":2,"output_index":1,"item":{"type":"web_search_call"}}`, continuationOverload),
	} {
		ctx := upstreamRetryTestContext(t, context.Background(), "503", 2)
		stream := continuationSSE(tool) + tail
		attempts := 0
		resp, err := DoWithConfiguredUpstreamRetry(continuationRequest(t, ctx), func(r *http.Request) (*http.Response, error) {
			attempts++
			_ = r.Body.Close()
			return continuationResponse(stream), nil
		})
		require.NoError(t, err)
		body, err := io.ReadAll(resp.Body)
		_ = resp.Body.Close()
		require.NoError(t, err)
		require.Equal(t, stream, string(body))
		require.Equal(t, 1, attempts)
	}
}

func TestOpenAIStreamContinuationMultipleAttemptsThroughGatewayParsers(t *testing.T) {
	gin.SetMode(gin.TestMode)
	for _, passthrough := range []bool{false, true} {
		t.Run(fmt.Sprint("passthrough=", passthrough), func(t *testing.T) {
			ctx := upstreamRetryTestContext(t, context.Background(), "503", 2)
			attempts := 0
			resp, err := DoWithConfiguredUpstreamRetry(continuationRequest(t, ctx), func(r *http.Request) (*http.Response, error) {
				attempts++
				body, err := io.ReadAll(r.Body)
				require.NoError(t, err)
				_ = r.Body.Close()
				require.Equal(t, attempts, len(gjson.GetBytes(body, "input").Array()))
				for i := 1; i < attempts; i++ {
					require.Equal(t, fmt.Sprint(i), gjson.GetBytes(body, fmt.Sprintf("input.%d.content.0.text", i)).String())
				}
				terminal := continuationOverload
				if attempts == 3 {
					terminal = `{"type":"response.completed","sequence_number":4,"response":{"id":"resp_3","status":"completed","output":[{"type":"message","id":"msg_3","role":"assistant","content":[{"type":"output_text","text":"3","annotations":[]}]}],"usage":{"input_tokens":10,"output_tokens":1}}}`
				} else {
					terminal = strings.Replace(terminal, `"error":`, `"usage":{"input_tokens":10,"output_tokens":1},"error":`, 1)
				}
				return continuationResponse(continuationSSE(
					fmt.Sprintf(`{"type":"response.created","sequence_number":0,"response":{"id":"resp_%d"}}`, attempts),
					fmt.Sprintf(`{"type":"response.output_item.added","sequence_number":1,"output_index":0,"item":{"type":"message","id":"msg_%d","role":"assistant","content":[]}}`, attempts),
					fmt.Sprintf(`{"type":"response.output_text.delta","sequence_number":2,"output_index":0,"content_index":0,"item_id":"msg_%d","delta":"%d"}`, attempts, attempts),
					fmt.Sprintf(`{"type":"response.output_text.done","sequence_number":3,"output_index":0,"content_index":0,"item_id":"msg_%d","text":"%d"}`, attempts, attempts),
					terminal,
				)), nil
			})
			require.NoError(t, err)
			defer resp.Body.Close()
			recorder := httptest.NewRecorder()
			c, _ := gin.CreateTestContext(recorder)
			c.Request = httptest.NewRequest(http.MethodPost, "/v1/responses", nil).WithContext(ctx)
			svc := &OpenAIGatewayService{cfg: &config.Config{Gateway: config.GatewayConfig{MaxLineSize: defaultMaxLineSize}}}
			account := &Account{ID: 1, Platform: PlatformOpenAI, Type: AccountTypeOAuth}
			var usage *OpenAIUsage
			var responseID string
			if passthrough {
				result, forwardErr := svc.handleStreamingResponsePassthrough(ctx, resp, c, account, time.Now(), "test", "test")
				require.NoError(t, forwardErr)
				usage, responseID = result.usage, result.responseID
			} else {
				result, forwardErr := svc.handleStreamingResponse(ctx, resp, c, account, time.Now(), "test", "test")
				require.NoError(t, forwardErr)
				usage, responseID = result.usage, result.responseID
			}
			require.Equal(t, 3, attempts)
			require.Equal(t, 30, usage.InputTokens)
			require.Equal(t, 3, usage.OutputTokens)
			require.Equal(t, "resp_1", responseID)
			stream := recorder.Body.String()
			require.NotContains(t, stream, "response.failed")
			require.Equal(t, 3, strings.Count(stream, "event: response.output_text.done"), "已完成文本事件不能在续接时再发一次")
			var sequence int64 = -1
			for _, frame := range strings.Split(stream, "\n\n") {
				payload, kind := continuationEvent([]byte(frame))
				if !gjson.ValidBytes(payload) {
					continue
				}
				require.Greater(t, gjson.GetBytes(payload, "sequence_number").Int(), sequence)
				sequence = gjson.GetBytes(payload, "sequence_number").Int()
				if kind == "response.output_text.delta" {
					index := gjson.GetBytes(payload, "output_index").Int()
					require.Equal(t, fmt.Sprint(index+1), gjson.GetBytes(payload, "delta").String())
				}
				if kind == "response.completed" {
					require.Len(t, gjson.GetBytes(payload, "response.output").Array(), 3)
				}
			}
		})
	}
}

func TestOpenAIStreamContinuationSharesHTTPBudget(t *testing.T) {
	ctx := upstreamRetryTestContext(t, context.Background(), "503", 2)
	attempts := 0
	resp, err := DoWithConfiguredUpstreamRetry(continuationRequest(t, ctx), func(r *http.Request) (*http.Response, error) {
		attempts++
		_ = r.Body.Close()
		if attempts == 2 {
			return &http.Response{StatusCode: 503, Body: io.NopCloser(strings.NewReader("busy")), Header: http.Header{}}, nil
		}
		return continuationResponse(continuationSSE(continuationOverload)), nil
	})
	require.NoError(t, err)
	defer resp.Body.Close()
	body, err := io.ReadAll(resp.Body)
	require.NoError(t, err)
	require.Equal(t, 3, attempts, "HTTP 重试和 SSE 续接不能各自重新计算预算")
	require.Equal(t, 1, strings.Count(string(body), "event: response.failed"))
}

func TestOpenAIStreamContinuationRetryAfterBounds(t *testing.T) {
	ctx := upstreamRetryTestContext(t, context.Background(), "503", 2)
	b := &openAIContinuationBody{state: upstreamErrorRetryFromContext(ctx), headers: http.Header{}}
	for _, test := range []struct {
		value   string
		allowed bool
		want    time.Duration
	}{
		{"0", true, 0},
		{time.Now().Add(-time.Hour).UTC().Format(http.TimeFormat), true, 0},
		{"60", true, time.Minute},
		{"61", false, 0},
		{"184467440737095516160", false, 0},
		{"invalid", true, 100 * time.Millisecond},
	} {
		payload, err := json.Marshal(map[string]any{"error": map[string]any{"headers": map[string]string{"Retry-After": test.value}}})
		require.NoError(t, err)
		delay, allowed := b.retryDelay(payload)
		require.Equal(t, test.allowed, allowed, test.value)
		if allowed {
			require.Equal(t, test.want, delay, test.value)
		}
	}
	b.waited = 119 * time.Second
	_, allowed := b.retryDelay([]byte(`{"headers":{"retry-after":"2"}}`))
	require.False(t, allowed)
}

func TestOpenAIStreamContinuationBareFinalErrorRetainsUsage(t *testing.T) {
	ctx := upstreamRetryTestContext(t, context.Background(), "503", 1)
	attempts := 0
	resp, err := DoWithConfiguredUpstreamRetry(continuationRequest(t, ctx), func(r *http.Request) (*http.Response, error) {
		attempts++
		_ = r.Body.Close()
		if attempts == 1 {
			failure := strings.Replace(continuationOverload, `"error":`, `"usage":{"input_tokens":10,"output_tokens":2},"error":`, 1)
			return continuationResponse(continuationSSE(failure)), nil
		}
		return continuationResponse(continuationSSE(`{"type":"error","sequence_number":0,"error":{"code":"server_is_overloaded","message":"Our servers are currently overloaded","headers":{"Retry-After":"0"}},"usage":{"input_tokens":5,"output_tokens":1}}`)), nil
	})
	require.NoError(t, err)
	defer resp.Body.Close()
	body, err := io.ReadAll(resp.Body)
	require.NoError(t, err)
	foundError := false
	for _, frame := range strings.Split(string(body), "\n\n") {
		payload, kind := continuationEvent([]byte(frame))
		if kind == "error" {
			foundError = true
			usage, ok := extractOpenAIUsageFromJSONBytes(payload)
			require.True(t, ok)
			require.Equal(t, 15, usage.InputTokens)
			require.Equal(t, 3, usage.OutputTokens)
		}
	}
	require.True(t, foundError, "重试耗尽后必须保留最终错误")
	require.Equal(t, 2, attempts)
}

func TestOpenAIStreamContinuationConcurrentCloseUnblocksRead(t *testing.T) {
	parent, cancel := context.WithCancel(context.Background())
	defer cancel()
	ctx := upstreamRetryTestContext(t, parent, "503", 1)
	reader, writer := io.Pipe()
	defer writer.Close()
	resp, err := DoWithConfiguredUpstreamRetry(continuationRequest(t, ctx), func(r *http.Request) (*http.Response, error) {
		_ = r.Body.Close()
		return &http.Response{StatusCode: 200, Header: http.Header{"Content-Type": {"text/event-stream"}}, Body: reader}, nil
	})
	require.NoError(t, err)
	done := make(chan error, 1)
	go func() { _, readErr := io.ReadAll(resp.Body); done <- readErr }()
	var wg sync.WaitGroup
	for i := 0; i < 8; i++ {
		wg.Add(1)
		go func() { defer wg.Done(); _ = resp.Body.Close() }()
	}
	cancel()
	wg.Wait()
	select {
	case err := <-done:
		require.Error(t, err)
	case <-time.After(time.Second):
		t.Fatal("并发关闭未终止阻塞中的上游读取")
	}
}

func TestOpenAIStreamContinuationSkipsStatefulRequests(t *testing.T) {
	for _, extra := range []string{`"conversation":"conv_1"`, `"background":true`} {
		ctx := upstreamRetryTestContext(t, context.Background(), "503", 1)
		req, err := http.NewRequestWithContext(WithHTTPUpstreamProfile(ctx, HTTPUpstreamProfileOpenAI), http.MethodPost, "https://upstream.example/v1/responses", strings.NewReader(`{"stream":true,"input":"hello",`+extra+`}`))
		require.NoError(t, err)
		attempts := 0
		stream := continuationSSE(continuationOverload)
		resp, err := DoWithConfiguredUpstreamRetry(req, func(r *http.Request) (*http.Response, error) {
			attempts++
			_ = r.Body.Close()
			return continuationResponse(stream), nil
		})
		require.NoError(t, err)
		body, err := io.ReadAll(resp.Body)
		_ = resp.Body.Close()
		require.NoError(t, err)
		require.Equal(t, stream, string(body))
		require.Equal(t, 1, attempts)
	}
}
