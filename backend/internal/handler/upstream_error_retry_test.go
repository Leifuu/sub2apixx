//go:build unit

package handler

import (
	"bytes"
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/Wei-Shaw/sub2api/internal/config"
	"github.com/Wei-Shaw/sub2api/internal/pkg/tlsfingerprint"
	"github.com/Wei-Shaw/sub2api/internal/server/middleware"
	"github.com/Wei-Shaw/sub2api/internal/service"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
)

type configuredRetrySettingRepo struct{ service.SettingRepository }

func (*configuredRetrySettingRepo) GetValue(_ context.Context, key string) (string, error) {
	if key == service.SettingKeyUpstreamErrorRetry {
		return `{"enabled":true,"max_retries":2,"delay_ms":100,"errors":"currently overloaded"}`, nil
	}
	return "", service.ErrSettingNotFound
}

type configuredRetryHTTPUpstream struct {
	calls          int
	streamFailure  bool
	exhaustAccount bool
	accountIDs     []int64
}

func (u *configuredRetryHTTPUpstream) Do(req *http.Request, _ string, accountID int64, _ int) (*http.Response, error) {
	return service.DoWithConfiguredUpstreamRetry(req, func(r *http.Request) (*http.Response, error) {
		u.calls++
		u.accountIDs = append(u.accountIDs, accountID)
		if r.Body != nil {
			_, _ = io.Copy(io.Discard, r.Body)
			_ = r.Body.Close()
		}
		status := 200
		body := "event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"delta\":\"hello\"}\n\nevent: response.completed\ndata: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_ok\",\"status\":\"completed\",\"output\":[{\"type\":\"message\",\"content\":[{\"type\":\"output_text\",\"text\":\"hello\"}]}],\"usage\":{\"input_tokens\":1,\"output_tokens\":1}}}\n\n"
		contentType := "text/event-stream"
		if (!u.exhaustAccount && u.calls <= 2) || (u.exhaustAccount && accountID == u.accountIDs[0]) {
			body = `{"error":{"code":"custom_busy","message":"Our servers are CURRENTLY OVERLOADED. Please try again later."}}`
			status, contentType = 503, "application/json"
			if u.streamFailure {
				status, contentType = 200, "text/event-stream"
				body = "event: response.created\ndata: {\"type\":\"response.created\",\"response\":{\"id\":\"failed\"}}\n\nevent: error\ndata: " + `{"type":"error","error":{"code":"custom_busy","message":"Our servers are CURRENTLY OVERLOADED. Please try again later."}}` + "\n\n"
				// OAuth 在真正生成内容前会发送 metadata，不能将其当成输出提交。
				body = "event: response.metadata\ndata: {\"type\":\"response.metadata\",\"sequence_number\":0}\n\n" + body
				if u.exhaustAccount {
					body = strings.ReplaceAll(body, "custom_busy", "server_error")
					body += "event: response.failed\ndata: " + `{"type":"response.failed","response":{"id":"failed","status":"failed","error":{"code":"server_error","type":"service_unavailable_error","message":"Our servers are CURRENTLY OVERLOADED. Please try again later."}}}` + "\n\n"
				}
			}
		}
		return &http.Response{StatusCode: status, Header: http.Header{"Content-Type": {contentType}}, Body: io.NopCloser(strings.NewReader(body))}, nil
	})
}
func (u *configuredRetryHTTPUpstream) DoWithTLS(req *http.Request, proxy string, account int64, concurrency int, _ *tlsfingerprint.Profile) (*http.Response, error) {
	return u.Do(req, proxy, account, concurrency)
}

// Exercise the complete Responses handler: two upstream failures must produce
// one successful client response and release every acquired concurrency slot.
func TestUpstreamErrorRetryResponsesRecoversWithinOneClientRequest(t *testing.T) {
	gin.SetMode(gin.TestMode)
	for _, test := range []struct {
		name                                        string
		streamFailure, exhaustAccount, largeRequest bool
	}{
		{"http_error", false, false, false},
		{"sse_error", true, false, false},
		{"exhausted_sse_switches_account", true, true, false},
		{"large_sse_switches_account", true, true, true},
		{"exhausted_http_switches_account", false, true, false},
	} {
		t.Run(test.name, func(t *testing.T) {
			cfg := &config.Config{RunMode: config.RunModeSimple}
			cfg.Gateway.MaxAccountSwitches = 0
			repo := &grokCredentialHandlerRepo{accounts: []service.Account{{ID: 801, Name: "test", Platform: service.PlatformOpenAI, Type: service.AccountTypeAPIKey, Status: service.StatusActive, Schedulable: true, Concurrency: 1, Credentials: map[string]any{"api_key": "test"}}}}
			if test.exhaustAccount {
				cfg.Gateway.MaxAccountSwitches = 1
				// 禁用账号级额外重试，明确验证管理员预算耗尽后的切号。
				repo.accounts[0].Credentials["pool_mode"] = true
				repo.accounts[0].Credentials["pool_mode_retry_count"] = 0
				other := repo.accounts[0]
				other.ID = 802
				repo.accounts = append(repo.accounts, other)
			}
			upstream := &configuredRetryHTTPUpstream{streamFailure: test.streamFailure, exhaustAccount: test.exhaustAccount}
			billingCache := service.NewBillingCacheService(nil, nil, nil, nil, nil, nil, cfg, nil)
			gateway := service.NewOpenAIGatewayService(repo, nil, nil, nil, nil, nil, nil, cfg, nil, nil, service.NewBillingService(cfg, nil), nil, billingCache, upstream, &service.DeferredService{}, nil, nil, nil, nil, nil, nil, nil)
			defer billingCache.Stop()
			defer gateway.CloseOpenAIWSPool()
			acquired := 0
			cache := &concurrencyCacheMock{
				acquireUserSlotFn:    func(context.Context, int64, int, string) (bool, error) { return true, nil },
				acquireAccountSlotFn: func(context.Context, int64, int, string) (bool, error) { acquired++; return true, nil },
			}
			h := NewOpenAIGatewayHandler(gateway, service.NewConcurrencyService(cache), billingCache, &service.APIKeyService{}, nil, nil, nil, nil, cfg)
			groupID := int64(901)
			key := &service.APIKey{ID: 902, GroupID: &groupID, User: &service.User{ID: 903, Status: service.StatusActive}, Group: &service.Group{ID: groupID, Platform: service.PlatformOpenAI, Status: service.StatusActive}}
			router := gin.New()
			router.Use(middleware.UpstreamErrorRetry(service.NewSettingService(&configuredRetrySettingRepo{}, cfg)))
			router.Use(func(c *gin.Context) {
				c.Set(string(middleware.ContextKeyAPIKey), key)
				c.Set(string(middleware.ContextKeyUser), middleware.AuthSubject{UserID: 903, Concurrency: 1})
				c.Next()
			})
			router.POST("/v1/responses", h.Responses)
			input := "hello"
			if test.largeRequest {
				input = strings.Repeat("x", (8<<20)+1)
			}
			req := httptest.NewRequest(http.MethodPost, "/v1/responses", bytes.NewBufferString(`{"model":"gpt-5","input":"`+input+`","stream":true}`))
			req.Header.Set("Content-Type", "application/json")
			rec := httptest.NewRecorder()
			router.ServeHTTP(rec, req)
			require.Equal(t, http.StatusOK, rec.Code, rec.Body.String())
			wantCalls := 3
			if test.exhaustAccount {
				wantCalls = 4
				require.Len(t, upstream.accountIDs, wantCalls)
				require.NotEqual(t, upstream.accountIDs[0], upstream.accountIDs[len(upstream.accountIDs)-1])
				require.Equal(t, []int64{upstream.accountIDs[0], upstream.accountIDs[0], upstream.accountIDs[0]}, upstream.accountIDs[:3])
				require.NotContains(t, rec.Body.String(), "response.metadata", "切号必须丢弃失败账号的暂存事件")
			}
			require.Equal(t, wantCalls, upstream.calls)
			require.Equal(t, 1, strings.Count(rec.Body.String(), "event: response.completed"))
			require.NotContains(t, rec.Body.String(), "custom_busy")
			require.NotContains(t, rec.Body.String(), "currently overloaded")
			require.NotContains(t, rec.Body.String(), "response.failed")
			require.Empty(t, repo.setErrorIDs)
			require.Empty(t, repo.setTempIDs)
			require.Equal(t, int32(acquired), atomic.LoadInt32(&cache.releaseAccountCalled))
		})
	}
}
