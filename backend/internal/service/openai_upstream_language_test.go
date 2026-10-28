package service

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
)

func TestOpenAIUpstreamLanguageHTTPAndWS(t *testing.T) {
	for _, accountType := range []string{AccountTypeOAuth, AccountTypeAPIKey} {
		for _, language := range []string{"", "zh-CN,zh;q=0.9", "en-US"} {
			t.Run(accountType+"/"+language, func(t *testing.T) {
				account := &Account{ID: 95, Platform: PlatformOpenAI, Type: accountType,
					Credentials: map[string]any{"header_override_enabled": true,
						"header_overrides": map[string]any{"accept-language": "zh-TW", "oai-language": "zh-TW"}}}
				for _, path := range []string{"gateway", "test", "ws"} {
					t.Run(path, func(t *testing.T) {
						req := httptest.NewRequest(http.MethodPost, "https://example.com/v1/responses", nil)
						if language != "" {
							req.Header.Set("Accept-Language", language)
							req.Header["aCcEpT-LaNgUaGe"] = []string{"zh-TW", "en-GB"}
						}
						account.ApplyHeaderOverrides(req.Header)
						probe := &fingerprintTransportProbe{}
						var headers http.Header
						if path == "ws" {
							c, _ := gin.CreateTestContext(httptest.NewRecorder())
							c.Request = req
							var err error
							headers, _, err = (&OpenAIGatewayService{}).buildOpenAIWSHeaders(
								context.Background(), c, account, "token",
								OpenAIWSProtocolDecision{Transport: OpenAIUpstreamTransportResponsesWebsocketV2},
								true, "", "", "", "", "")
							require.NoError(t, err)
						} else {
							var response *http.Response
							var err error
							if path == "gateway" {
								response, err = (&OpenAIGatewayService{httpUpstream: probe}).doOpenAIUpstream(req, "", account)
							} else {
								response, err = (&AccountTestService{httpUpstream: probe}).doOpenAIAccountTestUpstream(req, "", account, false)
							}
							require.NoError(t, err)
							require.NoError(t, response.Body.Close())
							headers = req.Header
						}
						var values []string
						for name, value := range headers {
							if strings.EqualFold(name, "Accept-Language") {
								values = append(values, value...)
							}
						}
						require.Equal(t, []string{"ja-JP"}, values)
						require.Equal(t, []string{"ja-JP"}, headers.Values("Oai-Language"))
					})
				}
			})
		}
	}
	otherHeaders := http.Header{"Accept-Language": {"zh-CN"}}
	applyOpenAIUpstreamLanguage(otherHeaders, &Account{Platform: PlatformGrok})
	require.Equal(t, "zh-CN", otherHeaders.Get("Accept-Language"))
	quotaHeaders := buildCodexCommonHeaders("token", "account", false)
	require.Equal(t, "ja-JP", quotaHeaders["accept-language"])
	require.Equal(t, "ja-JP", quotaHeaders["oai-language"])
}
