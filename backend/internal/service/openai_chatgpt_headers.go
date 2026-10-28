package service

import (
	"context"
	"net/http"
	"strings"
)

const openAIUpstreamLanguageTag = "ja-JP"

// applyOpenAIUpstreamLanguage 固定 OpenAI 出站语言，清除客户端或账号覆写产生的重复头。
func applyOpenAIUpstreamLanguage(headers http.Header, account *Account) {
	if headers == nil || account == nil || account.Platform != PlatformOpenAI {
		return
	}
	for name := range headers {
		if strings.EqualFold(name, "Accept-Language") || strings.EqualFold(name, "Oai-Language") {
			delete(headers, name)
		}
	}
	headers.Set("Accept-Language", openAIUpstreamLanguageTag)
	headers.Set("Oai-Language", openAIUpstreamLanguageTag)
}

func setOpenAIChatGPTAccountHeaders(headers http.Header, account *Account) {
	if headers == nil || account == nil || !account.IsOpenAIOAuthLike() {
		return
	}
	if chatgptAccountID := account.GetChatGPTAccountID(); chatgptAccountID != "" {
		headers.Set("chatgpt-account-id", chatgptAccountID)
	}
	if account.IsChatGPTAccountFedRAMP() {
		headers.Set("x-openai-fedramp", "true")
	} else {
		headers.Del("x-openai-fedramp")
	}
}

// resolveAndSetOpenAIChatGPTAccountHeaders 解析 spark 影子账号至其母账号（凭据透传），
// 再调用 setOpenAIChatGPTAccountHeaders 写入 chatgpt-account-id / x-openai-fedramp 头。
// 普通账号（非影子）为直通，行为与直接调用 setOpenAIChatGPTAccountHeaders 一致。
func resolveAndSetOpenAIChatGPTAccountHeaders(ctx context.Context, repo AccountRepository, headers http.Header, account *Account) error {
	credAccount, err := resolveCredentialAccount(ctx, repo, account)
	if err != nil {
		return err
	}
	setOpenAIChatGPTAccountHeaders(headers, credAccount)
	return nil
}
