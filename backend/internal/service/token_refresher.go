package service

import (
	"context"
	"fmt"
	"log/slog"
	"strings"
	"time"
)

const (
	openAICLITokenRefreshWindow   = 5 * time.Minute
	openAICLITokenRefreshInterval = 8 * 24 * time.Hour
	openAILastRefreshAtExtraKey   = "openai_last_refresh_at"
)

// TokenRefresher 定义平台特定的token刷新策略接口
// 通过此接口可以扩展支持不同平台（Anthropic/OpenAI/Gemini）
type TokenRefresher interface {
	// CanRefresh 检查此刷新器是否能处理指定账号
	CanRefresh(account *Account) bool

	// NeedsRefresh 检查账号的token是否需要刷新
	NeedsRefresh(account *Account, refreshWindow time.Duration) bool

	// Refresh 执行token刷新，返回更新后的credentials
	// 注意：返回的map应该保留原有credentials中的所有字段，只更新token相关字段
	Refresh(ctx context.Context, account *Account) (map[string]any, error)
}

// ClaudeTokenRefresher 处理Anthropic/Claude OAuth token刷新
type ClaudeTokenRefresher struct {
	oauthService *OAuthService
}

// NewClaudeTokenRefresher 创建Claude token刷新器
func NewClaudeTokenRefresher(oauthService *OAuthService) *ClaudeTokenRefresher {
	return &ClaudeTokenRefresher{
		oauthService: oauthService,
	}
}

// CacheKey 返回用于分布式锁的缓存键
func (r *ClaudeTokenRefresher) CacheKey(account *Account) string {
	return ClaudeTokenCacheKey(account)
}

// CanRefresh 检查是否能处理此账号
// 处理 anthropic 平台的 oauth 与 setup-token 类型账号。
// 两者的 access_token 均为短期令牌（expires_in=28800，即 8h），到期都需刷新；
// setup-token 之前被排除会导致其 access_token 过期后请求 401。
// 此处与手动刷新入口（account.IsOAuth()）保持一致，实际是否刷新由 NeedsRefresh
// 基于 expires_at 门控，并在分布式锁保护下执行，不会造成过度刷新。
func (r *ClaudeTokenRefresher) CanRefresh(account *Account) bool {
	return account.Platform == PlatformAnthropic && account.IsOAuth()
}

// NeedsRefresh 检查token是否需要刷新
// 基于 expires_at 字段判断是否在刷新窗口内
func (r *ClaudeTokenRefresher) NeedsRefresh(account *Account, refreshWindow time.Duration) bool {
	expiresAt := account.GetCredentialAsTime("expires_at")
	if expiresAt == nil {
		return false
	}
	return time.Until(*expiresAt) < refreshWindow
}

// Refresh 执行token刷新
// 保留原有credentials中的所有字段，只更新token相关字段
func (r *ClaudeTokenRefresher) Refresh(ctx context.Context, account *Account) (map[string]any, error) {
	tokenInfo, err := r.oauthService.RefreshAccountToken(ctx, account)
	if err != nil {
		return nil, err
	}

	newCredentials := BuildClaudeAccountCredentials(tokenInfo)
	newCredentials = MergeCredentials(account.Credentials, newCredentials)

	return newCredentials, nil
}

// OpenAITokenRefresher 处理 OpenAI OAuth token刷新
type OpenAITokenRefresher struct {
	openaiOAuthService *OpenAIOAuthService
	accountRepo        AccountRepository
}

// NewOpenAITokenRefresher 创建 OpenAI token刷新器
func NewOpenAITokenRefresher(openaiOAuthService *OpenAIOAuthService, accountRepo AccountRepository) *OpenAITokenRefresher {
	return &OpenAITokenRefresher{
		openaiOAuthService: openaiOAuthService,
		accountRepo:        accountRepo,
	}
}

// CacheKey 返回用于分布式锁的缓存键
func (r *OpenAITokenRefresher) CacheKey(account *Account) string {
	return OpenAITokenCacheKey(account)
}

// CanRefresh 检查是否能处理此账号
func (r *OpenAITokenRefresher) CanRefresh(account *Account) bool {
	if account.IsCredentialShadow() {
		return false
	}
	return account.Platform == PlatformOpenAI && account.Type == AccountTypeOAuth
}

// NeedsRefresh follows the ChatGPT Codex CLI policy for OpenAI OAuth tokens.
// The generic refresh window is intentionally ignored so other providers keep
// their existing background refresh policy.
func (r *OpenAITokenRefresher) NeedsRefresh(account *Account, _ time.Duration) bool {
	return needsOpenAICLITokenRefresh(account, time.Now())
}

func needsOpenAICLITokenRefresh(account *Account, now time.Time) bool {
	if account == nil {
		return false
	}
	if account.IsOpenAIPersonalAccessToken() {
		return false
	}
	if strings.TrimSpace(account.GetOpenAIRefreshToken()) == "" {
		return false
	}
	expiresAt := account.GetCredentialAsTime("expires_at")
	if expiresAt != nil {
		return !expiresAt.After(now.Add(openAICLITokenRefreshWindow))
	}
	if account.IsRateLimited() {
		return true
	}

	lastRefreshRaw, ok := account.Extra[openAILastRefreshAtExtraKey]
	if !ok {
		return true
	}
	lastRefresh, err := time.Parse(time.RFC3339, strings.TrimSpace(fmt.Sprint(lastRefreshRaw)))
	if err != nil {
		return true
	}
	return lastRefresh.Before(now.Add(-openAICLITokenRefreshInterval))
}

func recordOpenAITokenRefresh(ctx context.Context, repo AccountRepository, account *Account) {
	if repo == nil || account == nil || account.Platform != PlatformOpenAI || account.Type != AccountTypeOAuth || account.IsOpenAIPersonalAccessToken() {
		return
	}

	refreshedAt := time.Now().UTC().Format(time.RFC3339)
	if err := repo.UpdateExtra(ctx, account.ID, map[string]any{openAILastRefreshAtExtraKey: refreshedAt}); err != nil {
		slog.Warn("openai_token_refresh_timestamp_persist_failed", "account_id", account.ID, "error", err)
		return
	}
	if account.Extra == nil {
		account.Extra = make(map[string]any)
	}
	account.Extra[openAILastRefreshAtExtraKey] = refreshedAt
}

// Refresh 执行token刷新
// 保留原有credentials中的所有字段，只更新token相关字段
func (r *OpenAITokenRefresher) Refresh(ctx context.Context, account *Account) (map[string]any, error) {
	tokenInfo, err := r.openaiOAuthService.RefreshAccountToken(ctx, account)
	if err != nil {
		return nil, err
	}

	// 使用服务提供的方法构建新凭证，并保留原有字段
	newCredentials := r.openaiOAuthService.BuildAccountCredentials(tokenInfo)
	newCredentials = MergeCredentials(account.Credentials, newCredentials)
	newCredentials = NormalizeOpenAIPersonalAccessTokenCredentials(account, tokenInfo, newCredentials)

	return newCredentials, nil
}
