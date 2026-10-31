package service

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"testing"
	"time"

	"github.com/Wei-Shaw/sub2api/internal/config"
	"github.com/stretchr/testify/require"
)

type openAIAPIKeyHealthSettingRepo struct {
	SettingRepository
	value    string
	getCalls int
}

func (r *openAIAPIKeyHealthSettingRepo) GetValue(context.Context, string) (string, error) {
	r.getCalls++
	return r.value, nil
}

type openAIAPIKeyHealthAccountRepo struct {
	AccountRepository
	setCalls int
	reason   string
}

func (r *openAIAPIKeyHealthAccountRepo) SetTempUnschedulable(_ context.Context, _ int64, _ time.Time, reason string) error {
	r.setCalls++
	r.reason = reason
	return nil
}

type openAIAPIKeyHealthCacheStub struct {
	TempUnschedCache
	recordCalls int
	resetCalls  int
	setCalls    int
	tripped     bool
	counts      map[int64]int64
}

func (c *openAIAPIKeyHealthCacheStub) RecordOpenAIAPIKeyHealthFailure(_ context.Context, accountID int64, _ int, threshold int) (int64, bool, error) {
	c.recordCalls++
	if c.counts == nil {
		c.counts = make(map[int64]int64)
	}
	c.counts[accountID]++
	count := c.counts[accountID]
	return count, c.tripped || count >= int64(threshold), nil
}

func (c *openAIAPIKeyHealthCacheStub) ResetOpenAIAPIKeyHealthFailures(_ context.Context, accountID int64) error {
	c.resetCalls++
	if c.counts != nil {
		delete(c.counts, accountID)
	}
	return nil
}

func (c *openAIAPIKeyHealthCacheStub) SetTempUnsched(context.Context, int64, *TempUnschedState) error {
	c.setCalls++
	return nil
}

type openAIAPIKeyHealthRuntimeBlocker struct{ calls int }

func (b *openAIAPIKeyHealthRuntimeBlocker) BlockAccountScheduling(*Account, time.Time, string) {
	b.calls++
}
func (*openAIAPIKeyHealthRuntimeBlocker) ClearAccountSchedulingBlock(int64) {}

func openAIHealthPoolAccount() *Account {
	return &Account{
		ID:       42,
		Platform: PlatformOpenAI,
		Type:     AccountTypeAPIKey,
		Credentials: map[string]any{
			"pool_mode": true,
		},
	}
}

func openAIHealthOAuthAccount() *Account {
	return &Account{ID: 43, Platform: PlatformOpenAI, Type: AccountTypeOAuth}
}

func TestClassifyOpenAIAPIKeyHealthFailureExclusions(t *testing.T) {
	tests := []struct {
		name     string
		err      error
		eligible bool
	}{
		{name: "account attributed 502", err: &UpstreamFailoverError{StatusCode: http.StatusBadGateway}, eligible: true},
		{name: "request scoped capacity", err: &UpstreamFailoverError{StatusCode: 529, RequestScopedTransient: true}},
		{name: "provider scoped overload", err: &UpstreamFailoverError{StatusCode: 529, Scope: GatewayFailureScopeProvider}},
		{name: "account retryable rate limit", err: &UpstreamFailoverError{StatusCode: http.StatusTooManyRequests, RetryableOnSameAccount: true}, eligible: true},
		{name: "request scoped retryable capacity", err: &UpstreamFailoverError{StatusCode: 529, RetryableOnSameAccount: true, RequestScopedTransient: true}},
		{name: "credential disable path", err: &UpstreamFailoverError{StatusCode: http.StatusUnauthorized, Stage: GatewayFailureStageAccountAuth, Scope: GatewayFailureScopeAccount}},
		{name: "client request", err: &UpstreamFailoverError{StatusCode: http.StatusBadRequest}},
		{name: "responses stream EOF", err: fmt.Errorf("stream read error: %w", io.ErrUnexpectedEOF), eligible: true},
		{name: "client cancellation", err: fmt.Errorf("stream read error: %w", context.Canceled)},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, _, eligible := classifyOpenAIAPIKeyHealthFailure(tt.err)
			require.Equal(t, tt.eligible, eligible)
		})
	}
}

func TestOpenAIAPIKeyHealthBreakerDefaultEnabled(t *testing.T) {
	settings := NewSettingService(&openAIAPIKeyHealthSettingRepo{}, &config.Config{})
	cache := &openAIAPIKeyHealthCacheStub{tripped: true}
	svc := NewRateLimitService(&openAIAPIKeyHealthAccountRepo{}, nil, &config.Config{}, nil, cache)
	svc.SetSettingService(settings)
	svc.SetOpenAIAPIKeyHealthCache(cache)

	require.True(t, svc.ObserveOpenAIAPIKeyHealthFailure(context.Background(), openAIHealthOAuthAccount(), &UpstreamFailoverError{StatusCode: http.StatusBadGateway}))
	require.Equal(t, 1, cache.recordCalls)
}

func TestOpenAIAPIKeyHealthBreakerTripsPersistedAndRuntimeState(t *testing.T) {
	encoded, err := json.Marshal(OpenAIAPIKeyHealthBreakerSettings{Enabled: true, WindowMinutes: 1, FailureThreshold: 3, CooldownMinutes: 5})
	require.NoError(t, err)
	settings := NewSettingService(&openAIAPIKeyHealthSettingRepo{value: string(encoded)}, &config.Config{})
	cache := &openAIAPIKeyHealthCacheStub{tripped: true}
	repo := &openAIAPIKeyHealthAccountRepo{}
	blocker := &openAIAPIKeyHealthRuntimeBlocker{}
	svc := NewRateLimitService(repo, nil, &config.Config{}, nil, cache)
	svc.SetSettingService(settings)
	svc.SetOpenAIAPIKeyHealthCache(cache)
	svc.SetAccountRuntimeBlocker(blocker)
	account := openAIHealthPoolAccount()

	require.True(t, svc.ObserveOpenAIAPIKeyHealthFailure(context.Background(), account, &UpstreamFailoverError{StatusCode: http.StatusBadGateway, ResponseBody: []byte(`{"error":"upstream"}`)}))
	require.Equal(t, 1, cache.recordCalls)
	require.Equal(t, 1, cache.setCalls)
	require.Equal(t, 1, repo.setCalls)
	require.Equal(t, 1, blocker.calls)
	require.NotNil(t, account.TempUnschedulableUntil)
	require.Contains(t, repo.reason, openAIAPIKeyHealthBreakerReason)
}

func TestOpenAIAPIKeyHealthBreakerRequiresThreeConsecutiveFailures(t *testing.T) {
	encoded, err := json.Marshal(OpenAIAPIKeyHealthBreakerSettings{Enabled: true, WindowMinutes: 1, FailureThreshold: 3, CooldownMinutes: 5})
	require.NoError(t, err)
	settings := NewSettingService(&openAIAPIKeyHealthSettingRepo{value: string(encoded)}, &config.Config{})
	cache := &openAIAPIKeyHealthCacheStub{}
	repo := &openAIAPIKeyHealthAccountRepo{}
	svc := NewRateLimitService(repo, nil, &config.Config{}, nil, cache)
	svc.SetSettingService(settings)
	svc.SetOpenAIAPIKeyHealthCache(cache)
	account := openAIHealthOAuthAccount()
	errValue := &UpstreamFailoverError{StatusCode: http.StatusBadGateway}

	require.False(t, svc.ObserveOpenAIAPIKeyHealthFailure(context.Background(), account, errValue))
	require.False(t, svc.ObserveOpenAIAPIKeyHealthFailure(context.Background(), account, errValue))
	require.Equal(t, 0, repo.setCalls)
	require.True(t, svc.ObserveOpenAIAPIKeyHealthFailure(context.Background(), account, errValue))
	require.Equal(t, 1, repo.setCalls)
}

func TestOpenAIAPIKeyHealthSuccessClearsConsecutiveFailures(t *testing.T) {
	encoded, err := json.Marshal(OpenAIAPIKeyHealthBreakerSettings{Enabled: true, WindowMinutes: 1, FailureThreshold: 3, CooldownMinutes: 5})
	require.NoError(t, err)
	settings := NewSettingService(&openAIAPIKeyHealthSettingRepo{value: string(encoded)}, &config.Config{})
	cache := &openAIAPIKeyHealthCacheStub{}
	repo := &openAIAPIKeyHealthAccountRepo{}
	svc := NewRateLimitService(repo, nil, &config.Config{}, nil, cache)
	svc.SetSettingService(settings)
	svc.SetOpenAIAPIKeyHealthCache(cache)
	account := openAIHealthOAuthAccount()
	errValue := &UpstreamFailoverError{StatusCode: http.StatusBadGateway}

	require.False(t, svc.ObserveOpenAIAPIKeyHealthFailure(context.Background(), account, errValue))
	require.False(t, svc.ObserveOpenAIAPIKeyHealthFailure(context.Background(), account, errValue))
	svc.ObserveOpenAIAPIKeyHealthSuccess(context.Background(), account)
	require.Equal(t, 1, cache.resetCalls)
	require.False(t, svc.ObserveOpenAIAPIKeyHealthFailure(context.Background(), account, errValue))
	require.Equal(t, 0, repo.setCalls)
}
