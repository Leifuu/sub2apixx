package service

import (
	"testing"

	"github.com/Wei-Shaw/sub2api/internal/pkg/tlsfingerprint"
	"github.com/stretchr/testify/require"
)

func TestTLSProfileForPluginPreservesCodexProfile(t *testing.T) {
	profile := tlsfingerprint.NewMacCodexProfile()
	converted := tlsProfileForPlugin(profile)

	require.NotNil(t, converted)
	require.Equal(t, profile.Name, converted.Name)
	require.Equal(t, uint16sToUint32s(profile.CipherSuites), converted.CipherSuites)
	require.Equal(t, uint16sToUint32s(profile.Curves), converted.Curves)
	require.Equal(t, uint16sToUint32s(profile.PointFormats), converted.PointFormats)
	require.Equal(t, profile.EnableGREASE, converted.EnableGrease)
	require.Equal(t, uint16sToUint32s(profile.SignatureAlgorithms), converted.SignatureAlgorithms)
	require.Equal(t, profile.ALPNProtocols, converted.AlpnProtocols)
	require.Equal(t, uint16sToUint32s(profile.SupportedVersions), converted.SupportedVersions)
	require.Equal(t, uint16sToUint32s(profile.KeyShareGroups), converted.KeyShareGroups)
	require.Equal(t, uint16sToUint32s(profile.PSKModes), converted.PskModes)
	require.Equal(t, uint16sToUint32s(profile.Extensions), converted.Extensions)
}

func TestTLSProfileForPluginOmitsProfileWhenUnset(t *testing.T) {
	require.Nil(t, tlsProfileForPlugin(nil))
}
