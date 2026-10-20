"""Server credential fallback must not expose keys or leak across cloud users."""
import pytest
import ugc_credentials


@pytest.mark.parametrize('name', ['FAL_API_KEY', 'ELEVENLABS_API_KEY'])
def test_self_host_fallback_and_explicit_header_override(monkeypatch, name):
    monkeypatch.setenv(name, 'server-secret')
    assert ugc_credentials.resolve_key(name) == 'server-secret'
    assert ugc_credentials.resolve_key(name, 'browser-secret') == 'browser-secret'
    assert ugc_credentials.resolve_key(name, '', managed=True) == ''
    assert ugc_credentials.resolve_key(name, 'browser-secret', managed=True) == 'browser-secret'


def test_config_returns_presence_only_and_cloud_returns_nothing(monkeypatch):
    for name in ['FAL_API_KEY', 'ELEVENLABS_API_KEY', 'OPENROUTER_API_KEY']:
        monkeypatch.setenv(name, 'private-test-value')
    monkeypatch.setenv('OPENROUTER_MODEL', 'vendor/model')
    assert ugc_credentials.public_config() == {
        'fal': True, 'elevenlabs': True, 'openrouter': True, 'model': 'vendor/model',
    }
    assert 'private-test-value' not in str(ugc_credentials.public_config())
    assert ugc_credentials.public_config(managed=True) == {}


def test_unrelated_app_credentials_are_not_accepted():
    with pytest.raises(ValueError):
        ugc_credentials.resolve_key('DATABASE_URL')
