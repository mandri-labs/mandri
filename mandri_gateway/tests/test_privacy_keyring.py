import base64
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from keyring.errors import KeyringLocked
from mandri.core.types.execution import ProtectionError
from mandri.gateway import privacy_key_lock, privacy_keys
from mandri.gateway.privacy_keys import FilePrivacyKey, KeyringPrivacyKey


class SecureBackend:
    __module__ = "keyring.backends.SecretService"

    def __init__(self):
        self.values = {}
        self.writes = 0

    def get_password(self, service, account):
        return self.values.get((service, account))

    def set_password(self, service, account, value):
        self.values[service, account] = value
        self.writes += 1


class InsecureBackend(SecureBackend):
    __module__ = "keyrings.alt.file"


def test_os_keyring_keeps_stable_installation_scoped_key(monkeypatch, tmp_path):
    backend = SecureBackend()
    monkeypatch.setattr(privacy_keys.keyring, "get_keyring", lambda: backend)
    provider = KeyringPrivacyKey(tmp_path / "installation-one")
    with pytest.raises(ProtectionError):
        provider.load()
    key = provider.load(create=True)
    assert len(key) == 32
    assert KeyringPrivacyKey(tmp_path / "installation-one").load() == key
    assert KeyringPrivacyKey(tmp_path / "installation-two").load(create=True) != key
    assert backend.writes == 2


def test_plaintext_keyring_is_never_an_implicit_fallback(monkeypatch, tmp_path):
    backend = InsecureBackend()
    monkeypatch.setattr(privacy_keys.keyring, "get_keyring", lambda: backend)
    with pytest.raises(ProtectionError, match="OS credential store"):
        KeyringPrivacyKey(tmp_path).load(create=True)
    assert backend.values == {}


def test_locked_and_invalid_stores_do_not_reset_keys(monkeypatch, tmp_path):
    backend = SecureBackend()
    monkeypatch.setattr(privacy_keys.keyring, "get_keyring", lambda: backend)
    provider = KeyringPrivacyKey(tmp_path)
    provider.load(create=True)
    before = dict(backend.values)
    monkeypatch.setattr(backend, "get_password", lambda *args: base64.b64encode(b"short").decode())
    with pytest.raises(ProtectionError, match="invalid length"):
        provider.load(create=True)
    assert backend.values == before

    def locked(*args):
        raise KeyringLocked("synthetic")

    monkeypatch.setattr(backend, "get_password", locked)
    with pytest.raises(ProtectionError, match="locked"):
        provider.load(create=True)
    assert backend.values == before


def test_windows_file_key_requires_os_store(monkeypatch, tmp_path):
    monkeypatch.setattr(privacy_keys, "sys", SimpleNamespace(platform="win32"))
    with pytest.raises(ProtectionError, match="OS credential store"):
        FilePrivacyKey(tmp_path / "key").load(create=True)
    assert not (tmp_path / "key").exists()


def test_concurrent_os_key_creation_cannot_replace_the_first_scope_key(monkeypatch, tmp_path):
    backend = SecureBackend()
    entered, contended, release = (threading.Event() for _ in range(3))
    original_read = backend.get_password
    original_lock = privacy_key_lock._try_lock

    def first_read(service, account):
        if not entered.is_set():
            entered.set()
            assert release.wait(3)
        return original_read(service, account)

    def observed_lock(handle):
        acquired = original_lock(handle)
        if not acquired:
            contended.set()
        return acquired

    monkeypatch.setattr(backend, "get_password", first_read)
    monkeypatch.setattr(privacy_keys.keyring, "get_keyring", lambda: backend)
    monkeypatch.setattr(privacy_key_lock, "_try_lock", observed_lock)
    first, second = KeyringPrivacyKey(tmp_path), KeyringPrivacyKey(tmp_path)
    with ThreadPoolExecutor(max_workers=2) as executor:
        one = executor.submit(first.load, create=True)
        try:
            assert entered.wait(3)
            two = executor.submit(second.load, create=True)
            assert contended.wait(3)
        finally:
            release.set()
        assert one.result(3) == two.result(3)
    assert backend.writes == 1
    assert first.load() == second.load()


def test_keyring_initialization_error_releases_the_creation_lock(monkeypatch, tmp_path):
    backend = SecureBackend()
    monkeypatch.setattr(privacy_keys.keyring, "get_keyring", lambda: backend)
    original_read = backend.get_password

    def locked(*args):
        raise KeyringLocked("synthetic")

    monkeypatch.setattr(backend, "get_password", locked)
    with pytest.raises(ProtectionError, match="locked"):
        KeyringPrivacyKey(tmp_path).load(create=True)
    monkeypatch.setattr(backend, "get_password", original_read)
    assert len(KeyringPrivacyKey(tmp_path).load(create=True)) == 32
    assert backend.writes == 1


@pytest.mark.skipif(os.name == "nt", reason="POSIX lock permissions")
def test_insecure_creation_lock_does_not_access_os_credentials(monkeypatch, tmp_path):
    path = tmp_path / ".privacy-key.lock"
    path.write_bytes(b"x")
    path.chmod(0o666)
    monkeypatch.setattr(
        privacy_keys.keyring,
        "get_keyring",
        lambda: pytest.fail("An invalid lock must fail before reading credentials"),
    )
    with pytest.raises(ProtectionError, match="key lock"):
        KeyringPrivacyKey(tmp_path).load(create=True)
