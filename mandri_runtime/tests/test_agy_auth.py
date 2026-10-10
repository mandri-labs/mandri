import base64
import ctypes
import json
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest
from mandri.core.ids import HarnessKind
from mandri.runtime import (
    agy_auth,
    agy_credentials_linux,
    agy_credentials_macos,
    agy_credentials_windows,
)
from mandri.runtime.native_catalog import discover_models
from mandri.runtime.native_command_catalog import discover_commands
from mandri.runtime.native_usage import read_native_usage
from mandri.runtime.usage_accounts import _agy_usage


@pytest.mark.skipif(sys.platform not in {"win32", "darwin"}, reason="Native desktop keychain")
def test_native_desktop_keychain_returns_no_credential_without_interaction():
    reader = agy_credentials_windows if sys.platform == "win32" else agy_credentials_macos
    assert reader.read_credential(service=f"mandri-test-{uuid4()}", account="missing") is None


@pytest.mark.parametrize(
    "platform,reader",
    [
        ("win32", "agy_credentials_windows"),
        ("darwin", "agy_credentials_macos"),
        ("linux", "agy_credentials_linux"),
    ],
)
def test_platform_credential_reader(monkeypatch, agy_oauth_token, platform, reader):
    monkeypatch.setattr(agy_auth.sys, "platform", platform)
    read = Mock(return_value=agy_oauth_token)
    monkeypatch.setattr(getattr(agy_auth, reader), "read_credential", read)
    assert agy_auth._stored_credential() == agy_oauth_token
    read.assert_called_once_with()


@pytest.mark.parametrize("argument", ["separate", "equals"])
async def test_file_credentials_use_selected_profile(
    monkeypatch, tmp_path, agy_oauth_token, argument
):
    monkeypatch.setattr(agy_auth, "_stored_credential", Mock(return_value=None))
    store = tmp_path / "profile/antigravity-cli"
    store.mkdir(parents=True)
    (store / "antigravity-oauth-token").write_text(agy_oauth_token, encoding="utf-8")
    argv = ["--gemini_dir", "profile"] if argument == "separate" else ["--gemini_dir=profile"]
    await agy_auth.require_agy_authentication(["agy", *argv], tmp_path)


async def test_file_fallback_marker_skips_unavailable_keyring(
    monkeypatch, tmp_path, agy_oauth_token
):
    read = Mock(side_effect=AssertionError("Must not access unavailable keyring"))
    monkeypatch.setattr(agy_auth, "_stored_credential", read)
    store = tmp_path / "antigravity-cli"
    (store / "cache").mkdir(parents=True)
    (store / "cache/antigravity-keyring-unavailable").touch()
    (store / "antigravity-oauth-token").write_text(agy_oauth_token, encoding="utf-8")
    await agy_auth.require_agy_authentication(["agy", "--gemini_dir", str(tmp_path)], tmp_path)
    read.assert_not_called()


@pytest.mark.parametrize(
    "raw",
    [None, "invalid", "[]", "{}", '{"auth_method":"enterprise","token":{}}'],
)
async def test_missing_or_unknown_credentials_refuse_authentication(monkeypatch, tmp_path, raw):
    monkeypatch.setattr(agy_auth, "_stored_credential", Mock(return_value=raw))
    with pytest.raises(PermissionError):
        await agy_auth.require_agy_authentication(["agy", "--gemini_dir", str(tmp_path)], tmp_path)


@pytest.mark.parametrize(
    "field,value",
    [
        ("access_token", ""),
        ("refresh_token", None),
        ("token_type", "unknown"),
        ("expiry", "2000-01-01T00:00:00Z"),
        ("expiry", "2099-01-01T00:00:00"),
        ("expiry", "invalid"),
    ],
)
async def test_unusable_token_is_rejected(monkeypatch, tmp_path, agy_oauth_token, field, value):
    payload = json.loads(agy_oauth_token)
    payload["token"][field] = value
    monkeypatch.setattr(agy_auth, "_stored_credential", lambda: json.dumps(payload))
    with pytest.raises(PermissionError):
        await agy_auth.require_agy_authentication(["agy"], tmp_path)


@pytest.mark.parametrize("probe", ["models", "commands", "quota", "legacy_quota"])
async def test_unauthenticated_passive_probes_never_spawn_agy(monkeypatch, tmp_path, probe):
    monkeypatch.setattr(agy_auth, "_stored_credential", Mock(return_value=None))
    monkeypatch.setattr(agy_auth, "default_agy_root", lambda: tmp_path)
    spawn = AsyncMock(side_effect=AssertionError("Unauthenticated agy must not be launched"))
    command = ["agy", "--gemini_dir", str(tmp_path)]
    with pytest.raises(PermissionError):
        if probe == "models":
            await discover_models(HarnessKind.AGY, command, tmp_path, {}, spawn)
        elif probe == "commands":
            await discover_commands(HarnessKind.AGY, command, tmp_path, {}, spawn)
        elif probe == "quota":
            await read_native_usage(HarnessKind.AGY, "profile", command, tmp_path, {}, spawn)
        else:
            monkeypatch.setattr("mandri.runtime.usage_accounts.spawn", spawn)
            await _agy_usage(tmp_path / "agy", tmp_path, {})
    spawn.assert_not_awaited()


@pytest.mark.parametrize("locked", ["collection", "item", None])
def test_linux_reads_login_collection_without_unlocking(monkeypatch, agy_oauth_token, locked):
    connection = Mock()
    item = Mock()
    item.is_locked.return_value = locked == "item"
    item.get_secret.return_value = agy_oauth_token.encode()
    collection = Mock(collection_path="/org/freedesktop/secrets/collection/login")
    collection.is_locked.return_value = locked == "collection"
    collection.search_items.return_value = iter([item])
    storage = SimpleNamespace(
        dbus_init=Mock(return_value=connection),
        get_all_collections=Mock(return_value=iter([collection])),
        Collection=Mock(side_effect=AssertionError("Must use login collection")),
    )
    monkeypatch.setattr(
        agy_credentials_linux.SecretService, "secretstorage", storage, raising=False
    )
    if locked:
        with pytest.raises(PermissionError):
            agy_credentials_linux.read_credential()
        item.get_secret.assert_not_called()
    else:
        assert agy_credentials_linux.read_credential() == agy_oauth_token
        collection.search_items.assert_called_once_with(
            {"service": "gemini", "username": "antigravity"}
        )
    collection.unlock.assert_not_called()
    item.unlock.assert_not_called()
    connection.close.assert_called_once()


def test_linux_missing_login_collection_queries_default_without_creating_it(monkeypatch):
    connection = Mock()
    collection = Mock()
    collection.is_locked.return_value = False
    collection.search_items.return_value = iter([])
    storage = SimpleNamespace(
        dbus_init=Mock(return_value=connection),
        get_all_collections=Mock(return_value=iter([])),
        Collection=Mock(return_value=collection),
        create_collection=Mock(side_effect=AssertionError("Must not create a collection")),
    )
    monkeypatch.setattr(
        agy_credentials_linux.SecretService, "secretstorage", storage, raising=False
    )
    assert agy_credentials_linux.read_credential() is None
    storage.Collection.assert_called_once_with(connection)
    storage.create_collection.assert_not_called()
    collection.unlock.assert_not_called()
    connection.close.assert_called_once()


@pytest.mark.parametrize("failure", [PermissionError("locked"), OSError("unavailable")])
async def test_inaccessible_keyring_prevents_launch_even_with_a_token_file(
    monkeypatch, tmp_path, agy_oauth_token, failure
):
    monkeypatch.setattr(agy_auth, "_stored_credential", Mock(side_effect=failure))
    store = tmp_path / "antigravity-cli"
    store.mkdir()
    (store / "antigravity-oauth-token").write_text(agy_oauth_token, encoding="utf-8")
    with pytest.raises(PermissionError):
        await agy_auth.require_agy_authentication(["agy", "--gemini_dir", str(tmp_path)], tmp_path)


def test_windows_credential_is_utf8_and_freed(monkeypatch, agy_oauth_token):
    encoded = agy_oauth_token.encode()
    blob = ctypes.create_string_buffer(encoded)
    credential = agy_credentials_windows._Credential()
    credential.CredentialBlobSize = len(encoded)
    credential.CredentialBlob = ctypes.cast(blob, ctypes.POINTER(ctypes.c_ubyte))
    api = Mock()

    def read(target, kind, flags, result):
        assert (target, kind, flags) == ("gemini:antigravity", 1, 0)
        pointer = ctypes.POINTER(agy_credentials_windows._Credential)
        ctypes.cast(result, ctypes.POINTER(pointer))[0] = ctypes.pointer(credential)
        return True

    api.CredReadW.side_effect = read
    monkeypatch.setattr(agy_credentials_windows.sys, "platform", "win32")
    monkeypatch.setattr(
        agy_credentials_windows.ctypes, "WinDLL", Mock(return_value=api), raising=False
    )
    assert agy_credentials_windows.read_credential() == agy_oauth_token
    api.CredFree.assert_called_once()


@pytest.mark.parametrize("encoding", ["plain", "hex", "base64"])
def test_macos_decodes_go_keyring_secrets(agy_oauth_token, encoding):
    raw = agy_oauth_token
    if encoding == "hex":
        raw = "go-keyring-encoded:" + raw.encode().hex()
    elif encoding == "base64":
        raw = "go-keyring-base64:" + base64.b64encode(raw.encode()).decode()
    assert agy_credentials_macos._decode_credential(raw) == agy_oauth_token


@pytest.mark.parametrize("status", [0, -25300, -25308])
def test_macos_keychain_query_forbids_ui_and_releases_resources(
    monkeypatch, agy_oauth_token, status
):
    payload = ctypes.create_string_buffer(agy_oauth_token.encode())
    core = Mock()
    core.CFStringCreateWithCString.side_effect = [101, 102]
    core.CFDictionaryCreate.return_value = 103
    core.CFDataGetLength.return_value = len(agy_oauth_token.encode())
    core.CFDataGetBytePtr.return_value = ctypes.addressof(payload)
    security = Mock()

    def lookup(query, result):
        assert query == 103
        if status == 0:
            ctypes.cast(result, ctypes.POINTER(ctypes.c_void_p))[0] = 104
        return status

    security.SecItemCopyMatching.side_effect = lookup
    monkeypatch.setattr(agy_credentials_macos.ctypes, "CDLL", Mock(side_effect=[security, core]))
    names = {}

    def constant(library, name):
        return names.setdefault(name, len(names) + 1000)

    monkeypatch.setattr(agy_credentials_macos, "_constant", constant)
    if status == -25308:
        with pytest.raises(OSError):
            agy_credentials_macos.read_credential()
    else:
        result = agy_credentials_macos.read_credential()
        assert result == (agy_oauth_token if status == 0 else None)
    query = core.CFDictionaryCreate.call_args.args
    pairs = dict(zip(query[1], query[2], strict=True))
    assert pairs[names["kSecUseAuthenticationUI"]] == names["kSecUseAuthenticationUIFail"]
    assert pairs[names["kSecAttrService"]] == 101
    assert pairs[names["kSecAttrAccount"]] == 102
    released = [call.args[0] for call in core.CFRelease.call_args_list]
    assert released == ([104, 103, 102, 101] if status == 0 else [103, 102, 101])
