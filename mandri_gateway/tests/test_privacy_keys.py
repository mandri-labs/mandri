import os
from pathlib import Path

import pytest
from mandri.core.types.execution import ProtectionError
from mandri.gateway.privacy_keys import FilePrivacyKey

pytestmark = pytest.mark.skipif(os.name == "nt", reason="Private file keys require POSIX ownership")


def test_explicit_file_key_is_private_and_stable(tmp_path: Path) -> None:
    path = tmp_path / "keys" / "privacy.key"
    provider = FilePrivacyKey(path)
    with pytest.raises(ProtectionError):
        provider.load()
    key = provider.load(create=True)
    assert len(key) == 32
    assert path.stat().st_mode & 0o777 == 0o600
    assert FilePrivacyKey(path).load(create=True) == key


def test_insecure_file_is_not_accepted_or_repaired(tmp_path: Path) -> None:
    path = tmp_path / "privacy.key"
    path.write_bytes(b"k" * 32)
    path.chmod(0o644)
    with pytest.raises(ProtectionError, match="owner-only"):
        FilePrivacyKey(path).load(create=True)
    assert path.stat().st_mode & 0o777 == 0o644


def test_wrong_length_does_not_generate_replacement_keys(tmp_path: Path) -> None:
    path = tmp_path / "target.key"
    path.write_bytes(b"short")
    path.chmod(0o600)
    with pytest.raises(ProtectionError):
        FilePrivacyKey(path).load(create=True)
    assert path.read_bytes() == b"short"
