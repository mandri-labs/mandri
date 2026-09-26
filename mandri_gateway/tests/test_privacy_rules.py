import copy
import json
import re
import sys

import pytest
from mandri.core.types.config import PrivacySettings
from mandri.core.types.execution import ProtectionError
from mandri.database.privacy import StoredPrivacyScope
from mandri.gateway.privacy_rules import load_extensions
from mandri.gateway.privacy_scopes import PrivacyScopes
from mandri.gateway.surrogate import SurrogateEngine, SurrogateScope


@pytest.fixture
def private_rules(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    file = tmp_path / "private-rules.json"
    file.write_text(
        json.dumps(
            {
                "version": 1,
                "exact_values": [{"value": "Synthetic Confidential Company", "kind": "identity"}],
                "rules": [
                    {
                        "name": "private-ticket",
                        "pattern": "TICKET-[0-9]{6}",
                        "fields": ["reference"],
                    }
                ],
            }
        )
    )
    file.chmod(0o600)
    monkeypatch.setattr("mandri.gateway.privacy_context.getpass.getuser", lambda: "synthetic-user")
    monkeypatch.setattr(
        "mandri.gateway.privacy_context.socket.gethostname", lambda: "synthetic-host"
    )
    monkeypatch.setattr(
        "mandri.gateway.privacy_context.Path.home", lambda: tmp_path / "synthetic-home"
    )
    return file, workspace


class MemoryRepository:
    def __init__(self):
        self.scopes = {}

    async def create(self, scope_id, payload):
        self.scopes[scope_id] = StoredPrivacyScope(scope_id, 0, copy.deepcopy(payload), b"k" * 32)
        return self.scopes[scope_id]

    async def load(self, scope_id):
        return copy.deepcopy(self.scopes[scope_id])

    async def revision(self, scope_id):
        scope = self.scopes[scope_id]
        return scope.revision, scope.fingerprint

    async def save(self, scope, payload):
        updated = StoredPrivacyScope(
            scope.scope_id, scope.revision + 1, copy.deepcopy(payload), scope.key
        )
        self.scopes[scope.scope_id] = updated
        return updated


@pytest.mark.skipif(sys.platform == "win32", reason="Private rule files require POSIX ownership")
async def test_local_extensions_are_persisted_and_configuration_changes_do_not_change_continuation(
    private_rules,
):
    file, workspace = private_rules
    repository = MemoryRepository()
    settings = PrivacySettings(rules_file=str(file))
    service = PrivacyScopes(repository, settings)
    identity = await service.create(str(workspace))
    source = {"organization": "Synthetic Confidential Company", "reference": "TICKET-582046"}
    protected, _ = await service.prepare(identity, lambda engine: engine.protect(source))
    assert protected["organization"] != source["organization"]
    assert re.fullmatch(r"TICKET-[0-9]{6}", protected["reference"])
    assert protected["reference"] != source["reference"]
    file.write_text('{"version":999}')
    resumed = PrivacyScopes(repository, settings)
    second = {"organization": source["organization"], "reference": "TICKET-913275"}
    protected_second, engine = await resumed.prepare(
        identity, lambda engine: engine.protect(second)
    )
    assert protected_second["organization"] == protected["organization"]
    assert re.fullmatch(r"TICKET-[0-9]{6}", protected_second["reference"])
    assert protected_second["reference"] != second["reference"]
    assert engine.restore(protected_second) == second
    assert engine.protect({"public_note": "TICKET-913275"}) != {"public_note": "TICKET-913275"}
    with pytest.raises(ProtectionError):
        await resumed.create(str(workspace))


@pytest.mark.parametrize(
    "change",
    [
        "permissions",
        "symlink",
        "missing",
        "inside_workspace",
        "oversize",
        "duplicate_key",
        "unknown_field",
        "invalid_kind",
    ],
)
def test_invalid_local_rules_fail_closed_without_echoing_private_values(private_rules, change):
    file, workspace = private_rules
    if change == "permissions":
        file.chmod(0o644)
    elif change == "symlink":
        source = file.with_suffix(".source")
        file.rename(source)
        file.symlink_to(source)
    elif change == "missing":
        file.unlink()
    elif change == "inside_workspace":
        destination = workspace / file.name
        file.rename(destination)
        file = destination
    elif change == "oversize":
        file.write_bytes(b"x" * 65_537)
    elif change == "duplicate_key":
        file.write_text('{"version":1,"version":1}')
    elif change == "unknown_field":
        file.write_text('{"version":1,"callback":"Synthetic Confidential Company"}')
    else:
        file.write_text('{"version":1,"exact_values":[{"value":"private","kind":"path_root"}]}')
    with pytest.raises(ProtectionError) as caught:
        load_extensions(str(file), workspace)
    assert "Synthetic Confidential Company" not in str(caught.value)
    assert str(file) not in str(caught.value)


def test_legacy_scope_upgrades_without_changing_assigned_values():
    initial = SurrogateEngine(SurrogateScope("legacy"))
    source = {"email": "legacy@private.example"}
    protected = initial.protect(source)
    legacy = initial.scope.to_dict()
    legacy["version"] = 1
    legacy.pop("rules")
    upgraded = SurrogateScope.from_dict(legacy)
    assert upgraded.version == 2
    assert upgraded.rules == []
    resumed = SurrogateEngine(upgraded)
    assert resumed.protect(source) == protected
    assert resumed.restore(protected) == source
    assert resumed.scope.to_dict()["rules"] == []


@pytest.mark.parametrize(
    "rules",
    [
        "invalid",
        [{"name": "x"}],
        [{"name": "x", "pattern": "A", "kind": "identifier", "fields": [False]}],
    ],
)
def test_unknown_persisted_rules_do_not_break_continuation(rules):
    payload = SurrogateScope("invalid").to_dict()
    payload["rules"] = rules
    engine = SurrogateEngine(SurrogateScope.from_dict(payload))
    source = "private@example.fr"
    assert engine.restore_text(engine.protect_text(source)) == source


@pytest.mark.skipif(sys.platform == "win32", reason="Private rule files require POSIX ownership")
def test_unreplaceable_rules_are_omitted_without_blocking_valid_rules(private_rules):
    file, workspace = private_rules
    file.write_text(
        json.dumps(
            {
                "version": 1,
                "rules": [
                    {"name": "unknown", "pattern": "(a+)+$"},
                    {"name": "literal", "pattern": "PRIVATE"},
                    {"name": "supported", "pattern": "TICKET-[0-9]{4}"},
                ],
            }
        )
    )
    extensions = load_extensions(str(file), workspace)
    assert [rule.name for rule in extensions.rules] == ["supported"]
