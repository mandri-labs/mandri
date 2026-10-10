import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tomlkit
from mandri.core.types.mcp import McpServerConfig
from mandri.mcp.storage import SecretStore, atomic_write


@dataclass(frozen=True)
class Definition:
    id: str
    revision: int
    config: McpServerConfig


class Registry:
    def __init__(self, base_dir: Path) -> None:
        self.path = base_dir / "mcp.toml"
        self.secrets = SecretStore(base_dir / "mcp-secrets")
        self.definitions: dict[str, Definition] = {}
        self.errors: dict[str, str] = {}
        self.invalid: dict[str, Any] = {}

    def load(self) -> None:
        try:
            if not self.path.exists():
                return
            document = tomlkit.loads(self.path.read_text(encoding="utf-8"))
            if document.get("schema_version") != 1:
                raise ValueError("Unsupported MCP configuration version")
            entries = document.get("servers", [])
            if not isinstance(entries, list):
                raise ValueError("Expected server definitions")
        except Exception:
            self.errors["configuration"] = "Invalid mcp.toml"
            return
        for index, entry in enumerate(entries):
            identity = str(entry.get("id", "")) if isinstance(entry, dict) else ""
            try:
                uuid.UUID(identity)
                payload = dict(entry)
                payload.pop("id")
                revision = int(payload.pop("revision", 1))
                secret = self.secrets.load(identity)
                payload.update(env=secret.get("env", {}), headers=secret.get("headers", {}))
                config = McpServerConfig.model_validate(payload)
                self._check_name(config.name)
                if identity in self.definitions:
                    raise ValueError("Duplicate identity")
                self.definitions[identity] = Definition(identity, revision, config)
            except Exception:
                key = f"entry-{index}"
                self.errors[key] = "Invalid server definition"
                self.invalid[key] = entry

    def _check_name(self, name: str, exclude: str | None = None) -> None:
        if any(
            row.id != exclude and row.config.name.lower() == name.lower()
            for row in self.definitions.values()
        ):
            raise ValueError("Server name already exists")

    def get(self, identity: str) -> Definition:
        definition = self.definitions.get(identity)
        if definition is None:
            definition = next(
                (row for row in self.definitions.values() if row.config.name == identity), None
            )
        if definition is None:
            raise KeyError(identity)
        return definition

    def save(self) -> None:
        if "configuration" in self.errors:
            raise ValueError("Repair invalid mcp.toml before editing servers")
        entries: list[dict[str, Any]] = []
        for row in self.definitions.values():
            config = row.config.model_dump(exclude={"env", "headers"}, exclude_none=True)
            entries.append({"id": row.id, "revision": row.revision, **config})
        entries.extend(self.invalid.values())
        atomic_write(self.path, tomlkit.dumps({"schema_version": 1, "servers": entries}))

    def add(self, config: McpServerConfig) -> Definition:
        self._check_name(config.name)
        row = Definition(str(uuid.uuid4()), 1, config)
        self._persist(row)
        return row

    def update(self, identity: str, revision: int, values: dict[str, Any]) -> Definition:
        current = self.get(identity)
        if revision != current.revision:
            raise ValueError("Server changed; reload before saving")
        config = McpServerConfig.model_validate(current.config.model_dump() | values)
        self._check_name(config.name, current.id)
        row = Definition(current.id, current.revision + 1, config)
        self._persist(row)
        return row

    def _persist(self, row: Definition) -> None:
        if "configuration" in self.errors:
            raise ValueError("Repair invalid mcp.toml before editing servers")
        old = self.definitions.get(row.id)
        secret = self.secrets.load(row.id)
        updated = secret | {"env": row.config.env, "headers": row.config.headers}
        if old and (row.config.url != old.config.url or row.config.auth != old.config.auth):
            updated.pop("oauth", None)
        self.secrets.save(row.id, updated)
        self.definitions[row.id] = row
        try:
            self.save()
        except BaseException:
            if old is None:
                self.definitions.pop(row.id, None)
                self.secrets.delete(row.id)
            else:
                self.definitions[row.id] = old
                self.secrets.save(row.id, secret)
            raise

    def discard_invalid(self, identity: str) -> None:
        entry = self.invalid.pop(identity)
        error = self.errors.pop(identity)
        try:
            self.save()
        except BaseException:
            self.invalid[identity] = entry
            self.errors[identity] = error
            raise

    def remove(self, identity: str) -> Definition:
        row = self.get(identity)
        del self.definitions[row.id]
        try:
            self.save()
        except BaseException:
            self.definitions[row.id] = row
            raise
        self.secrets.delete(row.id)
        return row
