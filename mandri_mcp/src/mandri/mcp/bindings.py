import secrets
import time
import uuid
from dataclasses import dataclass

from mandri.core.types.mcp import McpBinding, McpEndpoint


@dataclass
class Grant:
    binding: McpBinding
    owner: str
    expires: float | None


class Bindings:
    def __init__(self) -> None:
        self.grants: dict[str, Grant] = {}

    def issue(self, owner: str, servers: list[McpEndpoint], *, lease: bool = False) -> McpBinding:
        self.sweep()
        self.release(owner)
        identity = uuid.uuid4().hex
        binding = McpBinding(
            id=identity,
            token=secrets.token_urlsafe(32),
            servers=[
                server.model_copy(update={"url": server.url.replace("{binding}", identity)})
                for server in servers
            ],
        )
        self.grants[identity] = Grant(binding, owner, time.monotonic() + 120 if lease else None)
        return binding

    def authenticate(self, identity: str, server_id: str, token: str) -> bool:
        grant = self.grants.get(identity)
        if grant is None:
            return False
        if grant.expires is not None and grant.expires <= time.monotonic():
            del self.grants[identity]
            return False
        return secrets.compare_digest(token.encode(), grant.binding.token.encode()) and any(
            server.id == server_id for server in grant.binding.servers
        )

    def renew(self, identity: str) -> None:
        grant = self.grants.get(identity)
        if grant is None or grant.expires is None or grant.expires <= time.monotonic():
            self.grants.pop(identity, None)
            raise KeyError(identity)
        grant.expires = time.monotonic() + 120

    def revoke(self, identity: str) -> None:
        self.grants.pop(identity, None)

    def release(self, owner: str) -> None:
        self.grants = {key: grant for key, grant in self.grants.items() if grant.owner != owner}

    def withdraw(self, identity: str) -> None:
        for grant in self.grants.values():
            grant.binding.servers = [row for row in grant.binding.servers if row.id != identity]

    def sweep(self) -> None:
        now = time.monotonic()
        self.grants = {
            key: grant
            for key, grant in self.grants.items()
            if grant.expires is None or grant.expires > now
        }

    def clear(self) -> None:
        self.grants.clear()
