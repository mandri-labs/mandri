from typing import Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

McpState = Literal["disabled", "starting", "ready", "reconnecting", "error", "auth_required"]
McpTransport = Literal["stdio", "streamable_http", "sse"]


class McpServerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,47}$")
    enabled: bool = True
    transport: McpTransport = "stdio"
    command: str | None = None
    args: list[str] = Field(default_factory=list, max_length=256)
    cwd: str | None = None
    env: dict[str, str] = Field(default_factory=dict)
    env_refs: dict[str, str] = Field(default_factory=dict)
    url: str | None = None
    headers: dict[str, str] = Field(default_factory=dict)
    auth: Literal["none", "oauth"] = "none"
    allow_protected: bool = False
    startup_timeout: float = Field(default=15, gt=0, le=120)
    call_timeout: float = Field(default=60, gt=0, le=3600)

    @field_validator("command", "cwd", "url")
    @classmethod
    def valid_string(cls, value: str | None) -> str | None:
        if value is not None and (not value.strip() or "\x00" in value):
            raise ValueError("Expected a non-empty value")
        return value

    @field_validator("env", "env_refs", "headers")
    @classmethod
    def valid_mapping(cls, value: dict[str, str]) -> dict[str, str]:
        if any(
            not key or any(char in key + entry for char in "\x00\r\n")
            for key, entry in value.items()
        ):
            raise ValueError("Invalid environment variable or header")
        return value

    @model_validator(mode="after")
    def valid_transport(self) -> "McpServerConfig":
        if any("\x00" in arg for arg in self.args):
            raise ValueError("Invalid command argument")
        if self.transport == "stdio":
            if not self.command or self.url or self.headers or self.auth != "none":
                raise ValueError("Local servers require a command and no remote authentication")
        else:
            parsed = urlsplit(self.url or "")
            if parsed.scheme not in {"http", "https"} or not parsed.hostname:
                raise ValueError("Remote servers require an HTTP or HTTPS URL")
            if parsed.username or parsed.password or parsed.fragment:
                raise ValueError("URL credentials and fragments are not supported")
            if self.command or self.args or self.cwd or self.env or self.env_refs:
                raise ValueError("Remote servers cannot contain local launch settings")
        return self


class McpServerView(BaseModel):
    id: str
    revision: int
    name: str
    enabled: bool
    transport: McpTransport
    command: str | None = None
    args: list[str] = Field(default_factory=list)
    cwd: str | None = None
    url: str | None = None
    env_keys: list[str] = Field(default_factory=list)
    env_refs: dict[str, str] = Field(default_factory=dict)
    header_keys: list[str] = Field(default_factory=list)
    auth: Literal["none", "oauth"] = "none"
    allow_protected: bool = False
    startup_timeout: float = 15
    call_timeout: float = 60
    state: McpState
    error: str | None = None
    tools: int = 0
    resources: int = 0
    prompts: int = 0
    authorize_url: str | None = None


class McpSnapshot(BaseModel):
    revision: int
    servers: list[McpServerView]


class McpEndpoint(BaseModel):
    id: str
    name: str
    url: str


class McpBinding(BaseModel):
    id: str
    token: str
    servers: list[McpEndpoint]


class McpEvent(BaseModel):
    revision: int
    server_id: str
    deleted: bool = False
