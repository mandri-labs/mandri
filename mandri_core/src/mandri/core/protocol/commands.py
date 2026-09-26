from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class CommandDescriptor(BaseModel):
    id: str
    name: str
    description: str = ""
    aliases: list[str] = Field(default_factory=list)
    argument_hint: str | None = None
    accepts_arguments: bool = True
    kind: str = "command"
    available: bool = True
    unavailable_reason: str | None = None


class CommandItem(BaseModel):
    title: str
    description: str | None = None


class CommandField(BaseModel):
    label: str
    value: str | int | float | bool | None


class CommandResult(BaseModel):
    kind: Literal["text", "list", "fields", "notice", "transcript"]
    title: str | None = None
    text: str | None = None
    message: str | None = None
    items: list[CommandItem] = Field(default_factory=list)
    fields: list[CommandField] = Field(default_factory=list)
    empty_message: str | None = None


class CommandSessionParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    session_id: str = Field(min_length=1)


class CommandInvokeParams(CommandSessionParams):
    invocation_id: str = Field(min_length=1, max_length=128)
    command_id: str = Field(min_length=1)
    arguments: str = Field(default="", max_length=100_000)


class CommandGetParams(CommandSessionParams):
    invocation_id: str = Field(min_length=1)


class CommandCatalog(BaseModel):
    commands: list[CommandDescriptor]
    reason: str | None = None


class CommandInvocation(BaseModel):
    invocation_id: str
    session_id: str
    command: CommandDescriptor
    arguments: str = ""
    state: Literal["running", "succeeded", "failed", "unknown", "interrupted"]
    result: CommandResult | None = None
    error: str | None = None
    cancellable: bool = False


class CommandSnapshot(BaseModel):
    invocations: list[CommandInvocation]


class CommandCatalogParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    harness: Literal["claude", "codex", "agy", "opencode", "pi"]
    force_refresh: bool = False
    cwd: str | None = None
    profile_id: str | None = None
    execution_backend: Literal["host", "docker"] = "host"
    privacy_mode: Literal["none", "surrogate"] = "none"


class CommandCatalogsParams(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ScopedCommandCatalog(CommandCatalog):
    harness: Literal["claude", "codex", "agy", "opencode", "pi"]
    cwd: str
    profile_id: str | None = None
    execution_backend: Literal["host", "docker"] = "host"
    privacy_mode: Literal["none", "surrogate"] = "none"
    state: Literal["ready", "unavailable"]


class CommandCatalogsSnapshot(BaseModel):
    default_cwd: str
    catalogs: list[ScopedCommandCatalog]
