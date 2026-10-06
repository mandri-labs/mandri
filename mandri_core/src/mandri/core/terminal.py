from mandri.core.ids import HarnessKind
from mandri.core.types.execution import ExecutionBackend, PrivacyMode
from pydantic import BaseModel, ConfigDict, Field


class TerminalSize(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rows: int = Field(default=24, ge=1, le=1000)
    columns: int = Field(default=80, ge=1, le=1000)


class TerminalStart(TerminalSize):
    harness: HarnessKind
    model: str = Field(min_length=3)
    cwd: str = Field(min_length=1)
    effort: str | None = None
    execution_backend: ExecutionBackend = ExecutionBackend.HOST
    privacy_mode: PrivacyMode = PrivacyMode.NONE
    args: list[str] = Field(default_factory=list)
    term: str = Field(default="xterm-256color", pattern=r"^[a-zA-Z0-9_.+-]{1,64}$")
