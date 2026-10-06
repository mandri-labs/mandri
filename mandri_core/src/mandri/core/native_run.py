from mandri.core.ids import HarnessKind
from mandri.core.types.execution import ExecutionBackend, PrivacyMode
from pydantic import BaseModel, ConfigDict, Field


class NativeRunStart(BaseModel):
    model_config = ConfigDict(extra="forbid")

    harness: HarnessKind
    model: str = Field(min_length=3)
    cwd: str = Field(min_length=1)
    effort: str | None = None
    execution_backend: ExecutionBackend = ExecutionBackend.HOST
    privacy_mode: PrivacyMode = PrivacyMode.NONE
    args: list[str] = Field(default_factory=list)
    tty: bool = False
    term: str = Field(default="xterm-256color", pattern=r"^[a-zA-Z0-9_.+-]{1,64}$")


class NativeRunPlan(BaseModel):
    id: str
    argv: list[str]
    env: dict[str, str] = Field(default_factory=dict)
