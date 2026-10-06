"""Run command domain types."""

import dataclasses
import pathlib

from mandri.core.ids import HarnessKind
from mandri.core.types.execution import ExecutionBackend, PrivacyMode


@dataclasses.dataclass(frozen=True)
class RunSpec:
    """Immutable launch specification for a harness run."""

    harness: HarnessKind
    model_arg: str
    base_dir: pathlib.Path
    cwd: pathlib.Path | None = None
    effort: str | None = None
    passthrough_args: tuple[str, ...] = ()
    execution_backend: ExecutionBackend | None = None
    privacy_mode: PrivacyMode | None = None
