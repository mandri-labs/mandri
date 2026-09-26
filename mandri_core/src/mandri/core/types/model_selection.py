import enum
from dataclasses import dataclass
from types import MappingProxyType

from mandri.core.ids import HarnessKind


class ModelSource(enum.StrEnum):
    GATEWAY = "gateway"
    NATIVE = "native"


@dataclass(frozen=True)
class HarnessModelCapabilities:
    sources: tuple[ModelSource, ...] = (ModelSource.GATEWAY,)
    gateway_model_requires_restart: bool = False
    gateway_effort_requires_restart: bool = False


_GATEWAY = HarnessModelCapabilities()
_NATIVE = (ModelSource.GATEWAY, ModelSource.NATIVE)
_HARNESS_MODELS = MappingProxyType(
    {
        HarnessKind.CODEX: HarnessModelCapabilities(_NATIVE),
        HarnessKind.CLAUDE: HarnessModelCapabilities(_NATIVE),
        HarnessKind.OPENCODE: HarnessModelCapabilities(gateway_model_requires_restart=True),
        HarnessKind.AGY: HarnessModelCapabilities(_NATIVE),
        HarnessKind.PI: HarnessModelCapabilities(
            _NATIVE,
            gateway_model_requires_restart=True,
            gateway_effort_requires_restart=True,
        ),
    }
)


def model_capabilities(harness: str) -> HarnessModelCapabilities:
    try:
        kind = HarnessKind(harness)
    except ValueError:
        return _GATEWAY
    return _HARNESS_MODELS[kind]
