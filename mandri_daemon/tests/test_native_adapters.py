from unittest.mock import Mock

from mandri.core.hub import Hub, Topic
from mandri.core.ids import HarnessKind, HarnessSessionId
from mandri.core.types.model_selection import ModelSource
from mandri.daemon.serve import build_harness_adapters
from mandri.runtime.adapters import AdapterContext


def test_native_codex_resume_keeps_provider_and_ultra():
    adapters = build_harness_adapters(
        AdapterContext(
            kind=HarnessKind.CODEX,
            process=Mock(),
            hub=Hub(),
            topic=Topic("native"),
            model="gpt-native",
            model_source=ModelSource.NATIVE,
            reasoning_effort="ultra",
            resume_thread_id=HarnessSessionId("existing-thread"),
        )
    )
    assert adapters is not None
    assert adapters.control._thread_start_params == {
        "model": "gpt-native",
        "modelProvider": "openai",
        "config": {"model_reasoning_effort": "ultra"},
    }
