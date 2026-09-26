"""Smoke tests for the runtime package surface."""

from mandri.core.ids import HarnessKind
from mandri.runtime.adapters import AdapterContext
from mandri.runtime.registry import SessionRegistry
from mandri.runtime.service import RuntimeService


def test_runtime_service_constructs_with_defaults() -> None:
    service = RuntimeService(harness_commands={})
    assert service.installed_harnesses() == []
    assert service.registry.live_ids() == []


def test_runtime_service_reports_installed_harnesses() -> None:
    service = RuntimeService(harness_commands={"claude": ["claude", "-p"]})
    assert service.installed_harnesses() == ["claude"]


def test_registry_marks_live_and_stopped() -> None:
    registry = SessionRegistry()
    registry.mark_live("s1")
    assert registry.status("s1") == "live"
    registry.mark_stopped("s1")
    assert registry.status("s1") == "stopped"


def test_adapter_context_defaults() -> None:
    context = AdapterContext(kind=HarnessKind.CLAUDE)
    assert context.kind is HarnessKind.CLAUDE
    assert context.process is None
    assert context.hub is None
