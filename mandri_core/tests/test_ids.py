"""Tests for core typed identifiers, enums, and claim evidence."""

import dataclasses

import pytest
from mandri.core.ids import (
    HARNESS_WIRE_FORMATS,
    ApprovalDecision,
    ApprovalId,
    ApprovalKind,
    ApprovalStatus,
    ClaimEvidence,
    CorrelationId,
    EpochMs,
    FsPath,
    HarnessKind,
    HarnessSessionId,
    ModelRef,
    PageToken,
    ProjectPath,
    ProviderKind,
    RawEvent,
    RouteId,
    SecretRef,
    SessionId,
    SessionState,
    SessionTitle,
    WireFormat,
)


def test_newtypes_are_str_based() -> None:
    assert isinstance(SessionId("abc"), str)
    assert isinstance(HarnessSessionId("abc"), str)
    assert isinstance(SessionTitle("t"), str)
    assert isinstance(ProjectPath("p"), str)
    assert isinstance(ModelRef("openrouter/m"), str)
    assert isinstance(RouteId("r"), str)
    assert isinstance(SecretRef("s"), str)
    assert isinstance(ApprovalId("a"), str)
    assert isinstance(CorrelationId("c"), str)
    assert isinstance(PageToken("t"), str)
    assert isinstance(FsPath("f"), str)
    assert isinstance(RawEvent("raw"), str)


def test_epoch_ms_is_int_based() -> None:
    value = EpochMs(1720000000000)
    assert isinstance(value, int)
    assert value + 1 == 1720000000001


def test_harness_kind_values() -> None:
    assert HarnessKind("codex") is HarnessKind.CODEX
    assert HarnessKind("claude") is HarnessKind.CLAUDE
    assert HarnessKind("opencode") is HarnessKind.OPENCODE
    with pytest.raises(ValueError):
        HarnessKind("unknown")


def test_provider_kind_values() -> None:
    assert ProviderKind("openrouter") is ProviderKind.OPENROUTER
    assert ProviderKind("opencode_go") is ProviderKind.OPENCODE_GO
    assert ProviderKind("lm_studio") is ProviderKind.LM_STUDIO
    assert ProviderKind("chatgpt") is ProviderKind.CHATGPT
    assert len(ProviderKind) == 10


def test_state_enums() -> None:
    assert SessionState("live") is SessionState.LIVE
    assert ApprovalStatus("pending") is ApprovalStatus.PENDING
    assert ApprovalDecision("acceptForSession") is ApprovalDecision.ACCEPT_FOR_SESSION
    assert ApprovalKind("command_execution") is ApprovalKind.COMMAND_EXECUTION


def test_harness_wire_formats_mapping() -> None:
    assert HARNESS_WIRE_FORMATS[HarnessKind.CODEX] == (WireFormat.OPENAI,)
    assert HARNESS_WIRE_FORMATS[HarnessKind.CLAUDE] == (WireFormat.ANTHROPIC,)
    assert HARNESS_WIRE_FORMATS[HarnessKind.OPENCODE] == (WireFormat.OPENAI,)


def test_claim_evidence_frozen() -> None:
    evidence = ClaimEvidence(
        harness=HarnessKind.CLAUDE,
        project_path=ProjectPath("C:/work"),
        captured_native_id=HarnessSessionId("native"),
        time_window=(EpochMs(1), EpochMs(2)),
    )
    assert evidence.harness is HarnessKind.CLAUDE
    with pytest.raises(dataclasses.FrozenInstanceError):
        evidence.harness = HarnessKind.CODEX
