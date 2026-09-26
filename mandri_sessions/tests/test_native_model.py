import json
from unittest.mock import AsyncMock

import pytest
from mandri.core.ids import HarnessKind, SessionId
from mandri.core.launch import build_harness_launch
from mandri.core.types.availability import SessionOwner
from mandri.sessions.native_activity import native_model
from mandri.sessions.ownership.service import NativeOwnership
from mandri.sessions.service import SessionsService
from mandri.sessions.transcripts.claude_transcripts import ClaudeTranscriptReader
from mandri.sessions.transcripts.codex_transcripts import CodexTranscriptReader
from mandri.sessions.transcripts.resolver import TranscriptResolver

from .substitutes import make_session


@pytest.mark.parametrize(
    "model",
    [
        "gpt-6-astra",
        "claude-fable",
        "claude-opus-custom",
        "claude-sonnet-custom",
        "claude-haiku-custom",
        "private/vendor/model-v2",
    ],
)
@pytest.mark.parametrize("kind", ["turn_context", "assistant", "system"])
def test_native_model_preserves_arbitrary_names(kind, model):
    entry = {"type": kind}
    if kind == "system":
        entry.update(subtype="init", model=model)
    else:
        entry["payload" if kind == "turn_context" else "message"] = {"model": model}
    noise = {"type": "tool", "payload": {"model": "wrong-model"}}
    synthetic = {"type": "assistant", "message": {"model": "<synthetic>"}}
    assert native_model([json.dumps(e) for e in (entry, noise, synthetic)]) == model


@pytest.mark.parametrize("role", ["user", "assistant"])
def test_opencode_latest_selection(role):
    model = {"providerID": "custom", "modelID": "vendor/model"}
    info = {"role": role, **({"model": model} if role == "user" else model)}
    entry = {"type": "message.updated", "properties": {"info": info}}
    assert native_model([json.dumps(entry)]) == "custom/vendor/model"


@pytest.mark.parametrize("harness", [HarnessKind.CODEX, HarnessKind.CLAUDE])
def test_mandri_run_model_is_preserved_in_native_metadata(harness):
    model = "custom/vendor/model"
    launch = build_harness_launch(harness, 1234, "route", model, "dummy")
    if harness is HarnessKind.CODEX:
        value = next(arg for arg in launch.args if arg.startswith('model="'))
        recorded = json.loads(value.partition("=")[2])
        entry = {"type": "turn_context", "payload": {"model": recorded}}
    else:
        recorded = launch.env["ANTHROPIC_MODEL"]
        entry = {"type": "assistant", "message": {"model": recorded}}
    assert native_model([json.dumps(entry)]) == model


async def test_model_before_recent_window_does_not_change_latest_activity(tmp_path, monkeypatch):
    path = tmp_path / "rollout-native.jsonl"
    path.write_text(
        json.dumps({"type": "turn_context", "payload": {"model": "gpt-6-astra"}})
        + "\n"
        + '{"type":"progress"}\n' * 600
        + json.dumps({"type": "event_msg", "payload": {"type": "task_started"}})
        + "\n",
        encoding="utf8",
    )
    reader = CodexTranscriptReader(tmp_path)
    service = SessionsService(
        AsyncMock(), AsyncMock(), TranscriptResolver({HarnessKind.CODEX: reader})
    )
    service.get_session = AsyncMock(
        return_value=make_session(
            harness=HarnessKind.CODEX,
            native_id="native",
            project_path="/workspace",
            model="pending/qwen",
        )
    )
    service.history = AsyncMock(side_effect=AssertionError("status must not load display history"))
    monkeypatch.setattr(
        "mandri.sessions.service.inspect_owner",
        lambda *args: NativeOwnership(SessionOwner.EXTERNAL),
    )
    assert await service.external_status(SessionId("one")) == (True, "gpt-6-astra")
    service.history.assert_not_awaited()


async def test_missing_native_model_never_uses_pending_selection(tmp_path):
    (tmp_path / "rollout-native.jsonl").write_bytes(b"")
    reader = CodexTranscriptReader(tmp_path)
    service = SessionsService(
        AsyncMock(), AsyncMock(), TranscriptResolver({HarnessKind.CODEX: reader})
    )
    service.get_session = AsyncMock(
        return_value=make_session(
            harness=HarnessKind.CODEX,
            native_id="native",
            project_path="/workspace",
            model="pending/qwen",
        )
    )
    assert await service.external_status(SessionId("one")) == (False, None)


@pytest.mark.parametrize("harness", [HarnessKind.CODEX, HarnessKind.CLAUDE])
async def test_native_model_survives_long_turn_and_changes_between_turns(
    tmp_path, monkeypatch, harness
):
    filename = "rollout-native.jsonl" if harness is HarnessKind.CODEX else "native.jsonl"
    path = tmp_path / filename
    kind = "turn_context" if harness is HarnessKind.CODEX else "assistant"
    key = "payload" if harness is HarnessKind.CODEX else "message"
    path.write_text(
        json.dumps({"type": kind, key: {"model": "custom/first"}})
        + "\n"
        + (json.dumps({"type": "progress"}) + "\n") * 600,
        encoding="utf8",
    )
    reader = (
        CodexTranscriptReader(tmp_path)
        if harness is HarnessKind.CODEX
        else ClaudeTranscriptReader(tmp_path)
    )
    service = SessionsService(AsyncMock(), AsyncMock(), TranscriptResolver({harness: reader}))
    monkeypatch.setattr(
        service,
        "get_session",
        AsyncMock(
            return_value=make_session(
                harness=harness,
                native_id="native",
                project_path="/workspace",
                model="pending/qwen",
            )
        ),
    )
    assert (await service.external_status(SessionId("one")))[1] == "custom/first"
    with path.open("a", encoding="utf8") as handle:
        handle.write(json.dumps({"type": kind, key: {"model": "custom/second"}}) + "\n")
    assert (await service.external_status(SessionId("one")))[1] == "custom/second"
