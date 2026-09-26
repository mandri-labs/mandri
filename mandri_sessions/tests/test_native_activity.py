import json

import pytest
from mandri.core.ids import HarnessKind
from mandri.sessions.native_activity import native_turn_busy


@pytest.mark.parametrize(
    "harness,events,busy",
    [
        (HarnessKind.CODEX, [{"type": "event_msg", "payload": {"type": "task_started"}}], True),
        (HarnessKind.CODEX, [{"type": "event_msg", "payload": {"type": "task_complete"}}], False),
        (
            HarnessKind.CODEX,
            [
                {"type": "event_msg", "payload": {"type": "task_started"}},
                {"type": "event_msg", "payload": {"type": "turn_aborted"}},
            ],
            False,
        ),
        (HarnessKind.CODEX, [{"type": "response_item"}], None),
        (
            HarnessKind.CLAUDE,
            [{"type": "assistant", "message": {"stop_reason": "end_turn"}}],
            False,
        ),
        (HarnessKind.CLAUDE, [{"type": "assistant", "message": {"stop_reason": "tool_use"}}], True),
        (
            HarnessKind.OPENCODE,
            [
                {
                    "type": "message.updated",
                    "properties": {"info": {"role": "assistant", "time": {"completed": 1}}},
                }
            ],
            False,
        ),
        (
            HarnessKind.OPENCODE,
            [{"type": "message.updated", "properties": {"info": {"role": "assistant"}}}],
            True,
        ),
    ],
)
def test_native_turn_boundaries(harness, events, busy):
    assert native_turn_busy(harness, [json.dumps(event) for event in events]) is busy
