import io
import json

import pytest
from mandri.sessions.transcripts.jsonl import page_from_jsonl
from mandri.sessions.transcripts.pi_pages import pi_page
from mandri.sessions.transcripts.recent import recent_jsonl
from mandri.sessions.transcripts.record_preview import (
    MAX_PREVIEW_BYTES,
    MAX_PREVIEW_CHARS,
    preview_from_prefix,
    record_preview,
)
from mandri.sessions.transcripts.records import RecordSpan


@pytest.mark.parametrize(
    "event,expected,kind",
    [
        (
            {
                "type": "event_msg",
                "payload": {
                    "type": "item_completed",
                    "item": {
                        "type": "CommandExecution",
                        "command": ["echo", "hello"],
                        "stdout": "output\n" * 20000,
                    },
                },
            },
            "echo hello\n\noutput\n",
            "command",
        ),
        (
            {
                "type": "assistant",
                "message": {"content": [{"type": "text", "text": "Claude\n" * 20000}]},
            },
            "Claude\n",
            None,
        ),
        (
            {
                "type": "message",
                "message": {
                    "role": "toolResult",
                    "content": [{"type": "text", "text": "Pi\n" * 20000}],
                },
            },
            "Pi\n",
            None,
        ),
        ({"type": "text", "text": "OpenCode\n" * 20000}, "OpenCode\n", None),
        ({"type": "step", "step": {"text": "Antigravity\n" * 20000}}, "Antigravity\n", None),
        (
            {"type": "compacted", "payload": {"message": "summary\n" * 20000}},
            "summary\n",
            "compaction",
        ),
        (
            {"type": "system", "subtype": "compact_boundary", "message": "summary" * 20000},
            "summary",
            "compaction",
        ),
        ({"type": "compaction", "summary": "summary" * 20000}, "summary", "compaction"),
    ],
)
def test_preview_decodes_native_content_with_bounded_reads(event, expected, kind):
    class BoundedFile(io.BytesIO):
        def read(self, size=-1):
            assert 0 < size <= MAX_PREVIEW_BYTES
            return super().read(size)

    raw = json.dumps(event).encode()
    result = record_preview(BoundedFile(b"previous\n" + raw), 9, len(raw))
    assert str(result["preview"]).startswith(expected)
    assert len(str(result["preview"])) <= MAX_PREVIEW_CHARS
    assert result["preview_truncated"] is True
    assert result.get("event_kind") == kind


@pytest.mark.parametrize("ending", [b"\\", b"\\u00", b"\xc3"])
def test_preview_handles_split_escapes_and_unicode(ending):
    result = preview_from_prefix(b'{"text":"readable prefix ' + ending)
    assert "readable prefix" in result["preview"]
    assert "\ufffd" not in result["preview"]


@pytest.mark.parametrize("prefix", [b"not json", b'{"type": [], "text": "visible"}', b"[" * 1000])
def test_malformed_and_unrecognized_records_keep_a_bounded_preview(prefix):
    result = preview_from_prefix(prefix)
    assert result["preview"]
    assert len(result["preview"]) <= MAX_PREVIEW_CHARS


@pytest.mark.parametrize("mode", ["recent", "forward", "pi_recent", "pi_forward"])
def test_every_jsonl_pagination_path_returns_large_record_preview(tmp_path, mode):
    path = tmp_path / "session.jsonl"
    raw = (json.dumps({"type": "unknown", "text": "readable output\n" * 100000}) + "\n").encode()
    path.write_bytes(raw + b'{"text":"after"}\n')
    if mode.startswith("pi_"):
        page = pi_page(
            path,
            [RecordSpan(0, len(raw)), RecordSpan(len(raw), path.stat().st_size)],
            None,
            10,
            recent=mode == "pi_recent",
        )
    else:
        page = (recent_jsonl if mode == "recent" else page_from_jsonl)(path, None, 10)
    marker = json.loads(page.entries[0])
    assert marker["type"] == "mandri.transcript_record"
    assert marker["preview"].startswith("readable output\n")
    assert marker["preview_truncated"] is True
    assert marker["byte_length"] == len(raw)
    assert json.loads(page.entries[1]) == {"text": "after"}


def test_forward_pagination_excludes_a_large_torn_tail(tmp_path):
    path = tmp_path / "session.jsonl"
    path.write_bytes(b'{"text":"before"}\n{"text":"' + b"x" * (2 * 1024**2))
    page = page_from_jsonl(path, None, 10)
    assert len(page.entries) == 1
    assert not page.has_more
    with path.open("ab") as handle:
        handle.write(b'"}\n')
    resumed = page_from_jsonl(path, page.next_token, 10)
    assert len(resumed.entries) == 1
    assert json.loads(resumed.entries[0])["preview"].startswith("x")
