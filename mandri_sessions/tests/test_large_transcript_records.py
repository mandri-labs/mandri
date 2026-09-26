import io
import json
import tracemalloc
from pathlib import Path

import pytest
from mandri.core.ids import PageToken
from mandri.sessions.transcripts import recent
from mandri.sessions.transcripts.errors import PageTokenInvalidError, PageTokenStaleError
from mandri.sessions.transcripts.record_download import open_record
from mandri.sessions.transcripts.record_metadata import MAX_HEADER_BYTES, record_metadata
from mandri.sessions.transcripts.records import CHUNK_SIZE, ReverseRecords
from mandri.sessions.transcripts.tokens import PageTokenData, decode_page_token, encode_page_token


def all_entries(path: Path) -> list[str]:
    cursor = None
    entries = []
    for _ in range(100):
        page = recent.recent_jsonl(path, cursor, 7)
        entries = page.entries + entries
        if not page.has_more:
            return entries
        assert page.next_token != cursor
        cursor = page.next_token
    pytest.fail("pagination did not advance")


def test_large_compaction_is_downloadable_without_blocking_surrounding_events(tmp_path):
    path = tmp_path / "session.jsonl"
    large = (
        json.dumps({"type": "compacted", "payload": {"image": "x" * (9 * 1024**2)}}).encode()
        + b"\n"
    )
    path.write_bytes(b'"before"\n' + large + b'"after"\n')
    entries = all_entries(path)
    assert entries[0] == '"before"'
    assert entries[-1] == '"after"'
    assert len(entries) == 3
    marker = json.loads(entries[1])
    assert marker["byte_length"] == len(large)
    assert marker["type"] == "mandri.transcript_record"
    assert marker["original_type"] == "compacted"
    download = open_record(path, PageToken(marker["record_token"]))
    assert b"".join(download.chunks()) == large
    assert download.handle.closed
    with path.open("ab") as handle:
        handle.write(b'"appended"\n')
    assert all_entries(path)[1] == entries[1]


@pytest.mark.parametrize(
    "prefix,expected",
    [
        (b'{"timestamp":"today","type":"compacted","payload":', "compacted"),
        (b'{"type":"response_item","payload":', "response_item"),
        (b'{"payload":{"type":"compacted"},"type":"unknown","data":', "unknown"),
        (b'{"payload":{"type":"compacted","data":', None),
        (b'{"type":42,"data":', None),
        (b"not json", None),
    ],
)
def test_large_record_metadata_is_bounded_and_only_uses_top_level_type(prefix, expected):
    class BoundedFile(io.BytesIO):
        def read(self, size=-1):
            assert 0 < size <= MAX_HEADER_BYTES
            return super().read(size)

    data = b"previous\n" + prefix + b"x" * (2 * MAX_HEADER_BYTES)
    metadata = record_metadata(BoundedFile(data), len(b"previous\n"), len(data))
    assert metadata.get("original_type") == expected


def test_partial_giant_tail_is_excluded_and_later_completion_is_visible(tmp_path, monkeypatch):
    monkeypatch.setattr(recent, "MAX_SCAN_BYTES", 128)
    path = tmp_path / "session.jsonl"
    path.write_bytes(b'"before"\n"' + b"x" * 1024)
    first = recent.recent_jsonl(path, None, 7)
    assert first.entries == [] and first.has_more
    with path.open("ab") as handle:
        handle.write(b'"\n')
    cursor = first.next_token
    older = []
    while cursor:
        page = recent.recent_jsonl(path, cursor, 7)
        older = page.entries + older
        cursor = page.next_token
    assert older == ['"before"']
    assert len(all_entries(path)) == 2


@pytest.mark.parametrize("item_type", ["CommandExecution", "ContextCompaction", "FileChange"])
def test_large_codex_completed_item_preserves_nested_types(tmp_path, item_type):
    path = tmp_path / "session.jsonl"
    event = {
        "timestamp": "2026-01-01T00:00:00Z",
        "ordinal": 1,
        "type": "event_msg",
        "payload": {
            "type": "item_completed",
            "thread_id": "thread",
            "turn_id": "turn",
            "item": {"type": item_type, "id": "item", "stdout": "x" * (2 * 1024**2)},
        },
    }
    original = (json.dumps(event) + "\n").encode()
    path.write_bytes(original)
    marker = json.loads(all_entries(path)[0])
    assert marker["original_type"] == "event_msg"
    assert marker["original_event_type"] == "item_completed"
    assert marker["original_item_type"] == item_type
    assert len(json.dumps(marker)) < 24 * 1024
    assert b"".join(open_record(path, PageToken(marker["record_token"])).chunks()) == original


def test_page_budget_counts_serialized_content_and_keeps_every_record(tmp_path, monkeypatch):
    monkeypatch.setattr(recent, "MAX_PAGE_BYTES", 10000)
    path = tmp_path / "session.jsonl"
    lines = [json.dumps({"id": i, "text": "é" * 1000}, ensure_ascii=False) for i in range(5)]
    path.write_text("\n".join(lines) + "\n", encoding="utf8")
    assert all_entries(path) == lines
    page = recent.recent_jsonl(path, None, 500)
    assert sum(len(json.dumps(e)) + 1 for e in page.entries) <= 10000


def test_reverse_scan_uses_bounded_reads_and_memory_across_giant_record():
    class TrackingFile(io.BytesIO):
        total = 0

        def read(self, size=-1):
            assert 0 < size <= CHUNK_SIZE
            self.total += size
            return super().read(size)

    data = b'"before"\n' + b"x" * (16 * 1024**2) + b'\n"after"\n'
    handle = TrackingFile(data)
    tracemalloc.start()
    try:
        scan = ReverseRecords(handle, len(data), len(data))
        spans = list(scan)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert len(spans) == 3
    assert handle.total == len(data)
    assert peak < 4 * CHUNK_SIZE


def test_record_reference_rejects_another_file_and_truncation(tmp_path, monkeypatch):
    monkeypatch.setattr(recent, "MAX_INLINE_BYTES", 4)
    path = tmp_path / "one.jsonl"
    other = tmp_path / "other.jsonl"
    path.write_bytes(b'"original"\n')
    other.write_bytes(path.read_bytes())
    marker = json.loads(all_entries(path)[0])
    token = PageToken(marker["record_token"])
    with pytest.raises(PageTokenStaleError):
        open_record(other, token)
    path.write_bytes(b'"x"\n')
    with pytest.raises(PageTokenStaleError):
        open_record(path, token)


@pytest.mark.parametrize("offset,end", [(101, None), (10, 9), (10, 101), (10, -1)])
def test_invalid_continuation_bounds_are_rejected(offset, end):
    token = encode_page_token(
        PageTokenData("jsonl-recent", offset=offset, file_size=100, record_end=end)
    )
    with pytest.raises(PageTokenInvalidError):
        decode_page_token(token, "jsonl-recent")


@pytest.mark.parametrize("content", [b"", b"partial", b"\n", b"1\n", b"1\n2\n", b"1\r\n2\r\ntail"])
def test_scan_boundaries_preserve_complete_records(content):
    spans = list(ReverseRecords(io.BytesIO(content), len(content), len(content)))
    expected = content.split(b"\n")[:-1]
    assert [content[s.start : s.end].rstrip(b"\r\n") for s in reversed(spans)] == [
        e.rstrip(b"\r") for e in expected
    ]
