import asyncio
from collections.abc import AsyncIterator
from types import SimpleNamespace

from mandri.runtime.pump import LineEventKind, LinePump, pump_process_streams


async def test_unlimited_protocol_lines_keep_chunked_utf8_and_following_response():
    async def chunks() -> AsyncIterator[bytes]:
        for chunk in [b'{"text":"', b"\xc3", b'\xa9"}\n{"id":2}\n']:
            yield chunk

    lines = [line async for line in LinePump(chunks, limit=None).lines()]
    assert [line.kind for line in lines] == [LineEventKind.LINE, LineEventKind.LINE]
    assert [line.text for line in lines] == ['{"text":"é"}', '{"id":2}']


async def test_explicit_line_limit_still_recovers_at_next_line():
    async def chunks() -> AsyncIterator[bytes]:
        for chunk in [b"oversized", b"-line\nok\n"]:
            yield chunk

    lines = [line async for line in LinePump(chunks, limit=4).lines()]
    assert [line.kind for line in lines] == [LineEventKind.OVERSIZE, LineEventKind.LINE]
    assert lines[-1].text == "ok"


async def test_process_line_limit_still_applies_to_both_streams():
    stdout, stderr = asyncio.StreamReader(), asyncio.StreamReader()
    for stream in (stdout, stderr):
        stream.feed_data(b"oversized\nok\n")
        stream.feed_eof()
    pumps = pump_process_streams(SimpleNamespace(stdout=stdout, stderr=stderr), limit=4)
    for pump in pumps:
        assert [line.kind async for line in pump.lines()] == [
            LineEventKind.OVERSIZE,
            LineEventKind.LINE,
        ]
