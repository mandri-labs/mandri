import json
from typing import BinaryIO

from mandri.sessions.transcripts.record_metadata import record_metadata
from mandri.sessions.transcripts.record_preview import record_preview
from mandri.sessions.transcripts.records import RecordSpan
from mandri.sessions.transcripts.tokens import PageTokenData, encode_page_token

MAX_INLINE_BYTES = 1024 * 1024


def large_record(handle: BinaryIO, span: RecordSpan, identity: str) -> str:
    reference = encode_page_token(
        PageTokenData("jsonl-record", offset=span.start, file_size=span.end, file_id=identity)
    )
    return json.dumps(
        {
            "type": "mandri.transcript_record",
            "byte_length": span.size,
            "record_token": reference,
            **record_metadata(handle, span.start, span.size),
            **record_preview(handle, span.start, span.size),
        },
        separators=(",", ":"),
    )
