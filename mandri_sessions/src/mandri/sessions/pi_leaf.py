import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PiLeaf:
    leaf_id: str | None
    modified_at: int


def read_pi_leaf(
    path: Path, native_id: str | None, signature: tuple[int, int, int]
) -> PiLeaf | None:
    pointer = path.with_name(path.name + ".mandri-leaf")
    try:
        with pointer.open("rb") as handle:
            record = json.loads(handle.read(65536))
        if (
            not isinstance(record, dict)
            or record.get("sessionId") != native_id
            or record.get("size") != str(signature[2])
            or record.get("mtimeNs") != str(signature[1])
            or "leafId" not in record
            or not (record["leafId"] is None or isinstance(record["leafId"], str))
        ):
            return None
        return PiLeaf(record["leafId"], pointer.stat().st_mtime_ns)
    except (OSError, ValueError, RecursionError):
        return None
