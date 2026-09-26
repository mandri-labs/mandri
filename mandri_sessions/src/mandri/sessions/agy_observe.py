import json
import sys
import time
from pathlib import Path
from typing import Any

from mandri.sessions.agy_profiles import read_agy_json, validate_agy_id, write_agy_json


def observe_agy(profile: Path, event: str, payload: dict[str, Any]) -> None:
    native_id = payload.get("conversationId")
    if not isinstance(native_id, str):
        return
    validate_agy_id(native_id)
    context = read_agy_json(profile / "mandri-launch.json")
    if context.get("native_id") is None and context.get("new_conversation") is True:
        cache = read_agy_json(profile / "antigravity-cli/cache/last_conversations.json")
        if cache.get(str(context.get("cwd", ""))) == native_id:
            context.update(native_id=native_id, is_mandri_root=True, new_conversation=False)
            write_agy_json(profile / "mandri-launch.json", context)
    selected = context.get("native_id") == native_id
    binding = {
        "native_id": native_id,
        "cwd": context.get("cwd", ""),
        "model": context.get("model"),
        "model_source": context.get("model_source", "gateway"),
    }
    if selected and context.get("is_mandri_root") is True:
        binding["is_mandri_root"] = True
    write_agy_json(profile / "bindings" / f"{native_id}.json", binding)
    if selected:
        write_agy_json(profile / "mandri-session.json", binding)
    with (profile / "mandri-events.jsonl").open("a", encoding="utf-8") as journal:
        record = {"event": event, **payload, "_mandri_recorded_at_ns": time.time_ns()}
        journal.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    try:
        payload = json.load(sys.stdin)
        if isinstance(payload, dict):
            observe_agy(Path(sys.argv[1]), sys.argv[2], payload)
    except (OSError, ValueError, IndexError):
        pass
    print("{}")


if __name__ == "__main__":
    main()
