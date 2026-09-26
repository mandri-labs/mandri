from collections.abc import Callable
from typing import Any


def context_edit(value: Any, transform: Callable[[Any], Any]) -> Any:
    if not isinstance(value, dict) or not isinstance(value.get("edits"), list):
        return value
    return {
        **value,
        "edits": [
            {**edit, "exclude_tools": transform(edit["exclude_tools"])}
            if isinstance(edit, dict)
            and edit.get("type") == "clear_tool_uses_20250919"
            and isinstance(edit.get("exclude_tools"), list)
            else edit
            for edit in value["edits"]
        ],
    }
