import json


def request_reference(identifier: object) -> str | None:
    if isinstance(identifier, bool) or not isinstance(identifier, (int, str)):
        return None
    return json.dumps(identifier, ensure_ascii=True, separators=(",", ":"))
