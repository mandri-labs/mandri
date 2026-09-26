import json


def is_internal_source(source: object) -> bool:
    if isinstance(source, str):
        try:
            source = json.loads(source)
        except ValueError:
            return False
    if not isinstance(source, dict):
        return False
    subagent = source.get("subagent", source.get("sub_agent", source.get("subAgent")))
    return isinstance(subagent, dict) and subagent.get("other") == "guardian"
