import json
import re
from typing import Any

_STRING = r'"(?:\\.|[^"\\])*"'
_COMMENTS = re.compile(_STRING + r"|//[^\r\n]*|/\*[\s\S]*?\*/")
_COMMAS = re.compile(_STRING + r"|,\s*(?=[}\]])")


def loads(content: str) -> Any:
    content = _COMMENTS.sub(
        lambda match: match.group() if match.group().startswith('"') else " ", content
    )
    content = _COMMAS.sub(
        lambda match: match.group() if match.group().startswith('"') else "", content
    )
    return json.loads(content)
