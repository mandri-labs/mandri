import base64
import json
import re
import string
from collections.abc import Iterator
from urllib.parse import quote, quote_plus

from mandri.core.types.execution import ProtectionError
from mandri.gateway.surrogate.matching import LiteralIndex
from mandri.gateway.surrogate.types import Mapping

ALPHABETS = {
    "base64": string.ascii_letters + string.digits + "+/=_-",
    "base32": string.ascii_letters + string.digits + "=",
    "hex": string.ascii_letters + string.digits,
    "percent": string.ascii_letters + string.digits + "%._~-",
    "escaped": string.ascii_letters + string.digits + "_\\",
}


def unicode_escape(value: str) -> str:
    encoded = value.encode("utf-16-be")
    return "".join(
        f"\\u{int.from_bytes(encoded[index : index + 2], 'big'):04x}"
        for index in range(0, len(encoded), 2)
    )


def punctuation_escape(value: str, *, percent: bool = False) -> str:
    return "".join(
        char
        if char.isalnum() or char in "-_"
        else "".join(f"%{byte:02X}" for byte in char.encode())
        if percent
        else unicode_escape(char)
        for char in value
    )


def renderings(value: str) -> dict[str, str]:
    raw = value.encode("utf-8")
    result = {
        "base64:standard": base64.b64encode(raw).decode(),
        "base64:url": base64.urlsafe_b64encode(raw).decode(),
        "base32:upper": base64.b32encode(raw).decode(),
        "hex:lower": raw.hex(),
        "hex:upper": raw.hex().upper(),
        "percent:component": quote(value, safe=""),
        "percent:form": quote_plus(value, safe=""),
        "percent:full": "".join(f"%{byte:02X}" for byte in raw),
        "percent:punctuation": punctuation_escape(value, percent=True),
        "escaped:json": json.dumps(value, ensure_ascii=True)[1:-1],
        "escaped:json:double": json.dumps(
            json.dumps(value, ensure_ascii=True)[1:-1], ensure_ascii=True
        )[1:-1],
        "escaped:unicode": unicode_escape(value),
        "escaped:punctuation": punctuation_escape(value),
    }
    for key, encoded in list(result.items()):
        if key.startswith(("base64:", "base32:")):
            result[key + ":unpadded"] = encoded.rstrip("=")
        elif key.startswith("percent:"):
            result[key + ":lower"] = re.sub(r"%[0-9A-F]{2}", lambda m: m.group().lower(), encoded)
    return result


def boundary(text: str, start: int, end: int, mapping: Mapping) -> bool:
    alphabet = ALPHABETS[mapping.context.split(":", 1)[0]]
    if mapping.context.endswith(":path_root"):
        return (not start or text[start - 1] not in alphabet) and (
            end == len(text) or text[end] not in alphabet or text[end] in "/\\"
        )
    return (not start or text[start - 1] not in alphabet) and (
        end == len(text) or text[end] not in alphabet
    )


class KnownRepresentations:
    def __init__(self, mappings: list[Mapping], observed: list[Mapping]) -> None:
        forward: dict[str, Mapping] = {}
        reverse: dict[str, Mapping] = {}
        ambiguous: set[str] = set()
        used = 0
        canonical = {item.original: item for item in mappings}
        for item in mappings:
            if not 6 <= len(item.original) <= 4096:
                continue
            try:
                originals, aliases = renderings(item.original), renderings(item.surrogate)
            except UnicodeError:
                continue
            for form, original in originals.items():
                alias = aliases[form]
                if original == item.original or alias == item.surrogate or len(original) < 8:
                    continue
                if (
                    original in forward
                    and forward[original].surrogate != alias
                    and item == canonical[item.original]
                ):
                    forward[original] = Mapping("encoded", original, alias, form)
                if alias in reverse and reverse[alias].original != original:
                    ambiguous.add(alias)
                context = (
                    form + ":path_root"
                    if item.kind == "path_root" and form.startswith("escaped:json")
                    else form
                )
                mapping = Mapping("encoded", original, alias, context)
                forward.setdefault(original, mapping)
                reverse.setdefault(alias, mapping)
                used += len(original) + len(alias)
                if used > 4 * 1024 * 1024:
                    raise ProtectionError(
                        "privacy_alias_capacity", "Encoded replacement capacity exceeded"
                    )
        self.forward = LiteralIndex(list(forward.values()))
        reverse = {alias: mapping for alias, mapping in reverse.items() if alias not in ambiguous}
        reverse.update({mapping.surrogate: mapping for mapping in observed})
        self.reverse = LiteralIndex(list(reverse.values()), reverse=True)
        self.mappings = list(reverse.values())

    def find(self, text: str, *, reverse: bool = False) -> Iterator[tuple[int, int, Mapping]]:
        index = self.reverse if reverse else self.forward
        for start, end, mapping in index.find(text):
            if boundary(text, start, end, mapping):
                yield start, end, mapping
