import re
from dataclasses import dataclass
from urllib.parse import quote

from mandri.gateway.surrogate.paths import case_like, root_occurrences
from mandri.gateway.surrogate.types import PathRoot

PERCENT_RUN = re.compile(r"(?:%[0-9a-fA-F]{2})+")


@dataclass(frozen=True, slots=True)
class EncodedRoot:
    start: int
    end: int
    token_end: int
    original: str
    surrogate: str


def encoded_roots(text: str, roots: list[PathRoot]) -> list[EncodedRoot]:
    if not roots or not PERCENT_RUN.search(text):
        return []
    decoded = []
    positions: list[tuple[int, int]] = []
    cursor = 0
    for match in PERCENT_RUN.finditer(text):
        for index in range(cursor, match.start()):
            decoded.append(text[index])
            positions.append((index, index + 1))
        raw = bytes.fromhex(match.group().replace("%", ""))
        try:
            unescaped = raw.decode("utf-8")
        except UnicodeError:
            for index in range(match.start(), match.end()):
                decoded.append(text[index])
                positions.append((index, index + 1))
            cursor = match.end()
            continue
        offset = match.start()
        for char in unescaped:
            width = len(char.encode("utf-8")) * 3
            decoded.append(char)
            positions.append((offset, offset + width))
            offset += width
        cursor = match.end()
    for index in range(cursor, len(text)):
        decoded.append(text[index])
        positions.append((index, index + 1))
    value = "".join(decoded)
    result = []
    for root in roots:
        for start, end, token_end in root_occurrences(value, root):
            raw_start, raw_end = positions[start][0], positions[end - 1][1]
            raw_original = text[raw_start:raw_end]
            if raw_original == value[start:end]:
                continue
            safe = "" if "%2f" in raw_original.lower() or "%5c" in raw_original.lower() else "/:\\"
            alias = quote(case_like(root.surrogate, value[start:end]), safe=safe)
            if re.search(r"%[0-9a-f]*[a-f]", raw_original) and not re.search(
                r"%[0-9A-F]*[A-F]", raw_original
            ):
                alias = re.sub(r"%[0-9A-F]{2}", lambda match: match.group().lower(), alias)
            result.append(
                EncodedRoot(raw_start, raw_end, positions[token_end - 1][1], raw_original, alias)
            )
    return result
