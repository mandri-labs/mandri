import ntpath
import posixpath
import re
from collections.abc import Iterator
from dataclasses import replace

from mandri.core.types.execution import ProtectionError
from mandri.gateway.surrogate.types import Mapping, PathRoot

PATH_TOKEN = re.compile(r"(?<![\w@])(?:[A-Za-z]:[\\/]|\\\\|/|(?:\.{1,2}/)?[\w.-]+/)[^\s\"'`<>|;,]*")
PATH_END = frozenset("\r\n\t\"'`<>|;, ")
NETWORK_TOKEN = re.compile(r"\b[\w+.-]+://\S+|(?<![\w./])(?:[\w.-]+@)?[\w.-]+:[\w.%~-]+/\S+")


def windows_path(value: str) -> bool:
    return bool(re.match(r"^[a-zA-Z]:[\\/]", value)) or value.startswith("\\\\")


def absolute_path(value: str) -> bool:
    return value.startswith("/") or windows_path(value)


def normalized(value: str, case_sensitive: bool = True) -> str:
    result = ntpath.normpath(value) if windows_path(value) else posixpath.normpath(value)
    return result if case_sensitive else result.casefold()


def descendant(path: str, root: str, *, case_sensitive: bool = True) -> str | None:
    separator = "\\" if windows_path(root) and "\\" in root else "/"
    left, right = path.rstrip("/\\"), root.rstrip("/\\")
    compared_left = left if case_sensitive else left.casefold()
    compared_right = right if case_sensitive else right.casefold()
    if compared_left == compared_right:
        return ""
    if compared_left.startswith(compared_right + separator):
        return path[len(right) :]
    return None


def root_occurrences(
    text: str, root: PathRoot, *, reverse: bool = False
) -> Iterator[tuple[int, int, int]]:
    needle = root.surrogate if reverse else root.original
    flags = 0 if root.case_sensitive or reverse else re.IGNORECASE
    pattern = (
        "".join(r"[/\\]" if char in "/\\" else re.escape(char) for char in needle)
        if windows_path(needle)
        else re.escape(needle)
    )
    for match in re.finditer(pattern, text, flags):
        start, end = match.span()
        if start and (text[start - 1].isalnum() or text[start - 1] in "_.-\\"):
            continue
        if start and text[start - 1] == "/" and not text[:start].endswith("file://"):
            continue
        if end < len(text) and (text[end].isalnum() or text[end] in "_.-"):
            continue
        token_end = end
        while token_end < len(text) and text[token_end] not in PATH_END:
            token_end += 1
        token = text[start:token_end]
        normalized_token = normalized(
            token.replace("/", "\\") if windows_path(needle) else token, root.case_sensitive
        )
        normalized_root = normalized(needle, root.case_sensitive)
        if (
            not reverse
            and descendant(normalized_token, normalized_root, case_sensitive=root.case_sensitive)
            is None
        ):
            continue
        yield start, end, token_end


def path_reservations(text: str) -> list[tuple[int, int]]:
    network_ranges = [match.span() for match in NETWORK_TOKEN.finditer(text)]
    return [
        match.span()
        for match in PATH_TOKEN.finditer(text)
        if not any(start <= match.start() and match.end() <= end for start, end in network_ranges)
    ]


def case_like(value: str, template: str) -> str:
    if len(value) != len(template):
        return value
    return "".join(
        source
        if char in "/\\" and source in "/\\" and (windows_path(value) or windows_path(template))
        else char.upper()
        if source.isupper()
        else char.lower()
        if source.islower()
        else char
        for char, source in zip(value, template, strict=True)
    )


def path_mappings(mappings: list[Mapping]) -> list[Mapping]:
    variants: dict[str, Mapping] = {}
    for item in mappings:
        if item.kind != "path_root" or not (
            windows_path(item.original) and windows_path(item.surrogate)
        ):
            continue
        for separator in ("/", "\\"):
            original = item.original.replace("\\", separator).replace("/", separator)
            surrogate = item.surrogate.replace("\\", separator).replace("/", separator)
            variant = replace(item, original=original, surrogate=surrogate)
            existing = variants.get(surrogate)
            if existing and existing.original != original:
                raise ProtectionError(
                    "privacy_alias_collision", "Path replacement spellings conflict"
                )
            variants[surrogate] = variant
    for item in mappings:
        existing = variants.get(item.surrogate)
        equivalent = (
            existing is not None
            and item.kind == existing.kind == "path_root"
            and ntpath.normcase(ntpath.normpath(item.original))
            == ntpath.normcase(ntpath.normpath(existing.original))
        )
        if existing and existing.original != item.original and not equivalent:
            raise ProtectionError("privacy_alias_collision", "Path replacement spellings conflict")
        variants[item.surrogate] = item
    return list(variants.values())
