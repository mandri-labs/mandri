import re
import secrets
import string
from collections.abc import Callable, Iterator
from itertools import islice

from mandri.core.types.execution import ProtectionError
from mandri.gateway.surrogate.formats import SurrogateGenerator, valid


def variants(value: str, alphabets: tuple[str, ...]) -> Iterator[str]:
    chars = list(value)
    while True:
        for index in range(len(chars) - 1, -1, -1):
            alphabet = alphabets[index]
            if not alphabet:
                continue
            position = alphabet.index(chars[index])
            chars[index] = alphabet[(position + 1) % len(alphabet)]
            if position + 1 < len(alphabet):
                break
        alias = "".join(chars)
        if alias == value:
            return
        yield alias


def unique_alias(
    original: str,
    candidate: str,
    available: Callable[[str, str], bool],
    *,
    kind: str = "identity",
    alphabets: tuple[str, ...] = (),
) -> str:
    if re.fullmatch(r"-?(?:0|[1-9]\d*)", original) and re.fullmatch(r"-?0\d+", candidate):
        start = int(candidate.startswith("-"))
        candidate = candidate[:start] + "1" + candidate[start + 1 :]

    def acceptable(alias: str) -> bool:
        return (
            not any(word in alias.casefold() for word in ("surrogate", "mandri"))
            and available(original, alias)
            and valid(kind, alias, original)
        )

    if acceptable(candidate):
        return candidate
    if not alphabets:
        alphabets = tuple(
            next(
                (
                    group
                    for group in (string.digits, string.ascii_lowercase, string.ascii_uppercase)
                    if char in group
                ),
                "",
            )
            for char in candidate
        )
    for alias in islice(variants(candidate, alphabets), 128):
        if acceptable(alias):
            return alias
    generator = SurrogateGenerator()
    for _ in range(128):
        alias = generator.generate(kind, original)
        if kind in {"identity", "username", "hostname", "git_owner", "git_repository", "address"}:
            alias = secrets.token_hex(12)
        elif kind == "path_root":
            alias = ("C:\\" if "\\" in original else "/") + secrets.token_hex(12)
        if acceptable(alias):
            return alias
    raise ProtectionError("privacy_alias_unavailable", "Unable to generate a distinct replacement")
