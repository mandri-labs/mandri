import re
import secrets
import string
from dataclasses import dataclass, field

from mandri.gateway.surrogate.types import KINDS, Span

RULE_TOKEN = re.compile(
    r"(?:\\[dws]|\\[^A-Za-z0-9]|\[(?:\\.|[^\]\\])+\]|[\w @:./-])(?:\{\d+(?:,\d+)?\})?"
)


@dataclass(frozen=True, slots=True)
class TypedRule:
    name: str
    pattern: str
    kind: str = "identifier"
    fields: frozenset[str] = frozenset()
    compiled: re.Pattern[str] = field(
        init=False, repr=False, compare=False, default=re.compile(r"(?!)")
    )
    masks: tuple[int, ...] = field(init=False, repr=False, compare=False, default=())
    width: int = field(init=False, repr=False, compare=False, default=0)

    def __post_init__(self) -> None:
        if not self.name or self.kind not in KINDS:
            return
        if not self.pattern or not self.pattern.isascii():
            return
        body = self.pattern.removeprefix("^").removesuffix("$")
        cursor = 0
        maximum_length = 0
        masks = [0] * 128
        mutable = False
        for token in RULE_TOKEN.finditer(body):
            if token.start() != cursor:
                return
            quantifier = re.search(r"\{(\d+)(?:,(\d+))?\}$", token.group())
            if quantifier:
                lower = int(quantifier.group(1))
                upper = int(quantifier.group(2) or quantifier.group(1))
                if lower < 1 or upper != lower:
                    return
            else:
                upper = 1
            atom = re.sub(r"\{\d+(?:,\d+)?\}$", "", token.group())
            bitmask = ((1 << upper) - 1) << maximum_length
            try:
                matcher = re.compile(atom, re.ASCII)
            except re.error:
                return
            alphabet = [code for code in range(128) if matcher.fullmatch(chr(code))]
            mutable |= len(alphabet) > 1
            for codepoint in alphabet:
                masks[codepoint] |= bitmask
            maximum_length += upper
            cursor = token.end()
        if cursor != len(body) or not mutable:
            return
        try:
            object.__setattr__(self, "compiled", re.compile(self.pattern, re.ASCII))
            object.__setattr__(self, "masks", tuple(masks))
            object.__setattr__(self, "width", maximum_length)
        except re.error:
            return

    def find(self, text: str, context: str) -> list[Span]:
        if not self.width or (self.fields and context.casefold() not in self.fields):
            return []
        state = 0
        terminal = 1 << (self.width - 1)
        result = []
        for index, char in enumerate(text):
            codepoint = ord(char)
            state = ((state << 1) | 1) & (self.masks[codepoint] if codepoint < 128 else 0)
            if state & terminal:
                start, end = index + 1 - self.width, index + 1
                if self.pattern.startswith("^") and start != 0:
                    break
                if (
                    self.pattern.endswith("$")
                    and end != len(text)
                    and not (end == len(text) - 1 and text[-1] == "\n")
                ):
                    continue
                result.append(Span(start, end, self.kind, 190, self.name))
                state = 0
        return result

    def generate(self, original: str) -> str:
        if not self.width:
            return original
        body = self.pattern.removeprefix("^").removesuffix("$")
        tokens = [match.group() for match in RULE_TOKEN.finditer(body)]
        groups = re.fullmatch("".join("(" + token + ")" for token in tokens), original)
        if groups is None:
            return original
        result = []
        for token, source in zip(tokens, groups.groups(), strict=True):
            atom = re.sub(r"\{\d+(?:,\d+)?\}$", "", token)
            alphabet = [
                char
                for char in string.ascii_letters + string.digits + "_- .@:/"
                if re.fullmatch(atom, char)
            ]
            if not alphabet:
                result.append(source)
                continue
            for original_char in source:
                typed_alphabet = [
                    char
                    for char in alphabet
                    if (char.isdigit() and original_char.isdigit())
                    or (
                        char.isalpha()
                        and original_char.isalpha()
                        and char.isupper() == original_char.isupper()
                    )
                    or char == original_char
                ]
                result.append(secrets.choice(typed_alphabet or alphabet))
        return "".join(result)
