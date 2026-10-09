import re
from collections.abc import Callable

from mandri.gateway.surrogate.types import Span

HEAD = re.compile(
    r"(?im)(?<![\w-])(?:[\"'](?P<quoted>[a-z][a-z0-9_-]{1,48})[\"']|"
    r"(?P<plain>tenant[ \t]+uuid|[a-z][a-z0-9_-]{1,48}))[ \t]*[:=][ \t]*"
)
END = re.compile(r"[\r\n,;}]")
PERSON = re.compile(r"[^\W\d_]+(?:[ .'\u2019-][^\W\d_]+){0,7}")
USER = re.compile(r"[\w.@-]+(?:\\[\w.@-]+)?")


def labelled_spans(
    text: str,
    classify: Callable[[str, str], str | None],
    validate: Callable[[str, str], bool],
) -> list[Span]:
    heads = list(HEAD.finditer(text))
    spans: list[Span] = []
    for index, head in enumerate(heads):
        start = head.end()
        end = heads[index + 1].start() if index + 1 < len(heads) else len(text)
        label = head.group("quoted") or head.group("plain")
        if start >= end:
            continue
        quote = text[start] if text[start] in "\"'`" else ""
        if quote:
            closing = text.find(quote, start + 1, end)
            if closing < 0:
                continue
            start, end = start + 1, closing
        else:
            stop = END.search(text, start, end)
            if stop:
                end = stop.start()
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        for wrapper in ("**", "__", "`", "*"):
            if text[start:end].startswith(wrapper) and text[start:end].endswith(wrapper):
                start += len(wrapper)
                end -= len(wrapper)
        value = text[start:end]
        kind = classify(label, value)
        if not kind or not value or not validate(kind, value):
            continue
        if kind == "identity" and PERSON.fullmatch(value) is None:
            continue
        if kind == "address" and not re.match(r"\d+[ -]+\w", value):
            continue
        if kind == "username":
            if USER.fullmatch(value) is None:
                continue
            if "\\" in value:
                host, user = value.split("\\", 1)
                spans.extend(
                    (
                        Span(start, start + len(host), "hostname", 176),
                        Span(end - len(user), end, "username", 176),
                    )
                )
                continue
        spans.append(Span(start, end, kind, 176, label))
    return spans
