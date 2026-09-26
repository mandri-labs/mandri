import re
from collections.abc import Iterable
from dataclasses import replace
from typing import Any

from mandri.gateway.surrogate.allocation import unique_alias
from mandri.gateway.surrogate.compound import transform_git, transform_url
from mandri.gateway.surrogate.detectors import context_kind, detect, select_spans
from mandri.gateway.surrogate.encoded_paths import encoded_roots
from mandri.gateway.surrogate.formats import SurrogateGenerator, valid
from mandri.gateway.surrogate.json_text import rewrite_json
from mandri.gateway.surrogate.matching import LiteralIndex
from mandri.gateway.surrogate.path_aliases import PathAliases
from mandri.gateway.surrogate.paths import (
    absolute_path,
    case_like,
    descendant,
    normalized,
    path_reservations,
    root_occurrences,
    windows_path,
)
from mandri.gateway.surrogate.public import PUBLIC_HOSTS, PUBLIC_PACKAGE_SCOPES
from mandri.gateway.surrogate.rules import TypedRule
from mandri.gateway.surrogate.types import (
    KINDS,
    JSONValue,
    Mapping,
    PathRoot,
    RuleSpec,
    Span,
    SurrogateScope,
)

PATH_EXEMPT_KINDS = frozenset(
    {
        "identity",
        "username",
        "hostname",
        "domain",
        "git_owner",
        "git_repository",
        "identifier",
        "account",
    }
)


def literal_occurrences(text: str, value: str) -> Iterable[tuple[int, int]]:
    cursor = 0
    while (start := text.find(value, cursor)) >= 0:
        end = start + len(value)
        cursor = end
        if value[0].isalnum() and start and (text[start - 1].isalnum() or text[start - 1] == "_"):
            continue
        if value[-1].isalnum() and end < len(text) and (text[end].isalnum() or text[end] == "_"):
            continue
        yield start, end


class SurrogateEngine:
    def __init__(
        self,
        scope: SurrogateScope,
        *,
        generator: SurrogateGenerator | None = None,
        rules: Iterable[TypedRule] = (),
        public_hosts: Iterable[str] = PUBLIC_HOSTS,
    ) -> None:
        self.scope = scope
        self.generator = generator or SurrogateGenerator()
        provided = tuple(rules)
        if provided:
            specs = [
                RuleSpec(rule.name, rule.pattern, rule.kind, tuple(sorted(rule.fields)))
                for rule in provided
            ]
            names = {rule.name for rule in scope.rules}
            scope.rules.extend(rule for rule in specs if rule.name not in names)
        self.rules = tuple(
            TypedRule(rule.name, rule.pattern, rule.kind, frozenset(rule.fields))
            for rule in scope.rules
        )
        self.public_hosts = frozenset(host.casefold().rstrip(".") for host in public_hosts)
        self._private_public_hosts = {
            item.original.casefold().rstrip(".")
            for item in scope.mappings
            if item.original.casefold().rstrip(".") in self.public_hosts
        }
        self._originals = {item.original: item for item in scope.mappings}
        self._surrogates = {item.surrogate: item for item in scope.mappings}
        self._private_values = {
            item.original.casefold() for item in scope.mappings if len(item.original) >= 4
        }
        self._private_lengths = {len(value) for value in self._private_values}
        self._path_aliases = PathAliases(scope.roots)
        self._reserved: tuple[str, ...] = ()
        self._forward = LiteralIndex(scope.mappings)
        self._host_index = self._hosts()
        self._reverse = LiteralIndex(scope.mappings, reverse=True)
        self._indexed_size = len(scope.mappings)
        self._neutral_roots: list[PathRoot] = []
        self._legacy = LiteralIndex([])
        self._upgrade_aliases()
        self._legacy = LiteralIndex(
            [
                Mapping(item.kind, item.surrogate, self._originals[item.original].surrogate)
                for item in scope.mappings
                if any(word in item.surrogate.casefold() for word in ("surrogate", "mandri"))
            ]
        )

    def _upgrade_aliases(self) -> None:
        mappings = sorted(
            self._originals.values(), key=lambda item: item.kind in {"url", "git_remote"}
        )
        for item in mappings:
            if not any(word in item.surrogate.casefold() for word in ("surrogate", "mandri")):
                continue
            candidate = (
                self._compound(item.kind, item.original)
                if item.kind in {"url", "git_remote"}
                else self.generator.generate(item.kind, item.original)
            )
            self._record(item.kind, item.original, candidate, item.context, replace=True)
        self.scope.roots = [
            replace(root, surrogate=self._originals[root.original].surrogate)
            if root.original in self._originals
            else root
            for root in self.scope.roots
        ]
        self._path_aliases = PathAliases(self.scope.roots)
        self._reset()

    def _replace_legacy(self, value: str) -> str:
        matches = {(start, end): item for start, end, item in self._legacy.find(value)}
        spans = select_spans(
            [Span(start, end, item.kind) for (start, end), item in matches.items()]
        )
        pieces: list[str] = []
        cursor = 0
        for span in spans:
            pieces.extend((value[cursor : span.start], matches[span.start, span.end].surrogate))
            cursor = span.end
        pieces.append(value[cursor:])
        return "".join(pieces)

    def register(self, value: str, kind: str = "identity", context: str | None = None) -> str:
        return self._allocate(kind, value, context or "")

    def reserve_root(self, path: str) -> None:
        if absolute_path(path) and not any(root.original == path for root in self._neutral_roots):
            self._neutral_roots.append(
                PathRoot(normalized(path), normalized(path), not windows_path(path))
            )

    def register_root(
        self,
        path: str,
        surrogate: str | None = None,
        *,
        case_sensitive: bool | None = None,
        independent: bool = False,
    ) -> str:
        if not absolute_path(path) or path in {"/", "\\"}:
            return path
        path = path.rstrip("/\\")
        existing = next((root for root in self.scope.roots if root.original == path), None)
        if existing:
            return existing.surrogate
        case_sensitive = not windows_path(path) if case_sensitive is None else case_sensitive
        parent = max(
            (
                root
                for root in self.scope.roots
                if descendant(path, root.original, case_sensitive=root.case_sensitive) is not None
            ),
            key=lambda root: len(root.original),
            default=None,
        )
        if parent and not independent:
            surrogate = parent.surrogate + (
                descendant(path, parent.original, case_sensitive=parent.case_sensitive) or ""
            )
        elif not independent:
            for child in self.scope.roots:
                suffix = descendant(child.original, path, case_sensitive=case_sensitive)
                if suffix and child.surrogate.endswith(suffix):
                    surrogate = child.surrogate[: -len(suffix)]
                    break
        alias = (
            self._record("path_root", path, surrogate.rstrip("/\\"), "")
            if surrogate and absolute_path(surrogate)
            else self._allocate("path_root", path, "")
        )
        root = PathRoot(path, alias, case_sensitive)
        self.scope.roots.append(root)
        self._path_aliases.remember(root)
        return alias

    def protect(self, value: JSONValue) -> JSONValue:
        self._reset()
        self._reserved = tuple(self._strings(value))
        self._walk(value, "", "discover")
        self._reset()
        result = self._walk(value, "", "protect")
        return result

    def discover(self, value: JSONValue) -> None:
        self._reset()
        self._reserved = tuple(self._strings(value))
        self._walk(value, "", "discover")

    def restore(self, value: JSONValue) -> JSONValue:
        self._reset()
        return self._walk(value, "", "restore")

    def protect_text(self, value: str, context: str | None = None) -> str:
        self._reset()
        self._reserved = (value,)
        self._text(value, context or "", "discover")
        self._reset()
        result = self._text(value, context or "", "protect")
        return result

    def restore_text(self, value: str) -> str:
        self._reset()
        return self._text(value, "", "restore")

    def restore_offsets(self, value: str, offsets: list[int]) -> list[int]:
        self._reset()
        spans = self._restoration_spans(value)
        result = []
        for offset in offsets:
            if type(offset) is not int or not 0 <= offset <= len(value):
                result.append(offset)
                continue
            shift = 0
            for span in spans:
                if span.start < offset < span.end:
                    original = self._surrogates[value[span.start : span.end]].original
                    shift += min(offset - span.start, len(original)) - (offset - span.start)
                    break
                if span.end <= offset:
                    original = self._surrogates[value[span.start : span.end]].original
                    shift += len(original) - (span.end - span.start)
            result.append(offset + shift)
        return result

    def _reset(self) -> None:
        if self._indexed_size != len(self.scope.mappings):
            self._forward = LiteralIndex(self.scope.mappings)
            self._host_index = self._hosts()
            self._reverse = LiteralIndex(self.scope.mappings, reverse=True)
            self._indexed_size = len(self.scope.mappings)

    def _hosts(self) -> LiteralIndex:
        names = {
            item.original.casefold().rstrip("."): item
            for item in self.scope.mappings
            if item.kind in {"domain", "hostname"}
        }
        return LiteralIndex(
            [Mapping(item.kind, name, item.surrogate, item.context) for name, item in names.items()]
        )

    def _strings(self, value: JSONValue) -> Iterable[str]:
        pending = [value]
        while pending:
            item = pending.pop()
            if isinstance(item, str):
                yield item
            elif isinstance(item, dict):
                pending.extend(item.keys())
                pending.extend(item.values())
            elif isinstance(item, list):
                pending.extend(item)

    def _walk(self, value: JSONValue, context: str, operation: str) -> JSONValue:
        result: list[JSONValue] = [None]
        pending: list[tuple[JSONValue, str, Any, Any]] = [(value, context, result, 0)]
        while pending:
            item, key_context, target, key = pending.pop()
            if isinstance(item, dict):
                mapped: dict[str, JSONValue] = {}
                target[key] = mapped
                keys = {
                    name: self._text(name, "", operation) if isinstance(name, str) else name
                    for name in item
                }
                counts: dict[str, int] = {}
                for name in keys.values():
                    counts[name] = counts.get(name, 0) + 1
                for name, child in reversed(tuple(item.items())):
                    alias = keys[name] if counts[keys[name]] == 1 else name
                    original = (
                        self._restore_plain(name)
                        if operation == "restore" and isinstance(name, str)
                        else name
                    )
                    pending.append(
                        (child, original if isinstance(original, str) else "", mapped, alias)
                    )
            elif isinstance(item, list):
                children: list[JSONValue] = [None] * len(item)
                target[key] = children
                pending.extend(
                    (child, key_context, children, index)
                    for index, child in reversed(tuple(enumerate(item)))
                )
            elif isinstance(item, str):
                target[key] = self._text(item, key_context, operation)
            elif (
                isinstance(item, int)
                and not isinstance(item, bool)
                and context_kind(key_context, str(item))
            ):
                text = self._plain(str(item), key_context, operation)
                target[key] = int(text) if re.fullmatch(r"-?(?:0|[1-9]\d*)", text) else item
            else:
                target[key] = item
        return result[0]

    def _text(self, value: str, context: str, operation: str) -> str:
        nested = rewrite_json(value, lambda item: self._walk(item, context, operation))
        return nested if nested is not None else self._plain(value, context, operation)

    def _plain(self, value: str, context: str, operation: str) -> str:
        if operation == "restore":
            return self._restore_plain(value, context)
        value = self._replace_legacy(value)
        reservations = (
            [] if context in {"$ref", "$dynamicRef", "$url_path"} else path_reservations(value)
        )
        neutral_reservations = []
        for root in () if context == "$url_path" else self._neutral_roots:
            for start, _, end in root_occurrences(value, root):
                reservations.append((start, end))
                neutral_reservations.append((start, end))
        spans = detect(
            value,
            context,
            self.rules,
            public_package_scopes=PUBLIC_PACKAGE_SCOPES - self._originals.keys(),
        )
        root_spans: list[Span] = []
        for encoded in encoded_roots(value, self.scope.roots):
            if any(
                left <= encoded.start and encoded.token_end <= right
                for left, right in neutral_reservations
            ):
                continue
            if operation == "protect":
                self._record("path_root", encoded.original, encoded.surrogate, "")
            root_spans.append(Span(encoded.start, encoded.end, "path_root", 200))
            reservations.append((encoded.end, encoded.token_end))
        for root in self.scope.roots:
            for start, end, token_end in root_occurrences(value, root):
                if any(
                    left <= start and token_end <= right for left, right in neutral_reservations
                ):
                    continue
                original = value[start:end]
                if operation == "protect" and original != root.original:
                    self._record("path_root", original, case_like(root.surrogate, original), "")
                root_spans.append(Span(start, end, "path_root", 200))
                reservations.append((end, token_end))
        folded = value.translate(
            str.maketrans("ABCDEFGHIJKLMNOPQRSTUVWXYZ", "abcdefghijklmnopqrstuvwxyz")
        )
        for start, end, mapping in self._host_index.find(folded):
            spans.append(Span(start, end, mapping.kind, 155, mapping.context))
        for start, end, mapping in self._forward.find(value):
            if mapping.kind == "path_root":
                continue
            spans.append(Span(start, end, mapping.kind, 150, mapping.context))
        spans = [
            span
            for span in spans
            if span.kind not in PATH_EXEMPT_KINDS
            or not any(start <= span.start and span.end <= end for start, end in reservations)
        ]
        aliases = [(start, end) for start, end, _ in self._reverse.find(value)]
        spans = select_spans(
            [
                span
                for span in root_spans + spans
                if not any(start <= span.start and span.end <= end for start, end in aliases)
            ]
        )
        if operation == "discover":
            for span in spans:
                if span.kind in {"url", "git_remote"}:
                    self._compound(span.kind, value[span.start : span.end], operation="discover")
                elif span.kind in {"cloud_resource", "private_package"}:
                    self._resource(span.kind, value[span.start : span.end])
                elif span.kind != "path_root":
                    self._allocate(span.kind, value[span.start : span.end], span.context)
            return value
        pieces: list[str] = []
        cursor = 0
        for span in spans:
            original = value[span.start : span.end]
            surrogate = self._allocate(span.kind, original, span.context)
            pieces.extend((value[cursor : span.start], surrogate))
            cursor = span.end
        pieces.append(value[cursor:])
        return "".join(pieces)

    def _restore_plain(self, value: str, context: str = "") -> str:
        pieces: list[str] = []
        cursor = 0
        for span in self._restoration_spans(value, context):
            mapping = self._surrogates[value[span.start : span.end]]
            pieces.extend((value[cursor : span.start], mapping.original))
            cursor = span.end
        pieces.append(value[cursor:])
        return "".join(pieces)

    def _restoration_spans(self, value: str, context: str = "") -> list[Span]:
        spans = []
        for start, end, mapping in self._reverse.find(value):
            if mapping.kind == "path_root":
                root = PathRoot(mapping.original, mapping.surrogate)
                if not any(
                    left == start for left, _, _ in root_occurrences(value, root, reverse=True)
                ):
                    continue
            priority = (
                300
                if mapping.kind in {"url", "git_remote", "cloud_resource", "private_package"}
                else 200
                if mapping.kind == "path_root"
                else 100
            )
            spans.append(Span(start, end, mapping.kind, priority))
        return select_spans(spans)

    def _compound(self, kind: str, original: str, *, operation: str = "protect") -> str:
        if kind == "git_remote":
            return transform_git(
                original, self._allocate, self.public_hosts - self._private_public_hosts
            )
        return transform_url(
            original,
            self._allocate,
            lambda text, context: self._plain(text, context, operation),
            self.public_hosts - self._private_public_hosts,
        )

    def _resource(self, kind: str, original: str) -> str:
        if kind == "private_package":
            scope, slash, package = original.removeprefix("@").partition("/")
            if not slash:
                return original
            return (
                "@"
                + self._allocate("git_owner", scope, "")
                + "/"
                + self._allocate("git_repository", package, "")
            )
        parts = original.split(":", 5)
        if len(parts) != 6 or parts[0] != "arn":
            return original
        if parts[4]:
            parts[4] = self._allocate("account", parts[4], "account_id")
        parts[5] = self._allocate("identifier", parts[5], "resource")
        return ":".join(parts)

    def _allocate(self, kind: str, original: str, context: str) -> str:
        existing = self._originals.get(original)
        if existing:
            if kind in {"url", "git_remote"}:
                candidate = self._compound(kind, original)
                if candidate not in {original, existing.surrogate}:
                    return self._record(kind, original, candidate, context, replace=True)
            return existing.surrogate
        if original in self._surrogates or kind not in KINDS or not valid(kind, original):
            return original
        rule = next((item for item in self.rules if item.name == context), None)
        if rule is not None:
            candidate = rule.generate(original)
        elif kind in {"url", "git_remote"}:
            candidate = self._compound(kind, original)
            if candidate == original:
                return original
        elif kind in {"cloud_resource", "private_package"}:
            candidate = self._resource(kind, original)
        elif kind == "path_root":
            candidate = self._path_aliases.generate(
                original, self._originals, self.generator, self._available
            )
        elif kind == "username" and original in self._path_aliases.components:
            candidate = self._path_aliases.components[original]
        else:
            candidate = self.generator.generate(kind, original)
        return self._record(kind, original, candidate, context)

    def _available(self, original: str, candidate: str) -> bool:
        if (
            not candidate
            or candidate == original
            or candidate in self._originals
            or candidate in self._surrogates
            or (len(original) >= 4 and original.casefold() in candidate.casefold())
            or self._contains_private(candidate)
        ):
            return False
        return not any(candidate in text for text in self._reserved)

    def _contains_private(self, candidate: str) -> bool:
        folded = candidate.casefold()
        return any(
            folded[start : start + width] in self._private_values
            for width in self._private_lengths
            for start in range(len(folded) - width + 1)
        )

    def _record(
        self, kind: str, original: str, surrogate: str, context: str, *, replace: bool = False
    ) -> str:
        existing = self._originals.get(original)
        if existing and not replace:
            return existing.surrogate
        rule = next((item for item in self.rules if item.name == context), None)
        alphabets = (
            tuple(
                "".join(chr(code) for code, mask in enumerate(rule.masks) if mask & (1 << index))
                for index in range(rule.width)
            )
            if rule is not None and rule.width == len(surrogate)
            else ()
        )
        surrogate = unique_alias(
            original, surrogate, self._available, kind=kind, alphabets=alphabets
        )
        mapping = Mapping(kind, original, surrogate, context)
        self.scope.mappings.append(mapping)
        self._originals[original] = mapping
        if len(original) >= 4:
            self._private_values.add(original.casefold())
            self._private_lengths.add(len(original.casefold()))
        if original.casefold().rstrip(".") in self.public_hosts:
            self._private_public_hosts.add(original.casefold().rstrip("."))
        self._surrogates[surrogate] = mapping
        return surrogate
