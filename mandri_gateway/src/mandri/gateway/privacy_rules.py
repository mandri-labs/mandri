import json
import os
import stat
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mandri.core.types.execution import ProtectionError
from mandri.gateway.privacy_context import metadata_read_flags
from mandri.gateway.surrogate import TypedRule
from mandri.gateway.surrogate.types import KINDS, RuleSpec


@dataclass(frozen=True, slots=True)
class ExactValue:
    value: str
    kind: str = "identity"
    context: str = ""


@dataclass(frozen=True, slots=True)
class PrivacyExtensions:
    exact: tuple[ExactValue, ...] = ()
    rules: tuple[RuleSpec, ...] = ()


def invalid() -> ProtectionError:
    return ProtectionError(
        "privacy_rules_invalid", "Local privacy rules are invalid or unavailable"
    )


def unique_object(items: list[tuple[str, Any]]) -> dict[str, Any]:
    value = {}
    for key, child in items:
        if key in value:
            raise invalid()
        value[key] = child
    return value


def read_rules(path: Path, workspace: Path) -> bytes:
    if sys.platform == "win32":
        raise ProtectionError(
            "privacy_platform_unsupported",
            "Secure local metadata reads are unavailable on this platform",
        )
    else:
        flags = metadata_read_flags()
        if not path.is_absolute() or path.resolve().is_relative_to(workspace.resolve()):
            raise invalid()
        try:
            descriptor = os.open(path, flags)
        except OSError:
            raise invalid() from None
        try:
            metadata = os.fstat(descriptor)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_uid != os.geteuid()
                or metadata.st_mode & 63
            ):
                raise invalid()
            with os.fdopen(descriptor, "rb", closefd=False) as file:
                raw = file.read()
            return raw
        except OSError:
            raise invalid() from None
        finally:
            os.close(descriptor)


def load_extensions(filename: str | None, workspace: Path) -> PrivacyExtensions:
    if filename is None:
        return PrivacyExtensions()
    raw = read_rules(Path(filename).expanduser(), workspace)
    try:
        value = json.loads(raw, object_pairs_hook=unique_object)
    except (ValueError, UnicodeError, RecursionError):
        raise invalid() from None
    if (
        not isinstance(value, dict)
        or set(value) - {"version", "exact_values", "rules"}
        or type(value.get("version")) is not int
        or (value["version"] != 1)
    ):
        raise invalid()
    raw_exact, raw_rules = (value.get("exact_values", []), value.get("rules", []))
    if not isinstance(raw_exact, list) or (not isinstance(raw_rules, list)):
        raise invalid()
    exact = []
    for item in raw_exact:
        if not isinstance(item, dict) or set(item) - {"value", "kind", "context"}:
            raise invalid()
        original, kind, context = (
            item.get("value"),
            item.get("kind", "identity"),
            item.get("context", ""),
        )
        if (
            not isinstance(original, str)
            or not original
            or (not isinstance(kind, str))
            or (kind not in KINDS - {"path_root"})
            or (not isinstance(context, str))
        ):
            raise invalid()
        exact.append(ExactValue(original, kind, context))
    rules: list[RuleSpec] = []
    for item in raw_rules:
        if not isinstance(item, dict) or set(item) - {"name", "pattern", "kind", "fields"}:
            raise invalid()
        name, pattern, kind, fields = (
            item.get("name"),
            item.get("pattern"),
            item.get("kind", "identifier"),
            item.get("fields", []),
        )
        if (
            not isinstance(name, str)
            or name in {rule.name for rule in rules}
            or (not isinstance(pattern, str))
            or (not isinstance(kind, str))
            or (kind not in KINDS - {"path_root"})
            or (not isinstance(fields, list))
            or any(not isinstance(field, str) or not field for field in fields)
        ):
            raise invalid()
        rule = TypedRule(name, pattern, kind, frozenset(field.casefold() for field in fields))
        if not rule.width:
            continue
        rules.append(RuleSpec(rule.name, rule.pattern, rule.kind, tuple(sorted(rule.fields))))
    return PrivacyExtensions(tuple(exact), tuple(rules))
