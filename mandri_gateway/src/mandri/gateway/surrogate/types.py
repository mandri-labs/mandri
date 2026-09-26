from dataclasses import asdict, dataclass, field
from typing import Any, Literal, get_args

type JSONValue = str | int | float | bool | list[JSONValue] | dict[str, JSONValue] | None
type EntityKind = Literal[
    "email",
    "identity",
    "username",
    "hostname",
    "domain",
    "git_owner",
    "git_repository",
    "git_remote",
    "url",
    "path_root",
    "ipv4",
    "ipv6",
    "cidr",
    "mac",
    "uuid",
    "identifier",
    "secret",
    "basic",
    "phone",
    "date_of_birth",
    "address",
    "iban",
    "card",
    "account",
    "cloud_resource",
    "private_package",
]
KINDS = frozenset(get_args(EntityKind.__value__))


@dataclass(frozen=True, slots=True)
class Mapping:
    kind: str
    original: str
    surrogate: str
    context: str = ""


@dataclass(frozen=True, slots=True)
class PathRoot:
    original: str
    surrogate: str
    case_sensitive: bool = True


@dataclass(frozen=True, slots=True)
class RuleSpec:
    name: str
    pattern: str
    kind: str = "identifier"
    fields: tuple[str, ...] = ()


@dataclass(slots=True)
class SurrogateScope:
    scope_id: str
    version: int = 2
    mappings: list[Mapping] = field(default_factory=list)
    roots: list[PathRoot] = field(default_factory=list)
    rules: list[RuleSpec] = field(default_factory=list)

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "scope_id": self.scope_id,
            "version": self.version,
            "mappings": [dict(asdict(item)) for item in self.mappings],
            "roots": [dict(asdict(item)) for item in self.roots],
            "rules": [
                {
                    "name": item.name,
                    "pattern": item.pattern,
                    "kind": item.kind,
                    "fields": list(item.fields),
                }
                for item in self.rules
            ],
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SurrogateScope":
        mappings = [
            Mapping(item["kind"], item["original"], item["surrogate"], item.get("context") or "")
            for item in records(value.get("mappings"))
            if isinstance(item, dict)
            and all(isinstance(item.get(key), str) for key in ("kind", "original", "surrogate"))
            and item["original"]
            and item["surrogate"]
            and isinstance(item.get("context", ""), (str, type(None)))
        ]
        roots = [
            PathRoot(item["original"], item["surrogate"], bool(item.get("case_sensitive", True)))
            for item in records(value.get("roots"))
            if isinstance(item, dict)
            and isinstance(item.get("original"), str)
            and isinstance(item.get("surrogate"), str)
            and item["original"]
            and item["surrogate"]
        ]
        rules = [
            RuleSpec(
                item["name"],
                item["pattern"],
                item.get("kind", "identifier"),
                tuple(field for field in item.get("fields", []) if isinstance(field, str)),
            )
            for item in records(value.get("rules"))
            if isinstance(item, dict)
            and isinstance(item.get("name"), str)
            and isinstance(item.get("pattern"), str)
            and isinstance(item.get("kind", "identifier"), str)
            and isinstance(item.get("fields", []), list)
        ]
        return cls(str(value.get("scope_id", "")), mappings=mappings, roots=roots, rules=rules)


def records(value: Any) -> list[dict[str, Any]]:
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


@dataclass(frozen=True, slots=True)
class Span:
    start: int
    end: int
    kind: str
    priority: int = 50
    context: str = ""
