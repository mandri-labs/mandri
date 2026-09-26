from mandri.gateway.surrogate.types import Mapping


def redact(value: str, kind: str) -> str:
    if kind == "email" and "@" in value:
        local, domain = value.rsplit("@", 1)
        prefix = local[: min(3, len(local) // 3)]
        suffix = domain[-min(7, len(domain) // 2) :] if len(domain) > 2 else ""
        return prefix + "****@****" + suffix
    if kind == "secret" and len(value) > 8:
        return "********" + value[-4:]
    if kind in {"secret", "basic"} or len(value) < 6:
        return "********"
    visible = min(3, len(value) // 4)
    return value[:visible] + "****" + value[-visible:]


def inventory(mappings: list[Mapping]) -> list[dict[str, str]]:
    current = {item.original: item for item in mappings}
    return [
        {"kind": item.kind, "redacted": redact(item.original, item.kind)}
        for item in current.values()
    ]
