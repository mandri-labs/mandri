import re

_COMPONENT = re.compile(r"[a-z0-9]+(?:(?:[._]|__|-+)[a-z0-9]+)*")
_DOMAIN = re.compile(
    r"[a-zA-Z0-9](?:[a-zA-Z0-9-]*[a-zA-Z0-9])?(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]*[a-zA-Z0-9])?)*(?::[0-9]{1,5})?"
)
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
_TAG = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}")


def valid_image_reference(value: str, *, digest_required: bool = False) -> bool:
    if not isinstance(value, str) or len(value) > 512:
        return False
    if _DIGEST.fullmatch(value):
        return not digest_required
    reference, separator, digest = value.partition("@")
    if separator and not _DIGEST.fullmatch(digest):
        return False
    if digest_required and not separator:
        return False
    parts = reference.split("/")
    if len(parts) > 1 and ("." in parts[0] or ":" in parts[0] or parts[0] == "localhost"):
        domain = parts.pop(0)
        if not _DOMAIN.fullmatch(domain):
            return False
        if ":" in domain and not 1 <= int(domain.rsplit(":", 1)[1]) <= 65535:
            return False
    if ":" in parts[-1]:
        parts[-1], tag = parts[-1].rsplit(":", 1)
        if not _TAG.fullmatch(tag):
            return False
    return bool(parts) and all(_COMPONENT.fullmatch(part) for part in parts)
