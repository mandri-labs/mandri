import ipaddress
import re
from bisect import bisect_left

from mandri.gateway.surrogate.formats import EMAIL, MAC, UUID, valid
from mandri.gateway.surrogate.labelled import labelled_spans
from mandri.gateway.surrogate.public import PUBLIC_PACKAGE_SCOPES
from mandri.gateway.surrogate.rules import TypedRule
from mandri.gateway.surrogate.types import Span

URL = re.compile(
    r"(?i)\b(?:https?|ssh|git|postgres(?:ql)?|mysql|mongodb(?:\+srv)?|redis|rediss|amqps?|s3)://[^\s<>\"'`]+"
)
GIT_SCP = re.compile(r"(?<![\w./])(?:[\w.-]+@)?[\w.-]+:(?:[\w.%~-]+/)+[\w.%~-]+(?:\.git)?")
IPV4 = re.compile(r"(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?:/\d{1,2})?(?![\w/]|\.[\w.])")
IPV6 = re.compile(
    r"(?<![\w:])(?:[0-9a-fA-F]{0,4}:){2,}[0-9a-fA-F:.]*(?:%[\w.-]+)?(?:/\d{1,3})?(?![\w:])"
)
PRIVATE_DOMAIN = re.compile(
    r"(?<![\w.-])(?:[a-zA-Z0-9][\w-]*\.)+(?:internal|local|corp|lan|private|test)(?![\w.-])"
)
API_SECRET = re.compile(
    r"(?<![\w-])(?:sk-(?:proj-|ant-api\d+-)?[a-zA-Z0-9_-]{16,}"
    r"|gh[pousr]_[a-zA-Z0-9]{20,}|github_pat_[a-zA-Z0-9_]{20,}"
    r"|(?:AKIA|ASIA)[A-Z0-9]{16})(?![\w-])"
)
AUTH = re.compile(r"(?i)\b(Bearer|Basic)\s+([a-zA-Z0-9+/_.=~-]{4,})")
ARN = re.compile(r"\barn:(?:aws|aws-cn|aws-us-gov):[a-z0-9-]+:[a-z0-9-]*:[0-9]*:[\w./:@+=,-]+")
PRIVATE_PACKAGE = re.compile(r"(?<![\w./])@([\w.-]+)/[\w.-]+")
CONTEXTS = {
    "email": "email",
    "email_address": "email",
    "author_email": "email",
    "username": "username",
    "user_name": "username",
    "login": "username",
    "handle": "username",
    "name": "identity",
    "full_name": "identity",
    "first_name": "identity",
    "last_name": "identity",
    "author": "identity",
    "customer": "identity",
    "customer_name": "identity",
    "organization": "identity",
    "organisation": "identity",
    "company": "identity",
    "team": "identity",
    "hostname": "hostname",
    "host": "hostname",
    "domain": "domain",
    "server": "hostname",
    "phone": "phone",
    "telephone": "phone",
    "mobile": "phone",
    "phone_number": "phone",
    "dob": "date_of_birth",
    "date_of_birth": "date_of_birth",
    "birthdate": "date_of_birth",
    "address": "address",
    "street_address": "address",
    "postal_address": "address",
    "iban": "iban",
    "card_number": "card",
    "credit_card": "card",
    "payment_card": "card",
    "account": "account",
    "account_id": "account",
    "account_number": "account",
    "bank_account": "account",
    "tenant_id": "identifier",
    "tenant_uuid": "uuid",
    "customer_id": "identifier",
    "user_id": "identifier",
    "device_id": "identifier",
    "ticket_id": "identifier",
    "project_id": "identifier",
    "bucket": "identifier",
    "bucket_name": "identifier",
    "resource": "identifier",
    "arn": "cloud_resource",
    "ssn": "identifier",
    "national_id": "identifier",
    "passport": "identifier",
    "postal_code": "identifier",
    "password": "secret",
    "passwd": "secret",
    "secret": "secret",
    "token": "secret",
    "api_key": "secret",
    "apikey": "secret",
    "access_token": "secret",
    "refresh_token": "secret",
    "client_secret": "secret",
    "session_cookie": "secret",
    "cookie": "secret",
}


def context_kind(context: str, value: str) -> str | None:
    normalized = re.sub(r"[ \t-]+", "_", context.casefold())
    kind = CONTEXTS.get(normalized)
    if kind is None:
        kind = next(
            (
                "secret"
                for suffix in ("_api_key", "_token", "_secret", "_password")
                if normalized.endswith(suffix)
            ),
            None,
        )
    if kind is None and normalized.endswith("_id"):
        kind = "identifier"
    if kind in {"identifier", "account"} and UUID.fullmatch(value):
        return "uuid"
    if kind == "identity" and EMAIL.fullmatch(value):
        return "email"
    return kind


def valid_context(kind: str, value: str) -> bool:
    if kind in {"phone", "email", "date_of_birth", "iban", "card", "hostname", "domain"}:
        return valid(kind, value)
    return bool(value.strip()) and valid(kind, value)


def detect(
    text: str,
    context: str = "",
    rules: tuple[TypedRule, ...] = (),
    *,
    public_package_scopes: frozenset[str] = PUBLIC_PACKAGE_SCOPES,
) -> list[Span]:
    spans = []
    kind = context_kind(context, text)
    if kind and valid_context(kind, text):
        uuid_prefix = UUID.match(text) if kind in {"identifier", "account"} else None
        if uuid_prefix and re.fullmatch(r":\d+", text[uuid_prefix.end() :]):
            spans.append(Span(0, uuid_prefix.end(), "uuid", 165, context))
        else:
            spans.append(Span(0, len(text), kind, 190 if kind == "email" else 160, context))
    labels = labelled_spans(text, context_kind, valid_context)
    for pattern, entity, priority in (
        (GIT_SCP, "git_remote", 180),
        (URL, "url", 180),
        (EMAIL, "email", 175),
        (API_SECRET, "secret", 175),
        (PRIVATE_DOMAIN, "domain", 65),
        (MAC, "mac", 165),
        (ARN, "cloud_resource", 180),
        (PRIVATE_PACKAGE, "private_package", 180),
    ):
        for match in pattern.finditer(text):
            if entity == "private_package" and match.group(1) in public_package_scopes:
                continue
            end = match.end()
            start = match.start()
            if entity in {"url", "git_remote"}:
                while end > match.start() and text[end - 1] in ".),]":
                    end -= 1
            if entity == "url" and text[match.start() : end].endswith("://"):
                continue
            if entity == "email" and not valid("email", text[match.start() : end]):
                continue
            if entity == "email" and text[start] == "`" and text[end : end + 1] == "`":
                start += 1
            spans.append(Span(start, end, entity, priority))
    for match in IPV4.finditer(text):
        prefix = text[max(0, match.start() - 20) : match.start()]
        if re.search(r"(?i)(?:version|release|v)\s*[:=]?\s*$", prefix):
            continue
        value = match.group().rstrip(".")
        entity = "cidr" if "/" in value else "ipv4"
        if valid(entity, value):
            spans.append(Span(match.start(), match.start() + len(value), entity, 165))
    for match in IPV6.finditer(text):
        value = match.group().rstrip(".")
        entity = "cidr" if "/" in value else "ipv6"
        try:
            valid_ip = (
                ipaddress.ip_network(value, strict=False)
                if "/" in value
                else ipaddress.ip_address(value)
            )
        except ValueError:
            continue
        if valid_ip.version == 6:
            spans.append(Span(match.start(), match.start() + len(value), entity, 165))
    for match in AUTH.finditer(text):
        entity = "basic" if match.group(1).lower() == "basic" else "secret"
        if entity == "basic" and not valid("basic", match.group(2)):
            continue
        if entity == "basic" and context.casefold() not in {"authorization", "proxy-authorization"}:
            prefix = text[max(0, match.start() - 24) : match.start()]
            if not re.search(r"(?i)authorization:[ \t]*$", prefix) and not (
                match.start() == 0 and match.group(1) == "Basic" and len(match.group(2)) >= 16
            ):
                continue
        spans.append(Span(match.start(2), match.end(2), entity, 175))
    spans.extend(labels)
    for rule in rules:
        spans.extend(rule.find(text, context))
    return spans


def select_spans(spans: list[Span]) -> list[Span]:
    selected: list[Span] = []
    starts: list[int] = []
    for span in sorted(
        spans, key=lambda item: (-item.priority, -(item.end - item.start), item.start, item.kind)
    ):
        if span.end <= span.start:
            continue
        index = bisect_left(starts, span.start)
        overlaps_previous = index > 0 and selected[index - 1].end > span.start
        overlaps_next = index < len(selected) and selected[index].start < span.end
        if not overlaps_previous and not overlaps_next:
            selected.insert(index, span)
            starts.insert(index, span.start)
    return selected
