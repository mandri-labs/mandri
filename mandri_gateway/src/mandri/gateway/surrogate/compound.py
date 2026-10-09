import ipaddress
import re
from collections.abc import Callable
from urllib.parse import quote, unquote, urlsplit, urlunsplit

from mandri.gateway.surrogate.detectors import context_kind
from mandri.gateway.surrogate.public import PUBLIC_FORGES

GIT_BOUNDARIES = frozenset(
    {"issues", "pull", "pulls", "merge_requests", "tree", "blob", "commit", "commits", "-"}
)
DATABASE_SCHEMES = frozenset({"postgres", "postgresql", "mysql", "mongodb", "mongodb+srv"})
Allocate = Callable[[str, str, str], str]
Transform = Callable[[str, str], str]


def encoded_component(value: str, kind: str, allocate: Allocate) -> str:
    try:
        decoded = unquote(value, errors="strict")
    except UnicodeError:
        return value
    surrogate = allocate(kind, decoded, "")
    return quote(surrogate, safe="@._~-")


def git_path(path: str, allocate: Allocate) -> str:
    leading = "/" if path.startswith("/") else ""
    components = path.lstrip("/").split("/")
    boundary = next(
        (index for index, item in enumerate(components) if item in GIT_BOUNDARIES), len(components)
    )
    if boundary < 2:
        return path
    result = []
    for index, component in enumerate(components):
        if index >= boundary or not component:
            result.append(component)
        elif index == boundary - 1:
            suffix = ".git" if component.endswith(".git") else ""
            name = component[:-4] if suffix else component
            result.append(encoded_component(name, "git_repository", allocate) + suffix)
        else:
            result.append(encoded_component(component, "git_owner", allocate))
    return leading + "/".join(result)


def private_host(host: str, allocate: Allocate, public_hosts: frozenset[str]) -> str:
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        if host.casefold().rstrip(".") in public_hosts:
            return host
        return allocate("domain" if "." in host else "hostname", host, "")
    return allocate("ipv4" if address.version == 4 else "ipv6", host, "")


def transform_url(
    value: str, allocate: Allocate, transform: Transform, public_hosts: frozenset[str]
) -> str:
    try:
        parts = urlsplit(value)
        host = parts.hostname
        port = parts.port
    except ValueError:
        return value
    if not host or not parts.netloc:
        return value
    raw_authority = parts.netloc.rsplit("@", 1)[-1]
    raw_host = (
        raw_authority[1 : raw_authority.index("]")]
        if raw_authority.startswith("[")
        else raw_authority.rsplit(":", 1)[0]
        if port is not None
        else raw_authority
    )
    transformed_host = private_host(raw_host, allocate, public_hosts)
    if ":" in transformed_host:
        transformed_host = "[" + transformed_host + "]"
    if port is not None:
        transformed_host += ":" + str(port)
    userinfo = ""
    if "@" in parts.netloc:
        raw_userinfo = parts.netloc.rsplit("@", 1)[0]
        username, separator, password = raw_userinfo.partition(":")
        if username == "git" and parts.scheme in {"ssh", "git"}:
            userinfo = username
        else:
            userinfo = encoded_component(username, "username", allocate)
        if separator:
            userinfo += ":" + encoded_component(password, "secret", allocate)
        userinfo += "@"
    path = parts.path
    is_git = (
        host.casefold().rstrip(".") in PUBLIC_FORGES
        or path.endswith(".git")
        or parts.scheme in {"ssh", "git"}
    )
    path = (
        git_path(path, allocate)
        if is_git
        else "/".join(
            encoded_component(component, "identifier", allocate) if component else component
            for component in path.split("/")
        )
        if parts.scheme in DATABASE_SCHEMES
        else transform(path, "" if host.casefold().rstrip(".") in public_hosts else "$url_path")
    )
    if parts.scheme == "s3":
        transformed_host = allocate("identifier", host, "bucket")
    query = transform_query(parts.query, allocate, transform)
    fragment = transform_query(parts.fragment, allocate, transform)
    return urlunsplit((parts.scheme, userinfo + transformed_host, path, query, fragment))


def transform_query(value: str, allocate: Allocate, transform: Transform) -> str:
    result = []
    for part in re.split(r"([&;])", value):
        key, equals, raw = part.partition("=")
        if not equals:
            result.append(transform(part, ""))
            continue
        try:
            decoded_key = unquote(key, errors="strict")
            decoded_value = unquote(raw.replace("+", " "), errors="strict")
        except UnicodeError:
            result.append(part)
            continue
        kind = context_kind(decoded_key, decoded_value)
        replacement = (
            allocate(kind, decoded_value, decoded_key)
            if kind and decoded_value
            else transform(decoded_value, "$url_query")
        )
        new_raw = raw if replacement == decoded_value else quote(replacement, safe="")
        if "+" in raw:
            new_raw = new_raw.replace("%20", "+")
        result.append(key + equals + new_raw)
    return "".join(result)


def transform_git(value: str, allocate: Allocate, public_hosts: frozenset[str]) -> str:
    authority, separator, path = value.partition(":")
    if not separator:
        return value
    username, at, host = authority.rpartition("@")
    if not at:
        host, username = authority, ""
    rendered_user = (
        username if username == "git" else allocate("username", username, "") if username else ""
    )
    rendered_host = private_host(host, allocate, public_hosts)
    return (rendered_user + "@" if at else "") + rendered_host + ":" + git_path(path, allocate)
