import ipaddress
import json
import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

import pytest
from mandri.gateway.surrogate import SurrogateEngine, SurrogateScope


@dataclass(frozen=True)
class PrivateCase:
    family: str
    field: str
    value: Any
    private_spans: tuple[str, ...]
    public_spans: tuple[str, ...] = ()


_PRIVATE = (
    PrivateCase(
        "email", "message", "résumé+qualif@courrier.example", ("résumé+qualif@courrier.example",)
    ),
    PrivateCase(
        "email", "message", "Build.Owner+ci@notify.example", ("Build.Owner+ci@notify.example",)
    ),
    PrivateCase(
        "email", "message", "operator_43@xn--caf-dma.example", ("operator_43@xn--caf-dma.example",)
    ),
    PrivateCase(
        "git",
        "origin",
        "git@gitlab.com:heldout-crew/client-api.git",
        ("heldout-crew", "client-api"),
        ("git@gitlab.com:", ".git"),
    ),
    PrivateCase(
        "git",
        "origin",
        "ssh://git@github.com:2222/quiet-otter/ledger-ui.git",
        ("quiet-otter", "ledger-ui"),
        ("ssh://git@github.com:2222/", ".git"),
    ),
    PrivateCase(
        "git",
        "origin",
        "https://gitlab.com/sector-nine/blue-team/build-jobs.git",
        ("sector-nine", "blue-team", "build-jobs"),
        ("https://gitlab.com/", ".git"),
    ),
    PrivateCase("network", "address", "172.29.16.41", ("172.29.16.41",)),
    PrivateCase("network", "address", "2001:db8:89::46", ("2001:db8:89::46",)),
    PrivateCase("network", "subnet", "10.89.32.0/20", ("10.89.32.0/20",), ("/20",)),
    PrivateCase("network", "subnet", "fd72:849a:6700::/40", ("fd72:849a:6700::/40",), ("/40",)),
    PrivateCase("network", "message", "b2:4c:16:a8:20:7e", ("b2:4c:16:a8:20:7e",)),
    PrivateCase(
        "service",
        "message",
        "https://gateway.private:8443/healthz",
        ("gateway.private",),
        (":8443/healthz",),
    ),
    PrivateCase(
        "service",
        "message",
        "redis://release-bot:SecretForFixture9@queue.internal:6379/2",
        ("release-bot", "SecretForFixture9", "queue.internal"),
        (
            "redis://",
            ":6379/2",
        ),
    ),
    PrivateCase(
        "service",
        "message",
        "postgresql://reporter:NoRealPassword7@reports.corp:5432/sales",
        ("reporter", "NoRealPassword7", "reports.corp"),
        (
            "postgresql://",
            ":5432/",
        ),
    ),
    PrivateCase("identity", "username", "build-operator-73", ("build-operator-73",)),
    PrivateCase("identity", "full_name", "Émilie Example", ("Émilie Example",)),
    PrivateCase(
        "identity",
        "organization",
        "Synthetic Apricot Cooperative",
        ("Synthetic Apricot Cooperative",),
    ),
    PrivateCase("identity", "bucket_name", "synthetic-ledger-874", ("synthetic-ledger-874",)),
    PrivateCase(
        "identifier",
        "tenant_id",
        "fe2da53a-afd0-42bf-aabd-4983716af085",
        ("fe2da53a-afd0-42bf-aabd-4983716af085",),
    ),
    PrivateCase("identifier", "customer_id", "CLIENT-789134", ("CLIENT-789134",)),
    PrivateCase("identifier", "account_id", 234567890123, ("234567890123",)),
    PrivateCase("personal", "phone", "+44 7700 900123", ("+44 7700 900123",)),
    PrivateCase("personal", "dob", "1992-02-29", ("1992-02-29",)),
    PrivateCase("personal", "postal_code", "90417", ("90417",)),
    PrivateCase("personal", "passport", "ZX7654321", ("ZX7654321",)),
    PrivateCase(
        "secret",
        "message",
        "Authorization: Bearer Z7fixturetokenX2946Alpha",
        ("Z7fixturetokenX2946Alpha",),
        ("Authorization: Bearer ",),
    ),
    PrivateCase(
        "secret", "message", "password='FixtureOnly!7294'", ("FixtureOnly!7294",), ("password='",)
    ),
    PrivateCase(
        "secret",
        "message",
        "sk-proj-SyntheticOnlyForTest7391405628",
        ("sk-proj-SyntheticOnlyForTest7391405628",),
    ),
)

_PUBLIC = (
    ("package", "@types/react"),
    ("package", "@tanstack/react-query"),
    ("package", "requests>=2.31.0"),
    ("version", "version: 2.19.34.61"),
    ("version", "release 21.39.47.53"),
    ("version", "v3.18.29.44"),
    ("control", "call_Z9ak3ps72w8Dd10"),
    ("control", "7a561e46-cda2-4ac3-8383-0f3cc69f08cd"),
    ("control", "c37aa8ce9189eb3dcb83998c88b9180f024ae766"),
    ("path", "mandri/mandri_api/src/mandri/api/app.py"),
    ("path", "./nested/project/src/EmailValidator.ts"),
    ("path", "../fixtures/account_identifier.json"),
    ("code", 'const usernameField = "username";'),
    ("code", 'SELECT account_id FROM customer WHERE status = "ready";'),
    ("technical-url", "https://docs.python.org/3.13/library/pathlib.html"),
    ("technical-url", "https://developer.mozilla.org/en-US/docs/Web/JavaScript"),
)


@pytest.mark.parametrize(
    "case", _PRIVATE, ids=lambda case: f"{case.family}-{case.field}-{case.value}"
)
def test_independent_private_corpus_has_no_missed_spans(case: PrivateCase) -> None:
    engine = SurrogateEngine(SurrogateScope("heldout-private"))
    source = {"result": {"rows": [[{case.field: case.value}]]}}
    result = engine.protect(source)
    value = result["result"]["rows"][0][0][case.field]
    serialized = json.dumps(value, ensure_ascii=False)
    assert type(value) is type(case.value)
    assert value != case.value
    for span in case.private_spans:
        assert span not in serialized
    for span in case.public_spans:
        assert span in value
    if case.family == "email":
        assert re.fullmatch(r"[a-z]+\.[a-z]+@[a-z0-9]+\.com", value)
    if case.family == "network" and case.field == "address":
        assert ipaddress.ip_address(value).version == ipaddress.ip_address(case.value).version
    if case.family == "network" and case.field == "subnet":
        assert ipaddress.ip_network(value).prefixlen == ipaddress.ip_network(case.value).prefixlen
    if case.family == "service":
        original, protected = urlsplit(case.value), urlsplit(value)
        assert original.scheme == protected.scheme
        assert original.port == protected.port
    assert engine.restore(result) == source
    assert engine.protect(source) == result


@pytest.mark.parametrize("family,value", _PUBLIC)
def test_independent_public_corpus_has_no_changed_cells(family: str, value: str) -> None:
    engine = SurrogateEngine(SurrogateScope(f"heldout-public-{family}"))
    source = {"result": {"rows": [[{"text": value}]]}}
    assert engine.protect(source) == source


def test_heldout_nested_workspace_paths_preserve_every_public_suffix() -> None:
    engine = SurrogateEngine(SurrogateScope("heldout-tree"))
    root = "/home/synthetic-operator/SyntheticWorkspace"
    engine.register_root(root)
    suffixes = (
        "/mandri/mandri_api/src/mandri/api/app.py",
        "/mandri/mandri_api/src/newly-created-729/entry.ts",
        "/mandri/mandri_app/components/SendButton.tsx",
        "/mandri/fixtures/test.email-844.json",
    )
    source = [{"cwd": root + "/mandri", "files": [root + suffix for suffix in suffixes]}]
    result = engine.protect(source)
    surrogate_root = result[0]["cwd"].removesuffix("/mandri")
    assert surrogate_root != root
    assert result[0]["files"] == [surrogate_root + suffix for suffix in suffixes]
    assert engine.protect({"files": [suffix.lstrip("/") for suffix in suffixes]}) == {
        "files": [suffix.lstrip("/") for suffix in suffixes]
    }
    assert engine.restore(result) == source


def test_unrecognized_package_scope_retains_conservative_classification() -> None:
    engine = SurrogateEngine(SurrogateScope("heldout-unknown-scope"))
    source = "@example/fetch-helper"
    result = engine.protect_text(source)
    assert "example" not in result
    assert "fetch-helper" not in result
    assert engine.restore_text(result) == source


@pytest.mark.parametrize(
    "original,kind,source",
    [
        ("developer.mozilla.org", "domain", "https://developer.mozilla.org/reference"),
        ("docs.python.org", "domain", "https://docs.python.org/3/library/json.html"),
        ("@tanstack/private-addon", "private_package", 'import "@tanstack/private-addon"'),
        ("tanstack", "git_owner", 'import "@tanstack/private-addon"'),
    ],
)
def test_public_technical_inventory_does_not_override_registered_private_values(
    original: str, kind: str, source: str
) -> None:
    engine = SurrogateEngine(SurrogateScope("heldout-private-override"))
    engine.register(original, kind)
    result = engine.protect_text(source)
    assert original not in result
    assert engine.restore_text(result) == source


def test_public_url_keeps_private_query_values_protected() -> None:
    engine = SurrogateEngine(SurrogateScope("heldout-public-query"))
    source = (
        "https://developer.mozilla.org/reference?email=heldout%40example.test&token=FixtureToken738"
    )
    result = engine.protect_text(source)
    assert result.startswith("https://developer.mozilla.org/reference?")
    assert "heldout" not in result and "FixtureToken738" not in result
    assert engine.restore_text(result) == source
