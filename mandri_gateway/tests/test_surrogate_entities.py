import base64
import ipaddress
import json
import re
import uuid
from urllib.parse import parse_qs, urlsplit

import pytest
from mandri.gateway.surrogate import SurrogateEngine, SurrogateScope


@pytest.fixture
def engine():
    return SurrogateEngine(SurrogateScope("synthetic-entity-test"))


@pytest.mark.parametrize(
    "email",
    [
        "alice@example.fr",
        "Alice.Smith+build@corp.example.org",
        "éloïse@équipe.fr",
        "a_b-c'd@sub.example.net",
        "first.last@xn--quipe-9ra.fr",
    ],
)
@pytest.mark.parametrize("wrapper", ["{}", "Contact <{}>.", "mailto:{}", "Author: {}\n"])
def test_email_surrogates_are_reserved_valid_stable_and_reversible(engine, email, wrapper):
    source = wrapper.format(email)
    result = engine.protect_text(source)
    assert email not in result
    mapping = next(item for item in engine.scope.mappings if item.original == email)
    assert re.fullmatch(r"[a-z0-9-]+@[a-z0-9]+\.invalid", mapping.surrogate)
    assert engine.protect_text(source) == result
    assert engine.restore_text(result) == source


def test_git_compounds_share_owner_repository_and_keep_operational_suffixes(engine):
    scp = "git@github.com:private-team/private-repo.git"
    first = engine.protect_text(scp)
    owner, repository = first.split(":", 1)[1].removesuffix(".git").split("/")
    browser = engine.protect_text("https://github.com/private-team/private-repo/issues/42")
    ssh = engine.protect_text("ssh://git@github.com:2222/private-team/private-repo.git")
    assert browser == f"https://github.com/{owner}/{repository}/issues/42"
    assert ssh == f"ssh://git@github.com:2222/{owner}/{repository}.git"
    assert engine.restore_text(f"https://github.com/{owner}/{repository}/pull/99") == (
        "https://github.com/private-team/private-repo/pull/99"
    )
    assert engine.restore_text(first) == scp


def test_nested_gitlab_groups_private_host_credentials_and_ports(engine):
    source = "https://build-user:swordfish@git.internal:8443/team/platform/backend.git"
    result = engine.protect_text(source)
    parsed = urlsplit(result)
    assert parsed.port == 8443
    assert parsed.hostname.endswith(".invalid")
    assert parsed.username != "build-user"
    assert parsed.password != "swordfish"
    assert parsed.path.endswith(".git")
    assert len(parsed.path.split("/")) == 4
    assert engine.restore_text(result) == source


def test_url_query_repeated_keys_percent_encoded_emails_and_secrets(engine):
    source = "https://api.internal:9443/v1?email=alice%40example.fr&token=Abc12345&email=alice%40example.fr#password=Zyx32100"
    result = engine.protect_text(source)
    parsed = urlsplit(result)
    query = parse_qs(parsed.query)
    assert parsed.port == 9443
    assert parsed.path == "/v1"
    assert len(query["email"]) == 2 and query["email"][0] == query["email"][1]
    assert query["email"][0].endswith(".invalid")
    assert query["token"] != ["Abc12345"]
    assert "Zyx32100" not in parsed.fragment
    assert engine.restore_text(result) == source


@pytest.mark.parametrize(
    "source",
    [
        "postgresql://admin:very-secret@db.internal:5432/customer_data",
        "redis://user:12345678@cache.internal:6379/0",
        "mongodb+srv://client:top-secret@db.corp/records",
    ],
)
def test_database_connection_strings_keep_parser_structure(engine, source):
    result = engine.protect_text(source)
    original_parts, parts = urlsplit(source), urlsplit(result)
    assert parts.scheme == original_parts.scheme
    assert parts.port == original_parts.port
    assert parts.username != original_parts.username
    assert parts.password != original_parts.password
    assert parts.hostname != original_parts.hostname
    assert engine.restore_text(result) == source


@pytest.mark.parametrize(
    "source", ["https://docs.python.org/3/library/json.html", "https://example.com/status"]
)
def test_public_technical_urls_without_private_content_remain_unchanged(engine, source):
    assert engine.protect_text(source) == source


@pytest.mark.parametrize(
    "source",
    ["10.24.32.1", "172.20.9.90", "192.168.2.8", "8.8.4.4", "2001:db8:1::12", "fe80::1234%eth0"],
)
def test_network_addresses_remain_parseable_same_ip_version(engine, source):
    result = engine.protect_text(source)
    assert result != source
    assert ipaddress.ip_address(result).version == ipaddress.ip_address(source).version
    assert engine.restore_text(result) == source


@pytest.mark.parametrize("source", ["10.42.0.0/16", "192.168.0.0/24", "fd12:3456::/48"])
def test_cidr_keeps_network_and_prefix_semantics(engine, source):
    result = engine.protect_text(source)
    original_network = ipaddress.ip_network(source)
    surrogate_network = ipaddress.ip_network(result)
    assert surrogate_network.version == original_network.version
    assert surrogate_network.prefixlen == original_network.prefixlen
    assert surrogate_network != original_network
    assert engine.restore_text(result) == source


@pytest.mark.parametrize("source", ["00:1a:2b:3c:4d:5e", "AA-BB-CC-DD-EE-FF"])
def test_mac_keeps_separator_and_case(engine, source):
    result = engine.protect_text(source)
    assert re.fullmatch(r"[0-9a-fA-F]{2}([:-][0-9a-fA-F]{2}){5}", result)
    assert (":" in result) == (":" in source)
    assert result != source
    assert engine.restore_text(result) == source


@pytest.mark.parametrize(
    "source",
    [
        "version: 1.2.3.4",
        "release 10.20.30.40",
        "v1.2.3.4",
        "999.999.999.999",
        "123.4.5",
        "the counter is 12345678",
        "tool_call_abcdef123456",
        "143185e9-5255-41eb-ac6f-df30d9deadc2",
        "abcdefabcdefabcdefabcdefabcdefabcdefabcd",
        "@types/node",
        "src/module/example.py",
        "safe public source_identifier and sourceIdentifier",
    ],
)
def test_unlabelled_control_like_values_are_not_classified_as_private(engine, source):
    assert engine.protect_text(source) == source


@pytest.mark.parametrize(
    "field,value",
    [
        ("tenant_id", "143185e9-5255-41eb-ac6f-df30d9deadc2"),
        ("device_id", "143185E9-5255-41EB-AC6F-DF30D9DEADC2"),
    ],
)
def test_labelled_uuids_keep_version_shape_and_case(engine, field, value):
    result = engine.protect({field: value})
    alias = result[field]
    assert uuid.UUID(alias).version == uuid.UUID(value).version
    assert alias != value
    assert alias.isupper() == value.isupper()
    assert engine.restore(result) == {field: value}


@pytest.mark.parametrize(
    "field,value",
    [
        ("username", "jane-doe"),
        ("full_name", "Marie Curie"),
        ("organization", "Cedar Labs"),
        ("customer_id", "CUS-004281"),
        ("bucket_name", "customer-archive-2019"),
        ("account_id", "00001234"),
        ("account_id", 123456789012),
        ("address", "12 avenue des Fleurs, Lyon"),
        ("postal_code", "07500"),
        ("ssn", "123-45-6789"),
        ("passport", "AB1234567"),
        ("phone", "+33 6 12 34 56 78"),
        ("phone", "06.12.34.56.78"),
        ("phone", "+1 (202) 555-0123"),
        ("dob", "1987-09-24"),
        ("dob", "24/09/1987"),
    ],
)
def test_contextual_private_fields_preserve_json_type_and_roundtrip(engine, field, value):
    source = {field: value}
    result = engine.protect(source)
    assert type(result[field]) is type(value)
    assert result[field] != value
    assert engine.restore(result) == source


def independent_iban_checksum(value):
    compact = value.replace(" ", "").upper()
    expanded = "".join(
        str(ord(char) - ord("A") + 10) if char.isalpha() else char
        for char in compact[4:] + compact[:4]
    )
    return int(expanded) % 97


@pytest.mark.parametrize(
    "value",
    ["FR76 3000 6000 0112 3456 7890 189", "GB82 WEST 1234 5698 7654 32", "DE89370400440532013000"],
)
def test_iban_has_independently_validated_checksum_and_format(engine, value):
    result = engine.protect({"iban": value})
    alias = result["iban"]
    assert alias[:2] == value[:2]
    assert len(alias) == len(value)
    assert independent_iban_checksum(alias) == 1
    assert alias != value
    assert engine.restore(result) == {"iban": value}


def test_card_checksum_is_valid_and_separators_preserved(engine):
    source = {"card_number": "4111 1111 1111 1111"}
    result = engine.protect(source)
    alias = result["card_number"]
    digits = list(map(int, alias.replace(" ", "")))
    doubled = [number * 2 for number in digits[::2]]
    assert (
        sum(number - 9 if number > 9 else number for number in doubled) + sum(digits[1::2])
    ) % 10 == 0
    assert len(alias) == 19 and alias[4] == alias[9] == alias[14] == " "
    assert alias != source["card_number"]
    assert engine.restore(result) == source


@pytest.mark.parametrize(
    "source",
    [
        "sk-proj-AbCd1234567890ABCDEFG",
        "ghp_abcdefghijklmnopqrstuvwxyz1234567890",
        "AKIAABCDEFGHIJKLMNOP",
        "Bearer AbcdEFGH1234567890",
        "password='aB3dE!fG'",
        "OPENROUTER_API_KEY=Abcd123456789012345",
        "host=api.internal password=VerySecret123",
    ],
)
def test_secrets_in_content_are_replaced_and_exactly_restored(engine, source):
    result = engine.protect_text(source)
    assert result != source
    assert engine.restore_text(result) == source
    assert "VerySecret123" not in result


def test_basic_auth_substitution_preserves_decodable_credential_pair(engine):
    original = base64.b64encode(b"private-user:private-pass").decode()
    result = engine.protect_text("Basic " + original)
    assert b":" in base64.b64decode(result.removeprefix("Basic "), validate=True)
    assert original not in result
    assert engine.restore_text(result) == "Basic " + original


def test_aws_arn_retains_service_syntax_and_changes_resource_identity(engine):
    source = "arn:aws:lambda:eu-west-1:123456789012:function:private-service"
    result = engine.protect_text(source)
    assert result.startswith("arn:aws:lambda:eu-west-1:")
    assert re.fullmatch(r"arn:aws:lambda:eu-west-1:\d{12}:[a-z]+:[a-z-]+", result)
    assert "123456789012" not in result and "private-service" not in result
    assert engine.restore_text(result) == source


def test_private_package_retains_scoped_package_grammar(engine):
    result = engine.protect_text("Install @private-team/private-library")
    assert re.fullmatch(r"Install @[a-z-]+/[a-z-]+", result)
    assert "private-team" not in result and "private-library" not in result
    assert engine.restore_text(result) == "Install @private-team/private-library"


@pytest.mark.parametrize(
    "source",
    [
        "-----BEGIN PRIVATE KEY-----\nnot-a-real-key\n-----END PRIVATE KEY-----",
        "-----BEGIN RSA PRIVATE KEY-----\nsynthetic\n-----END RSA PRIVATE KEY-----",
        "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJhbGljZSJ9.dGVzdC1zaWduYXR1cmU",
        {"dob": "31/02/2000"},
        {"iban": "FR0000000000000000000000000"},
        {"card_number": "4111 1111 1111 1112"},
    ],
)
def test_unknown_or_invalid_typed_material_passes_through(engine, source):
    assert engine.protect(source) == source


def test_nested_json_is_transformed_in_keys_values_and_embedded_arguments(engine):
    payload = {
        "entries": [{"alice@example.fr": {"owner": "alice@example.fr"}}],
        "arguments": json.dumps(
            {"contact": "alice@example.fr", "ids": [{"account_id": "00001234"}]}
        ),
    }
    result = engine.protect(payload)
    encoded = json.dumps(result)
    assert "alice@example.fr" not in encoded and "00001234" not in encoded
    assert isinstance(json.loads(result["arguments"]), dict)
    assert engine.restore(result) == payload


def test_private_url_path_masks_seeded_identity_without_treating_it_as_workspace_suffix(engine):
    engine.register("private-customer")
    source = "https://service.private.example/api/private-customer/issues"
    result = engine.protect_text(source)
    assert "private-customer" not in result
    assert "/api/" in result and result.endswith("/issues")
    assert engine.restore_text(result) == source


def test_late_label_discovery_precedes_compound_allocation(engine):
    source = {
        "url": "https://service.private.example/api/private-customer/issues",
        "organization": "private-customer",
    }
    result = engine.protect(source)
    assert "private-customer" not in json.dumps(result)
    assert result["organization"] in result["url"]
    assert engine.restore(result) == source


@pytest.mark.parametrize(
    "text",
    [
        "Use basic searches for symbols.",
        "Basic configuration is sufficient.",
        "A basic description of folders.",
    ],
)
def test_basic_auth_word_in_technical_prose_is_not_a_credential(engine, text):
    assert engine.protect_text(text) == text


@pytest.mark.parametrize("value", ["searches", "bm9fY29sb24=", "@@@@"])
def test_unknown_basic_auth_format_passes_through(engine, value):
    source = {"authorization": "Basic " + value}
    assert engine.protect(source) == source


@pytest.mark.parametrize("registered", ["Developer.Mozilla.Org", "developer.mozilla.org."])
@pytest.mark.parametrize(
    "spelling", ["developer.mozilla.org", "DEVELOPER.MOZILLA.ORG", "DeVeLoPeR.Mozilla.Org."]
)
def test_private_dns_case_variants_override_public_inventory_and_restore_exactly(
    registered, spelling
):
    engine = SurrogateEngine(SurrogateScope("private-dns-case"))
    engine.register(registered, "domain")
    source = {"text": f"Host {spelling} and https://{spelling}:443/reference"}
    protected = engine.protect(source)
    assert "mozilla" not in protected["text"].casefold()
    assert engine.restore(protected) == source
    restarted = SurrogateEngine(SurrogateScope.from_dict(engine.scope.to_dict()))
    assert restarted.protect(source) == protected
    assert restarted.restore(protected) == source


@pytest.mark.parametrize(
    "source",
    ["https://DEVELOPER.MOZILLA.ORG/reference", "https://Docs.Python.Org./3/library/json.html"],
)
def test_unregistered_public_dns_spelling_is_preserved(engine, source):
    assert engine.protect_text(source) == source


def test_private_hostname_ascii_fold_keeps_unicode_text_offsets_exact(engine):
    engine.register("PRIVATE-HOST", "hostname")
    original = "İstanbul: Private-Host is reachable"
    protected = engine.protect_text(original)
    assert protected.startswith("İstanbul: ")
    assert "private-host" not in protected.casefold()
    assert engine.restore_text(protected) == original
