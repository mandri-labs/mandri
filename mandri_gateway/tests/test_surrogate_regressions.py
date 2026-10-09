import json
import ntpath
import posixpath
import re
from urllib.parse import urlsplit

import pytest
from mandri.core.types.execution import ProtectionError
from mandri.gateway.privacy_known import KnownValues
from mandri.gateway.surrogate import SurrogateEngine, SurrogateScope
from mandri.gateway.surrogate.formats import valid
from mandri.gateway.surrogate.stream import StreamRestorer
from mandri.gateway.surrogate.types import Mapping


@pytest.fixture
def engine():
    return SurrogateEngine(SurrogateScope("regression"))


@pytest.mark.parametrize(
    "source",
    [
        "Acknowledge received only: password=FAKE_PASSWORD_102;",
        "Fictional config:\npassword=FAKE_PASSWORD_102",
        '1\t{\n2\t  "password": "FAKE_PASSWORD_102"\n3\t}',
    ],
)
def test_secret_labels_are_detected_inside_surrounding_text(engine, source):
    protected = engine.protect_text(source)
    assert "FAKE_PASSWORD_102" not in protected
    assert engine.restore_text(protected) == source


@pytest.mark.parametrize(
    "source",
    [
        '1\t{\n2\t  "full_name": "Alice Smith",\n3\t  "phone": "+33612345678"\n4\t}',
        '2→  "full_name": "Alice Smith",\n3→  "phone": "+33612345678"',
    ],
)
def test_numbered_values_keep_their_surrounding_rendering(engine, source):
    protected = engine.protect_text(source)
    assert "Alice Smith" not in protected and "+33612345678" not in protected
    assert engine.restore_text(protected) == source


def test_labelled_email_does_not_register_the_label(engine):
    protected = engine.protect_text("email=alice@example.org")
    alias = engine.register("alice@example.org", "email")
    assert protected == "email=" + alias
    assert engine.restore_text(json.dumps({"email": alias})) == '{"email": "alice@example.org"}'
    assert all(not m.original.startswith("email=") for m in engine.scope.mappings)


def test_known_windows_identity_keeps_component_aliases_without_markdown(engine):
    host = engine.register("PRIVATE-HOST", "hostname")
    user = engine.register("privateuser", "username")
    source = "Username: `PRIVATE-HOST\\privateuser`"
    protected = engine.protect_text(source)
    assert protected == f"Username: `{host}\\{user}`"
    assert engine.restore_text(json.dumps({"username": host + "\\" + user})) == json.dumps(
        {"username": "PRIVATE-HOST\\privateuser"}
    )


@pytest.mark.parametrize(
    "source",
    [
        "The `name:` slug. Link with `[[name]]` to other documents.",
        'Names are the address: send with `SendMessage({to: "<name>", message: "..."})`.',
    ],
)
def test_instruction_prose_and_tool_examples_are_not_entities(engine, source):
    assert engine.protect_text(source) == source


@pytest.mark.parametrize("value", ["10.24.46.68", "fd12:3456:789a::731"])
@pytest.mark.parametrize("suffix", [".", ",", ";", ")", "]"])
def test_network_address_terminal_punctuation_is_outside_entity(engine, value, suffix):
    source = "Connect to " + value + suffix
    protected = engine.protect_text(source)
    assert value not in protected
    assert protected.endswith(suffix)
    assert engine.restore_text(protected) == source


def test_url_collision_preserves_scheme_and_assignment_across_repeated_discovery(engine):
    engine.register("owner", "git_owner")
    source = "https://payments.private.test/ownerextra/usd"
    first = engine.protect_text(source)
    assert urlsplit(first).scheme == "https"
    for _ in range(5):
        assert engine.protect_text(source) == first
    assert engine.restore_text(first) == source


def test_emails_have_plausible_name_components_and_com_domain(engine):
    value = engine.register("alice@example.org", "email")
    local, domain = value.split("@")
    assert len(local.split(".")) == 2
    assert all(part.isalpha() for part in local.split("."))
    assert domain.endswith(".com")
    assert engine.restore_text(value) == "alice@example.org"


def test_url_query_repository_is_not_reserved_as_a_filesystem_path(engine):
    owner = engine.register("private-owner", "git_owner")
    repo = engine.register("private-repo", "git_repository")
    source = "https://stats.private.test/svg?repos=private-owner/private-repo&type=Date"
    protected = engine.protect_text(source)
    assert "private-owner" not in protected and "private-repo" not in protected
    assert f"repos={owner}%2F{repo}&type=Date" in protected
    assert engine.restore_text(protected) == source
    assert engine.protect_text(source) == protected


def test_numbered_json_preserves_escaped_string_semantics(engine):
    value = {
        "full_name": 'A"lice Smith',
        "password": "Fake\nPassword!",
        "username": "test\\operator",
    }
    source = "\n".join(
        f"{i}\t{line}" for i, line in enumerate(json.dumps(value, indent=2).splitlines(), 1)
    )
    protected = engine.protect_text(source)
    decoded = json.loads("\n".join(line.split("\t", 1)[1] for line in protected.splitlines()))
    assert decoded["password"] != value["password"]
    assert decoded["full_name"] != value["full_name"]
    assert engine.restore_text(protected) == source


def test_url_allocation_uses_the_same_case_rules_as_literal_replacement(engine):
    engine.register("Audio", "identity")
    engine.register("Video", "identity")
    source = "https://www.chromium.org/audio-video"
    protected = engine.protect_text(source)
    assert urlsplit(protected).scheme == "https"
    assert urlsplit(protected).path == "/audio-video"
    assert engine.restore_text(protected) == source
    assert engine.protect_text(source) == protected


@pytest.mark.parametrize("scheme", ["postgresql", "mysql", "mongodb+srv"])
def test_database_url_names_are_private_components(engine, scheme):
    source = f"{scheme}://dbuser:dbpassword@db.private.test:5432/customerdb?sslmode=require"
    protected = engine.protect_text(source)
    assert "customerdb" not in protected
    assert urlsplit(protected).scheme == scheme
    assert urlsplit(protected).port == 5432
    assert urlsplit(protected).query == "sslmode=require"
    assert engine.restore_text(protected) == source
    assert engine.protect_text(source) == protected


@pytest.mark.parametrize(
    "root",
    [
        r"D:\Private\Project",
        "C:/Private/Project",
        r"\\server\share\Project",
        "/home/private/project",
    ],
)
def test_root_alias_keeps_platform_and_separator_style(engine, root):
    alias = engine.register_root(root)
    assert valid("path_root", alias, root)
    assert re.findall(r"[/\\]", alias) == re.findall(r"[/\\]", root)
    path_module = posixpath if root.startswith("/") else ntpath
    assert path_module.isabs(alias)
    if root[1:2] == ":":
        assert alias[:2] == root[:2]
    suffix = r"\fixtures\data.json" if "\\" in root else "/fixtures/data.json"
    source = root + suffix
    assert engine.protect_text(source) == alias + suffix
    assert engine.restore_text(alias + suffix) == source


@pytest.mark.parametrize(
    "original,alias",
    [
        (r"D:\Private\Project", "/private/project"),
        ("/home/private/project", r"D:\Private\Project"),
        (r"D:\Private\Project", "D:/Private/Project"),
    ],
)
def test_root_alias_validation_rejects_cross_platform_and_separator_changes(original, alias):
    assert not valid("path_root", alias, original)


@pytest.mark.parametrize(
    "root", [r"D:\Private\Project", "C:/Private/Project", r"\\server\share\Project"]
)
@pytest.mark.parametrize("separator", ["/", "\\"])
def test_windows_known_path_restores_model_separator_changes_after_reload(root, separator):
    engine = SurrogateEngine(SurrogateScope("native-path"))
    alias = engine.register_root(root)
    original = root.replace("\\", separator).replace("/", separator)
    rendered = alias.replace("\\", separator).replace("/", separator)
    suffix = separator + "fixtures" + separator + "data.json"
    resumed = SurrogateEngine(SurrogateScope.from_dict(engine.scope.to_dict()))
    assert resumed.restore_text(rendered + suffix) == original + suffix
    assert KnownValues(resumed).text(original + suffix) == rendered + suffix
    assert resumed.protect_text(original + suffix) == rendered + suffix
    assert resumed.restore({"file_path": rendered + suffix}) == {"file_path": original + suffix}
    for width in (1, 3, 7):
        stream = StreamRestorer(resumed)
        result = "".join(
            stream.feed((rendered + suffix)[i : i + width])
            for i in range(0, len(rendered + suffix), width)
        )
        assert result + stream.finish() == original + suffix


def test_separator_variant_cannot_overwrite_an_unrelated_reverse_alias():
    scope = SurrogateScope(
        "native-path-collision",
        mappings=[
            Mapping("path_root", r"D:\Private\Project", r"D:\Fictional\Example"),
            Mapping("secret", "another-private-secret", "D:/Fictional/Example"),
        ],
    )
    with pytest.raises(ProtectionError, match="Path replacement spellings conflict"):
        SurrogateEngine(scope)
