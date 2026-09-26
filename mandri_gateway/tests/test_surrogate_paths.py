import json
from urllib.parse import quote

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from mandri.gateway.surrogate import SurrogateEngine, SurrogateScope

ROOT = "/home/private-user/Project"
ALIAS = "/home/river-user/Orchard"


@pytest.fixture
def engine():
    result = SurrogateEngine(SurrogateScope("synthetic-path-test"))
    result.register_root(ROOT, ALIAS)
    return result


def test_workspace_root_is_distinct_from_nested_repositories_and_cwd(engine):
    engine.register("mandri", "identity")
    engine.register("mandri_api", "identity")
    source = {
        "cwd": ROOT + "/mandri/mandri_api",
        "files": [
            ROOT + "/mandri/mandri_api/src/api.py",
            ROOT + "/mandri-app/src/App.tsx",
            ROOT + "/mandri-e2e/tests/test_api.py",
        ],
    }
    result = engine.protect(source)
    assert result["cwd"] == ALIAS + "/mandri/mandri_api"
    assert result["files"] == [item.replace(ROOT, ALIAS, 1) for item in source["files"]]
    assert engine.restore(result) == source
    assert len(engine.scope.roots) == 1


def test_neutral_runtime_roots_preserve_names_without_creating_mapping_or_alias(engine):
    alias = engine.register("mandri", "identity")
    before = engine.scope.to_dict()
    engine.reserve_root("/workspace")
    engine.reserve_root("/home/worker")
    assert engine.scope.to_dict() == before
    source = {
        "path": "/workspace/mandri/mandri_api/src/api.py",
        "home": "/home/worker/mandri/config",
        "description": "mandri",
    }
    result = engine.protect(source)
    assert result == {**source, "description": alias}
    assert engine.restore(result) == source
    assert engine.scope.to_dict() == before


def test_trusted_home_prefix_preserves_private_workspace_prefix_and_public_descendants(engine):
    engine.register("mandri", "identity")
    home = "/home/private-user"
    home_alias = engine.register_root(home, independent=True)
    assert next(root.surrogate for root in engine.scope.roots if root.original == ROOT) == ALIAS
    source = [ROOT + "/mandri/src/new.py", home + "/.config/tool.json", "mandri/src/new.py"]
    protected = engine.protect(source)
    assert protected == [
        ALIAS + "/mandri/src/new.py",
        home_alias + "/.config/tool.json",
        "mandri/src/new.py",
    ]
    assert "Project" not in protected[0]
    resumed = SurrogateEngine(SurrogateScope.from_dict(engine.scope.to_dict()))
    assert resumed.restore(protected) == source
    assert resumed.protect(source) == protected
    child_alias = engine.register_root(ROOT + "/mandri")
    assert "mandri" not in child_alias.casefold()
    assert engine.restore_text(child_alias + "/src/new.py") == ROOT + "/mandri/src/new.py"


@pytest.mark.parametrize(
    "private_root", ["/workspace", "/home/worker", "/home", "/workspace/Project"]
)
def test_neutral_runtime_paths_override_identically_spelled_host_prefixes(private_root):
    engine = SurrogateEngine(SurrogateScope("same-spelling"))
    engine.register_root(private_root)
    engine.register("Project")
    before = engine.scope.to_dict()
    engine.reserve_root("/workspace")
    engine.reserve_root("/home/worker")
    source = ["/workspace/Project/api/src/new.py", "/home/worker/Project/config"]
    assert engine.protect(source) == source
    assert engine.restore(source) == source
    assert engine.scope.to_dict() == before
    host_engine = SurrogateEngine(SurrogateScope.from_dict(before))
    host_path = private_root + "/mandri/api.py"
    host_protected = host_engine.protect_text(host_path)
    assert host_protected != host_path
    assert host_protected.endswith("/mandri/api.py")
    assert host_engine.restore_text(host_protected) == host_path
    assert (
        engine.protect_text("Project is a private organization")
        != "Project is a private organization"
    )


@pytest.mark.parametrize(
    "value",
    [
        "mandri/mandri_api/src/api.py",
        "./mandri/mandri_api/src/",
        "../mandri-app/src/App.tsx",
        "src/sibling/../api.py",
        "/workspace/mandri/mandri_api/src/api.py",
    ],
)
def test_relative_and_runtime_paths_keep_every_descendant_component(engine, value):
    engine.register("mandri", "identity")
    engine.register("mandri_api", "identity")
    assert engine.protect_text(value) == value


@pytest.mark.parametrize(
    "value",
    [
        ROOT + "/src/new.py",
        ROOT + "/src/deep/deeper/latest.py",
        ROOT + "/.git/config",
        ROOT + "/src/../tests/check.py",
        ROOT + "/src/api.py:42:8",
    ],
)
def test_prefix_mapping_restores_unseen_model_generated_descendant_paths(engine, value):
    assert not any(mapping.original == value for mapping in engine.scope.mappings)
    alias = value.replace(ROOT, ALIAS, 1)
    assert engine.restore_text(alias) == value
    assert engine.protect_text(value) == alias


@pytest.mark.parametrize(
    "neighbor", [ROOT + "-old/file.py", ROOT + "2/file.py", ROOT + "_copy/file.py"]
)
def test_root_prefix_does_not_match_neighboring_names(engine, neighbor):
    assert engine.protect_text(neighbor) == neighbor
    assert engine.restore_text(neighbor.replace(ROOT, ALIAS, 1)) == neighbor.replace(ROOT, ALIAS, 1)


def test_nested_registered_roots_never_generate_reserved_words(engine):
    child = ROOT + "/mandri/mandri_api"
    alias = engine.register_root(child)
    assert "mandri" not in alias.casefold()
    source = child + "/src/server.py"
    assert engine.protect_text(source) == alias + "/src/server.py"
    assert engine.restore_text(alias + "/src/fresh.py") == child + "/src/fresh.py"
    assert engine.register_root(child, "/home/river-user/Elsewhere") == alias


def test_unrelated_roots_do_not_inherit_another_workspaces_alias(engine):
    other = engine.register_root("/opt/private-build/Service")
    assert not other.startswith(ALIAS)
    assert engine.protect_text("/opt/private-build/Service/src/a.py") == other + "/src/a.py"


@pytest.mark.parametrize("source", [ROOT + "/../outside/key", ROOT + "/src/../../outside/key"])
def test_paths_outside_registered_root_pass_through(engine, source):
    assert engine.protect_text(source) == source


def test_file_uri_retains_scheme_and_root_mapping(engine):
    source = "file://" + ROOT + "/mandri/src/file.py"
    result = engine.protect_text(source)
    assert result == "file://" + ALIAS + "/mandri/src/file.py"
    assert engine.restore_text(result) == source


@pytest.mark.parametrize(
    "root,alias,path",
    [
        (
            r"C:\Users\Private\Project",
            r"C:\Users\Orchard\Orchard",
            r"C:\Users\Private\Project\mandri\src\file.py",
        ),
        (
            r"\\PrivateHost\PrivateShare\Project",
            r"\\OrchardHost\OrchardShare\Orchard",
            r"\\PrivateHost\PrivateShare\Project\mandri\src\file.py",
        ),
        (
            "C:/Users/Private/Project",
            "C:/Users/Orchard/Orchard",
            "C:/Users/Private/Project/mandri/src/file.py",
        ),
    ],
)
def test_windows_and_unc_roots_keep_descendants_and_escaped_json(root, alias, path):
    engine = SurrogateEngine(SurrogateScope("windows"))
    engine.register_root(root, alias)
    source = {"path": path, "args": json.dumps({"cwd": root, "path": path}, indent=2)}
    result = engine.protect(source)
    assert result["path"] == alias + path[len(root) :]
    assert json.loads(result["args"])["path"] == alias + path[len(root) :]
    assert engine.restore(result) == source


def test_case_insensitive_root_keeps_each_occurrences_original_spelling():
    engine = SurrogateEngine(SurrogateScope("windows-casing"))
    engine.register_root(r"C:\Users\Alice\Project", r"C:\Users\Cedar\Orchard")
    source = [r"C:\Users\Alice\Project\src\App.py", r"c:\users\alice\project\src\App.py"]
    result = engine.protect(source)
    assert result[0].endswith(r"\src\App.py") and result[1].endswith(r"\src\App.py")
    assert "Alice" not in result[0] and "alice" not in result[1]
    assert engine.restore(result) == source


def test_quoted_roots_with_spaces_and_unicode_survive_nested_json():
    engine = SurrogateEngine(SurrogateScope("quoted"))
    original = "/home/élise/My Project"
    alias = "/home/river/New Orchard"
    engine.register_root(original, alias)
    source = {
        "listing": [{"path": original + "/mandri/src/a.py"}],
        "args": json.dumps({"path": original + "/mandri/src/été.py"}, ensure_ascii=True),
    }
    result = engine.protect(source)
    assert result["listing"][0]["path"] == alias + "/mandri/src/a.py"
    assert json.loads(result["args"])["path"] == alias + "/mandri/src/été.py"
    assert engine.restore(result) == source


@pytest.mark.parametrize("safe", ["/", ""])
@pytest.mark.parametrize("lowercase", [False, True])
def test_percent_encoded_roots_keep_exact_encoded_suffix_and_roundtrip(safe, lowercase):
    engine = SurrogateEngine(SurrogateScope("percent-path"))
    root, alias = "/home/élise/My Project", "/home/river/New Orchard"
    engine.register_root(root, alias)
    path = quote(root + "/mandri/src/été.py", safe=safe)
    if lowercase:
        path = path.replace("%2F", "%2f").replace("%C3", "%c3").replace("%A9", "%a9")
    result = engine.protect_text(path)
    original_root = quote(root, safe=safe)
    if lowercase:
        original_root = (
            original_root.replace("%2F", "%2f").replace("%C3", "%c3").replace("%A9", "%a9")
        )
    assert original_root not in result
    assert result.endswith(path[len(original_root) :])
    assert engine.restore_text(result) == path


@settings(max_examples=80, deadline=None)
@given(st.lists(st.from_regex(r"[a-z]{1,12}", fullmatch=True), min_size=1, max_size=12))
def test_arbitrary_unseen_descendants_preserve_complete_tree(components):
    engine = SurrogateEngine(SurrogateScope("tree-property"))
    engine.register_root(ROOT, ALIAS)
    suffix = "/" + "/".join(components) + "/new_file.py"
    assert engine.protect_text(ROOT + suffix) == ALIAS + suffix
    assert engine.restore_text(ALIAS + suffix) == ROOT + suffix
