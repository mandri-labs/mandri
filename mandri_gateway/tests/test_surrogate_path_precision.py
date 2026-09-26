import json
from urllib.parse import quote

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from mandri.gateway.surrogate import SurrogateEngine, SurrogateScope


def test_path_components_do_not_rewrite_tool_descriptions_or_identifiers():
    engine = SurrogateEngine(SurrogateScope("path-only"))
    root = "/home/example-user/ExampleProject/artifacts/runs/synthetic"
    alias = engine.register_root(root)
    source = {
        "description": "This command runs synthetic checks against a remote endpoint.",
        "code": "runs = synthetic(remote)",
        "path": root + "/mandri/mandri_api/src/remote.py",
        "relative": "mandri/mandri_api/src/runs/synthetic.py",
    }
    protected = engine.protect(source)
    assert protected == {**source, "path": alias + "/mandri/mandri_api/src/remote.py"}
    assert engine.restore(protected) == source
    assert all(mapping.kind == "path_root" for mapping in engine.scope.mappings)


@pytest.mark.parametrize("register_first", [True, False])
def test_explicit_username_remains_protected_with_same_path_alias(register_first):
    engine = SurrogateEngine(SurrogateScope("explicit-user"))
    root = "/home/example-user/ExampleProject"
    if register_first:
        user = engine.register("example-user", "username", "process_username")
    alias = engine.register_root(root)
    if not register_first:
        user = engine.register("example-user", "username", "process_username")
    assert alias.split("/")[2] == user
    source = ["example-user runs synthetic checks", root + "/mandri/src/remote.py"]
    protected = engine.protect(source)
    assert protected == [user + " runs synthetic checks", alias + "/mandri/src/remote.py"]
    assert engine.restore(protected) == source
    resumed = SurrogateEngine(SurrogateScope.from_dict(engine.scope.to_dict()))
    assert resumed.protect(source) == protected
    assert resumed.restore(protected) == source


@pytest.mark.parametrize(
    "root",
    [
        "/home/example-user/ExampleProject/runs/synthetic",
        r"C:\Users\ExampleUser\ExampleProject\runs\synthetic",
    ],
)
def test_parent_sibling_prefixes_remain_coherent_across_scope_reload(root):
    separator = "\\" if root.startswith("C:") else "/"
    engine = SurrogateEngine(SurrogateScope("parents"))
    alias = engine.register_root(root)
    resumed = SurrogateEngine(SurrogateScope.from_dict(engine.scope.to_dict()))
    parent = root.rsplit(separator, 1)[0]
    parent_alias = resumed.register_root(parent, independent=True)
    assert alias.rsplit(separator, 1)[0] == parent_alias
    suffixes = [
        separator + "mandri" + separator + "mandri_api" + separator + "src" + separator + "new.py",
        separator + "sibling" + separator + "runs.py",
    ]
    source = [root + suffixes[0], parent + suffixes[1], "runs synthetic remote"]
    protected = resumed.protect(source)
    assert protected == [alias + suffixes[0], parent_alias + suffixes[1], source[2]]
    reloaded = SurrogateEngine(SurrogateScope.from_dict(resumed.scope.to_dict()))
    assert reloaded.protect(source) == protected
    assert reloaded.restore(protected) == source


@pytest.mark.parametrize("encoding", ["plain", "percent", "json", "nested_json"])
def test_encoded_root_keeps_nested_descendants_and_unrelated_words(encoding):
    engine = SurrogateEngine(SurrogateScope("encoded-precision"))
    root = "/home/private person/Mandri/runs/synthetic"
    alias = engine.register_root(root)
    path = root + "/mandri/mandri_api/src/runs.py"
    if encoding == "percent":
        source = quote(path, safe="")
    elif encoding == "json":
        source = json.dumps({"path": path, "description": "runs synthetic checks"})
    elif encoding == "nested_json":
        source = json.dumps({"arguments": json.dumps({"path": path}), "text": "runs synthetic"})
    else:
        source = path
    protected = engine.protect_text(source)
    assert protected != source
    assert engine.restore_text(protected) == source
    assert engine.protect_text("runs synthetic checks") == "runs synthetic checks"
    assert engine.protect_text(path) == alias + "/mandri/mandri_api/src/runs.py"


@settings(max_examples=60, deadline=None)
@given(
    st.lists(
        st.sampled_from(["runs", "synthetic", "remote", "mandri", "mandri_api", "src"]),
        min_size=1,
        max_size=12,
    )
)
def test_generated_nested_suffixes_never_register_plain_directory_names(parts):
    engine = SurrogateEngine(SurrogateScope("nested-property"))
    root = "/home/private/Mandri/runs/synthetic"
    alias = engine.register_root(root)
    suffix = "/" + "/".join(parts) + "/file.py"
    prose = " ".join(parts)
    source = {"nested": [{"path": root + suffix, "description": prose}]}
    protected = engine.protect(source)
    assert protected == {"nested": [{"path": alias + suffix, "description": prose}]}
    assert engine.restore(protected) == source
