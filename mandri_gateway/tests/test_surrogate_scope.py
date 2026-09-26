import json

import pytest
from mandri.gateway.surrogate import SurrogateEngine, SurrogateGenerator, SurrogateScope, TypedRule


class SequenceGenerator(SurrogateGenerator):
    def __init__(self, candidates):
        self.candidates = iter(candidates)

    def generate(self, kind, original):
        return next(self.candidates)


def engine(**kwargs):
    return SurrogateEngine(SurrogateScope("synthetic-scope"), **kwargs)


def test_scope_snapshot_is_json_serializable_and_resumes_exact_assignments():
    first = engine()
    first.register_root("/home/private/Project", "/home/orchard/Service")
    source = {"email": "alice@example.fr", "file": "/home/private/Project/src/new.py"}
    protected = first.protect(source)
    serialized = json.loads(json.dumps(first.scope.to_dict()))
    resumed = SurrogateEngine(SurrogateScope.from_dict(serialized))
    assert resumed.protect(source) == protected
    assert resumed.restore(protected) == source
    assert (
        resumed.restore_text("/home/orchard/Service/src/future.py")
        == "/home/private/Project/src/future.py"
    )


def test_scopes_are_independent_and_fork_snapshots_do_not_mutate_parent():
    parent = engine()
    first = parent.protect_text("alice@example.fr")
    unrelated = engine().protect_text("alice@example.fr")
    assert first != unrelated
    fork_scope = SurrogateScope.from_dict(parent.scope.to_dict())
    fork_scope.scope_id = "synthetic-fork"
    fork = SurrogateEngine(fork_scope)
    assert fork.protect_text("alice@example.fr") == first
    fork.protect_text("second@example.fr")
    assert len(parent.scope.mappings) == 1 and len(fork.scope.mappings) == 2


def test_rejected_candidate_equal_to_later_natural_text_is_reallocated():
    subject = engine(generator=SequenceGenerator(["NaturalCandidate", "ReservedIdentity"]))
    source = {"full_name": "OriginalIdentity", "extra": "NaturalCandidate"}
    protected = subject.protect(source)
    assert protected["full_name"] not in {"OriginalIdentity", "NaturalCandidate"}
    assert protected["extra"] == "NaturalCandidate"
    assert subject.restore(protected) == source


def test_generator_collision_with_existing_assignment_retries_without_renumbering():
    subject = engine(generator=SequenceGenerator(["FirstAlias", "FirstAlias", "SecondAlias"]))
    first = subject.register("OriginalOne")
    second = subject.register("OriginalTwo")
    assert first == "FirstAlias" and second not in {first, "OriginalTwo"}
    assert subject.register("OriginalOne") == first


def test_repeating_generator_allocates_unique_values_without_exhaustion():
    subject = engine(generator=SequenceGenerator(["Same"] * 100))
    aliases = [subject.register(f"Original{index}") for index in range(100)]
    assert len(set(aliases)) == 100
    assert all(
        subject.restore_text(alias) == f"Original{index}" for index, alias in enumerate(aliases)
    )


def test_existing_surrogate_is_idempotent():
    subject = engine()
    alias = subject.register("PrivateIdentity")
    assert subject.protect_text(f"Natural new value: {alias}") == f"Natural new value: {alias}"
    assert subject.restore_text(alias + "suffix") == alias + "suffix"
    assert subject.restore_text(alias[:-1]) == alias[:-1]


def test_restoration_is_exact_and_does_not_recursively_expand_restored_content():
    subject = engine(generator=SequenceGenerator(["Cedar", "Orchard"]))
    subject.register("Willow")
    subject.register("Private Cedar")
    assert subject.restore_text("Orchard") == "Private Cedar"


def test_late_context_registration_covers_earlier_prose_in_same_payload():
    subject = engine()
    source = {"description": "PrivatePerson must review this change", "full_name": "PrivatePerson"}
    protected = subject.protect(source)
    assert "PrivatePerson" not in json.dumps(protected)
    assert subject.restore(protected) == source


def test_explicit_literal_matching_does_not_replace_inside_other_words():
    subject = engine()
    alias = subject.register("ann", "username")
    assert subject.protect_text("ann annex Joanne ann_data") == f"{alias} annex Joanne ann_data"


def test_dynamically_discovered_values_are_registered_from_real_later_payloads():
    subject = engine()
    subject.protect({"initial": "No identity is present"})
    assert subject.scope.mappings == []
    result = subject.protect({"tool_result": [{"email": "discovered@example.fr"}]})
    assert "discovered@example.fr" not in json.dumps(result)
    assert subject.restore(result) == {"tool_result": [{"email": "discovered@example.fr"}]}


@pytest.mark.parametrize(
    "mutation",
    [
        lambda state: state.update(version=999),
        lambda state: state.update(scope_id=""),
        lambda state: state.update(mappings="wrong"),
        lambda state: state["mappings"].append(state["mappings"][0]),
        lambda state: state["mappings"][0].update(kind="arbitrary-code"),
        lambda state: state["mappings"][0].update(surrogate=state["mappings"][0]["original"]),
        lambda state: state["mappings"][0].update(unknown="field"),
        lambda state: state["roots"].append(
            {"original": "/unknown", "surrogate": "/unknown2", "case_sensitive": True}
        ),
    ],
)
def test_scope_loading_tolerates_unknown_fields_and_obsolete_metadata(mutation):
    subject = engine()
    subject.protect_text("alice@example.fr")
    state = subject.scope.to_dict()
    mutation(state)
    SurrogateEngine(SurrogateScope.from_dict(state)).protect_text("second@example.fr")


@pytest.mark.parametrize("value", [b"binary", float("nan"), float("inf"), {123: "non-string-key"}])
def test_unknown_values_pass_through(value):
    result = engine().protect(value)
    assert result is value or result == value


def test_custom_bounded_typed_rule_preserves_its_literal_prefix_and_regex():
    rule = TypedRule("customer-ticket", r"TICKET-[A-Z]{3}-[0-9]{6}")
    subject = engine(rules=[rule])
    source = "Handle TICKET-ABC-012345 and TICKET-ABC-012345"
    result = subject.protect_text(source)
    aliases = rule.compiled.findall(result)
    assert len(aliases) == 2 and aliases[0] == aliases[1]
    assert aliases[0] != "TICKET-ABC-012345"
    assert subject.restore_text(result) == source


@pytest.mark.parametrize(
    "pattern",
    [
        r"(a+)+$",
        r"a.*b",
        r"[A-Z]+",
        r"(foo|bar)",
        r"[0-9]{0,8}",
        r"(?=private)",
        r"a{1,128}a{1,128}Z",
    ],
)
def test_unsupported_custom_regex_shapes_do_not_detect(pattern):
    assert TypedRule("unknown", pattern).find("private foo ABC123", "") == []


def test_explicit_field_custom_rule_only_matches_its_configured_context():
    subject = engine(
        rules=[TypedRule("internal", r"TEAM-[0-9]{4}", fields=frozenset({"internal_tag"}))]
    )
    result = subject.protect({"public_tag": "TEAM-1234", "internal_tag": "TEAM-5678"})
    assert result["public_tag"] == "TEAM-1234"
    assert result["internal_tag"] != "TEAM-5678"


def test_embedded_json_whitespace_and_untouched_tokens_are_preserved():
    subject = engine()
    original = '{\n  "x": [ "alice@example.fr", 1.000, true ], "untouched": "\\u00e9"\n}'
    result = subject.protect_text(original)
    assert '\n  "x": [ "' in result
    assert ', 1.000, true ], "untouched": "\\u00e9"\n}' in result
    assert subject.restore_text(result) == original


def test_schema_property_keys_required_and_pointer_references_stay_coherent():
    subject = engine()
    subject.register("PrivateProject")
    source = {
        "$defs": {"PrivateProject": {"type": "string"}},
        "properties": {"PrivateProject": {"$ref": "#/$defs/PrivateProject"}},
        "required": ["PrivateProject"],
    }
    result = subject.protect(source)
    alias = result["required"][0]
    assert set(result["$defs"]) == {alias}
    assert set(result["properties"]) == {alias}
    assert result["properties"][alias]["$ref"] == f"#/$defs/{alias}"
    assert subject.restore(result) == source


@pytest.mark.parametrize("suffix", ["", "@", ".invalid"])
def test_long_non_entity_runs_do_not_require_unbounded_pattern_backtracking(suffix):
    subject = engine()
    source = "x" * 200000 + suffix
    assert subject.protect_text(source) == source


def test_duplicate_nested_json_object_keys_are_preserved():
    source = '{"private": "first", "private": "second"}'
    assert engine().protect_text(source) == source


def test_new_context_updates_compound_and_preserves_previous_restoration():
    subject = engine()
    source = "https://service.private.example/api/private-customer/issues"
    previous = subject.protect_text(source)
    subject.register("private-customer")
    current = subject.protect_text(source)
    assert "private-customer" not in current
    assert subject.protect_text(source) == current
    assert subject.restore_text(current) == source
    assert subject.restore_text(previous) == source
    restarted = SurrogateEngine(SurrogateScope.from_dict(subject.scope.to_dict()))
    assert restarted.protect_text(source) == current
    assert restarted.restore_text(previous) == source


def test_fixed_width_rule_uses_bounded_matching_on_long_near_misses():
    rule = TypedRule("long-fixed", "[a-z]{1024}Z")
    assert rule.find("a" * 200_000, "") == []
    text = "prefix " + "a" * 1024 + "Z suffix"
    assert [(span.start, span.end) for span in rule.find(text, "")] == [(7, 1032)]


def test_large_text_and_many_nodes_are_transformed_without_privacy_limits():
    subject = engine()
    text = "ordinary text " * 160_000 + " contact alice@example.fr"
    protected = subject.protect_text(text)
    assert "alice@example.fr" not in protected
    assert subject.restore_text(protected) == text
    payload = [None] * 100_001 + ["alice@example.fr"]
    protected_nodes = subject.protect(payload)
    assert protected_nodes[-1] != payload[-1]
    assert subject.restore(protected_nodes) == payload


def test_deep_structures_use_iterative_transformation():
    subject = engine()
    payload = "alice@example.fr"
    for _ in range(1200):
        payload = [payload]
    protected = subject.protect(payload)
    restored = subject.restore(protected)
    for _ in range(1200):
        protected = protected[0]
        restored = restored[0]
    assert protected != "alice@example.fr"
    assert restored == "alice@example.fr"


def test_scope_grows_beyond_previous_mapping_limit_and_resumes():
    subject = engine()
    originals = [f"PrivateValue{index}" for index in range(50_001)]
    aliases = [subject.register(value) for value in originals]
    assert len(set(aliases)) == len(originals)
    resumed = SurrogateEngine(SurrogateScope.from_dict(subject.scope.to_dict()))
    for index in (0, 25_000, 50_000):
        assert resumed.register(originals[index]) == aliases[index]
        assert resumed.restore_text(aliases[index]) == originals[index]


def test_repeated_payload_transformation_does_not_allocate_aliases_for_aliases():
    subject = engine()
    source = {"email": "alice@example.fr", "url": "https://private.example/path?token=Secret123"}
    protected = subject.protect(source)
    snapshot = subject.scope.to_dict()
    for _ in range(3):
        assert subject.protect(protected) == protected
        assert subject.protect(source) == protected
        assert subject.restore(protected) == source
    assert subject.scope.to_dict() == snapshot


def test_fixed_width_rule_collision_keeps_prefix_and_matching_shape(monkeypatch):
    subject = engine(rules=[TypedRule("ticket", "TICKET-[0-9]{4}")])
    monkeypatch.setattr(TypedRule, "generate", lambda self, original: "TICKET-3333")
    first = subject.protect_text("TICKET-1000")
    second = subject.protect_text("TICKET-2000")
    assert first != second
    assert subject.rules[0].compiled.fullmatch(second)
    assert subject.restore_text(second) == "TICKET-2000"


def test_collision_search_wraps_without_leaving_the_rule_alphabet(monkeypatch):
    subject = engine(rules=[TypedRule("ticket", "TICKET-[0-9]")])
    monkeypatch.setattr(TypedRule, "generate", lambda self, original: "TICKET-9")
    first = subject.protect_text("TICKET-4")
    second = subject.protect_text("TICKET-5")
    assert first == "TICKET-9"
    assert second == "TICKET-0"
    assert subject.restore_text(second) == "TICKET-5"


def test_literal_only_rule_does_not_detect_unreplaceable_content():
    rule = TypedRule("literal", "PRIVATE")
    assert rule.find("PRIVATE", "") == []
    assert engine(rules=[rule]).protect_text("PRIVATE") == "PRIVATE"
