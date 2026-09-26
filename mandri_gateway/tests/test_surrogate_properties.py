import json

from hypothesis import given, settings
from hypothesis import strategies as st
from mandri.gateway.surrogate import SurrogateEngine, SurrogateScope
from mandri.gateway.surrogate.detectors import select_spans
from mandri.gateway.surrogate.types import Span

LOCAL = st.text(alphabet="abcdefghijklmnopqrstuvwxyz0123456789", min_size=5, max_size=32)
EMAILS = LOCAL.map(lambda local: local + "@private.example.fr")


@settings(max_examples=150, deadline=None)
@given(st.integers(min_value=100, max_value=10**18))
def test_numeric_private_ids_keep_json_number_format_and_exact_value(value):
    engine = SurrogateEngine(SurrogateScope("numeric-property"))
    source = {"account_id": value}
    protected = engine.protect(source)
    assert type(protected["account_id"]) is int
    assert protected["account_id"] != value
    assert engine.restore(json.loads(json.dumps(protected))) == source


@settings(max_examples=100, deadline=None)
@given(st.lists(EMAILS, min_size=1, max_size=20))
def test_reversible_bijective_email_assignments_preserve_non_entity_text(emails):
    engine = SurrogateEngine(SurrogateScope("email-property"))
    source = "\n".join("<" + email + ">: keep punctuation & whitespace." for email in emails)
    result = engine.protect_text(source)
    assert all(email not in result for email in emails)
    assert result.count(": keep punctuation & whitespace.") == len(emails)
    assert engine.restore_text(result) == source
    mappings = [item for item in engine.scope.mappings if item.kind == "email"]
    assert len(mappings) == len(set(emails))
    assert len({item.surrogate for item in mappings}) == len(mappings)
    assert engine.protect_text(source) == result


CONTENT = st.recursive(
    st.one_of(EMAILS, st.none(), st.booleans(), st.integers(min_value=-1000, max_value=1000)),
    lambda children: st.one_of(
        st.lists(children, max_size=4),
        st.dictionaries(
            st.sampled_from(["items", "content", "data", "result"]), children, max_size=4
        ),
    ),
    max_leaves=25,
)


@settings(max_examples=100, deadline=None)
@given(CONTENT)
def test_nested_json_roundtrip_keeps_structure_types_and_values(source):
    engine = SurrogateEngine(SurrogateScope("json-property"))
    result = engine.protect(source)
    assert engine.restore(result) == source
    assert json.loads(json.dumps(engine.restore(result))) == source


@settings(max_examples=100, deadline=None)
@given(
    st.permutations(
        [
            Span(0, 25, "url", 180),
            Span(5, 12, "secret", 120),
            Span(8, 15, "identity", 150),
            Span(30, 40, "email", 115),
            Span(30, 40, "identity", 100),
        ]
    )
)
def test_overlapping_span_selection_does_not_depend_on_detector_order(spans):
    assert select_spans(list(spans)) == [Span(0, 25, "url", 180), Span(30, 40, "email", 115)]


@settings(max_examples=100, deadline=None)
@given(EMAILS)
def test_serialization_restart_and_scope_validation_do_not_change_alias(email):
    engine = SurrogateEngine(SurrogateScope("restart-property"))
    alias = engine.protect_text(email)
    restored_scope = SurrogateScope.from_dict(json.loads(json.dumps(engine.scope.to_dict())))
    resumed = SurrogateEngine(restored_scope)
    assert resumed.protect_text(email) == alias
    assert resumed.restore_text(alias) == email
    before = resumed.scope.to_dict()
    assert resumed.scope.to_dict() == before
