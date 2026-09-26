import json
from itertools import pairwise

from hypothesis import given, settings
from hypothesis import strategies as st
from mandri.gateway.surrogate import StreamRestorer, SurrogateEngine, SurrogateScope


def subject():
    engine = SurrogateEngine(SurrogateScope("synthetic-stream"))
    engine.register_root("/home/private/Project", "/home/orchard/Service")
    protected = engine.protect_text("alice@example.fr")
    source = f"Begin é {protected}; /home/orchard/Service/mandri/src/new.py. End"
    return engine, source


def test_every_single_split_matches_complete_restoration():
    engine, source = subject()
    expected = engine.restore_text(source)
    for split in range(len(source) + 1):
        stream = StreamRestorer(engine)
        actual = stream.feed(source[:split]) + stream.feed(source[split:]) + stream.finish()
        assert actual == expected, split


def test_single_character_deltas_restore_email_and_new_descendant_path():
    engine, source = subject()
    stream = StreamRestorer(engine)
    actual = "".join(stream.feed(char) for char in source) + stream.finish()
    assert actual == engine.restore_text(source)


def test_incomplete_alias_is_retained_until_its_boundary_is_known():
    engine, _ = subject()
    alias = next(item.surrogate for item in engine.scope.mappings if item.kind == "email")
    stream = StreamRestorer(engine)
    assert stream.feed("prefix " + alias[:12]) == "prefix "
    assert stream.feed(alias[12:]) == ""
    assert stream.feed(".") == "alice@example.fr."
    assert stream.finish() == ""


def test_unknown_partial_alias_is_never_guessed_at_end_of_stream():
    engine, _ = subject()
    alias = next(item.surrogate for item in engine.scope.mappings if item.kind == "email")
    stream = StreamRestorer(engine)
    assert stream.feed(alias[:-3]) == ""
    assert stream.finish() == alias[:-3]


def test_word_boundaries_are_preserved_across_deltas():
    engine, _ = subject()
    alias = next(item.surrogate for item in engine.scope.mappings if item.kind == "email")
    for prefix, suffix in [("x", ""), ("", "suffix"), ("_", "suffix"), ("", "")]:
        source = prefix + alias + suffix
        stream = StreamRestorer(engine)
        actual = "".join(stream.feed(char) for char in source) + stream.finish()
        assert actual == engine.restore_text(source)


def test_interleaved_lanes_keep_independent_buffers():
    engine, first = subject()
    second = first[::-1]
    lanes = {"text": StreamRestorer(engine), "reasoning": StreamRestorer(engine)}
    results = {"text": "", "reasoning": ""}
    for index in range(max(len(first), len(second))):
        for lane, source in (("text", first), ("reasoning", second)):
            results[lane] += lanes[lane].feed(source[index : index + 1])
    for lane, source in (("text", first), ("reasoning", second)):
        results[lane] += lanes[lane].finish()
        assert results[lane] == engine.restore_text(source)


def test_scope_snapshot_has_a_fixed_mapping_set_for_the_response():
    engine, _ = subject()
    stream = StreamRestorer(engine)
    second = engine.protect_text("second@example.fr")
    assert stream.feed(second) + stream.finish() == second
    assert engine.restore_text(second) == "second@example.fr"


def test_finished_stream_can_restore_another_batch():
    engine, source = subject()
    stream = StreamRestorer(engine)
    stream.feed(source)
    stream.finish()
    assert stream.finish() == ""
    assert stream.feed("late content") + stream.finish() == "late content"


@settings(max_examples=100, deadline=None)
@given(st.lists(st.integers(min_value=0, max_value=200), min_size=0, max_size=30))
def test_arbitrary_fragment_partitions_have_identical_restore(cuts):
    engine, source = subject()
    positions = sorted({0, len(source), *(min(cut, len(source)) for cut in cuts)})
    stream = StreamRestorer(engine)
    actual = "".join(stream.feed(source[left:right]) for left, right in pairwise(positions))
    assert actual + stream.finish() == engine.restore_text(source)


def test_tool_arguments_are_collected_as_json_before_semantic_restoration():
    engine, _ = subject()
    original = {
        "path": "/home/private/Project/mandri/src/new.py",
        "value": 'quote: " and newline:\n',
    }
    protected = engine.protect(original)
    wire = json.dumps(protected)
    assembled = "".join(wire[index : index + 3] for index in range(0, len(wire), 3))
    restored = engine.restore(json.loads(assembled))
    assert restored == original
    assert json.loads(json.dumps(restored)) == original


def test_stream_restores_known_identity_aliases_in_relative_paths():
    engine, _ = subject()
    alias = engine.register("PrivateIdentity")
    for source in [
        f"src/{alias}/file.py",
        f"{alias}/src/file.py",
        f"/workspace/src/{alias}/file.py",
    ]:
        stream = StreamRestorer(engine)
        actual = "".join(stream.feed(char) for char in source) + stream.finish()
        assert actual == engine.restore_text(source) == source.replace(alias, "PrivateIdentity")


def test_stream_restores_model_generated_related_git_urls():
    engine, _ = subject()
    remote = engine.protect_text("git@github.com:private-team/private-repo.git")
    owner, repo = remote.split(":", 1)[1].removesuffix(".git").split("/")
    source = f"https://github.com/{owner}/{repo}/pull/123"
    stream = StreamRestorer(engine)
    actual = "".join(stream.feed(char) for char in source) + stream.finish()
    assert (
        actual
        == engine.restore_text(source)
        == "https://github.com/private-team/private-repo/pull/123"
    )
