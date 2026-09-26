import json

import pytest
from mandri.gateway.privacy_known import KnownValues
from mandri.gateway.privacy_protocol import transform_content, visit_content
from mandri.gateway.surrogate import SurrogateEngine, SurrogateScope


@pytest.mark.parametrize(
    "source",
    [
        "cat /tmp/FictionalOwner/report.txt",
        "FictionalOwner/file.txt",
        '{"filename": "/tmp/FictionalOwner/report.txt"}',
        "Bearer prefixopaque-test-key123456suffix",
        "prefixopaque-test-key123456suffix",
    ],
)
def test_registered_values_are_replaced_in_paths_and_embedded_credentials(source):
    engine = SurrogateEngine(SurrogateScope("synthetic"))
    engine.register("FictionalOwner", "username")
    engine.register("opaque-test-key123456", "secret")
    body = {"messages": [{"role": "user", "content": source}]}
    protected = transform_content(body, engine)
    protected = visit_content(protected, KnownValues(engine).replace)
    assert "FictionalOwner" not in json.dumps(protected)
    assert "opaque-test-key123456" not in json.dumps(protected)
    assert KnownValues(engine).replace(protected) == protected
    assert transform_content(protected, engine, restore=True) == body


def test_known_numeric_identifiers_are_replaced_without_field_context():
    engine = SurrogateEngine(SurrogateScope("synthetic"))
    engine.register("123456789", "identifier")
    protected = KnownValues(engine).replace({"arbitrary": [123456789]})
    assert protected != {"arbitrary": [123456789]}
    assert type(protected["arbitrary"][0]) is int


def test_unknown_model_options_are_detected_and_masked():
    engine = SurrogateEngine(SurrogateScope("synthetic"))
    original = {"messages": [], "custom_metadata": {"contact": "fictional@example.invalid"}}
    protected = transform_content(original, engine)
    assert "fictional@example.invalid" not in json.dumps(protected)
    assert transform_content(protected, engine, restore=True) == original
