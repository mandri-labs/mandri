import base64
import json
from urllib.parse import quote

import pytest
from mandri.core.types.execution import ProtectionError
from mandri.gateway.privacy_known import KnownValues
from mandri.gateway.privacy_protocol import transform_content
from mandri.gateway.surrogate import SurrogateEngine, SurrogateScope
from mandri.gateway.surrogate.stream import StreamRestorer
from mandri.gateway.surrogate.types import Mapping

ENCODERS = [
    lambda value: base64.b64encode(value.encode()).decode(),
    lambda value: base64.b64encode(value.encode()).decode().rstrip("="),
    lambda value: base64.urlsafe_b64encode(value.encode()).decode(),
    lambda value: base64.b32encode(value.encode()).decode(),
    lambda value: value.encode().hex(),
    lambda value: value.encode().hex().upper(),
    lambda value: quote(value, safe=""),
    lambda value: "".join(f"%{byte:02X}" for byte in value.encode()),
    lambda value: value.replace(".", "%2E").replace("@", "%40"),
    lambda value: value.replace(".", r"\u002e").replace("@", r"\u0040"),
    lambda value: "".join(f"\\u{ord(char):04x}" for char in value),
]


@pytest.mark.parametrize("depth", [1, 2])
@pytest.mark.parametrize("original", [r"D:\Private\Project", "/home/private/project"])
def test_json_escaped_root_in_markdown_restores_descendants_after_reload(depth, original):
    engine = SurrogateEngine(SurrogateScope("escaped-path"))
    alias = engine.register_root(original)
    suffix = r"\fixtures\person.json" if original.startswith("D:") else "/fixtures/person.json"

    def encode(value):
        for _ in range(depth):
            value = json.dumps(value)[1:-1]
        return value

    source = 'Result:\n```json\n{"file": "' + encode(original + suffix) + '"}\n```'
    protected = 'Result:\n```json\n{"file": "' + encode(alias + suffix) + '"}\n```'
    assert engine.protect_text(source) == protected
    assert engine.restore_text(protected) == source
    resumed = SurrogateEngine(SurrogateScope.from_dict(engine.scope.to_dict()))
    assert resumed.restore_text(protected) == source
    for width in (1, 3, 7):
        stream = StreamRestorer(resumed)
        restored = "".join(
            stream.feed(protected[i : i + width]) for i in range(0, len(protected), width)
        )
        assert restored + stream.finish() == source


@pytest.mark.parametrize("encode", ENCODERS)
def test_known_representation_reuses_the_same_alias_and_restores_after_reload(encode):
    engine = SurrogateEngine(SurrogateScope("encoding"))
    original = "alice.smith@private.test"
    alias = engine.register(original, "email")
    source = "Value: " + encode(original) + ";"
    protected = engine.protect_text(source)
    assert encode(original) not in protected
    expected = encode(alias) if encode(original).endswith("=") else encode(alias).rstrip("=")
    assert expected in protected
    assert engine.protect_text(protected) == protected
    assert engine.restore_text(protected) == source
    resumed = SurrogateEngine(SurrogateScope.from_dict(engine.scope.to_dict()))
    assert resumed.protect_text(source) == protected
    assert resumed.restore_text(protected) == source
    known = KnownValues(resumed)
    assert known.text(source) == protected
    for width in (1, 3, 7):
        stream = StreamRestorer(resumed)
        restored = "".join(
            stream.feed(protected[i : i + width]) for i in range(0, len(protected), width)
        )
        assert restored + stream.finish() == source


def test_unknown_opaque_and_longer_base64_tokens_remain_unchanged():
    engine = SurrogateEngine(SurrogateScope("encoding"))
    original = "private-person"
    encoded = base64.b64encode(original.encode()).decode()
    assert engine.protect_text(encoded) == encoded
    engine.register(original)
    unrelated = base64.b64encode(b"unrelated opaque content").decode()
    for value in (unrelated, "X" + encoded, encoded + "X", "prefix" + encoded + "suffix"):
        assert engine.protect_text(value) == value


def test_discovery_later_in_request_masks_earlier_encoded_value():
    original = "alice.smith@private.test"
    encoded = base64.b64encode(original.encode()).decode()
    engine = SurrogateEngine(SurrogateScope("encoding"))
    payload = {"input": "Known representation: " + encoded, "metadata": {"email": original}}
    protected = transform_content(payload, engine)
    assert encoded not in protected["input"]
    assert transform_content(protected, engine, restore=True) == payload


@pytest.mark.parametrize(
    "block",
    [
        {"type": "input_image", "image_url": "https://media.private.test/private-person.png"},
        {
            "type": "image_url",
            "image_url": {"url": "https://media.private.test/private-person.png", "detail": "high"},
        },
        {
            "type": "image",
            "source": {"type": "url", "url": "https://media.private.test/private-person.png"},
        },
        {
            "fileData": {
                "fileUri": "https://media.private.test/private-person.png",
                "mimeType": "image/png",
            }
        },
    ],
)
def test_media_textual_urls_are_transformed_and_restored(block):
    engine = SurrogateEngine(SurrogateScope("media-url"))
    engine.register("private-person", "username")
    payload = {"input": [{"role": "user", "content": [block]}]}
    protected = transform_content(payload, engine)
    assert "private-person" not in json.dumps(protected)
    assert transform_content(protected, engine, restore=True) == payload
    assert "private-person" not in json.dumps(KnownValues(engine).replace(block))


@pytest.mark.parametrize(
    "block",
    [
        {"type": "input_image", "image_url": "data:image/png;base64,cHJpdmF0ZS1wZXJzb24="},
        {"type": "input_audio", "input_audio": {"data": "cHJpdmF0ZS1wZXJzb24=", "format": "wav"}},
        {
            "type": "image",
            "source": {"type": "base64", "data": "cHJpdmF0ZS1wZXJzb24=", "media_type": "image/png"},
        },
        {"inlineData": {"mimeType": "image/png", "data": "cHJpdmF0ZS1wZXJzb24="}},
    ],
)
def test_opaque_media_never_enters_known_encoded_matching(block):
    engine = SurrogateEngine(SurrogateScope("media-data"))
    engine.register("private-person")
    payload = {"input": [{"role": "user", "content": [block]}]}
    assert transform_content(payload, engine) == payload
    assert KnownValues(engine).replace(block) == block


def test_legacy_scope_retains_aliases_and_uses_the_current_serialization_policy():
    original = "alice.smith@private.test"
    historical = "a1b2c3d4@old.invalid"
    scope = SurrogateScope.from_dict(
        {
            "scope_id": "legacy",
            "version": 2,
            "mappings": [{"kind": "email", "original": original, "surrogate": historical}],
        }
    )
    engine = SurrogateEngine(scope)
    assert scope.version == 3
    current = engine.protect_text(original)
    assert current != historical and current.endswith(".com")
    assert engine.protect_text(historical) == current
    assert engine.restore_text(historical) == original
    fresh = engine.register("bob.jones@private.test", "email")
    assert fresh.endswith(".com")
    resumed = SurrogateEngine(SurrogateScope.from_dict(scope.to_dict()))
    assert resumed.protect_text(original) == current
    assert resumed.restore_text(historical) == original
    assert resumed.restore_text(fresh) == "bob.jones@private.test"


def test_encoded_alias_cannot_overwrite_a_canonical_reverse_mapping():
    original, alias = "private-person", "fictional-person"
    encoded_alias = base64.b64encode(alias.encode()).decode()
    engine = SurrogateEngine(
        SurrogateScope(
            "collision",
            mappings=[
                Mapping("identity", original, alias),
                Mapping("secret", "another-private-secret", encoded_alias),
            ],
        )
    )
    with pytest.raises(ProtectionError, match="Encoded and canonical"):
        engine.protect_text(base64.b64encode(original.encode()).decode())


def test_conflicting_exact_encoded_spellings_fail_before_egress():
    engine = SurrogateEngine(
        SurrogateScope(
            "ambiguous",
            mappings=[
                Mapping("identity", "private", "fictional"),
            ],
        )
    )
    padded = base64.b64encode(b"private").decode()
    unpadded = padded.rstrip("=")
    protected = engine.protect_text(padded)
    assert engine.restore_text(protected) == padded
    with pytest.raises(ProtectionError, match="spellings conflict"):
        engine.protect_text(unpadded)


def test_legacy_invalid_url_scheme_is_repaired_without_losing_the_old_reverse_alias():
    original = "https://private.test/document"
    previous = "invented://fictional.com/document"
    scope = SurrogateScope.from_dict(
        {
            "scope_id": "invalid-url",
            "version": 2,
            "mappings": [{"kind": "url", "original": original, "surrogate": previous}],
        }
    )
    engine = SurrogateEngine(scope)
    current = engine.protect_text(original)
    assert current.startswith("https://")
    assert engine.restore_text(previous) == original
    assert engine.protect_text(previous) == current
    resumed = SurrogateEngine(SurrogateScope.from_dict(scope.to_dict()))
    assert resumed.protect_text(original) == current
    assert resumed.restore_text(previous) == original
