import pytest
from mandri.core.image_reference import valid_image_reference

DIGEST = "sha256:" + "a" * 64


@pytest.mark.parametrize(
    "reference",
    [
        "worker",
        "worker:latest",
        "namespace/worker:v1.2",
        "registry.invalid:5000/team/worker",
        "localhost:5000/worker@" + DIGEST,
        "worker:release@" + DIGEST,
        DIGEST,
        "127.0.0.1:34567/worker@" + DIGEST,
    ],
)
def test_valid_local_or_registry_image_reference(reference):
    assert valid_image_reference(reference)


@pytest.mark.parametrize(
    "reference",
    [
        "",
        "https://registry.invalid/worker",
        "user:password@registry.invalid/worker",
        "registry.invalid/worker?token=secret",
        "registry.invalid/worker#fragment",
        "worker\n",
        "worker\x00",
        "worker@sha256:wrong",
        "worker@" + DIGEST + "@extra",
        "Uppercase",
        "../worker",
        "worker//name",
        "registry..invalid/worker",
        "registry.-bad/worker",
        "localhost:0/worker",
        "localhost:65536/worker",
        "worker:" + "x" * 129,
    ],
)
def test_reference_rejects_credentials_urls_and_invalid_syntax(reference):
    assert not valid_image_reference(reference)


@pytest.mark.parametrize("reference", ["worker", "worker:latest", DIGEST])
def test_acquisition_requires_repository_digest(reference):
    assert not valid_image_reference(reference, digest_required=True)
