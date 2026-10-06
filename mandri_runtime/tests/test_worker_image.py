from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from mandri.runtime.docker_backend import DockerBackend, DockerReadiness
from mandri.runtime.docker_config import DockerConfig
from mandri.runtime.errors.docker import DockerExecutionError
from mandri.runtime.worker_image import MANAGED_IMAGE, SOURCE_LABEL, prepare_worker, source_digest


async def test_managed_sources_build_once_then_refresh_when_source_changes(tmp_path, monkeypatch):
    sources = {"Dockerfile": b"FROM fixture\n", "worker/worker.py": b"original worker"}
    monkeypatch.setattr("mandri.runtime.worker_image.build_sources", lambda: dict(sources))
    cached = {}
    observed = []

    async def run(*args):
        return cached.get(args[-1], "")

    async def inspect(*args):
        return [{"Config": {"Labels": {SOURCE_LABEL: cached[args[-1]]}}}]

    async def build(context, reference, label, timeout):
        observed.append(reference)
        assert {
            p.relative_to(context).as_posix(): p.read_bytes()
            for p in Path(context).rglob("*")
            if p.is_file()
        } == sources
        cached[reference] = label.split("=", 1)[1]

    client = SimpleNamespace(run=run, json=inspect, build=build)
    first = await prepare_worker(client, tmp_path / "state", 30)
    assert await prepare_worker(client, tmp_path / "state", 30) == first
    assert observed == [first]
    sources["worker/worker.py"] = b"updated worker"
    second = await prepare_worker(client, tmp_path / "state", 30)
    assert second != first
    assert observed == [first, second]
    assert cached[first] != cached[second]
    assert not list(tmp_path.glob("worker-build-*"))


def test_source_digest_includes_paths_and_content_independently_of_order():
    assert source_digest({"a": b"x", "b": b"y"}) == source_digest({"b": b"y", "a": b"x"})
    assert source_digest({"a": b"x"}) != source_digest({"b": b"x"})
    assert source_digest({"a": b"x"}) != source_digest({"a": b"y"})


async def test_build_failure_removes_context_without_using_old_image(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "mandri.runtime.worker_image.build_sources", lambda: {"Dockerfile": b"FROM fixture"}
    )
    client = SimpleNamespace(
        run=AsyncMock(return_value=""), build=AsyncMock(side_effect=RuntimeError("build failed"))
    )
    with pytest.raises(RuntimeError, match="build failed"):
        await prepare_worker(client, tmp_path / "state", 30)
    assert not list(tmp_path.glob("worker-build-*"))


async def test_legacy_default_configuration_automatically_selects_managed_image(
    tmp_path, monkeypatch
):
    backend = DockerBackend(DockerConfig(MANAGED_IMAGE, tmp_path, pull_policy="never"))
    ready = DockerReadiness("image-id", "engine", "amd64", ("codex",), True)
    readiness = AsyncMock(return_value=ready)
    prepare = AsyncMock(return_value="mandri-worker:source-current")
    monkeypatch.setattr(backend, "readiness", readiness)
    monkeypatch.setattr("mandri.runtime.docker_backend.prepare_worker", prepare)
    assert await backend.prepare_image() is ready
    assert backend._image_reference == "mandri-worker:source-current"
    readiness.assert_any_await("mandri-worker:source-current")
    prepare.assert_awaited_once()


async def test_unqualified_engine_is_rejected_before_build(tmp_path, monkeypatch):
    backend = DockerBackend(DockerConfig(MANAGED_IMAGE, tmp_path))
    prepare = AsyncMock()
    monkeypatch.setattr("mandri.runtime.docker_backend.prepare_worker", prepare)
    monkeypatch.setattr(
        backend,
        "readiness",
        AsyncMock(side_effect=DockerExecutionError("docker_unavailable", "No engine")),
    )
    with pytest.raises(DockerExecutionError):
        await backend.prepare_image()
    prepare.assert_not_awaited()
