import asyncio
import uuid
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from mandri.core.types.execution import ExecutionBackend, ProtectionError
from mandri.runtime.docker_backend import DockerBackend, DockerReadiness
from mandri.runtime.docker_config import DockerConfig
from mandri.runtime.docker_image import image_options, prepare_image
from mandri.runtime.errors.docker import DockerExecutionError
from mandri.runtime.service import RuntimeService

DIGEST = "registry.invalid/worker@sha256:" + "a" * 64
READY = DockerReadiness("sha256:" + "b" * 64, "engine", "amd64", ("codex",))


def missing():
    return DockerExecutionError("docker_image_missing", "Missing")


async def test_missing_default_image_never_downloads(tmp_path):
    config = DockerConfig(DIGEST, tmp_path)
    client = SimpleNamespace(pull=AsyncMock())
    with pytest.raises(DockerExecutionError) as failure:
        await prepare_image(client, config, AsyncMock(side_effect=missing()))
    assert failure.value.reason == "docker_image_missing"
    client.pull.assert_not_awaited()


async def test_cold_download_revalidates_and_warm_cache_never_pulls(tmp_path):
    config = DockerConfig(DIGEST, tmp_path, pull_policy="if-missing", pull_timeout_seconds=15)
    client = SimpleNamespace(pull=AsyncMock())
    readiness = AsyncMock(side_effect=[missing(), READY, READY])
    assert await prepare_image(client, config, readiness) == READY
    assert await prepare_image(client, config, readiness) == READY
    client.pull.assert_awaited_once_with(DIGEST, 15)


@pytest.mark.parametrize(
    "reason",
    [
        "docker_unavailable",
        "docker_image_incompatible",
        "docker_platform_unqualified",
        "docker_operation_timed_out",
    ],
)
async def test_only_missing_image_permits_acquisition(tmp_path, reason):
    config = DockerConfig(DIGEST, tmp_path, pull_policy="if-missing")
    client = SimpleNamespace(pull=AsyncMock())
    with pytest.raises(DockerExecutionError) as failure:
        await prepare_image(
            client, config, AsyncMock(side_effect=DockerExecutionError(reason, "No"))
        )
    assert failure.value.reason == reason
    client.pull.assert_not_awaited()


@pytest.mark.parametrize("image", ["worker:latest", "sha256:" + "a" * 64, "https://bad/worker"])
async def test_missing_unpinned_image_cannot_download_even_with_direct_config(tmp_path, image):
    config = DockerConfig(image, tmp_path, pull_policy="if-missing")
    client = SimpleNamespace(pull=AsyncMock())
    with pytest.raises(DockerExecutionError) as failure:
        await prepare_image(client, config, AsyncMock(side_effect=missing()))
    assert failure.value.reason == "docker_image_reference_invalid"
    client.pull.assert_not_awaited()
    assert not image_options(config).can_prepare


@pytest.mark.parametrize(
    "reason,expected",
    [
        ("docker_image_missing", "docker_image_pull_failed"),
        ("docker_image_incompatible", "docker_image_incompatible"),
        ("docker_platform_unqualified", "docker_platform_unqualified"),
        ("docker_unavailable", "docker_unavailable"),
    ],
)
async def test_downloaded_image_must_pass_full_readiness(tmp_path, reason, expected):
    config = DockerConfig(DIGEST, tmp_path, pull_policy="if-missing")
    readiness = AsyncMock(side_effect=[missing(), DockerExecutionError(reason, "Bad image")])
    with pytest.raises(DockerExecutionError) as failure:
        await prepare_image(SimpleNamespace(pull=AsyncMock()), config, readiness)
    assert failure.value.reason == expected


async def test_concurrent_starts_share_one_acquisition_and_recheck_cache(tmp_path, monkeypatch):
    backend = DockerBackend(DockerConfig(DIGEST, tmp_path, pull_policy="if-missing"))
    entered, release = asyncio.Event(), asyncio.Event()
    cached = False

    async def readiness():
        if not cached:
            raise missing()
        return READY

    async def pull(*args):
        nonlocal cached
        entered.set()
        await release.wait()
        cached = True

    monkeypatch.setattr(backend, "readiness", readiness)
    backend.client.pull = AsyncMock(side_effect=pull)
    first = asyncio.create_task(backend.prepare_image())
    await entered.wait()
    cancelled = asyncio.create_task(backend.prepare_image())
    second = asyncio.create_task(backend.prepare_image())
    await asyncio.sleep(0)
    cancelled.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cancelled
    assert not first.done()
    release.set()
    assert await asyncio.gather(first, second) == [READY, READY]
    backend.client.pull.assert_awaited_once()


async def test_cancelling_owner_releases_acquisition_for_next_start(tmp_path, monkeypatch):
    backend = DockerBackend(DockerConfig(DIGEST, tmp_path, pull_policy="if-missing"))
    entered = asyncio.Event()

    async def pull(*args):
        entered.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(backend, "readiness", AsyncMock(side_effect=missing()))
    backend.client.pull = AsyncMock(side_effect=pull)
    first = asyncio.create_task(backend.prepare_image())
    await entered.wait()
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    monkeypatch.setattr(backend, "readiness", AsyncMock(return_value=READY))
    assert await asyncio.wait_for(backend.prepare_image(), 1) == READY


@pytest.mark.parametrize("explicit_id", [False, True])
async def test_start_prepares_in_operation_before_policy_scope_or_routes(
    tmp_path, monkeypatch, explicit_id
):
    runtime = RuntimeService({"codex": ["codex"]}, docker_config=DockerConfig(DIGEST, tmp_path))
    observed = []

    async def prepare():
        observed.append(runtime._start_operations.current_id)
        raise missing()

    monkeypatch.setattr(runtime._docker, "prepare_image", prepare)
    scope = AsyncMock()
    policy = AsyncMock()
    route = AsyncMock()
    monkeypatch.setattr(runtime, "_create_privacy_scope", scope)
    monkeypatch.setattr(runtime, "_validate_policy", policy)
    monkeypatch.setattr(runtime, "_bind_route", route)
    identifier = str(uuid.uuid4()) if explicit_id else None
    with pytest.raises(DockerExecutionError):
        await runtime.start_session(
            "codex",
            "provider/model",
            tmp_path,
            execution_backend=ExecutionBackend.DOCKER,
            operation_id=identifier,
        )
    assert len(observed) == 1 and uuid.UUID(observed[0])
    if identifier:
        assert observed[0] == identifier
    scope.assert_not_awaited()
    policy.assert_not_awaited()
    route.assert_not_awaited()


async def test_explicit_cancel_during_acquisition_has_operation_code_and_no_session(
    tmp_path, monkeypatch
):
    runtime = RuntimeService({"codex": ["codex"]}, docker_config=DockerConfig(DIGEST, tmp_path))
    entered, cleaned = asyncio.Event(), asyncio.Event()

    async def prepare():
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaned.set()

    monkeypatch.setattr(runtime._docker, "prepare_image", prepare)
    identifier = str(uuid.uuid4())
    start = asyncio.create_task(
        runtime.start_session(
            "codex",
            "provider/model",
            tmp_path,
            execution_backend=ExecutionBackend.DOCKER,
            operation_id=identifier,
        )
    )
    await entered.wait()
    await runtime.cancel_start(identifier)
    assert cleaned.is_set()
    with pytest.raises(ProtectionError) as failure:
        await start
    assert failure.value.code == "operation_cancelled"
    assert not runtime.registry.live_ids()


def test_capability_does_not_publish_unvalidated_credentials(tmp_path):
    config = DockerConfig("user:secret@registry.invalid/worker", tmp_path, pull_policy="if-missing")
    assert image_options(config).configured_image is None
    assert not image_options(config).can_prepare
    assert image_options(replace(config, image=DIGEST)).can_prepare
