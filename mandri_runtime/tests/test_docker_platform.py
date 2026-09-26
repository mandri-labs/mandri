from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from mandri.runtime.docker_backend import DockerBackend
from mandri.runtime.docker_config import DockerConfig
from mandri.runtime.errors.docker import DockerExecutionError


@pytest.mark.parametrize("platform", ["darwin", "win32"])
async def test_unqualified_daemon_platform_never_probes_engine(tmp_path, monkeypatch, platform):
    monkeypatch.setattr("mandri.runtime.docker_backend.sys.platform", platform)
    backend = DockerBackend(DockerConfig("image", tmp_path))
    backend.client = SimpleNamespace(json=AsyncMock())
    with pytest.raises(DockerExecutionError) as error:
        await backend.readiness()
    assert error.value.reason == "docker_platform_unqualified"
    backend.client.json.assert_not_awaited()


@pytest.mark.parametrize(
    "mutation",
    [
        {"OperatingSystem": "Docker Desktop"},
        {"Name": "docker-desktop"},
        {"Architecture": "aarch64"},
        {"Architecture": "unknown"},
    ],
)
async def test_unqualified_engine_topologies_fail_before_any_worker(
    tmp_path, monkeypatch, mutation
):
    monkeypatch.setattr("mandri.runtime.docker_backend.sys.platform", "linux")
    backend = DockerBackend(DockerConfig("image", tmp_path))
    info = {"OSType": "linux", "Architecture": "x86_64", **mutation}
    backend.client = SimpleNamespace(
        json=AsyncMock(
            side_effect=[[{"Endpoints": {"docker": {"Host": "unix:///var/run/docker.sock"}}}], info]
        )
    )
    with pytest.raises(DockerExecutionError) as error:
        await backend.readiness()
    assert error.value.reason == "docker_platform_unqualified"
    assert backend.client.json.await_count == 2
