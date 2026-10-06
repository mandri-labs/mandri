from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from mandri.runtime.docker_backend import DockerBackend
from mandri.runtime.docker_config import DockerConfig
from mandri.runtime.errors.docker import DockerExecutionError


@pytest.mark.parametrize("platform", ["linux", "darwin", "win32"])
@pytest.mark.parametrize("desktop", [False, True])
async def test_linux_worker_can_run_on_any_docker_host(tmp_path, monkeypatch, platform, desktop):
    monkeypatch.setattr("mandri.runtime.docker_backend.sys.platform", platform)
    backend = DockerBackend(DockerConfig("image", tmp_path))
    info = {
        "OSType": "linux",
        "ID": "engine",
        "NCPU": 8,
        "MemTotal": 16 * 1024**3,
        "OperatingSystem": "Docker Desktop" if desktop else "Linux",
    }
    image = {
        "Id": "worker",
        "Os": "linux",
        "Architecture": "amd64",
        "Config": {
            "Labels": {
                "io.mandri.worker.version": "1",
                "io.mandri.worker.workspace-identity": "1",
                "io.mandri.worker.native-run": "1",
                "io.mandri.worker.harnesses": "codex",
            }
        },
    }
    backend.client = SimpleNamespace(json=AsyncMock(side_effect=[info, [image]]))
    ready = await backend.readiness()
    assert ready.image_id == "worker"
    assert ready.native
    assert backend.desktop == (desktop or platform != "linux")
    assert backend.client.json.call_args_list[0].args[0] == "info"


async def test_windows_containers_cannot_run_linux_worker(tmp_path):
    backend = DockerBackend(DockerConfig("image", tmp_path))
    backend.client = SimpleNamespace(json=AsyncMock(return_value={"OSType": "windows"}))
    with pytest.raises(DockerExecutionError, match="Linux container engine"):
        await backend.readiness()
