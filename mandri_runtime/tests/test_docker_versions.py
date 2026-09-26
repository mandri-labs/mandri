from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from mandri.runtime.docker_versions import DockerVersions
from mandri.runtime.errors.docker import DockerExecutionError


async def test_native_version_probe_is_offline_isolated_and_cached_by_image():
    client = SimpleNamespace(run=AsyncMock(return_value="codex-cli 0.154.0\n"), remove=AsyncMock())
    versions = DockerVersions(client, "owner")
    assert await versions.version("sha256:one", "codex") == "0.154.0"
    assert await versions.version("sha256:one", "codex") == "0.154.0"
    assert client.run.await_count == 1
    arguments = client.run.call_args.args
    assert arguments[arguments.index("--network") + 1] == "none"
    assert "--mount" not in arguments and "--privileged" not in arguments
    assert arguments[-3:] == ("codex", "sha256:one", "--version")
    assert await versions.version("sha256:two", "codex") == "0.154.0"
    assert client.run.await_count == 2 and client.remove.await_count == 2


async def test_invalid_native_version_fails_closed_and_cleans_up():
    client = SimpleNamespace(run=AsyncMock(return_value="unknown"), remove=AsyncMock())
    with pytest.raises(DockerExecutionError, match="version is unavailable"):
        await DockerVersions(client, "owner").version("sha256:one", "codex")
    client.remove.assert_awaited_once()
