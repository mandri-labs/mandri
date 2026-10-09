import asyncio
import sys
from contextlib import suppress
from pathlib import Path

import psutil
import pytest
from mandri.runtime.docker_client import DockerClient
from mandri.runtime.errors.docker import DockerExecutionError


@pytest.fixture
def cli(tmp_path, monkeypatch):
    script = tmp_path / "docker"
    script.write_text(
        f"#!{sys.executable}\n"
        "import os, sys, time\n"
        "from pathlib import Path\n"
        "mode = sys.argv[-1]\n"
        "if mode == 'slow':\n"
        "    Path(__file__).with_suffix('.pid').write_text(str(os.getpid()))\n"
        "    time.sleep(30)\n"
        "messages = {'missing': 'Error: No such image: private', "
        "'engine': 'Cannot connect to the Docker daemon at unix:///var/run/docker.sock', "
        "'bad': 'digest mismatch secret-registry-data'}\n"
        "if mode in messages:\n"
        "    sys.stderr.write(messages[mode])\n"
        "    sys.exit(1)\n"
        "if mode == 'large':\n"
        "    sys.stderr.write('x' * 1000000)\n"
        "    sys.exit(1)\n"
    )
    script.chmod(0o700)
    spawn = asyncio.create_subprocess_exec

    async def spawn_python(binary, *args, **kwargs):
        assert binary == str(script)
        return await spawn(sys._base_executable, str(script), *args, **kwargs)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn_python)
    return DockerClient(str(script), timeout=1)


def assert_process_stopped(client):
    identifier = int(Path(client.binary).with_suffix(".pid").read_text())
    with suppress(psutil.NoSuchProcess):
        psutil.Process(identifier).wait(timeout=1)


@pytest.mark.parametrize(
    "mode,code",
    [
        ("bad", "docker_image_pull_failed"),
        ("engine", "docker_unavailable"),
        ("large", "docker_image_pull_failed"),
    ],
)
async def test_pull_error_is_typed_and_does_not_publish_registry_stderr(cli, mode, code):
    with pytest.raises(DockerExecutionError) as failure:
        await cli.pull(mode, 2)
    assert failure.value.reason == code
    assert "secret-registry-data" not in str(failure.value)
    assert len(str(failure.value)) < 200


async def test_successful_pull_is_not_subject_to_ordinary_inspect_timeout(cli):
    cli.timeout = 0.000001
    await cli.pull("success", 2)


async def test_image_missing_and_operation_timeout_have_distinct_codes(cli):
    with pytest.raises(DockerExecutionError) as missing:
        await cli.run("image", "inspect", "missing")
    assert missing.value.reason == "docker_image_missing"
    with pytest.raises(DockerExecutionError) as timeout:
        await cli.run("slow")
    assert timeout.value.reason == "docker_operation_timed_out"
    assert_process_stopped(cli)


async def test_pull_timeout_kills_and_waits_for_cli(cli):
    with pytest.raises(DockerExecutionError) as failure:
        await cli.pull("slow", 1)
    assert failure.value.reason == "docker_image_pull_timeout"
    assert_process_stopped(cli)


async def test_pull_cancel_is_not_reported_as_timeout_or_engine_loss(cli):
    task = asyncio.create_task(cli.pull("slow", 5))
    async with asyncio.timeout(2):
        while not Path(cli.binary).with_suffix(".pid").exists():
            await asyncio.sleep(0.005)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert_process_stopped(cli)


async def test_missing_cli_is_typed_unavailable(tmp_path):
    with pytest.raises(DockerExecutionError) as failure:
        await DockerClient(str(tmp_path / "missing")).pull("image", 1)
    assert failure.value.reason == "docker_unavailable"
