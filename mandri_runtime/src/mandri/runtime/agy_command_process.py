from mandri.runtime.docker_client import DockerClient
from mandri.runtime.process import ManagedProcess, spawn

_CONTAINER_COMMAND = """import os, signal, subprocess, sys
process = subprocess.Popen(sys.argv[1:], stdin=subprocess.DEVNULL, start_new_session=True)
try:
    code = process.wait(timeout=20)
except subprocess.TimeoutExpired:
    code = 124
finally:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()
sys.exit(code)
"""


async def spawn_container_command(
    client: DockerClient, container_id: str, argv: list[str]
) -> ManagedProcess:
    return await spawn(
        [client.binary, "exec", container_id, "python3", "-c", _CONTAINER_COMMAND, *argv],
        env=client.environment(),
    )
