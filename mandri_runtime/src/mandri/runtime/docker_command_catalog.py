import base64
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any

import httpx
from mandri.core.ids import HarnessKind
from mandri.runtime.control.errors import ControlError
from mandri.runtime.docker_backend import DockerBackend
from mandri.runtime.docker_config import (
    MANAGED_LABEL,
    NATIVE_HOME,
    OWNER_LABEL,
    SESSION_LABEL,
    WORKER_PROGRAM,
    WORKSPACE_ROOT,
)
from mandri.runtime.docker_process import DockerProcess
from mandri.runtime.docker_workspace import workspace_root
from mandri.runtime.native_command_catalog import discover_commands
from mandri.runtime.process import ManagedProcess, spawn

_HTTP_PROGRAM = """import base64,json,os,urllib.request,urllib.error
request=json.loads(os.environ['MANDRI_CATALOG_HTTP'])
try:
    response=urllib.request.urlopen(urllib.request.Request(request['url'],headers=request['headers']),timeout=15)
except urllib.error.HTTPError as error:
    response=error
with response:
    body=response.read(4194305)
    if len(body)>4194304:
        raise RuntimeError('Command catalog exceeds output limit')
    print(json.dumps({'status':response.status,'body':base64.b64encode(body).decode(),'headers':dict(response.headers)}))
"""


_WORKER_INIT = """import json,os,sys
from pathlib import Path
home=Path.home()
for name in ('.codex','.claude','.config','.cache','.local/share','.gemini','.pi/agent'):
    (home/name).mkdir(parents=True,exist_ok=True)
if sys.argv[1]=='agy':
    root=home/'.gemini/antigravity-cli'
    root.mkdir(parents=True,exist_ok=True)
    (root/'settings.json').write_text(json.dumps({'enableTelemetry':False,'useG1Credits':False,'modelProvider':'gemini','customModelsConfig':{'customModels':{'mandri':{'modelName':'mandri-route'}}}}))
os.execvp(sys.argv[1],sys.argv[1:])
"""


class DockerCatalogTransport(httpx.AsyncBaseTransport):
    def __init__(self, backend: DockerBackend) -> None:
        self.backend = backend
        self.container: str | None = None

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if self.container is None or request.method != "GET" or request.url.host != "127.0.0.1":
            raise ControlError("Invalid Docker catalog request")
        payload = json.dumps({"url": str(request.url), "headers": dict(request.headers)})
        output = await self.backend.client.run(
            "exec",
            "--env",
            "MANDRI_CATALOG_HTTP",
            self.container,
            "python3",
            "-c",
            _HTTP_PROGRAM,
            env={"MANDRI_CATALOG_HTTP": payload},
        )
        result = json.loads(output)
        return httpx.Response(
            result["status"],
            content=base64.b64decode(result["body"]),
            headers=result.get("headers"),
        )


class DockerCommandDiscovery:
    def __init__(self, backend: DockerBackend) -> None:
        self.backend = backend

    async def discover(
        self, harness: HarnessKind, command: list[str], cwd: str | Path
    ) -> list[dict[str, Any]]:
        ready = await self.backend.readiness()
        if (
            harness.value not in ready.harnesses
            or harness.value not in self.backend.config.image_harnesses
        ):
            raise ControlError("The configured worker image does not include this harness")
        root = workspace_root(cwd)
        transport = DockerCatalogTransport(self.backend)
        config = self.backend.config
        if sys.platform == "win32":
            uid, gid = 1000, 1000
        else:
            uid, gid = os.getuid(), os.getgid()
        if uid == 0:
            uid, gid = 1000, 1000
        metadata = root.stat()
        identity = f"{metadata.st_dev}:{metadata.st_ino}"

        async def launch(
            argv: list[str], *, cwd: str | Path, env: dict[str, str]
        ) -> ManagedProcess:
            name = f"mandri-catalog-{uuid.uuid4().hex}"
            worker_env = {
                "HOME": NATIVE_HOME,
                "USER": "worker",
                "LOGNAME": "worker",
                "PATH": "/usr/local/bin:/usr/bin:/bin",
                "TMPDIR": "/tmp",
                "LANG": "C.UTF-8",
                "XDG_CONFIG_HOME": f"{NATIVE_HOME}/.config",
                "XDG_DATA_HOME": f"{NATIVE_HOME}/.local/share",
                "XDG_CACHE_HOME": f"{NATIVE_HOME}/.cache",
                "CODEX_HOME": f"{NATIVE_HOME}/.codex",
                "CLAUDE_CONFIG_DIR": f"{NATIVE_HOME}/.claude",
                "AGY_CLI_DISABLE_AUTO_UPDATE": "true",
                "CI": "1",
                "MANDRI_WORKSPACE_IDENTITY": identity,
                **env,
            }
            if harness is HarnessKind.AGY:
                worker_env.update(
                    GEMINI_API_KEY="command-catalog",
                    GOOGLE_GEMINI_BASE_URL="http://127.0.0.1:1",
                )
                argv = [*argv, "--gemini_dir", f"{NATIVE_HOME}/.gemini", "--model", "mandri"]
            attached = None
            try:
                await self.backend.client.run(
                    "create",
                    "--name",
                    name,
                    "--interactive",
                    "--network",
                    "none",
                    "--label",
                    f"{MANAGED_LABEL}=true",
                    "--label",
                    f"{OWNER_LABEL}={self.backend.owner}",
                    "--label",
                    f"{SESSION_LABEL}={name}",
                    "--read-only",
                    "--cap-drop",
                    "ALL",
                    "--security-opt",
                    "no-new-privileges",
                    "--user",
                    f"{uid}:{gid}",
                    "--cpus",
                    str(config.cpus),
                    "--memory",
                    f"{config.memory_mb}m",
                    "--pids-limit",
                    str(config.pids_limit),
                    "--tmpfs",
                    f"/tmp:rw,nosuid,nodev,size={config.tmpfs_mb}m,uid={uid},gid={gid}",
                    "--tmpfs",
                    f"{NATIVE_HOME}:rw,nosuid,nodev,size={config.tmpfs_mb}m,uid={uid},gid={gid}",
                    "--workdir",
                    WORKSPACE_ROOT,
                    "--mount",
                    f"type=bind,src={root},dst={WORKSPACE_ROOT},readonly,bind-propagation=rprivate",
                    *[part for key in worker_env for part in ("--env", key)],
                    "--entrypoint",
                    "python3",
                    ready.image_id,
                    WORKER_PROGRAM,
                    "run",
                    *(["--watch-stdin"] if harness is HarnessKind.OPENCODE else []),
                    "python3",
                    "-c",
                    _WORKER_INIT,
                    *argv,
                    env=worker_env,
                )
                transport.container = name
                attached = await spawn(
                    [self.backend.client.binary, "start", "--attach", "--interactive", name],
                    env=self.backend.client.environment(),
                )
                await self.backend._await_workspace(attached, identity)
            except BaseException:
                try:
                    if attached is not None:
                        await attached.stop(grace=1)
                finally:
                    await self.backend.client.remove([name])
                raise

            async def cleanup() -> None:
                await self.backend.client.remove([name])

            return DockerProcess(attached, self.backend.client, name, cleanup)

        return await discover_commands(
            harness,
            [harness.value, *command[1:]],
            WORKSPACE_ROOT,
            {},
            launch,
            http_transport=transport,
        )
