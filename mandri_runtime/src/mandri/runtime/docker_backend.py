import asyncio
import base64
import hashlib
import ipaddress
import json
import os
import re
import sys
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from mandri.runtime.codex_catalog import materialize_catalog
from mandri.runtime.docker_client import DockerClient
from mandri.runtime.docker_config import (
    GENERATION_LABEL,
    MANAGED_LABEL,
    NATIVE_HOME,
    OWNER_LABEL,
    SESSION_LABEL,
    WORKER_PROGRAM,
    WORKSPACE_ROOT,
    DockerConfig,
)
from mandri.runtime.docker_image import prepare_image
from mandri.runtime.docker_ingress import WorkerIngress
from mandri.runtime.docker_network import local_networks, network_exception
from mandri.runtime.docker_process import DockerProcess
from mandri.runtime.docker_state import StateLease
from mandri.runtime.docker_versions import DockerVersions
from mandri.runtime.docker_workspace import (
    execution_context,
    native_state,
    read_context,
    workspace_root,
    write_context,
)
from mandri.runtime.errors.docker import DockerExecutionError
from mandri.runtime.pi_session_paths import docker_resume_args
from mandri.runtime.process import ManagedProcess, spawn


@dataclass(frozen=True)
class DockerReadiness:
    image_id: str
    engine_id: str
    architecture: str
    harnesses: tuple[str, ...]


class DockerBackend:
    def __init__(self, config: DockerConfig) -> None:
        self.config = config
        self.client = DockerClient(config.docker_binary)
        self.owner = hashlib.sha256(str(config.state_root.resolve()).encode()).hexdigest()[:24]
        self.versions = DockerVersions(self.client, self.owner)
        self._image_lock = asyncio.Lock()

    async def prepare_image(self) -> DockerReadiness:
        async with self._image_lock:
            return await prepare_image(self.client, self.config, self.readiness)

    async def readiness(self) -> DockerReadiness:
        config = self.config
        if sys.platform != "linux":
            raise DockerExecutionError(
                "docker_platform_unqualified", "Only native Linux Docker execution is qualified"
            )
        if min(config.cpus, config.memory_mb, config.pids_limit, config.tmpfs_mb) <= 0:
            raise DockerExecutionError(
                "docker_resource_limit_invalid", "Docker limits must be positive"
            )
        for exception in config.network_exceptions:
            network_exception(exception)
        context = await self.client.json("context", "inspect")
        endpoint = context[0]["Endpoints"]["docker"]["Host"]
        if not endpoint.startswith("unix://"):
            raise DockerExecutionError(
                "docker_unavailable", "Only local Linux Docker engines are qualified"
            )
        info = await self.client.json("info", "--format", "{{json .}}")
        if info.get("OSType") != "linux":
            raise DockerExecutionError("docker_unavailable", "A Linux container engine is required")
        if (
            "desktop" in str(info.get("OperatingSystem", "")).lower()
            or info.get("Name") == "docker-desktop"
        ):
            raise DockerExecutionError(
                "docker_platform_unqualified", "Docker Desktop execution is not qualified"
            )
        if info.get("Architecture") not in {"x86_64", "amd64"}:
            raise DockerExecutionError(
                "docker_platform_unqualified", "The Docker engine architecture is not qualified"
            )
        if any("rootless" in str(item) for item in info.get("SecurityOptions", [])):
            raise DockerExecutionError(
                "docker_unavailable", "Rootless network isolation is not qualified"
            )
        if config.cpus > int(info["NCPU"]) or config.memory_mb * 1024 * 1024 > int(
            info["MemTotal"]
        ):
            raise DockerExecutionError(
                "docker_resource_limit_invalid", "Configured Docker limits exceed engine capacity"
            )
        if not config.image:
            raise DockerExecutionError(
                "docker_image_incompatible", "A worker image must be configured"
            )
        image = (await self.client.json("image", "inspect", config.image))[0]
        labels = image.get("Config", {}).get("Labels") or {}
        if labels.get("io.mandri.worker.version") != "1" or image.get("Os") != "linux":
            raise DockerExecutionError(
                "docker_image_incompatible", "The image does not implement the worker contract"
            )
        if image.get("Architecture") != "amd64":
            raise DockerExecutionError(
                "docker_platform_unqualified", "The worker image architecture is not qualified"
            )
        if labels.get("io.mandri.worker.workspace-identity") != "1":
            raise DockerExecutionError(
                "docker_image_incompatible", "The image cannot verify the selected workspace mount"
            )
        harnesses = tuple(labels.get("io.mandri.worker.harnesses", "").split(","))
        return DockerReadiness(
            image["Id"],
            str(info["ID"]),
            str(image["Architecture"]),
            harnesses,
        )

    async def reconcile(self, active_session_ids: frozenset[str] = frozenset()) -> list[str]:
        ids = (
            await self.client.run("ps", "-aq", "--filter", f"label={OWNER_LABEL}={self.owner}")
        ).split()
        removed: list[str] = []
        for container in ids:
            details = (await self.client.json("inspect", container))[0]
            labels = details.get("Config", {}).get("Labels") or {}
            session_id = labels.get(SESSION_LABEL)
            if not session_id or session_id in active_session_ids:
                continue
            lease_path = self.config.state_root / f".{session_id}.lock"
            try:
                lease = StateLease(lease_path)
            except DockerExecutionError:
                continue
            try:
                await self.client.remove([container])
                removed.append(container)
            finally:
                lease.close()
        networks = (
            await self.client.run(
                "network", "ls", "-q", "--filter", f"label={OWNER_LABEL}={self.owner}"
            )
        ).split()
        for network in networks:
            details = (await self.client.json("network", "inspect", network))[0]
            if not details.get("Containers"):
                await self.client.run("network", "rm", network, check=False)
        return removed

    def context(
        self, session_id: str, cwd: str | Path, *, resume: bool, image_id: str | None = None
    ) -> dict[str, str]:
        root = workspace_root(cwd)
        state_root = self.config.state_root.expanduser().resolve()
        if state_root.is_relative_to(root) or root.is_relative_to(state_root):
            raise DockerExecutionError(
                "workspace_unavailable", "Workspace and native state storage must not overlap"
            )
        state = native_state(self.config.state_root, session_id, resume=resume)
        if state.is_relative_to(root) or root.is_relative_to(state):
            raise DockerExecutionError(
                "workspace_unavailable", "Workspace and native state must not overlap"
            )
        context = execution_context(root, state, image_id or self.config.image)
        if resume:
            stored = read_context(state)
            if stored.get("version") != "1" or any(
                stored.get(key) != context[key]
                for key in ("workspace_root", "workspace_device", "workspace_inode")
            ):
                raise DockerExecutionError(
                    "native_state_incompatible", "Native workspace identity changed"
                )
            if image_id is not None and stored.get("image") != image_id:
                raise DockerExecutionError(
                    "native_state_incompatible",
                    "The native state requires its original worker image",
                )
        return context

    async def spawn(
        self,
        session_id: str,
        harness: str,
        argv: list[str],
        cwd: str | Path,
        env: Mapping[str, str],
        ingress: WorkerIngress,
        *,
        listen_port: int | None = None,
        resume: bool = False,
    ) -> DockerProcess:
        ready = await self.readiness()
        if harness not in ready.harnesses or harness not in self.config.image_harnesses:
            raise DockerExecutionError(
                "docker_image_incompatible", "The image does not contain the selected harness"
            )
        context = self.context(session_id, cwd, resume=resume, image_id=ready.image_id)
        state = Path(context["native_state_root"])
        argv, env = list(argv), dict(env)
        lease = StateLease(state.parent / f".{session_id}.lock")
        generation = uuid.uuid4().hex
        name = f"mandri-{generation}"
        network = f"{name}-net"
        guard = f"{name}-guard"
        created: list[str] = []

        async def cleanup() -> None:
            try:
                await self.client.remove(list(reversed(created)), network)
            finally:
                try:
                    await ingress.aclose()
                finally:
                    lease.close()

        labels = [
            "--label",
            f"{MANAGED_LABEL}=true",
            "--label",
            f"{OWNER_LABEL}={self.owner}",
            "--label",
            f"{SESSION_LABEL}={session_id}",
            "--label",
            f"{GENERATION_LABEL}={generation}",
        ]
        try:
            existing = await self.client.run(
                "ps",
                "-q",
                "--filter",
                f"label={OWNER_LABEL}={self.owner}",
                "--filter",
                f"label={SESSION_LABEL}={session_id}",
            )
            if existing:
                raise DockerExecutionError(
                    "native_state_in_use", "A previous worker still owns this native state"
                )
            if harness == "pi":
                argv = [
                    "/opt/mandri/pi_gateway.ts"
                    if argument.endswith("/resources/pi_gateway.ts")
                    or argument.endswith("\\resources\\pi_gateway.ts")
                    else "/opt/mandri/pi_managed.ts"
                    if argument.endswith("/resources/pi_managed.ts")
                    or argument.endswith("\\resources\\pi_managed.ts")
                    else argument
                    for argument in argv
                ]
                if resume:
                    argv = docker_resume_args(argv, context)
            materialize_catalog(argv, env, state, NATIVE_HOME)
            await self.client.run("network", "create", "--driver", "bridge", *labels, network)
            if ingress.port is None:
                await ingress.start(self.config.ingress_bind)
            guard_policy = base64.urlsafe_b64encode(
                json.dumps(
                    {
                        "ingress_host": "127.0.0.1"
                        if ingress.socket_path
                        else self.config.ingress_host,
                        "ingress_socket": "/run/mandri-gateway.sock"
                        if ingress.socket_path
                        else None,
                        "ingress_port": ingress.port,
                        "denied": local_networks(),
                        "exceptions": [
                            network_exception(item) for item in self.config.network_exceptions
                        ],
                    }
                ).encode()
            ).decode()
            socket_args: list[str] = []
            if ingress.socket_path:
                if sys.platform == "win32":
                    raise DockerExecutionError(
                        "docker_unavailable", "Unix worker sockets are unavailable on Windows"
                    )
                else:
                    socket_args = [
                        "--group-add",
                        str(os.getgid()),
                        "--mount",
                        f"type=bind,src={ingress.socket_path},dst=/run/mandri-gateway.sock,readonly",
                    ]
            port_args = (
                ["--publish", f"127.0.0.1:{listen_port}:{listen_port}"] if listen_port else []
            )
            await self.client.run(
                "create",
                "--name",
                guard,
                *labels,
                "--network",
                network,
                "--add-host",
                "host.docker.internal:host-gateway",
                *port_args,
                *socket_args,
                "--read-only",
                "--cap-drop",
                "ALL",
                "--cap-add",
                "NET_ADMIN",
                "--security-opt",
                "no-new-privileges",
                "--user",
                "0:0",
                "--memory",
                "128m",
                "--pids-limit",
                "32",
                "--cpus",
                "0.25",
                "--tmpfs",
                "/run:rw,nosuid,nodev,noexec,size=1m",
                "--entrypoint",
                "python3",
                ready.image_id,
                WORKER_PROGRAM,
                "guard",
                guard_policy,
            )
            created.append(guard)
            await self.client.run("start", guard)
            await self._await_guard(guard)
            if ingress.socket_path:
                try:
                    await self.client.run(
                        "exec",
                        guard,
                        "python3",
                        "-c",
                        "import http.client,sys; "
                        "c=http.client.HTTPConnection('127.0.0.1',int(sys.argv[1]),timeout=3); "
                        "c.request('GET','/'); r=c.getresponse(); "
                        "assert r.status == 403, r.status; c.close()",
                        str(ingress.port),
                    )
                except DockerExecutionError:
                    raise DockerExecutionError(
                        "docker_network_unavailable", "The worker cannot reach its local gateway"
                    ) from None
            try:
                ingress_address = (
                    "127.0.0.1"
                    if ingress.socket_path
                    else str(ipaddress.ip_address(self.config.ingress_host))
                )
            except ValueError:
                ingress_address = (
                    await self.client.run(
                        "exec",
                        guard,
                        "python3",
                        "-c",
                        "import socket,sys; print(socket.gethostbyname(sys.argv[1]))",
                        self.config.ingress_host,
                    )
                ).strip()
                try:
                    ipaddress.IPv4Address(ingress_address)
                except ValueError:
                    raise DockerExecutionError(
                        "docker_network_unavailable", "Worker gateway address is unavailable"
                    ) from None
            configured_origin = f"http://{self.config.ingress_host}:{ingress.port}"
            resolved_origin = f"http://{ingress_address}:{ingress.port}"
            argv = [argument.replace(configured_origin, resolved_origin) for argument in argv]
            env = {
                key: value.replace(configured_origin, resolved_origin) for key, value in env.items()
            }
            worker_env = {
                "HOME": NATIVE_HOME,
                "USER": "worker",
                "LOGNAME": "worker",
                "PATH": "/usr/local/bin:/usr/bin:/bin",
                "TMPDIR": "/tmp",
                "XDG_CONFIG_HOME": f"{NATIVE_HOME}/.config",
                "XDG_DATA_HOME": f"{NATIVE_HOME}/.local/share",
                "XDG_CACHE_HOME": f"{NATIVE_HOME}/.cache",
                "CODEX_HOME": f"{NATIVE_HOME}/.codex",
                "CLAUDE_CONFIG_DIR": f"{NATIVE_HOME}/.claude",
                "CI": "1",
                "LANG": "C.UTF-8",
                **env,
                "MANDRI_WORKSPACE_IDENTITY": (
                    f"{context['workspace_device']}:{context['workspace_inode']}"
                ),
            }
            if any(not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key) for key in worker_env):
                raise DockerExecutionError(
                    "docker_image_incompatible", "Invalid worker environment key"
                )
            env_args = [argument for key in worker_env for argument in ("--env", key)]
            if sys.platform == "win32":
                uid, gid = 1000, 1000
            else:
                uid, gid = os.getuid(), os.getgid()
            if sys.platform != "win32" and uid == 0:
                uid, gid = 1000, 1000
                os.chown(state, uid, gid)
                for directory in state.rglob("*"):
                    if not directory.is_symlink():
                        os.chown(directory, uid, gid)
            watch_args = ["--watch-stdin"] if harness == "opencode" else []
            namespace_args = (
                ["--security-opt", "seccomp=unconfined", "--security-opt", "apparmor=unconfined"]
                if harness == "codex"
                else []
            )
            await self.client.run(
                "create",
                "--name",
                name,
                *labels,
                "--interactive",
                "--network",
                f"container:{guard}",
                "--user",
                f"{uid}:{gid}",
                "--read-only",
                "--cap-drop",
                "ALL",
                "--security-opt",
                "no-new-privileges",
                *namespace_args,
                "--cpus",
                str(self.config.cpus),
                "--memory",
                f"{self.config.memory_mb}m",
                "--memory-swap",
                f"{self.config.memory_mb}m",
                "--pids-limit",
                str(self.config.pids_limit),
                "--tmpfs",
                f"/tmp:rw,nosuid,nodev,size={self.config.tmpfs_mb}m,uid={uid},gid={gid}",
                "--workdir",
                WORKSPACE_ROOT,
                "--mount",
                f"type=bind,src={context['workspace_root']},dst={WORKSPACE_ROOT},bind-propagation=rprivate",
                "--mount",
                f"type=bind,src={state},dst={NATIVE_HOME},bind-propagation=rprivate",
                *env_args,
                "--entrypoint",
                "python3",
                ready.image_id,
                WORKER_PROGRAM,
                "run",
                *watch_args,
                *argv,
                env=worker_env,
            )
            created.append(name)
            details = (await self.client.json("inspect", name))[0]
            context.update(
                image=ready.image_id, generation=generation, container_id=str(details["Id"])
            )
            write_context(state, context)
            attached = await spawn(
                [self.client.binary, "start", "--attach", "--interactive", name],
                env=self.client.environment(),
            )
            try:
                await self._await_workspace(attached, worker_env["MANDRI_WORKSPACE_IDENTITY"])
            except BaseException:
                await attached.stop(2.0)
                raise
            process = DockerProcess(attached, self.client, str(details["Id"]), cleanup)
            process.execution_context = context
            return process
        except BaseException:
            await cleanup()
            raise

    async def _await_workspace(self, attached: ManagedProcess, identity: str) -> None:
        stdout = attached.process.stdout
        if stdout is None:
            raise DockerExecutionError(
                "workspace_identity_changed", "The worker cannot verify its workspace mount"
            )
        try:
            line = await asyncio.wait_for(stdout.readline(), 15.0)
        except (TimeoutError, OSError, ValueError):
            raise DockerExecutionError(
                "workspace_identity_changed", "The worker did not verify its workspace mount"
            ) from None
        if line != f"MANDRI_WORKSPACE_READY {identity}\n".encode():
            raise DockerExecutionError(
                "workspace_identity_changed", "The selected workspace changed before execution"
            )

    async def _await_guard(self, guard: str) -> None:
        deadline = asyncio.get_running_loop().time() + 15
        logs = ""
        while asyncio.get_running_loop().time() < deadline:
            logs = await self.client.run("logs", guard, include_stderr=True)
            if "MANDRI_NETWORK_READY" in logs:
                return
            state = await self.client.json("inspect", "--format", "{{json .State}}", guard)
            if not state.get("Running"):
                break
            await asyncio.sleep(0.1)
        raise DockerExecutionError(
            "docker_network_policy_unavailable",
            f"The worker network guard failed to become ready: {logs}",
        )
