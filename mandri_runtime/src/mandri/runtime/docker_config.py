from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DockerConfig:
    image: str
    state_root: Path
    pull_policy: str = "never"
    pull_timeout_seconds: int = 600
    cpus: float = 4.0
    memory_mb: int = 8192
    pids_limit: int = 1024
    tmpfs_mb: int = 1024
    ingress_bind: str = "0.0.0.0"
    ingress_host: str = "host.docker.internal"
    docker_binary: str = "docker"
    network_exceptions: tuple[str, ...] = ()
    image_harnesses: tuple[str, ...] = ("codex", "claude", "opencode", "agy", "pi")


WORKSPACE_ROOT = "/workspace"
NATIVE_HOME = "/home/worker"
WORKER_PROGRAM = "/opt/mandri/worker.py"
MANAGED_LABEL = "io.mandri.managed"
OWNER_LABEL = "io.mandri.owner"
SESSION_LABEL = "io.mandri.session"
GENERATION_LABEL = "io.mandri.generation"
