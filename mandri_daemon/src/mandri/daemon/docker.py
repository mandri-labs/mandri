from pathlib import Path

from mandri.core.types.config import DockerSettings
from mandri.runtime.docker_config import DockerConfig


def docker_config(settings: DockerSettings, base_dir: Path) -> DockerConfig | None:
    if settings.image is None:
        return None
    return DockerConfig(
        image=settings.image,
        state_root=Path(settings.state_root) if settings.state_root else base_dir / "worker-state",
        pull_policy=settings.pull_policy,
        pull_timeout_seconds=settings.pull_timeout_seconds,
        cpus=settings.cpus,
        memory_mb=settings.memory_mb,
        pids_limit=settings.pids_limit,
        tmpfs_mb=settings.tmpfs_mb,
        ingress_bind=settings.ingress_bind,
        ingress_host=settings.ingress_host,
        docker_binary=settings.docker_binary,
        network_exceptions=settings.network_exceptions,
    )
