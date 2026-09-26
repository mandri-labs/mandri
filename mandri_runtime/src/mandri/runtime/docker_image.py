from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Literal

from mandri.core.image_reference import valid_image_reference
from mandri.runtime.docker_client import DockerClient
from mandri.runtime.docker_config import DockerConfig
from mandri.runtime.errors.docker import DockerExecutionError


@dataclass(frozen=True)
class DockerImageOptions:
    configured_image: str | None = None
    pull_policy: Literal["never", "if-missing"] = "never"
    can_prepare: bool = False


def image_options(config: DockerConfig | None) -> DockerImageOptions:
    if config is None:
        return DockerImageOptions()
    return DockerImageOptions(
        configured_image=config.image if valid_image_reference(config.image) else None,
        pull_policy="if-missing" if config.pull_policy == "if-missing" else "never",
        can_prepare=can_acquire(config),
    )


def can_acquire(config: DockerConfig) -> bool:
    return config.pull_policy == "if-missing" and valid_image_reference(
        config.image, digest_required=True
    )


async def prepare_image[T](
    client: DockerClient, config: DockerConfig, readiness: Callable[[], Awaitable[T]]
) -> T:
    if config.pull_policy not in {"never", "if-missing"}:
        raise DockerExecutionError(
            "docker_image_reference_invalid", "Unknown worker image pull policy"
        )
    try:
        return await readiness()
    except DockerExecutionError as error:
        if error.reason != "docker_image_missing" or config.pull_policy == "never":
            raise
    if not can_acquire(config):
        raise DockerExecutionError(
            "docker_image_reference_invalid",
            "Worker image acquisition requires a repository digest",
        )
    await client.pull(config.image, config.pull_timeout_seconds)
    try:
        return await readiness()
    except DockerExecutionError as error:
        if error.reason == "docker_image_missing":
            raise DockerExecutionError(
                "docker_image_pull_failed", "The downloaded worker image is unavailable"
            ) from None
        raise
