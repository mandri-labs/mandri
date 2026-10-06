import hashlib
import tempfile
from pathlib import Path

from mandri.core import pi
from mandri.runtime import agy_hook
from mandri.runtime.docker_client import DockerClient

MANAGED_IMAGE = "mandri-worker:latest"
SOURCE_LABEL = "io.mandri.worker.source"


def build_sources() -> dict[str, bytes]:
    worker = Path(__file__).with_name("worker")
    core = Path(pi.__file__).with_name("resources")
    paths = {
        "Dockerfile": worker / "Dockerfile",
        "docker/worker/worker.py": worker / "worker.py",
        "mandri_runtime/src/mandri/runtime/agy_hook.py": Path(agy_hook.__file__),
        **{
            f"mandri_core/src/mandri/core/resources/{name}": core / name
            for name in ("pi_gateway.ts", "pi_managed.ts", "pi_permissions.ts")
        },
    }
    return {name: path.read_bytes() for name, path in paths.items()}


def source_digest(sources: dict[str, bytes]) -> str:
    digest = hashlib.sha256()
    for name, data in sorted(sources.items()):
        digest.update(name.encode() + b"\0" + hashlib.sha256(data).digest())
    return digest.hexdigest()


async def prepare_worker(client: DockerClient, state_root: Path, timeout: float) -> str:
    sources = build_sources()
    fingerprint = source_digest(sources)
    reference = f"mandri-worker:source-{fingerprint}"
    from_image = await client.run("image", "ls", "-q", "--no-trunc", reference)
    if from_image:
        image = (await client.json("image", "inspect", reference))[0]
        labels = image.get("Config", {}).get("Labels") or {}
        if labels.get(SOURCE_LABEL) == fingerprint:
            return reference
    state_root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="worker-build-", dir=state_root.parent) as directory:
        context = Path(directory)
        for name, data in sources.items():
            target = context / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        await client.build(context, reference, f"{SOURCE_LABEL}={fingerprint}", timeout)
    return reference
