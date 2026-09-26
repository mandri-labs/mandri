import json
import os
from pathlib import Path

from mandri.runtime.docker_config import NATIVE_HOME, WORKSPACE_ROOT
from mandri.runtime.docker_git import validate_git_directory
from mandri.runtime.errors.docker import DockerExecutionError


def workspace_root(path: str | Path) -> Path:
    try:
        root = Path(path).expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise DockerExecutionError(
            "workspace_unavailable", "Selected workspace is unavailable"
        ) from error
    if not root.is_dir() or root.parent == root or root == Path.home().resolve():
        raise DockerExecutionError("workspace_unavailable", "Select a project directory")
    if not os.access(root, os.R_OK | os.W_OK | os.X_OK):
        raise DockerExecutionError(
            "workspace_unavailable", "Workspace is not readable and writable"
        )
    validate_git_directory(root)
    return root


def native_state(root: Path, session_id: str, *, resume: bool) -> Path:
    if not session_id or any(
        char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
        for char in session_id
    ):
        raise DockerExecutionError("native_state_incompatible", "Invalid native state identity")
    parent = root.expanduser().resolve()
    path = parent / session_id
    if path.is_symlink() or (resume and not path.is_dir()):
        raise DockerExecutionError(
            "native_state_incompatible", "Native session state is unavailable"
        )
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.chmod(0o700)
    for name in (".codex", ".claude", ".config", ".cache", ".local/share", ".gemini", ".pi/agent"):
        directory = path / name
        if any(
            part.is_symlink()
            for part in (directory, *directory.parents)
            if part.is_relative_to(path)
        ):
            raise DockerExecutionError(
                "native_state_incompatible", "Native state directories cannot be symbolic links"
            )
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    return path


def translate_path(path: str, host_root: Path, state: Path) -> str:
    candidate = Path(path)
    for root, target in ((host_root, WORKSPACE_ROOT), (state, NATIVE_HOME)):
        if candidate.is_absolute() and candidate.is_relative_to(root):
            suffix = candidate.relative_to(root).as_posix()
            return target if suffix == "." else f"{target}/{suffix}"
    return path


def translate_value(value: str, gateway_port: int, ingress_url: str) -> str:
    return value.replace(f"http://127.0.0.1:{gateway_port}", ingress_url)


def execution_context(root: Path, state: Path, image: str) -> dict[str, str]:
    metadata = root.stat()
    return {
        "workspace_root": str(root),
        "container_root": WORKSPACE_ROOT,
        "native_state_root": str(state),
        "native_home": NATIVE_HOME,
        "image": image,
        "version": "1",
        "workspace_device": str(metadata.st_dev),
        "workspace_inode": str(metadata.st_ino),
    }


def write_context(state: Path, context: dict[str, str]) -> None:
    target = state.parent / f".{state.name}.context.json"
    temporary = target.with_suffix(".tmp")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(context, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(target)


def read_context(state: Path) -> dict[str, str]:
    target = state.parent / f".{state.name}.context.json"
    try:
        value = json.loads(target.read_text(encoding="utf-8"))
        if not isinstance(value, dict) or any(not isinstance(item, str) for item in value.values()):
            raise ValueError
    except (OSError, ValueError) as error:
        raise DockerExecutionError(
            "native_state_incompatible", "Native execution context is unavailable"
        ) from error
    return value
