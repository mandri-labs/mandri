import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


def default_agy_root() -> Path:
    return Path.home() / ".gemini"


def validate_agy_id(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", value):
        raise ValueError("Invalid Antigravity conversation or profile identifier")
    return value


def read_agy_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"Antigravity configuration must be an object: {path.name}")
    return value


def write_agy_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, ensure_ascii=False, indent=2)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def prepare_agy_profile(
    profiles_root: Path,
    profile_id: str,
    *,
    native: bool,
    canonical_root: Path | None = None,
) -> Path:
    canonical = (canonical_root or default_agy_root()).resolve()
    profile = (profiles_root.resolve() / validate_agy_id(profile_id)).resolve()
    if profile == canonical or canonical.is_relative_to(profile):
        raise ValueError("Antigravity profile must be separate from its native storage")
    profile.mkdir(parents=True, exist_ok=True, mode=0o700)
    store = profile / "antigravity-cli"
    store.mkdir(exist_ok=True)
    for name in ("conversations", "brain"):
        target = canonical / "antigravity-cli" / name
        target.mkdir(parents=True, exist_ok=True)
        _link_directory(store / name, target)
    settings_path = store / "settings.json"
    source = (
        settings_path if settings_path.exists() else canonical / "antigravity-cli/settings.json"
    )
    settings = read_agy_json(source)
    settings.update(enableTelemetry=False, useG1Credits=False)
    if native:
        settings.pop("modelProvider", None)
        if settings.get("model") in ("mandri", "mandri-route"):
            settings.pop("model", None)
        custom = settings.get("customModelsConfig", {}).get("customModels", {})
        custom.pop("mandri", None)
    else:
        settings["modelProvider"] = "gemini"
        custom = settings.setdefault("customModelsConfig", {}).setdefault("customModels", {})
        custom["mandri"] = {"modelName": "mandri-route"}
    write_agy_json(settings_path, settings)
    hooks_path = profile / "config/hooks.json"
    hooks = read_agy_json(hooks_path if hooks_path.exists() else canonical / "config/hooks.json")
    hooks.pop("mandri", None)
    write_agy_json(hooks_path, hooks)
    native_config = canonical / "config/config.json"
    profile_config = profile / "config/config.json"
    if native_config.is_file() and not profile_config.exists():
        shutil.copyfile(native_config, profile_config)
        profile_config.chmod(0o600)
    for project in (canonical / "config/projects").glob("*.json"):
        target = profile / "config/projects" / project.name
        if not target.exists():
            write_agy_json(target, read_agy_json(project))
    return profile


def merge_agy_hooks(profile_root: Path, hook_set: dict[str, Any]) -> None:
    path = profile_root / "config/hooks.json"
    hooks = read_agy_json(path)
    hooks["mandri"] = hook_set
    write_agy_json(path, hooks)


def _link_directory(link: Path, target: Path) -> None:
    if link.exists() or link.is_symlink():
        if link.resolve() != target.resolve():
            raise ValueError("Antigravity profile points to another native storage")
        return
    if sys.platform == "win32":
        environment = dict(os.environ, MANDRI_AGY_LINK=str(link), MANDRI_AGY_TARGET=str(target))
        subprocess.run(
            [
                "powershell.exe",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                "New-Item -ItemType Junction -Path $env:MANDRI_AGY_LINK "
                "-Target $env:MANDRI_AGY_TARGET -ErrorAction Stop | Out-Null",
            ],
            check=True,
            capture_output=True,
            env=environment,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
    else:
        link.symlink_to(target, target_is_directory=True)
