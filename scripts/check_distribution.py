import argparse
import hashlib
import json
import tarfile
import tomllib
import zipfile
from email.parser import BytesParser
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
COMPONENTS = {
    "api", "cli", "config", "core", "daemon", "database", "gateway", "providers",
    "runtime", "sessions",
}
SOURCE_SUFFIXES = {".py", ".typed"}
SOURCE_FILES = {"pyproject.toml", "README.md", "LICENSE", "PKG-INFO", ".gitignore"}
RESOURCE_FILES = {
    "core/resources/codex_prompt_0_154.md",
    "core/resources/codex_prompt_0_154.NOTICE",
    "core/resources/pi_gateway.ts",
    "core/resources/pi_managed.ts",
    "core/resources/pi_permissions.ts",
    "core/resources/pi_usage.ts",
    "daemon/usage_price_catalog.NOTICE",
    "gateway/tokenizers/9b5ad71b2ce5302211f9c61530b329a4922fc6a4",
    "gateway/tokenizers/LICENSE",
    "runtime/security/worker-seccomp.json",
    "runtime/security/third-party-license.txt",
    "runtime/security/NOTICE.txt",
}
ENTRY_POINTS = {
    "mandri = mandri.cli.main:main",
    "mandri-daemon = mandri.daemon.main:main",
}


def archive_files(path: Path) -> dict[str, bytes]:
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as archive:
            return {item.filename: archive.read(item) for item in archive.infolist()}
    with tarfile.open(path, "r:gz") as archive:
        result = {}
        for item in archive.getmembers():
            if item.isdir():
                continue
            if not item.isfile():
                raise ValueError(f"Unsupported archive entry: {item.name}")
            stream = archive.extractfile(item)
            if stream is None:
                raise ValueError(f"Unreadable archive entry: {item.name}")
            result[PurePosixPath(item.name).relative_to(item.name.split("/")[0]).as_posix()] = (
                stream.read()
            )
        return result


def allowed_wheel_path(path: PurePosixPath) -> bool:
    if path.as_posix() in {f"mandri/{name}" for name in RESOURCE_FILES}:
        return True
    if path.parts[0].endswith(".dist-info"):
        return path.name in {
            "METADATA", "WHEEL", "RECORD", "entry_points.txt", "LICENSE",
        }
    return (
        len(path.parts) >= 3
        and path.parts[0] == "mandri"
        and path.parts[1] in COMPONENTS
        and path.suffix in SOURCE_SUFFIXES
    )


def allowed_source_path(path: PurePosixPath) -> bool:
    if path.as_posix() in {
        f"mandri_{name.split('/')[0]}/src/mandri/{name}" for name in RESOURCE_FILES
    }:
        return True
    if len(path.parts) == 1:
        return path.name in SOURCE_FILES
    return (
        len(path.parts) >= 5
        and path.parts[0] == f"mandri_{path.parts[3]}"
        and path.parts[1:3] == ("src", "mandri")
        and path.parts[3] in COMPONENTS
        and path.suffix in SOURCE_SUFFIXES
    )


def check(path: Path, expected_version: str) -> dict[str, object]:
    files = archive_files(path)
    wheel = path.suffix == ".whl"
    allowed = allowed_wheel_path if wheel else allowed_source_path
    for name in files:
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts or not allowed(relative):
            raise ValueError(f"Unexpected distribution file: {name}")
        if "tests" in relative.parts or relative.name.startswith("test_"):
            raise ValueError(f"Test file in distribution: {name}")
    for component in COMPONENTS:
        prefix = "" if wheel else f"mandri_{component}/src/"
        for name in ("__init__.py", "py.typed"):
            if f"{prefix}mandri/{component}/{name}" not in files:
                raise ValueError(f"Missing {component}/{name}")
    for name in RESOURCE_FILES:
        prefix = "" if wheel else f"mandri_{name.split('/')[0]}/src/"
        if f"{prefix}mandri/{name}" not in files:
            raise ValueError(f"Missing resource: {name}")
    metadata_name = next(name for name in files if name.endswith("/METADATA") or name == "PKG-INFO")
    metadata = BytesParser().parsebytes(files[metadata_name])
    if metadata["Name"] != "mandri" or metadata["Version"] != expected_version:
        raise ValueError("Distribution name or version differs from pyproject.toml")
    dependencies = metadata.get_all("Requires-Dist", [])
    if any(value.startswith("mandri-") or "file:" in value for value in dependencies):
        raise ValueError("Internal or local dependency in public metadata")
    if wheel:
        entry_file = next(name for name in files if name.endswith("/entry_points.txt"))
        lines = {line.strip() for line in files[entry_file].decode().splitlines()}
        if not lines >= ENTRY_POINTS:
            raise ValueError("Missing CLI entry point")
    return {
        "file": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "files": len(files),
        "version": expected_version,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify public Python distribution contents.")
    parser.add_argument("artifacts", nargs="+", type=Path)
    args = parser.parse_args()
    version = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"][
        "version"
    ]
    print(json.dumps([check(path, version) for path in args.artifacts], indent=2))


if __name__ == "__main__":
    main()
