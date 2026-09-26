import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    kind = sys.argv[1]
    version = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
    source = (
        f"mandri=={version}"
        if kind == "pypi"
        else str(next((ROOT / "dist").glob("*.whl" if kind == "wheel" else "*.tar.gz")))
    )
    with tempfile.TemporaryDirectory(prefix="mandri-install-") as directory:
        target = Path(directory)
        python = target / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
        subprocess.run(["uv", "venv", "--python", "3.13", directory], check=True)
        subprocess.run(
            [
                "uv",
                "--no-config",
                "pip",
                "install",
                "--python",
                str(python),
                "--index-url",
                "https://pypi.org/simple",
                source,
            ],
            check=True,
        )
        subprocess.run(
            [
                str(python),
                "-I",
                str(ROOT / "scripts/check_installed.py"),
                version,
            ],
            cwd=target,
            check=True,
        )


if __name__ == "__main__":
    main()
