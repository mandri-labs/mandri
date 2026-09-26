import os
import re
import tomllib
import urllib.error
import urllib.request
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    version = tomllib.loads((root / "pyproject.toml").read_text())["project"]["version"]
    if not re.fullmatch(r"0\.\d+\.\d+b0", version):
        raise ValueError("Automatic publication is restricted to 0.x beta builds")
    if os.environ.get("GITHUB_REF") != f"refs/tags/v{version}":
        raise ValueError(f"Publication requires the tag v{version}")
    try:
        with urllib.request.urlopen(f"https://pypi.org/pypi/mandri/{version}/json", timeout=30):
            raise ValueError(f"Version {version} already exists on PyPI; increment the version")
    except urllib.error.HTTPError as error:
        if error.code != 404:
            raise
    print(f"Version {version} is available")


if __name__ == "__main__":
    main()
