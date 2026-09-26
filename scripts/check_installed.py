import argparse
import importlib
import importlib.metadata
import socket
import subprocess
import sys
from pathlib import Path


def reject_network(*args: object, **kwargs: object) -> None:
    raise RuntimeError("Network access during package import")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("version")
    args = parser.parse_args()
    socket.socket.connect = reject_network
    socket.create_connection = reject_network
    assert importlib.metadata.version("mandri") == args.version
    for component in (
        "api.app",
        "cli.main",
        "config",
        "core",
        "daemon.main",
        "database",
        "gateway",
        "providers",
        "runtime",
        "sessions",
    ):
        importlib.import_module(f"mandri.{component}")
    for name in ("mandri", "mandri-daemon"):
        executable = Path(sys.executable).parent / (
            f"{name}.exe" if sys.platform == "win32" else name
        )
        for option in ("--help", "--version"):
            result = subprocess.run(
                [str(executable), option],
                check=True,
                capture_output=True,
                text=True,
                timeout=60,
            )
            if option == "--version":
                assert args.version in result.stdout
    print(f"Installed Mandri {args.version}: imports and entry points passed")


if __name__ == "__main__":
    main()
