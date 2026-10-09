import os
import subprocess
import tempfile
from pathlib import Path

from mandri.core.types.execution import ProtectionError

KEYS = {"user.name": "author", "user.email": "email", "remote.origin.url": "origin"}


def git_values(root: Path) -> dict[str, str]:
    command = [
        "git",
        "--no-pager",
        "config",
        "--null",
        "--includes",
        "--get-regexp",
        r"^(user\.(name|email)|remote\.origin\.url)$",
    ]
    try:
        with tempfile.TemporaryFile() as output:
            result = subprocess.run(
                command,
                cwd=root,
                stdin=subprocess.DEVNULL,
                stdout=output,
                stderr=subprocess.DEVNULL,
                timeout=5,
                check=False,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            if result.returncode not in {0, 1}:
                raise ValueError
            output.seek(0)
            raw = output.read(65_537)
        if len(raw) > 65_536 or (raw and not raw.endswith(b"\0")):
            raise ValueError
        selected = {}
        for record in raw.decode("utf-8").split("\0"):
            if not record:
                continue
            key, separator, value = record.partition("\n")
            if not separator or key not in KEYS:
                raise ValueError
            selected[KEYS[key]] = value
        return {key: value for key, value in selected.items() if value}
    except (OSError, ValueError, subprocess.SubprocessError):
        raise ProtectionError(
            "privacy_context_unavailable", "Selected repository metadata is unavailable"
        ) from None
