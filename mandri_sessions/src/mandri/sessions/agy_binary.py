import os
import shutil
from pathlib import Path


def find_agy_binary() -> Path | None:
    found = shutil.which("agy")
    if found:
        return Path(found)
    home = Path.home()
    candidates = [home / ".local/bin/agy", home / ".agy/bin/agy"]
    local = os.environ.get("LOCALAPPDATA")
    if local:
        candidates.insert(0, Path(local) / "agy/bin/agy.exe")
    return next((candidate for candidate in candidates if candidate.is_file()), None)
