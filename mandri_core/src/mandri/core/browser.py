"""Cross-platform best-effort system browser launcher."""

import os
import shutil
import subprocess
import sys

LAUNCH_TIMEOUT_SECONDS = 15.0
_POSIX_OPENERS: tuple[tuple[str, ...], ...] = (
    ("xdg-open",),
    ("gio", "open"),
    ("gnome-open",),
    ("kde-open",),
    ("wslview",),
)


def open_browser(url: str) -> bool:
    """Open a URL in the system browser. Never writes to stdout or stderr."""
    if sys.platform == "win32":
        return _open_windows(url)
    if sys.platform == "darwin":
        return _run(("open", url))
    return _open_posix(url)


def _open_windows(url: str) -> bool:
    opener = getattr(os, "startfile", None)
    if opener is not None:
        try:
            opener(url)
        except OSError:
            return False
        return True
    return _run(("cmd", "/c", "start", "", url))


def _open_posix(url: str) -> bool:
    if not (_graphical_session() or _wsl()):
        return False
    return any(_run((*opener, url)) for opener in _available_openers())


def _available_openers() -> tuple[tuple[str, ...], ...]:
    return tuple(opener for opener in _POSIX_OPENERS if shutil.which(opener[0]))


def _graphical_session() -> bool:
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def _wsl() -> bool:
    return "microsoft" in os.uname().release.lower()


def _run(command: tuple[str, ...]) -> bool:
    try:
        result = subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            timeout=LAUNCH_TIMEOUT_SECONDS,
            check=False,
        )
    except OSError:
        return False
    except subprocess.TimeoutExpired:
        return True
    return result.returncode == 0
