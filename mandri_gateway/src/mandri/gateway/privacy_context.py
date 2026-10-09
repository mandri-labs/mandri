import getpass
import os
import socket
import stat
from pathlib import Path

from mandri.core.types.execution import ProtectionError
from mandri.gateway.privacy_git import git_values
from mandri.gateway.surrogate import SurrogateEngine
from mandri.gateway.surrogate.detectors import GIT_SCP, URL

_MAX_CONFIG_BYTES = 65_536
_WINDOWS = os.name == "nt"


def unavailable() -> ProtectionError:
    return ProtectionError(
        "privacy_context_unavailable", "Selected repository metadata is unavailable"
    )


def metadata_read_flags() -> int:
    if _WINDOWS:
        return os.O_RDONLY | int(getattr(os, "O_BINARY", 0)) | int(getattr(os, "O_NOINHERIT", 0))
    flags = [getattr(os, name, None) for name in ("O_NOFOLLOW", "O_NONBLOCK", "O_CLOEXEC")]
    if any(type(flag) is not int or flag == 0 for flag in flags):
        raise ProtectionError(
            "privacy_platform_unsupported",
            "Secure local metadata reads are unavailable on this platform",
        )
    result = os.O_RDONLY
    for flag in flags:
        if isinstance(flag, int):
            result |= flag
    return result


def local_text(path: Path, *, optional: bool = False) -> str | None:
    flags = metadata_read_flags()
    try:
        expected = os.lstat(path) if _WINDOWS else None
        if expected is not None and not stat.S_ISREG(expected.st_mode):
            raise unavailable()
        descriptor = os.open(path, flags)
    except FileNotFoundError:
        if optional:
            return None
        raise unavailable() from None
    except OSError:
        raise unavailable() from None
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_size > _MAX_CONFIG_BYTES
            or (
                expected is not None
                and (metadata.st_dev, metadata.st_ino) != (expected.st_dev, expected.st_ino)
            )
        ):
            raise unavailable()
        with os.fdopen(descriptor, "rb", closefd=False) as file:
            content = file.read(_MAX_CONFIG_BYTES + 1)
        if len(content) > _MAX_CONFIG_BYTES:
            raise unavailable()
        return content.decode("utf-8")
    except (OSError, UnicodeError):
        raise unavailable() from None
    finally:
        os.close(descriptor)


def seed_context(engine: SurrogateEngine, root: Path) -> None:
    username = getpass.getuser()
    hostname = socket.gethostname()
    if username:
        engine.register(username, "username", "process_username")
    if hostname:
        engine.register(hostname, "hostname", "process_hostname")
    for name in (
        "GIT_AUTHOR_NAME",
        "GIT_COMMITTER_NAME",
        "GIT_AUTHOR_EMAIL",
        "GIT_COMMITTER_EMAIL",
    ):
        if value := os.environ.get(name):
            engine.register(
                value, "email" if name.endswith("EMAIL") else "identity", "git_identity"
            )
    values = git_values(root)
    if values.get("author"):
        engine.register(values["author"], "identity", "git_author")
    if values.get("email"):
        engine.register(values["email"], "email", "git_email")
    if origin := values.get("origin"):
        if URL.fullmatch(origin) or GIT_SCP.fullmatch(origin):
            engine.protect_text(origin)
        else:
            engine.register(origin, "identifier", "git_origin")


def seed_home(engine: SurrogateEngine) -> None:
    home = Path.home()
    if str(home) == home.anchor:
        return
    engine.register_root(str(home), independent=True)
    canonical = home.resolve()
    if canonical != home:
        engine.register_root(str(canonical), independent=True)
