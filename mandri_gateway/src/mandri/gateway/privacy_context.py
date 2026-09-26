import configparser
import getpass
import os
import socket
import stat
from pathlib import Path

from mandri.core.types.execution import ProtectionError
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


def git_directories(root: Path) -> tuple[Path, ...]:
    marker = root / ".git"
    if marker.is_symlink():
        raise unavailable()
    if marker.is_dir():
        return (marker,)
    value = local_text(marker, optional=True)
    if value is None:
        return ()
    prefix, separator, target = value.strip().partition(": ")
    if prefix != "gitdir" or not separator or "\n" in target:
        raise unavailable()
    directory = (root / target).resolve()
    if directory.parent.name != "worktrees" or directory.parent.parent.name != ".git":
        raise unavailable()
    common = directory.parent.parent
    common_value = local_text(directory / "commondir")
    backpointer = local_text(directory / "gitdir")
    if common_value is None or backpointer is None:
        raise unavailable()
    if (directory / common_value.strip()).resolve() != common.resolve():
        raise unavailable()
    if Path(backpointer.strip()).resolve() != marker.resolve():
        raise unavailable()
    return (common, directory)


def git_values(root: Path) -> dict[str, str]:
    parser = configparser.ConfigParser(interpolation=None, strict=False)
    directories = git_directories(root)
    home = Path.home()
    paths = [home / ".config" / "git" / "config", home / ".gitconfig"]
    paths.extend(
        directory / ("config" if index == 0 else "config.worktree")
        for index, directory in enumerate(directories)
    )
    for path in paths:
        content = local_text(path, optional=True)
        if content is None:
            continue
        try:
            parser.read_string(content)
        except configparser.Error:
            raise unavailable() from None
    selected = {}
    for section, option, name in (
        ('remote "origin"', "url", "origin"),
        ("user", "name", "author"),
        ("user", "email", "email"),
    ):
        if not parser.has_option(section, option):
            continue
        value = parser.get(section, option, raw=True).strip()
        if value.startswith('"') and value.endswith('"'):
            value = value[1:-1].replace('\\"', '"').replace("\\\\", "\\")
        if value:
            selected[name] = value
    return selected


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
