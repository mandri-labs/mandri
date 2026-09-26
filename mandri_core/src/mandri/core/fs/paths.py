"""Windows extended-length path prefix handling for filesystem browsing."""

import ntpath
from os import path

from mandri.core.ids import FsPath

WINDOWS_EXTENDED_PREFIX = "\\\\?\\"


def strip_extended_prefix(raw: str) -> str:
    if raw.startswith(WINDOWS_EXTENDED_PREFIX + "UNC\\"):
        return "\\\\" + raw[len(WINDOWS_EXTENDED_PREFIX + "UNC\\") :]
    if raw.startswith(WINDOWS_EXTENDED_PREFIX):
        return raw[len(WINDOWS_EXTENDED_PREFIX) :]
    return raw


def normalize_fs_path(raw: str) -> FsPath:
    if not raw:
        return FsPath("")
    text = strip_extended_prefix(raw) if path is ntpath else raw
    return FsPath(path.normpath(text))
