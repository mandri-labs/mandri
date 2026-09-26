import ntpath
import posixpath

import pytest
from mandri.core.fs import paths


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("D:/work/project", r"D:\work\project"),
        (r"D:\work\project", r"D:\work\project"),
        ("//server/share/project", r"\\server\share\project"),
        (r"\\?\UNC\server\share\project", r"\\server\share\project"),
        (r"\\?\D:\work\project", r"D:\work\project"),
        ("", ""),
    ],
)
def test_windows_paths(monkeypatch, raw, expected):
    monkeypatch.setattr(paths, "path", ntpath)
    assert paths.normalize_fs_path(raw) == expected


@pytest.mark.parametrize("raw", ["/home/project", r"/home/a\b", r"\\?\folder", ""])
def test_posix_paths_preserve_names(monkeypatch, raw):
    monkeypatch.setattr(paths, "path", posixpath)
    assert paths.normalize_fs_path(raw) == raw
