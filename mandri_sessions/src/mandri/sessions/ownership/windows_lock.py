import ctypes
import sys
from ctypes import wintypes
from pathlib import Path


class UniqueProcess(ctypes.Structure):
    _fields_ = [("pid", wintypes.DWORD), ("started", wintypes.FILETIME)]


class ProcessInfo(ctypes.Structure):
    _fields_ = [
        ("process", UniqueProcess),
        ("name", wintypes.WCHAR * 256),
        ("service", wintypes.WCHAR * 64),
        ("application_type", wintypes.DWORD),
        ("status", wintypes.ULONG),
        ("session", wintypes.DWORD),
        ("restartable", wintypes.BOOL),
    ]


def locking_processes(path: Path) -> dict[int, float]:
    if sys.platform != "win32":
        raise OSError("Native Windows lock inspection requires Windows")
    manager = ctypes.WinDLL("rstrtmgr", use_last_error=True)
    handle = wintypes.DWORD()
    key = ctypes.create_unicode_buffer(33)
    if manager.RmStartSession(ctypes.byref(handle), 0, key) != 0:
        raise OSError("Cannot inspect native writer")
    try:
        files = (wintypes.LPCWSTR * 1)(str(path.resolve()))
        if manager.RmRegisterResources(handle, 1, files, 0, None, 0, None) != 0:
            raise OSError("Cannot inspect native writer")
        needed, count, reason = wintypes.UINT(), wintypes.UINT(), wintypes.DWORD()
        result = manager.RmGetList(
            handle, ctypes.byref(needed), ctypes.byref(count), None, ctypes.byref(reason)
        )
        if result == 0:
            return {}
        if result != 234:
            raise OSError("Cannot inspect native writer")
        count.value = needed.value
        processes = (ProcessInfo * count.value)()
        result = manager.RmGetList(
            handle, ctypes.byref(needed), ctypes.byref(count), processes, ctypes.byref(reason)
        )
        if result != 0:
            raise OSError("Native writer changed during inspection")
        return {
            int(info.process.pid): (
                (info.process.started.dwHighDateTime << 32) | info.process.started.dwLowDateTime
            )
            / 10_000_000
            - 11644473600
            for info in processes[: count.value]
        }
    finally:
        manager.RmEndSession(handle)
