import ctypes
import sys
from ctypes import wintypes


class _Credential(ctypes.Structure):
    _fields_ = [
        ("Flags", wintypes.DWORD),
        ("Type", wintypes.DWORD),
        ("TargetName", wintypes.LPWSTR),
        ("Comment", wintypes.LPWSTR),
        ("LastWritten", wintypes.FILETIME),
        ("CredentialBlobSize", wintypes.DWORD),
        ("CredentialBlob", ctypes.POINTER(ctypes.c_ubyte)),
        ("Persist", wintypes.DWORD),
        ("AttributeCount", wintypes.DWORD),
        ("Attributes", ctypes.c_void_p),
        ("TargetAlias", wintypes.LPWSTR),
        ("UserName", wintypes.LPWSTR),
    ]


def read_credential(*, service: str = "gemini", account: str = "antigravity") -> str | None:
    if sys.platform != "win32":
        raise OSError("Windows Credential Manager unavailable")
    api = ctypes.WinDLL("advapi32", use_last_error=True)
    api.CredReadW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.POINTER(ctypes.POINTER(_Credential)),
    ]
    api.CredReadW.restype = wintypes.BOOL
    api.CredFree.argtypes = [ctypes.c_void_p]
    api.CredFree.restype = None
    credential = ctypes.POINTER(_Credential)()
    if not api.CredReadW(f"{service}:{account}", 1, 0, ctypes.byref(credential)):
        error = ctypes.get_last_error()
        if error == 1168:
            return None
        raise OSError(error, "Antigravity credential unavailable")
    try:
        size = credential.contents.CredentialBlobSize
        if not 0 < size <= 65536:
            raise ValueError("Invalid Antigravity credential size")
        return ctypes.string_at(credential.contents.CredentialBlob, size).decode("utf-8")
    finally:
        api.CredFree(credential)
