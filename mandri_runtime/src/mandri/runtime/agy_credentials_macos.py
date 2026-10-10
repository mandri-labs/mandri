import base64
import ctypes


def _decode_credential(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("go-keyring-encoded:"):
        return bytes.fromhex(raw.removeprefix("go-keyring-encoded:")).decode("utf-8")
    if raw.startswith("go-keyring-base64:"):
        encoded = raw.removeprefix("go-keyring-base64:")
        return base64.b64decode(encoded, validate=True).decode("utf-8")
    return raw


def _constant(library: ctypes.CDLL, name: str) -> int:
    value = ctypes.c_void_p.in_dll(library, name).value
    if value is None:
        raise OSError("Unavailable Keychain constant")
    return value


def read_credential(*, service: str = "gemini", account: str = "antigravity") -> str | None:
    security = ctypes.CDLL("/System/Library/Frameworks/Security.framework/Security")
    core = ctypes.CDLL("/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation")
    core.CFStringCreateWithCString.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint32]
    core.CFStringCreateWithCString.restype = ctypes.c_void_p
    core.CFDictionaryCreate.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_ssize_t,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    core.CFDictionaryCreate.restype = ctypes.c_void_p
    core.CFRelease.argtypes = [ctypes.c_void_p]
    core.CFRelease.restype = None
    core.CFDataGetLength.argtypes = [ctypes.c_void_p]
    core.CFDataGetLength.restype = ctypes.c_ssize_t
    core.CFDataGetBytePtr.argtypes = [ctypes.c_void_p]
    core.CFDataGetBytePtr.restype = ctypes.c_void_p
    security.SecItemCopyMatching.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
    security.SecItemCopyMatching.restype = ctypes.c_int32
    service_ref = core.CFStringCreateWithCString(None, service.encode("utf-8"), 0x08000100)
    account_ref = core.CFStringCreateWithCString(None, account.encode("utf-8"), 0x08000100)
    query = None
    result = ctypes.c_void_p()
    try:
        if not service_ref or not account_ref:
            raise OSError("Unable to create Antigravity credential query")
        attributes = {
            "kSecClass": _constant(security, "kSecClassGenericPassword"),
            "kSecAttrService": service_ref,
            "kSecAttrAccount": account_ref,
            "kSecMatchLimit": _constant(security, "kSecMatchLimitOne"),
            "kSecReturnData": _constant(core, "kCFBooleanTrue"),
            "kSecUseAuthenticationUI": _constant(security, "kSecUseAuthenticationUIFail"),
        }
        keys = (ctypes.c_void_p * len(attributes))(
            *(_constant(security, name) for name in attributes)
        )
        values = (ctypes.c_void_p * len(attributes))(*attributes.values())
        query = core.CFDictionaryCreate(None, keys, values, len(attributes), None, None)
        if not query:
            raise OSError("Unable to create Antigravity credential query")
        status = security.SecItemCopyMatching(query, ctypes.byref(result))
        if status == -25300:
            return None
        if status != 0 or not result.value:
            raise OSError(status, "Antigravity credential unavailable")
        size = core.CFDataGetLength(result)
        if not 0 < size <= 65536:
            raise ValueError("Invalid Antigravity credential size")
        raw = ctypes.string_at(core.CFDataGetBytePtr(result), size).decode("utf-8")
        return _decode_credential(raw)
    finally:
        for reference in (result.value, query, account_ref, service_ref):
            if reference:
                core.CFRelease(reference)
