from contextlib import closing

from keyring.backends import SecretService


def read_credential(*, service: str = "gemini", account: str = "antigravity") -> str | None:
    storage = getattr(SecretService, "secretstorage", None)
    if storage is None:
        raise OSError("Secret Service unavailable")
    with closing(storage.dbus_init()) as connection:
        collection = next(
            (
                candidate
                for candidate in storage.get_all_collections(connection)
                if candidate.collection_path == "/org/freedesktop/secrets/collection/login"
            ),
            None,
        )
        if collection is None:
            collection = storage.Collection(connection)
        if collection.is_locked():
            raise PermissionError("Antigravity credential collection is locked")
        item = next(
            iter(collection.search_items({"service": service, "username": account})),
            None,
        )
        if item is None:
            return None
        if item.is_locked():
            raise PermissionError("Antigravity credential is locked")
        secret = item.get_secret()
        if not isinstance(secret, bytes) or not 0 < len(secret) <= 65536:
            raise ValueError("Invalid Antigravity credential size")
        return secret.decode("utf-8")
