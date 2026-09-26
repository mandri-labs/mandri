from collections.abc import Callable, Iterable
from pathlib import Path

from mandri.core.ports.database import DatabasePort
from mandri.core.ports.privacy_keys import PrivacyKeyProvider
from mandri.core.types.config import PrivacySettings
from mandri.core.types.execution import ProtectionError
from mandri.database.privacy import PrivacyRepository
from mandri.gateway.privacy import GatewayPrivacy
from mandri.gateway.privacy_keys import FilePrivacyKey, KeyringPrivacyKey
from mandri.gateway.privacy_scopes import PrivacyScopes


def build_privacy(
    settings: PrivacySettings,
    base_dir: Path,
    db: DatabasePort,
    credentials: Callable[[], Iterable[str]] = tuple,
) -> GatewayPrivacy:
    keys: PrivacyKeyProvider
    if settings.key_file is not None:
        path = Path(settings.key_file).expanduser()
        if not path.is_absolute() or path.resolve().is_relative_to(base_dir.resolve()):
            raise ProtectionError(
                "privacy_key_unavailable", "Privacy key must be outside the daemon state directory"
            )
        keys = FilePrivacyKey(path)
    else:
        keys = KeyringPrivacyKey(base_dir, settings.key_service)
    scopes = PrivacyScopes(PrivacyRepository(db, keys), settings, credentials)
    return GatewayPrivacy(scopes)
