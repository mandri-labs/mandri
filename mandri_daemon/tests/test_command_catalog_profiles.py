from unittest.mock import AsyncMock

import pytest
from mandri.core.ids import HarnessKind
from mandri.core.protocol.commands import CommandCatalogParams
from mandri.core.types.config import SessionsConfig
from mandri.daemon.command_catalogs import create_command_catalog_cache


@pytest.mark.parametrize("configured", [True, False])
async def test_agy_command_catalog_uses_native_authentication_profile(
    monkeypatch, tmp_path, configured
):
    native = tmp_path / "native"
    monkeypatch.setattr("mandri.daemon.command_catalogs.default_agy_root", lambda: native)
    discover = AsyncMock(return_value=[])
    monkeypatch.setattr("mandri.daemon.command_catalogs.discover_commands", discover)
    spawn = AsyncMock(side_effect=AssertionError("Discovery is replaced"))
    profiles = tmp_path / "separate-profiles"
    catalog = create_command_catalog_cache(
        {"agy": ["agy"]},
        SessionsConfig(agy_home=str(native) if configured else None),
        spawn,
        default_cwd=str(tmp_path),
        profiles_dir=profiles,
    )
    try:
        result = await catalog.get(CommandCatalogParams(harness="agy"))
        assert result.state == "ready"
        arguments = discover.call_args.args
        assert arguments[0] is HarnessKind.AGY
        assert arguments[1] == ["agy", "--gemini_dir", str(native)]
        assert not profiles.exists()
    finally:
        await catalog.aclose()
