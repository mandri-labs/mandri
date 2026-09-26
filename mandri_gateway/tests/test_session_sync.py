import json

import pytest
from mandri.core.ids import HarnessKind, HarnessSessionId, SessionState, WireFormat
from mandri.core.types.config import DaemonConfig, ProviderConfig
from mandri.core.types.execution import PrivacyMode, ProtectionError
from mandri.database.sqlite_adapter import AiosqliteDatabase
from mandri.gateway.route_registry import RouteRegistry
from mandri.providers.service import ProvidersRegistry
from mandri.sessions.adapters.pi_sessions import PiSessionsAdapter
from mandri.sessions.service import SessionsService
from mandri.sessions.sync import SyncEngine

from mandri_gateway.tests.test_route_registry import FakeConfig, FakeRoutes, FakeVerifier


async def test_protected_pi_route_survives_sync_before_native_history_is_written(tmp_path):
    db = AiosqliteDatabase()
    await db.connect(tmp_path / "mandri.db")
    try:
        await db.migrate()
        providers = ProvidersRegistry(
            FakeConfig(
                DaemonConfig(
                    providers=[ProviderConfig(name="test", kind="opencode_go", api_key="synthetic")]
                )
            ),
            FakeRoutes(),
            FakeVerifier(),
        )
        routes = RouteRegistry(db, providers)
        route = await routes.create(
            "test",
            "test-model",
            (WireFormat.OPENAI,),
            privacy_mode=PrivacyMode.SURROGATE,
            privacy_scope_id="test-scope",
        )
        adapter = PiSessionsAdapter(tmp_path / "sessions")
        engine = SyncEngine(db, {HarnessKind.PI: adapter})
        sessions = SessionsService(db, engine)
        session = await sessions.create_session(
            HarnessKind.PI,
            model="test/test-model",
            route_id=route.id,
            project_path=str(tmp_path),
            privacy_mode=PrivacyMode.SURROGATE,
            privacy_scope_id="test-scope",
            initial_state=SessionState.LIVE,
        )
        await sessions.reveal_native_id(session.id, HarnessSessionId("native-pi"))
        history = adapter.store.root / "project" / "native-pi.jsonl"
        adapter.store.register("native-pi", history)

        assert (await routes.resolve(route.id)).conversation_id == session.id
        assert adapter.fetch() == []
        await engine.sync()
        assert not engine.harness_state(HarnessKind.PI).degraded
        assert (await routes.resolve(route.id)).conversation_id == session.id

        history.parent.mkdir(parents=True)
        history.write_text(
            json.dumps(
                {
                    "type": "session",
                    "version": 3,
                    "id": "native-pi",
                    "cwd": str(tmp_path),
                    "timestamp": "2026-01-01T12:00:00Z",
                }
            )
            + "\n"
        )
        await engine.sync()
        assert not engine.harness_state(HarnessKind.PI).degraded
        resolved = await routes.resolve(route.id)
        assert resolved.conversation_id == session.id
        assert resolved.privacy_mode is PrivacyMode.SURROGATE
        assert resolved.privacy_scope_id == "test-scope"

        await sessions.set_session_state(session.id, SessionState.STOPPED)
        await sessions.delete_session(session.id)
        await engine.sync()
        with pytest.raises(ProtectionError, match="active session"):
            await routes.resolve(route.id)
    finally:
        await db.close()
