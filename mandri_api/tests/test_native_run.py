from contextlib import asynccontextmanager, contextmanager
from types import SimpleNamespace

from fastapi.testclient import TestClient
from mandri.api.app import create_app
from mandri.core.native_run import NativeRunPlan
from mandri.core.types.execution import ProtectionError


@contextmanager
def client_for(context):
    app = create_app()
    with TestClient(app) as client:
        app.state.lifespan.runtime = SimpleNamespace(native_run=context)
        app.state.lifespan.gateway = SimpleNamespace(reasoning_catalog=None)
        yield client


def test_prepare_keeps_route_alive_until_cli_releases_run():
    observed = []
    stopped = []

    @asynccontextmanager
    async def session(spec):
        observed.append(spec)
        try:
            yield NativeRunPlan(id="run", argv=["docker", "exec", "-it", "worker", "codex"])
        finally:
            stopped.append(True)

    with client_for(session) as client:
        response = client.post(
            "/v1/runtime/runs",
            json={
                "harness": "codex",
                "model": "p/model",
                "cwd": "/workspace",
                "execution_backend": "docker",
                "privacy_mode": "surrogate",
                "tty": True,
            },
        )
        assert response.status_code == 201
        assert response.json()["argv"] == ["docker", "exec", "-it", "worker", "codex"]
        assert not stopped
        assert observed[0].privacy_mode.value == "surrogate"
        assert client.delete("/v1/runtime/runs/run").status_code == 204
        assert stopped == [True]
        assert client.delete("/v1/runtime/runs/run").status_code == 204
        assert stopped == [True]


def test_privacy_failure_never_produces_a_run_plan():
    @asynccontextmanager
    async def session(spec):
        raise ProtectionError("privacy_key_unavailable", "Privacy key unavailable")
        yield

    with client_for(session) as client:
        response = client.post(
            "/v1/runtime/runs",
            json={
                "harness": "codex",
                "model": "p/model",
                "cwd": "/workspace",
            },
        )
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "privacy_key_unavailable"


def test_app_shutdown_releases_prepared_runs():
    stopped = []

    @asynccontextmanager
    async def session(spec):
        try:
            yield NativeRunPlan(id="run", argv=["docker", "exec", "-i", "worker", "codex"])
        finally:
            stopped.append(True)

    app = create_app()
    with TestClient(app) as client:
        app.state.lifespan.runtime = SimpleNamespace(native_run=session)
        app.state.lifespan.gateway = SimpleNamespace(reasoning_catalog=None)
        assert (
            client.post(
                "/v1/runtime/runs",
                json={
                    "harness": "codex",
                    "model": "p/model",
                    "cwd": "/workspace",
                },
            ).status_code
            == 201
        )
        assert not stopped
    assert stopped == [True]
