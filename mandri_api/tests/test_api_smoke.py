"""Smoke tests for the api package surface."""

from fastapi import FastAPI
from mandri.api.app import create_app
from mandri.api.deps import LifespanState


def test_create_app_returns_fastapi_instance() -> None:
    app = create_app()
    assert isinstance(app, FastAPI)


def test_lifespan_state_defaults_are_none() -> None:
    state = LifespanState()
    assert state.db is None
    assert state.sessions is None
    assert state.runtime is None
    assert state.gateway is None
    assert state.heartbeat_configured is False
