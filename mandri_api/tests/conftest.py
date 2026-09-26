"""Shared fixtures for api router tests."""

import pytest
from fastapi.testclient import TestClient
from mandri.api.app import create_app

DependencyOverrides = dict


@pytest.fixture
def make_client():
    def factory(overrides: DependencyOverrides) -> TestClient:
        app = create_app()
        for dependency, provider in overrides.items():
            app.dependency_overrides[dependency] = provider
        return TestClient(app)

    return factory
