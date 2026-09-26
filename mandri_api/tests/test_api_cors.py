"""Tests for CORS handling in the daemon API."""

import pytest
from fastapi.testclient import TestClient
from mandri.api.app import create_app
from mandri.core.types.config import CorsOrigin

DEV_ORIGIN = "http://localhost:1420"
DEV_ORIGIN_ALT = "http://127.0.0.1:1420"
FOREIGN_ORIGIN = "https://evil.example"
ALLOW_METHODS = "GET, POST, PATCH, PUT, DELETE, OPTIONS"
ADDITIONAL_DEV_ORIGINS = [
    "http://tauri.localhost",
    "tauri://localhost",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:5174",
    "http://127.0.0.1:5174",
]


@pytest.mark.parametrize("origin", ADDITIONAL_DEV_ORIGINS)
def test_preflight_for_additional_dev_origin_is_allowed(origin: str) -> None:
    app = create_app()
    client = TestClient(app)
    response = client.options(
        "/v1/sessions",
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": "GET",
        },
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == origin


@pytest.mark.parametrize("origin", [DEV_ORIGIN, *ADDITIONAL_DEV_ORIGINS])
def test_plain_get_includes_allow_origin_header_for_dev_origins(origin: str) -> None:
    app = create_app()
    client = TestClient(app)
    response = client.get("/v1/asyncapi.json", headers={"Origin": origin})
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == origin


def test_preflight_allows_authorization_and_accept_headers() -> None:
    app = create_app()
    client = TestClient(app)
    response = client.options(
        "/v1/sessions",
        headers={
            "Origin": DEV_ORIGIN,
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Headers": "content-type, authorization, accept",
        },
    )
    assert response.status_code == 200
    allow_headers = response.headers["access-control-allow-headers"].lower()
    assert "content-type" in allow_headers
    assert "authorization" in allow_headers
    assert "accept" in allow_headers


def test_preflight_for_foreign_origin_gets_no_allow_headers() -> None:
    app = create_app()
    client = TestClient(app)
    response = client.options(
        "/v1/sessions",
        headers={
            "Origin": FOREIGN_ORIGIN,
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Headers": "Content-Type",
        },
    )
    assert response.status_code == 400
    assert "access-control-allow-origin" not in response.headers


def test_preflight_for_dev_origin_is_allowed() -> None:
    app = create_app()
    client = TestClient(app)
    response = client.options(
        "/v1/sessions",
        headers={
            "Origin": DEV_ORIGIN,
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Headers": "Content-Type",
        },
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == DEV_ORIGIN
    assert response.headers["access-control-allow-methods"] == ALLOW_METHODS


def test_preflight_for_alt_loopback_origin_is_allowed() -> None:
    app = create_app()
    client = TestClient(app)
    response = client.options(
        "/v1/sessions",
        headers={
            "Origin": DEV_ORIGIN_ALT,
            "Access-Control-Request-Method": "GET",
        },
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == DEV_ORIGIN_ALT


def test_foreign_origin_gets_no_allow_origin_header() -> None:
    app = create_app()
    client = TestClient(app)
    response = client.get("/v1/asyncapi.json", headers={"Origin": FOREIGN_ORIGIN})
    assert "access-control-allow-origin" not in response.headers


def test_request_without_origin_unchanged() -> None:
    app = create_app()
    client = TestClient(app)
    response = client.get("/v1/asyncapi.json")
    assert response.status_code == 200
    assert "access-control-allow-origin" not in response.headers


def test_custom_origins_replace_defaults() -> None:
    app = create_app(cors_origins=[CorsOrigin("https://ops.example")])
    client = TestClient(app)
    allowed = client.get("/v1/asyncapi.json", headers={"Origin": "https://ops.example"})
    assert allowed.headers["access-control-allow-origin"] == "https://ops.example"
    denied = client.get("/v1/asyncapi.json", headers={"Origin": DEV_ORIGIN})
    assert "access-control-allow-origin" not in denied.headers
