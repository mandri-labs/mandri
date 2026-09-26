"""Served API contracts stay connected to executable routes and actions."""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from mandri.api.app import create_app
from mandri.core.protocol.registry import ACTIONS, ActionSpec
from pydantic import BaseModel, Field

SNAPSHOTS = Path(__file__).resolve().parents[2] / "api-snapshots"


class ContractProbeParams(BaseModel):
    count: int = Field(ge=1)


@pytest.mark.parametrize("contract", ["openapi", "asyncapi"])
def test_served_contract_matches_generated_snapshot(contract: str) -> None:
    with TestClient(create_app()) as client:
        response = client.get(f"/v1/{contract}.json")
    assert response.status_code == 200
    assert response.json() == json.loads(
        (SNAPSHOTS / f"{contract}.json").read_text(encoding="utf-8")
    )


def test_asyncapi_route_observes_action_registration_after_an_earlier_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    action = ActionSpec("contract.probe", ContractProbeParams, "Live action registration.")
    with TestClient(create_app()) as client:
        assert action.name not in client.get("/v1/asyncapi.json").json()["operations"]
        monkeypatch.setitem(ACTIONS, action.name, action)
        document = client.get("/v1/asyncapi.json").json()
    assert action.name in document["operations"]
    assert "ContractProbeParams" in document["components"]["schemas"]


def test_openapi_route_uses_registered_application_routes() -> None:
    app = create_app()

    @app.post("/contract-probe")
    def probe(body: ContractProbeParams) -> ContractProbeParams:
        return body

    with TestClient(app) as client:
        response = client.get("/v1/openapi.json")
    assert response.status_code == 200
    document = response.json()
    assert document["paths"]["/contract-probe"]["post"]["requestBody"]["content"][
        "application/json"
    ]["schema"] == {"$ref": "#/components/schemas/ContractProbeParams"}
