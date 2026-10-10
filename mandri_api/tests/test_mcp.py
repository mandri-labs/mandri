import httpx
from mandri.api.app import create_app
from mandri.api.local_security import LocalSecurity
from mandri.mcp.service import McpService


async def test_mcp_crud_is_revisioned_and_never_returns_secrets(tmp_path):
    app = create_app()
    service = McpService(tmp_path, 9000)
    service.start()
    async with app.router.lifespan_context(app):
        app.state.lifespan.mcp = service
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:9000"
        ) as http:
            created = await http.post(
                "/v1/mcp/servers",
                json={
                    "name": "local",
                    "command": "fixture",
                    "enabled": False,
                    "env": {"FIRST": "private-first", "SECOND": "private-second"},
                },
            )
            assert created.status_code == 201
            assert "private" not in created.text
            row = created.json()
            assert row["state"] == "disabled"
            updated = await http.patch(
                f"/v1/mcp/servers/{row['id']}",
                json={
                    "revision": row["revision"],
                    "changes": {"env_updates": {"FIRST": "replacement", "THIRD": "third"}},
                },
            )
            assert updated.status_code == 200
            assert service.registry.get(row["id"]).config.env == {
                "FIRST": "replacement",
                "SECOND": "private-second",
                "THIRD": "third",
            }
            stale = await http.patch(
                f"/v1/mcp/servers/{row['id']}",
                json={
                    "revision": row["revision"],
                    "changes": {"enabled": True},
                },
            )
            assert stale.status_code == 409
            invalid = await http.post(
                "/v1/mcp/servers",
                json={
                    "name": "invalid",
                    "command": "fixture",
                    "url": "https://example.invalid",
                    "env": {"SECRET": "private-error"},
                },
            )
            assert invalid.status_code == 422
            assert "private-error" not in invalid.text
            await http.delete(f"/v1/mcp/servers/{row['id']}")
            assert (await http.get("/v1/mcp/servers")).json()["servers"] == []
    await service.aclose()


async def test_mcp_child_token_cannot_access_management(tmp_path):
    app = create_app()
    app.add_middleware(LocalSecurity, hosts=["127.0.0.1"], origins=[], token="management")
    service = McpService(tmp_path, 9000)
    service.start()
    async with app.router.lifespan_context(app):
        app.state.lifespan.mcp = service
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://127.0.0.1:9000",
            headers={"Authorization": "Bearer child"},
        ) as http:
            assert (await http.get("/v1/mcp/servers")).status_code == 401
            data = await http.post("/v1/mcp/connect/" + "0" * 32 + "/" + "0" * 36, json={})
            assert data.status_code == 401
            assert data.json() == {"error": "Access denied"}
            callback = await http.get(
                "/v1/mcp/oauth/callback/" + "0" * 36, params={"code": "fake", "state": "wrong"}
            )
            assert callback.status_code == 400
    await service.aclose()
