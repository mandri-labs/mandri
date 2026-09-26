from typing import Any

from mandri.core.protocol.commands import (
    CommandCatalogParams,
    CommandGetParams,
    CommandInvokeParams,
    CommandSessionParams,
)
from mandri.core.protocol.registry import ActionRegistry
from mandri.runtime.commands import CommandService
from pydantic import BaseModel


def register_command_actions(registry: ActionRegistry, service: CommandService) -> None:
    async def catalogs(params: BaseModel) -> dict[str, Any]:
        return (await service.catalogs.snapshot()).model_dump(mode="json")

    async def scoped_catalog(params: BaseModel) -> dict[str, Any]:
        assert isinstance(params, CommandCatalogParams)
        return (await service.catalogs.get(params)).model_dump(mode="json")

    async def catalog(params: BaseModel) -> dict[str, Any]:
        assert isinstance(params, CommandSessionParams)
        return (await service.catalog(params.session_id)).model_dump(mode="json")

    async def invoke(params: BaseModel) -> dict[str, Any]:
        assert isinstance(params, CommandInvokeParams)
        return (
            await service.invoke(
                params.session_id, params.invocation_id, params.command_id, params.arguments
            )
        ).model_dump(mode="json")

    async def get(params: BaseModel) -> dict[str, Any]:
        assert isinstance(params, CommandGetParams)
        return service.get(params.session_id, params.invocation_id).model_dump(mode="json")

    async def listing(params: BaseModel) -> dict[str, Any]:
        assert isinstance(params, CommandSessionParams)
        return {
            "invocations": [row.model_dump(mode="json") for row in service.list(params.session_id)]
        }

    async def cancel(params: BaseModel) -> dict[str, Any]:
        assert isinstance(params, CommandGetParams)
        return (await service.cancel(params.session_id, params.invocation_id)).model_dump(
            mode="json"
        )

    registry.register("command.catalogs", catalogs)
    registry.register("command.catalog", scoped_catalog)
    registry.register("session.commands", catalog)
    registry.register("command.invoke", invoke)
    registry.register("command.get", get)
    registry.register("command.list", listing)
    registry.register("command.cancel", cancel)
