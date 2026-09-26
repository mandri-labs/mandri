"""REST routes for filesystem browsing (listing-only, no content endpoints)."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, Query
from mandri.api.deps import Database
from mandri.api.errors import ERROR_RESPONSES, ApiError
from mandri.core.fs.errors import (
    FsError,
    FsNotADirectoryError,
    FsNotFoundError,
)
from mandri.core.fs.service import FsService, list_project_paths
from mandri.core.fs.types import FilesystemEntry, FsNode
from mandri.core.ids import FsPath
from pydantic import BaseModel

router = APIRouter(prefix="/fs", tags=["fs"])

FS_ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    status: ERROR_RESPONSES[status] for status in (400, 404, 500)
}


class FsEntryOut(BaseModel):
    name: str
    path: str
    is_dir: bool
    size: int | None
    modified_at: int | None


class FsNodeOut(BaseModel):
    entry: FsEntryOut
    children: list[FsNodeOut]


def _fs_service() -> FsService:
    return FsService()


FsServiceDep = Annotated[FsService, Depends(_fs_service)]


def _fs_api_error(exc: FsError, path: str) -> ApiError:
    if isinstance(exc, FsNotFoundError):
        return ApiError(
            code="fs_not_found",
            message=str(exc),
            status=404,
            detail={"path": path},
        )
    if isinstance(exc, FsNotADirectoryError):
        return ApiError(
            code="fs_not_a_directory",
            message=str(exc),
            status=400,
            detail={"path": path},
        )
    return ApiError(
        code="fs_read_error",
        message=str(exc),
        status=500,
        detail={"path": path},
    )


def _to_entry_out(entry: FilesystemEntry) -> FsEntryOut:
    return FsEntryOut(
        name=entry.name,
        path=str(entry.path),
        is_dir=entry.is_dir,
        size=entry.size,
        modified_at=entry.modified_at,
    )


def _to_node_out(node: FsNode) -> FsNodeOut:
    return FsNodeOut(
        entry=_to_entry_out(node.entry),
        children=[_to_node_out(child) for child in node.children],
    )


@router.get("/roots", operation_id="list_fs_roots", responses=FS_ERROR_RESPONSES)
async def list_fs_roots(service: FsServiceDep) -> list[FsEntryOut]:
    return [_to_entry_out(entry) for entry in await service.list_roots()]


@router.get("/list", operation_id="list_fs_dir", responses=FS_ERROR_RESPONSES)
async def list_fs_dir(service: FsServiceDep, path: str) -> list[FsEntryOut]:
    try:
        entries = await service.list_dir(FsPath(path))
    except FsError as error:
        raise _fs_api_error(error, path) from error
    return [_to_entry_out(entry) for entry in entries]


@router.get("/tree", operation_id="browse_fs_tree", responses=FS_ERROR_RESPONSES)
async def browse_fs_tree(
    service: FsServiceDep,
    path: str,
    depth: Annotated[int, Query(ge=0)] = 2,
) -> FsNodeOut:
    try:
        node = await service.tree(FsPath(path), depth)
    except FsError as error:
        raise _fs_api_error(error, path) from error
    return _to_node_out(node)


@router.get("/projects", operation_id="list_fs_projects", responses=FS_ERROR_RESPONSES)
async def list_fs_projects(db: Database) -> list[str]:
    return [str(project_path) for project_path in await list_project_paths(db)]
