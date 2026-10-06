"""FastAPI application factory for the Mandri daemon."""

from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from mandri.api.deps import LifespanState
from mandri.api.errors import register_error_handlers
from mandri.api.native_run import router as native_run_router
from mandri.api.routers.attachments import router as attachments_router
from mandri.api.routers.chatgpt import router as chatgpt_router
from mandri.api.routers.execution import router as execution_router
from mandri.api.routers.forks import router as forks_router
from mandri.api.routers.fs import router as fs_router
from mandri.api.routers.gateway import router as gateway_router
from mandri.api.routers.providers import router as providers_router
from mandri.api.routers.runtime import router as runtime_router
from mandri.api.routers.sessions import router as sessions_router
from mandri.api.routers.transcript_records import router as transcript_records_router
from mandri.api.routers.usage import router as usage_router
from mandri.api.routers.worktree_integration import router as worktree_integration_router
from mandri.api.ws import router as ws_router
from mandri.core.types.config import DEFAULT_CORS_ORIGINS, CorsOrigin
from mandri.core.version import __version__


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.lifespan = LifespanState()
    try:
        yield
    finally:
        state = app.state.lifespan
        for stack in state.native_runs.values():
            await stack.aclose()
        if state.agent_observer is not None:
            await state.agent_observer.close()


def create_app(cors_origins: Sequence[CorsOrigin] | None = None) -> FastAPI:
    app = FastAPI(
        title="Mandri Daemon",
        version=__version__,
        openapi_url="/v1/openapi.json",
        docs_url="/v1/docs",
        redoc_url="/v1/redoc",
        lifespan=_lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=(
            list(cors_origins) if cors_origins is not None else list(DEFAULT_CORS_ORIGINS)
        ),
        allow_credentials=False,
        allow_methods=["GET", "POST", "PATCH", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type", "Authorization", "Accept"],
    )
    register_error_handlers(app)
    app.include_router(sessions_router, prefix="/v1")
    app.include_router(worktree_integration_router, prefix="/v1")
    app.include_router(attachments_router, prefix="/v1")
    app.include_router(transcript_records_router, prefix="/v1")
    app.include_router(providers_router, prefix="/v1")
    app.include_router(chatgpt_router, prefix="/v1")
    app.include_router(gateway_router, prefix="/v1")
    app.include_router(runtime_router, prefix="/v1")
    app.include_router(execution_router, prefix="/v1")
    app.include_router(forks_router, prefix="/v1")
    app.include_router(fs_router, prefix="/v1")
    app.include_router(usage_router, prefix="/v1")
    app.include_router(ws_router, prefix="/v1")
    app.include_router(native_run_router, prefix="/v1")
    return app
