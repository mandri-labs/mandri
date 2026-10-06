from contextlib import AsyncExitStack

from fastapi import APIRouter, Request, Response
from mandri.api.deps import Gateway, Runtime, app_state
from mandri.api.errors import ApiError
from mandri.api.routers._effort import normalize_effort, validate_effort
from mandri.core.native_run import NativeRunPlan, NativeRunStart
from mandri.providers.errors import ProviderInvalidError, ProviderNotFoundError
from mandri.runtime.errors import HarnessNotInstalledError, ProcessSpawnError
from mandri.runtime.errors.docker import DockerExecutionError

router = APIRouter(tags=["runtime"])


@router.post("/runtime/runs", operation_id="prepare_native_run", status_code=201)
async def prepare_native_run(
    spec: NativeRunStart, request: Request, runtime: Runtime, gateway: Gateway
) -> NativeRunPlan:
    state = app_state(request.app)
    if state is None:
        raise ApiError("service_unavailable", "Runtime is unavailable", 503)
    validate_effort(gateway, spec.model, spec.effort)
    spec.effort = normalize_effort(spec.effort)
    stack = AsyncExitStack()
    try:
        plan = await stack.enter_async_context(runtime.native_run(spec))
    except (
        DockerExecutionError,
        HarnessNotInstalledError,
        ProcessSpawnError,
        ProviderInvalidError,
        ProviderNotFoundError,
        OSError,
    ) as error:
        await stack.aclose()
        raise ApiError("native_run_failed", str(error), 422) from None
    except BaseException:
        await stack.aclose()
        raise
    state.native_runs[plan.id] = stack
    return plan


@router.delete("/runtime/runs/{run_id}", operation_id="release_native_run", status_code=204)
async def release_native_run(run_id: str, request: Request) -> Response:
    state = app_state(request.app)
    if state is not None:
        stack = state.native_runs.get(run_id)
        if stack is not None:
            await stack.aclose()
            state.native_runs.pop(run_id, None)
    return Response(status_code=204)
