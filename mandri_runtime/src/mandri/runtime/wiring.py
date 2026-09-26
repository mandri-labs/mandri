"""Per-harness gateway launch plans, backed by the shared core launch module."""

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from mandri.core import launch
from mandri.core.ids import HarnessKind, HarnessSessionId
from mandri.core.launch import HarnessLaunch, merged_env
from mandri.gateway.model_metadata import ModelMetadata

__all__ = ["HarnessLaunch", "build_harness_launch", "claude_resume_args", "merged_env"]

_CLAUDE_RESUME_FLAG = "--resume"


def claude_resume_args(native_id: HarnessSessionId) -> tuple[str, ...]:
    return (_CLAUDE_RESUME_FLAG, str(native_id))


def build_harness_launch(
    harness: HarnessKind,
    gateway_port: int,
    route_id: str,
    model: str,
    token: str,
    metadata: ModelMetadata | None = None,
    opencode_config: Mapping[str, Any] | None = None,
    resume_native_id: HarnessSessionId | None = None,
    *,
    effort: str | None = None,
) -> launch.HarnessLaunch:
    plan = launch.build_harness_launch(
        harness, gateway_port, route_id, model, token, metadata, opencode_config, effort=effort
    )
    if resume_native_id is not None and harness is HarnessKind.PI:
        return replace(plan, args=(*plan.args, "--session", str(resume_native_id)))
    if resume_native_id is not None and harness is HarnessKind.AGY:
        return replace(plan, args=(*plan.args, "--conversation", str(resume_native_id)))
    if resume_native_id is None or harness is not HarnessKind.CLAUDE:
        return plan
    return replace(plan, args=(*plan.args, *claude_resume_args(resume_native_id)))
