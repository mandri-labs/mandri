"""Derive native turn ownership from persisted transcript boundaries."""

import json
from collections.abc import Sequence

from mandri.core.ids import HarnessKind


def native_turn_busy(harness: HarnessKind, entries: Sequence[str]) -> bool | None:
    for line in reversed(entries):
        try:
            event = json.loads(line)
        except (ValueError, TypeError):
            continue
        if not isinstance(event, dict):
            continue
        if harness is HarnessKind.CODEX:
            payload = event.get("payload", {})
            if event.get("type") != "event_msg" or not isinstance(payload, dict):
                continue
            kind = payload.get("type")
            if kind in ("task_complete", "task_aborted", "turn_aborted"):
                return False
            if kind == "task_started":
                return True
        elif harness is HarnessKind.CLAUDE:
            message = event.get("message", {})
            if event.get("type") == "assistant" and isinstance(message, dict):
                return message.get("stop_reason") not in ("end_turn", "stop_sequence", "max_tokens")
            if event.get("type") == "user":
                return True
        elif harness is HarnessKind.OPENCODE:
            properties = event.get("properties")
            info = properties.get("info") if isinstance(properties, dict) else None
            if event.get("type") == "message.updated" and isinstance(info, dict):
                if info.get("role") == "assistant":
                    if info.get("finish") == "tool-calls":
                        return True
                    timing = info.get("time")
                    return not (isinstance(timing, dict) and timing.get("completed"))
                if info.get("role") == "user":
                    return True
        elif harness is HarnessKind.AGY:
            if event.get("event") == "Stop" and isinstance(event.get("fullyIdle"), bool):
                return not event["fullyIdle"]
            if event.get("type") == "USER_INPUT" or event.get("status") in ("ACTIVE", "RUNNING"):
                return True
        elif harness is HarnessKind.PI:
            message = event.get("message")
            if event.get("type") != "message" or not isinstance(message, dict):
                continue
            if message.get("role") == "assistant":
                return message.get("stopReason") not in ("stop", "length", "error", "aborted")
            if message.get("role") in ("user", "toolResult"):
                return True
    return None if entries else False


def native_model(entries: Sequence[str]) -> str | None:
    for line in reversed(entries):
        try:
            event = json.loads(line)
        except (ValueError, TypeError):
            continue
        if not isinstance(event, dict):
            continue
        kind = event.get("type")
        item = None
        if kind == "turn_context":
            item = event.get("payload")
        elif kind == "assistant":
            item = event.get("message")
        elif kind == "message" and isinstance(message := event.get("message"), dict):
            if message.get("role") == "assistant":
                model, provider = message.get("model"), message.get("provider")
                if isinstance(model, str) and model:
                    return f"{provider}/{model}" if provider else model
        elif kind == "model_change" and isinstance(event.get("modelId"), str):
            model, provider = event["modelId"], event.get("provider")
            return f"{provider}/{model}" if provider else model
        elif kind == "system" and event.get("subtype") == "init":
            item = event
        elif event.get("event") == "init":
            item = event.get("init", event)
        elif isinstance(event.get("modelName"), str):
            return str(event["modelName"])
        if isinstance(item, dict):
            model = item.get("model")
            if isinstance(model, str) and model.strip() and model != "<synthetic>":
                return model
        properties = event.get("properties")
        info = properties.get("info") if isinstance(properties, dict) else None
        if kind == "message.updated" and isinstance(info, dict):
            model_info = info.get("model") if info.get("role") == "user" else info
            if isinstance(model_info, dict):
                model = model_info.get("modelID")
                provider = model_info.get("providerID")
                if isinstance(model, str) and model.strip():
                    return f"{provider}/{model}" if provider else model
    return None
