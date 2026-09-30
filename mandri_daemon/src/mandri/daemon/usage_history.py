import asyncio
import hashlib
import json
import logging
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from mandri.core.clock import system_now_ms
from mandri.core.ids import HarnessKind, HarnessSessionId, ProjectPath
from mandri.core.ports.database import DatabasePort
from mandri.core.ports.transcripts import SessionRef
from mandri.core.types.usage import UsageObservation
from mandri.daemon.usage_history_routes import attribute_route
from mandri.database.usage import UsageRepository
from mandri.sessions.transcripts.claude_transcripts import ClaudeTranscriptReader
from mandri.sessions.transcripts.codex_transcripts import CodexTranscriptReader
from mandri.sessions.transcripts.errors import TranscriptError
from mandri.sessions.transcripts.pi_transcripts import PiTranscriptReader
from mandri.sessions.usage.adapter import to_usage_observation
from mandri.sessions.usage.history import UsageCursor, read_usage_batch
from mandri.sessions.usage.types import NativeUsageContext
from mandri.sessions.usage_opencode import OpencodeUsageReader

logger = logging.getLogger(__name__)
RECONCILIATION_VERSION = 6


class UsageHistorySync:
    def __init__(
        self,
        database: DatabasePort,
        repository: UsageRepository,
        codex_home: Path,
        *,
        claude_home: Path | None = None,
        opencode_db: Path | None = None,
    ) -> None:
        self._database = database
        self._repository = repository
        self._codex = CodexTranscriptReader(codex_home / "sessions")
        self._claude = ClaudeTranscriptReader(claude_home / "projects") if claude_home else None
        self._opencode = OpencodeUsageReader(opencode_db) if opencode_db else None
        self._pi = PiTranscriptReader()
        self._after = ""
        self._round_pending = False

    async def reconcile(self) -> bool:
        rows = await self._database.fetch_all(
            "SELECT id,native_id,project_path,harness,parent_session_id,parent_native_id,"
            "model,model_source,gateway_route_id FROM session"
            " WHERE harness IN ('codex','claude','opencode','pi')"
            " AND native_id IS NOT NULL AND execution_backend='host' AND id>?"
            " ORDER BY id LIMIT 20",
            (self._after,),
        )
        if not rows:
            self._after = ""
            pending, self._round_pending = self._round_pending, False
            return pending
        for row in rows:
            self._after = str(row["id"])
            try:
                self._round_pending = await self._session(dict(row)) or self._round_pending
            except Exception:
                logger.exception("Usage history source failed; previous metrics retained")
            await asyncio.sleep(0)
        return True

    async def _session(self, row: dict[str, Any]) -> bool:
        root = row.get("parent_session_id")
        seen = {row["id"]}
        while root and root not in seen and len(seen) < 64:
            seen.add(root)
            parent = await self._database.fetch_one(
                "SELECT parent_session_id FROM session WHERE id=?",
                (root,),
            )
            if parent is None or parent["parent_session_id"] is None:
                break
            root = parent["parent_session_id"]
        row["root_session_id"] = root
        row["route_history"] = (
            await self._database.fetch_all(
                "SELECT * FROM gateway_route_history WHERE route_id=? ORDER BY effective_from,id",
                (row["gateway_route_id"],),
            )
            if row.get("gateway_route_id")
            else []
        )
        row["route_revision"] = hashlib.sha256(
            json.dumps(row["route_history"], sort_keys=True).encode()
        ).hexdigest()
        session_id, native_id, harness = str(row["id"]), str(row["native_id"]), str(row["harness"])
        source_key = harness + "-history:" + hashlib.sha256(native_id.encode()).hexdigest()
        stored = await self._repository.read_cursor(source_key)
        reference = SessionRef(
            harness=HarnessKind(harness),
            native_id=HarnessSessionId(native_id),
            project_path=ProjectPath(str(row["project_path"])),
        )
        if harness == "opencode":
            return await self._opencode_session(row, reference, source_key, stored)
        reader = {"codex": self._codex, "claude": self._claude, "pi": self._pi}.get(harness)
        if reader is None:
            return False
        try:
            revision = await asyncio.to_thread(reader.revision, reference)
        except (OSError, TranscriptError):
            await self._unavailable(source_key, session_id, stored)
            return False
        path, modified, size = str(revision[0]), revision[1], revision[2]
        source_revision = f"{modified}:{size}"
        if len(revision) > 3:
            source_revision += f":{revision[3]}"
        reset = (
            stored is None
            or stored.get("parser_version") != 2
            or stored.get("reconciliation_version") != RECONCILIATION_VERSION
            or stored.get("route_revision") != row["route_revision"]
            or stored.get("status") == "source_changed"
        )
        if (
            not reset
            and stored is not None
            and stored.get("source_revision") == source_revision
            and stored.get("phase") == "active"
            and stored.get("status") != "backfill"
        ):
            return False
        cursor = (
            None
            if reset or stored is None
            else UsageCursor(
                identity=str(stored["identity"]),
                offset=int(stored["offset"]),
                anchor=str(stored["anchor"]),
                parser_version=int(stored["parser_version"]),
                state_json=str(stored.get("state_json", "{}")),
            )
        )
        context = NativeUsageContext(
            session_id=session_id,
            native_id=native_id,
            harness=harness,
            process_epoch="history",
            project_path=str(row["project_path"]),
            routing="native",
            parent_session_id=row.get("parent_session_id"),
            inherited_history="unknown" if row.get("parent_native_id") else "none",
            billing_mode="unknown",
        )
        batch = await asyncio.to_thread(
            read_usage_batch,
            Path(path),
            context,
            cursor,
            observed_at_ms=int(system_now_ms()),
            max_records=1000,
            max_bytes=32 * 1024 * 1024,
            max_record_bytes=16 * 1024 * 1024,
        )
        if batch.status == "source_changed":
            batch = await asyncio.to_thread(
                read_usage_batch,
                Path(path),
                context,
                None,
                observed_at_ms=int(system_now_ms()),
                max_records=1000,
                max_bytes=32 * 1024 * 1024,
                max_record_bytes=16 * 1024 * 1024,
            )
            reset = True
        if batch.status == "unavailable":
            await self._unavailable(source_key, session_id, stored)
            return False
        if batch.cursor is None:
            return False
        values = [
            self._attributed(to_usage_observation(item, sequence=item.occurred_at_ms or 0), row)
            for item in batch.observations
        ]
        pending = batch.has_more and batch.status != "partial_record"
        partial = batch.status in {
            "oversize_record",
            "malformed_record",
            "partial_usage",
            "partial_record",
        } or (not reset and stored is not None and stored.get("status") == "partial")
        checkpoint = {
            **asdict(batch.cursor),
            "reconciliation_version": RECONCILIATION_VERSION,
            "route_revision": row["route_revision"],
            "session_id": session_id,
            "source_revision": source_revision,
            "status": "backfill" if pending else "partial" if partial else "ready",
            "authoritative": int(bool(stored and stored.get("authoritative"))),
        }
        rebuilding = reset or stored is None or stored.get("phase") != "active"
        if rebuilding:
            await self._repository.stage_history(
                values,
                source_key,
                checkpoint,
                reset=reset,
                complete=not pending,
            )
        else:
            await self._repository.record_batch(
                values,
                source_key,
                {**checkpoint, "phase": "active", "authoritative": 1},
                skip_invalid=True,
            )
        return pending

    def _attributed(self, value: UsageObservation, row: dict[str, Any]) -> UsageObservation:
        attributed = attribute_route(value, row.get("route_history", []))
        if attributed is not None:
            return replace(attributed, root_session_id=row.get("root_session_id"))
        pricing = dict(value.pricing_context)
        model = row.get("model")
        provider = value.provider
        if pricing.get("observed_provider") == "mandri" or (
            row.get("model_source") == "gateway"
            and isinstance(model, str)
            and "/" in model
            and value.model in {model, model.partition("/")[2]}
        ):
            pricing["attributed_source"] = "gateway"
            value = replace(value, pricing_context=pricing)
        if pricing.get("provider_kind"):
            return replace(
                value,
                provider=provider or str(pricing["provider_kind"]),
                root_session_id=row.get("root_session_id"),
            )
        if (
            not row.get("route_history")
            and row.get("model_source") != "native"
            and value.model
            and isinstance(model, str)
        ):
            alias, separator, upstream = model.partition("/")
            if separator and value.model in {model, upstream}:
                provider = alias
                pricing["provider_kind"] = {
                    "opencode-go": "opencode_go",
                    "opencode_go": "opencode_go",
                    "openrouter-free": "openrouter",
                    "openrouter": "openrouter",
                }.get(alias, alias)
                pricing["model_basis"] = "historical_model_matches_session_route"
                return replace(
                    value,
                    provider=provider,
                    model=upstream,
                    root_session_id=row.get("root_session_id"),
                    pricing_context=pricing,
                )
            pricing["provider_kind"] = None
        if (
            row.get("harness") == "claude"
            and row.get("model_source") == "native"
            and value.model
            and value.model.startswith("claude-")
        ):
            pricing["provider_kind"] = "anthropic"
            pricing["provider_basis"] = "canonical_model_api_equivalent"
        return replace(value, root_session_id=row.get("root_session_id"), pricing_context=pricing)

    async def _unavailable(self, key: str, session_id: str, stored: dict[str, Any] | None) -> None:
        checkpoint = {**(stored or {}), "session_id": session_id, "status": "unavailable"}
        await self._repository.record_batch([], key, checkpoint)

    async def _opencode_session(
        self,
        row: dict[str, Any],
        reference: SessionRef,
        source_key: str,
        stored: dict[str, Any] | None,
    ) -> bool:
        if self._opencode is None:
            return False
        session_id = str(row["id"])
        try:
            revision = json.dumps(await asyncio.to_thread(self._opencode.revision, reference))
        except (OSError, TranscriptError):
            await self._unavailable(source_key, session_id, stored)
            return False
        if (
            stored
            and stored.get("reconciliation_version") == RECONCILIATION_VERSION
            and stored.get("route_revision") == row["route_revision"]
            and stored.get("phase") == "active"
            and stored.get("source_revision") == revision
        ):
            return False
        reset = (
            stored is None
            or stored.get("phase") != "staging"
            or stored.get("reconciliation_version") != RECONCILIATION_VERSION
            or stored.get("route_revision") != row["route_revision"]
        )
        if not reset and stored is not None:
            revision = str(stored.get("source_revision", revision))
        offset = None if reset or stored is None else stored.get("offset")
        batch = await asyncio.to_thread(
            self._opencode.read_usage,
            reference,
            session_id=session_id,
            observed_at_ms=int(system_now_ms()),
            cursor=offset,
            root_session_id=row.get("root_session_id"),
            routing="native",
        )
        if batch.status == "unavailable":
            await self._unavailable(source_key, session_id, stored)
            return False
        values = [
            self._opencode_attributed(self._attributed(value, row)) for value in batch.observations
        ]
        checkpoint = {
            "session_id": session_id,
            "offset": batch.cursor,
            "source_revision": revision,
            "parser_version": 2,
            "reconciliation_version": RECONCILIATION_VERSION,
            "route_revision": row["route_revision"],
            "status": "backfill" if batch.has_more else batch.status,
        }
        await self._repository.stage_history(
            values, source_key, checkpoint, reset=reset, complete=not batch.has_more
        )
        return batch.has_more

    def _opencode_attributed(self, value: UsageObservation) -> UsageObservation:
        pricing = {**value.pricing_context, "evidence": "history_request"}
        if value.provider == "mandri":
            return replace(
                value, authoritative=False, pricing_context={**pricing, "routing": "gateway"}
            )
        return replace(value, pricing_context=pricing)
