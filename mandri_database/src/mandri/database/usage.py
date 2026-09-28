import hashlib
import json
import sqlite3
from collections.abc import Sequence
from dataclasses import replace
from typing import Any, cast

from mandri.core.clock import system_now_ms
from mandri.core.types.usage import UsageAccount, UsageFilters, UsageObservation, UsagePrice
from mandri.core.usage_normalization import normalize_usage
from mandri.core.usage_pricing import validate_price, value_usage
from mandri.database.sqlite_adapter import AiosqliteDatabase
from mandri.database.usage_accounting import advances, contribution, validate
from mandri.database.usage_coverage import select_coverage
from mandri.database.usage_prices import PriceIndex, prices_for
from mandri.database.usage_queries import advance, overview, revision
from mandri.database.usage_reconcile import stage_history
from mandri.database.usage_serialization import COUNTERS, encode, observation, price
from mandri.database.usage_transactions import transaction
from mandri.database.usage_valuation import revalue_batch


class UsageRepository:
    def __init__(self, database: AiosqliteDatabase) -> None:
        self._database = database

    async def revision(self) -> int:
        return await transaction(self._database, revision, write=False)

    async def has_gateway_usage(self, session_id: str, root_session_id: str | None = None) -> bool:
        return await transaction(
            self._database,
            lambda db: _has_gateway_usage(db, session_id, root_session_id),
            write=False,
        )

    async def record(self, value: UsageObservation) -> int:
        validate(value)
        return await transaction(self._database, lambda db: self._record(db, value), write=True)

    def _record(
        self,
        db: sqlite3.Connection,
        value: UsageObservation,
        prices: Sequence[UsagePrice] | None = None,
    ) -> int:
        value = normalize_usage(value)
        current = revision(db)
        if (
            value.source.startswith("native:")
            and value.pricing_context.get("evidence") != "history_request"
            and value.session_id is not None
            and db.execute(
                "SELECT 1 FROM usage_cursor WHERE session_id=?"
                " AND json_extract(payload,'$.authoritative')=1 LIMIT 1",
                (value.session_id,),
            ).fetchone()
        ):
            value = replace(value, authoritative=False)
        scopes = (value.session_id or value.root_session_id, *value.included_session_ids)
        if any(
            db.execute("SELECT 1 FROM usage_suppression WHERE session_id=?", (scope,)).fetchone()
            for scope in scopes
            if scope is not None
        ):
            return current
        if any(
            db.execute("SELECT 1 FROM usage_source_suppression WHERE identity=?", (key,)).fetchone()
            for key in _source_identities(value)
        ):
            return current
        if value.kind == "cumulative" and value.authoritative:
            prior = db.execute(
                "SELECT payload FROM usage_baseline WHERE source=? AND epoch=?",
                (value.source, value.epoch),
            ).fetchone()
            if prior and not advances(value, observation(prior[0])):
                return current
        old_row = db.execute(
            "SELECT payload FROM usage_observation WHERE source=? AND source_key=?",
            (value.source, value.source_key),
        ).fetchone()
        if old_row:
            old = observation(old_row[0])
            if value.sequence < old.sequence or encode(value) == old_row[0]:
                return current
            if value.sequence == old.sequence:
                raise ValueError("Conflicting usage observation at the same sequence")
            if value.fact_key != old.fact_key or value.session_id != old.session_id:
                raise ValueError("Usage source identity cannot change its attribution")
        fact = value if value.authoritative else None
        if value.kind == "cumulative" and value.authoritative:
            baseline_row = db.execute(
                "SELECT payload FROM usage_baseline WHERE source=? AND epoch=?",
                (value.source, value.epoch),
            ).fetchone()
            previous = observation(baseline_row[0]) if baseline_row else None
            if previous is not None and not advances(value, previous):
                return current
            if previous is not None and any(
                getattr(previous, name) is not None and getattr(value, name) is None
                for name in COUNTERS
            ):
                raise ValueError("Cumulative snapshot lost previously observed counters")
            fact = contribution(value, previous)
            db.execute(
                "INSERT INTO usage_baseline VALUES (?,?,?,?) ON CONFLICT(source,epoch)"
                " DO UPDATE SET payload=excluded.payload",
                (
                    value.source,
                    value.epoch,
                    value.session_id or value.root_session_id,
                    encode(value),
                ),
            )
            if fact is not None:
                identity = json.dumps((value.source, value.epoch, value.sequence, value.source_key))
                fact = replace(
                    fact, fact_key="cumulative:" + hashlib.sha256(identity.encode()).hexdigest()
                )
        db.execute(
            "INSERT INTO usage_observation VALUES (?,?,?,?) ON CONFLICT(source,source_key)"
            " DO UPDATE SET payload=excluded.payload",
            (
                value.source,
                value.source_key,
                value.session_id or value.root_session_id,
                encode(value),
            ),
        )
        if fact is not None and select_coverage(db, fact):
            self._save_fact(db, fact, prices)
        elif fact is None and value.kind == "delta":
            db.execute(
                "DELETE FROM usage_fact WHERE fact_key=?"
                " AND json_extract(payload,'$.source')=?"
                " AND json_extract(payload,'$.source_key')=?",
                (value.fact_key, value.source, value.source_key),
            )
        return advance(db)

    def _save_fact(
        self,
        db: sqlite3.Connection,
        fact: UsageObservation,
        prices: Sequence[UsagePrice] | None = None,
    ) -> None:
        existing = db.execute(
            "SELECT payload,price_id,model FROM usage_fact WHERE fact_key=?", (fact.fact_key,)
        ).fetchone()
        if existing:
            old = observation(existing["payload"])
            if old.source != fact.source or old.source_key != fact.source_key:
                return
            if old.complete and not fact.complete:
                return
            merged = {}
            for name in COUNTERS:
                before, after = getattr(old, name), getattr(fact, name)
                if before is not None and after is not None and after < before:
                    raise ValueError("Usage replacement counters decreased")
                merged[name] = before if after is None else after
            immutable = {
                name: getattr(old, name)
                for name in (
                    "session_id",
                    "root_session_id",
                    "project_path",
                    "harness",
                    "provider",
                    "billing_mode",
                    "account_id",
                    "occurred_at",
                    "interval_start",
                )
            }
            fact = replace(
                fact,
                **merged,
                **immutable,
                model=old.model or fact.model,
            )
        if prices is None:
            prices = prices_for(db, fact)
        fact = normalize_usage(fact)
        if existing and existing["price_id"] and existing["model"] == fact.model:
            prices = [item for item in prices if item.price_id == existing["price_id"]]
        amount, price_id = value_usage(fact, prices)
        db.execute(
            "INSERT INTO usage_fact VALUES (?,?,?,?,?,?,?,?,?) ON CONFLICT(fact_key) DO UPDATE SET"
            " payload=excluded.payload,usd_equivalent=excluded.usd_equivalent,"
            "price_id=excluded.price_id,model=excluded.model",
            (
                fact.fact_key,
                fact.session_id,
                fact.root_session_id,
                fact.project_path,
                fact.model,
                fact.occurred_at,
                encode(fact),
                str(amount) if amount is not None else None,
                price_id,
            ),
        )

    async def overview(self, filters: UsageFilters | None = None) -> dict[str, Any]:
        return await transaction(
            self._database, lambda db: overview(db, filters or UsageFilters()), write=False
        )

    async def accounts(self) -> list[dict[str, Any]]:
        return cast(list[dict[str, Any]], (await self.account_snapshot())["accounts"])

    async def read_cursor(self, source_key: str) -> dict[str, Any] | None:
        def read(db: sqlite3.Connection) -> dict[str, Any] | None:
            row = db.execute(
                "SELECT payload FROM usage_cursor WHERE source_key=?", (source_key,)
            ).fetchone()
            return cast(dict[str, Any], json.loads(row[0])) if row else None

        return await transaction(self._database, read, write=False)

    async def record_batch(
        self,
        observations: Sequence[UsageObservation],
        source_key: str,
        cursor: dict[str, Any],
        *,
        skip_invalid: bool = False,
    ) -> int:
        allowed = {
            "session_id",
            "identity",
            "offset",
            "anchor",
            "parser_version",
            "reconciliation_version",
            "status",
            "source_revision",
            "modified",
            "size",
            "state_json",
            "authoritative",
            "phase",
        }
        if not source_key or set(cursor) - allowed or not isinstance(cursor.get("session_id"), str):
            raise ValueError("Invalid usage cursor fields or session scope")
        if not cursor["session_id"] or len(observations) > 1000:
            raise ValueError("Usage batch requires a session and at most 1000 observations")
        if any(not isinstance(value, (str, int, type(None))) for value in cursor.values()):
            raise ValueError("Usage cursor values must be scalar")
        payload = json.dumps(cursor, sort_keys=True, separators=(",", ":"))
        if len(payload) > 65536:
            raise ValueError("Usage cursor is too large")
        for value in observations:
            if not skip_invalid:
                validate(value)
            if value.session_id != cursor["session_id"]:
                raise ValueError("Usage cursor and observation session scopes differ")

        def save(db: sqlite3.Connection) -> int:
            if db.execute(
                "SELECT 1 FROM usage_suppression WHERE session_id=?", (cursor["session_id"],)
            ).fetchone():
                return revision(db)
            rejected = False
            new_gap = False
            prices = PriceIndex()
            for value in observations:
                db.execute("SAVEPOINT usage_event")
                try:
                    validate(value)
                    self._record(db, value, prices.for_fact(db, normalize_usage(value)))
                except ValueError as error:
                    db.execute("ROLLBACK TO usage_event")
                    if not skip_invalid:
                        raise
                    event_key = hashlib.sha256(
                        json.dumps((value.source, value.source_key, value.sequence)).encode()
                    ).hexdigest()
                    inserted = db.execute(
                        "INSERT OR IGNORE INTO usage_gap VALUES (?,?,?,?)",
                        (source_key, event_key, cursor["session_id"], str(error)),
                    )
                    new_gap = new_gap or inserted.rowcount > 0
                    rejected = True
                finally:
                    db.execute("RELEASE usage_event")
            saved_payload = (
                json.dumps({**cursor, "status": "partial"}, sort_keys=True, separators=(",", ":"))
                if rejected
                else payload
            )
            old = db.execute(
                "SELECT payload,session_id FROM usage_cursor WHERE source_key=?", (source_key,)
            ).fetchone()
            if old and old[1] != cursor["session_id"]:
                raise ValueError("Usage cursor cannot change its session scope")
            if old and old[0] == saved_payload:
                return advance(db) if new_gap else revision(db)
            db.execute(
                "INSERT INTO usage_cursor VALUES (?,?,?) ON CONFLICT(source_key)"
                " DO UPDATE SET payload=excluded.payload",
                (source_key, cursor["session_id"], saved_payload),
            )
            if old and not new_gap:
                before, after = json.loads(old[0]), json.loads(saved_payload)
                fields = ("status", "phase", "authoritative")
                previous_reasons = json.loads(before.get("state_json", "{}")).get(
                    "discard_reasons", {}
                )
                current_reasons = json.loads(after.get("state_json", "{}")).get(
                    "discard_reasons", {}
                )
                if (
                    all(before.get(key) == after.get(key) for key in fields)
                    and previous_reasons == current_reasons
                ):
                    return revision(db)
            return advance(db)

        return await transaction(self._database, save, write=True)

    async def revalue_unpriced(
        self, *, limit: int = 1000, after_key: str | None = None, all_facts: bool = False
    ) -> dict[str, Any]:
        return await revalue_batch(self._database, limit, after_key, all_facts)

    async def revalue_pending(self, *, limit: int = 100) -> dict[str, Any]:
        state = await self._database.fetch_one(
            "SELECT catalog_revision,completed_revision,after_key FROM usage_valuation WHERE id=1"
        )
        assert state is not None
        if state["catalog_revision"] == state["completed_revision"]:
            return {"next_key": None, "updated": 0}
        result = await self.revalue_unpriced(
            limit=limit, after_key=state["after_key"], all_facts=True
        )
        await self._database.execute(
            "UPDATE usage_valuation SET after_key=?,"
            " completed_revision=CASE WHEN ? IS NULL THEN catalog_revision"
            " ELSE completed_revision END WHERE id=1 AND catalog_revision=?",
            (result["next_key"], result["next_key"], state["catalog_revision"]),
        )
        return result

    async def account_snapshot(self) -> dict[str, Any]:
        def read(db: sqlite3.Connection) -> dict[str, Any]:
            accounts = [
                json.loads(row[0])
                for row in db.execute("SELECT payload FROM usage_account ORDER BY account_id")
            ]
            return {
                "revision": revision(db),
                "as_of": system_now_ms(),
                "accounts": [
                    account for account in accounts if UsageAccount(**account).has_account_data
                ],
            }

        return await transaction(self._database, read, write=False)

    async def upsert_account(self, account: UsageAccount) -> int:
        if not account.account_id or account.observed_at < 0:
            raise ValueError("Invalid usage account")

        def save(db: sqlite3.Connection) -> int:
            if not account.has_account_data:
                return revision(db)
            row = db.execute(
                "SELECT payload,observed_at FROM usage_account WHERE account_id=?",
                (account.account_id,),
            ).fetchone()
            payload = encode(account)
            if row and (row[0] == payload or row[1] >= account.observed_at):
                return revision(db)
            db.execute(
                "INSERT INTO usage_account VALUES (?,?,?) ON CONFLICT(account_id)"
                " DO UPDATE SET observed_at=excluded.observed_at,payload=excluded.payload",
                (account.account_id, account.observed_at, payload),
            )
            return advance(db)

        return await transaction(self._database, save, write=True)

    async def add_price(self, value: UsagePrice) -> int:
        return await self.add_prices([value])

    async def add_prices(self, values: Sequence[UsagePrice]) -> int:
        for value in values:
            validate_price(value)

        def save(db: sqlite3.Connection) -> int:
            changed = False
            for value in values:
                payload = encode(value)
                row = db.execute(
                    "SELECT payload FROM usage_price WHERE price_id=?",
                    (value.price_id,),
                ).fetchone()
                if row:
                    if row[0] != payload and encode(price(row[0])) != payload:
                        raise ValueError("Price versions are immutable")
                    continue
                db.execute("INSERT INTO usage_price VALUES (?,?)", (value.price_id, payload))
                changed = True
            if changed:
                db.execute(
                    "UPDATE usage_valuation SET catalog_revision=catalog_revision+1,"
                    " after_key=NULL WHERE id=1"
                )
            return advance(db) if changed else revision(db)

        return await transaction(self._database, save, write=True)

    async def catalog_state(self, source: str = "public") -> dict[str, Any] | None:
        def read(db: sqlite3.Connection) -> dict[str, Any] | None:
            row = db.execute(
                "SELECT payload FROM usage_catalog_state WHERE source=?", (source,)
            ).fetchone()
            return cast(dict[str, Any], json.loads(row[0])) if row else None

        return await transaction(self._database, read, write=False)

    async def save_catalog_state(self, value: dict[str, Any], source: str = "public") -> int:
        def save(db: sqlite3.Connection) -> int:
            db.execute(
                "INSERT INTO usage_catalog_state VALUES (?,?) ON CONFLICT(source)"
                " DO UPDATE SET payload=excluded.payload",
                (source, json.dumps(value)),
            )
            return advance(db)

        return await transaction(self._database, save, write=True)

    async def stage_history(
        self,
        values: Sequence[UsageObservation],
        source_key: str,
        cursor: dict[str, Any],
        *,
        reset: bool = False,
        complete: bool = False,
    ) -> int:
        return await transaction(
            self._database,
            lambda db: stage_history(
                db,
                values,
                source_key,
                cursor,
                self._record,
                reset=reset,
                complete=complete,
            ),
            write=True,
        )

    async def erase_session(self, session_id: str) -> int:
        if not session_id:
            raise ValueError("Session identity is required")

        def erase(db: sqlite3.Connection) -> int:
            for row in db.execute("SELECT session_id,payload FROM usage_observation"):
                if (
                    row[0] != session_id
                    and session_id in json.loads(row[1])["included_session_ids"]
                ):
                    raise ValueError("Erasure requires the enclosing aggregate session scope")
            if db.execute(
                "SELECT 1 FROM usage_suppression WHERE session_id=?", (session_id,)
            ).fetchone():
                return revision(db)
            db.execute("INSERT INTO usage_suppression VALUES (?)", (session_id,))
            for row in db.execute(
                "SELECT payload FROM usage_observation WHERE session_id=?"
                " UNION ALL SELECT payload FROM usage_history_stage WHERE session_id=?",
                (session_id, session_id),
            ):
                for key in _source_identities(observation(row[0])):
                    db.execute("INSERT OR IGNORE INTO usage_source_suppression VALUES (?)", (key,))
            db.execute(
                "DELETE FROM usage_fact WHERE session_id IS NULL AND root_session_id=?",
                (session_id,),
            )
            for table in (
                "usage_observation",
                "usage_baseline",
                "usage_fact",
                "usage_cursor",
                "usage_gap",
                "usage_history_stage",
            ):
                db.execute(f"DELETE FROM {table} WHERE session_id=?", (session_id,))
            return advance(db)

        return await transaction(self._database, erase, write=True)


def _source_identities(value: UsageObservation) -> tuple[str, ...]:
    identifiers = {
        "record": value.source_key,
        "epoch": value.epoch,
        "native_session": value.native_session_id,
    }
    return tuple(
        hashlib.sha256(json.dumps((value.source, kind, identity)).encode()).hexdigest()
        for kind, identity in identifiers.items()
        if identity is not None
    )


def _has_gateway_usage(
    db: sqlite3.Connection, session_id: str | None, root_session_id: str | None
) -> bool:
    ids = tuple({key for key in (session_id, root_session_id) if key is not None})
    if not ids:
        return False
    placeholders = ",".join("?" for _ in ids)
    return (
        db.execute(
            "SELECT 1 FROM usage_fact WHERE source='gateway' AND "
            f"(session_id IN ({placeholders}) OR root_session_id IN ({placeholders})) LIMIT 1",
            (*ids, *ids),
        ).fetchone()
        is not None
    )
