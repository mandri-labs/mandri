import sqlite3

import aiosqlite
from mandri.database.errors import MigrationError
from mandri.database.usage_schema import USAGE_STATEMENTS

USAGE_TABLES = (
    "usage_revision",
    "usage_observation",
    "usage_fact",
    "usage_baseline",
    "usage_suppression",
    "usage_source_suppression",
    "usage_cursor",
    "usage_gap",
    "usage_account",
    "usage_price",
    "usage_history_stage",
    "usage_catalog_state",
)

FACT_FIELDS = (
    "source",
    "harness",
    "provider",
    "billing_mode",
    "interval_start",
    "observed_at",
    "complete",
    "reported_cost_usd",
)
TOKEN_FIELDS = (
    "input_tokens",
    "output_tokens",
    "cache_read_tokens",
    "cache_write_tokens",
    "reasoning_tokens",
    "total_tokens",
    "request_count",
)

USAGE_INITIAL = (
    *USAGE_STATEMENTS,
    "CREATE TABLE usage_history_stage (source_key TEXT NOT NULL, event_key TEXT NOT NULL,"
    " session_id TEXT NOT NULL, payload TEXT NOT NULL, PRIMARY KEY(source_key,event_key))",
    "CREATE INDEX ix_usage_stage_session ON usage_history_stage(session_id)",
    "CREATE TABLE usage_catalog_state (source TEXT PRIMARY KEY, payload TEXT NOT NULL)",
    "CREATE TABLE usage_valuation (id INTEGER PRIMARY KEY CHECK(id=1),"
    " catalog_revision INTEGER NOT NULL, completed_revision INTEGER NOT NULL, after_key TEXT)",
    "INSERT INTO usage_valuation VALUES (1,0,0,NULL)",
    *(
        f"ALTER TABLE usage_fact ADD COLUMN {field} GENERATED ALWAYS AS"
        f" (json_extract(payload,'$.{field}')) STORED"
        for field in FACT_FIELDS
    ),
    *(
        f"ALTER TABLE usage_fact ADD COLUMN {field} TEXT GENERATED ALWAYS AS"
        f" (NULLIF(payload -> '$.{field}', 'null')) STORED"
        for field in TOKEN_FIELDS
    ),
    "ALTER TABLE usage_observation ADD COLUMN root_session_id TEXT GENERATED ALWAYS AS"
    " (json_extract(payload,'$.root_session_id')) STORED",
    "ALTER TABLE usage_price ADD COLUMN provider TEXT GENERATED ALWAYS AS"
    " (json_extract(payload,'$.provider')) STORED",
    "ALTER TABLE usage_price ADD COLUMN model TEXT GENERATED ALWAYS AS"
    " (json_extract(payload,'$.model')) STORED",
    "ALTER TABLE usage_price ADD COLUMN valuation_basis TEXT GENERATED ALWAYS AS"
    " (json_extract(payload,'$.valuation_basis')) STORED",
    "CREATE INDEX ix_usage_observation_session ON usage_observation(session_id,source)",
    "CREATE INDEX ix_usage_observation_root ON usage_observation(root_session_id,source)",
    "CREATE INDEX ix_usage_gateway_session ON usage_fact(session_id) WHERE source='gateway'",
    "CREATE INDEX ix_usage_gateway_root ON usage_fact(root_session_id) WHERE source='gateway'",
    "CREATE INDEX ix_usage_unpriced ON usage_fact(fact_key) WHERE price_id IS NULL",
    "CREATE INDEX ix_usage_model_time ON usage_fact(model,occurred_at)",
    "CREATE INDEX ix_usage_harness_time ON usage_fact(harness,occurred_at)",
    "CREATE INDEX ix_usage_provider_time ON usage_fact(provider,occurred_at)",
    "CREATE INDEX ix_usage_price_model ON usage_price(model,provider)",
    "CREATE INDEX ix_usage_gap_session ON usage_gap(session_id)",
    "CREATE INDEX ix_usage_baseline_session ON usage_baseline(session_id)",
)


async def migrate_usage(connection: aiosqlite.Connection) -> None:
    async with connection.execute(
        "CREATE TABLE IF NOT EXISTS usage_schema (version INTEGER NOT NULL)"
    ):
        pass
    async with connection.execute("SELECT version FROM usage_schema") as cursor:
        row = await cursor.fetchone()
    if row is not None:
        if row[0] != 1:
            raise MigrationError(f"Unsupported usage schema {row[0]}")
        return
    try:
        await connection.execute("BEGIN IMMEDIATE")
        for statement in USAGE_INITIAL:
            async with connection.execute(statement):
                pass
        await connection.execute("INSERT INTO usage_schema VALUES (1)")
        await connection.commit()
    except sqlite3.Error as error:
        await connection.rollback()
        raise MigrationError(f"Usage migration failed: {error}") from error
