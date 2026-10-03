import sqlite3
from typing import NamedTuple

import aiosqlite
from mandri.database.errors import MigrationError
from mandri.database.usage_migrations import USAGE_TABLES
from mandri.database.usage_schema import USAGE_STATEMENTS

LATEST_VERSION = 19


class Migration(NamedTuple):
    version: int
    statements: tuple[str, ...]


MIGRATIONS: tuple[Migration, ...] = (
    Migration(
        version=1,
        statements=(
            """
            CREATE TABLE session (
              id TEXT PRIMARY KEY,
              harness TEXT NOT NULL,
              native_id TEXT UNIQUE,
              native_title TEXT,
              title_overlay TEXT,
              project_path TEXT NOT NULL,
              created_at INTEGER NOT NULL,
              updated_at INTEGER NOT NULL,
              state TEXT NOT NULL,
              model_id TEXT REFERENCES model(id),
              gateway_route_id TEXT,
              deleted INTEGER NOT NULL DEFAULT 0,
              last_synced_at INTEGER NOT NULL
            )
            """,
            """
            CREATE UNIQUE INDEX ux_session_harness_native
              ON session(harness, native_id) WHERE native_id IS NOT NULL AND deleted = 0
            """,
            """
            CREATE INDEX ix_session_state ON session(state) WHERE deleted = 0
            """,
            """
            CREATE TABLE model (
              id TEXT PRIMARY KEY,
              name TEXT NOT NULL UNIQUE,
              provider TEXT NOT NULL,
              model_ref TEXT NOT NULL,
              api_base TEXT,
              api_key TEXT NOT NULL,
              state TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE gateway_route (
              id TEXT PRIMARY KEY,
              model_id TEXT NOT NULL REFERENCES model(id),
              formats TEXT NOT NULL,
              created_at INTEGER NOT NULL
            )
            """,
        ),
    ),
    Migration(
        version=2,
        statements=(
            """
            CREATE TABLE session_v2 (
              id TEXT PRIMARY KEY,
              harness TEXT NOT NULL,
              native_id TEXT,
              native_title TEXT,
              title_overlay TEXT,
              project_path TEXT NOT NULL,
              created_at INTEGER NOT NULL,
              updated_at INTEGER NOT NULL,
              state TEXT NOT NULL,
              model_id TEXT REFERENCES model(id),
              gateway_route_id TEXT,
              deleted INTEGER NOT NULL DEFAULT 0,
              last_synced_at INTEGER NOT NULL,
              interaction_mode TEXT
            )
            """,
            """
            INSERT INTO session_v2 (
              id, harness, native_id, native_title, title_overlay, project_path,
              created_at, updated_at, state, model_id, gateway_route_id,
              deleted, last_synced_at
            )
            SELECT
              id, harness, native_id, native_title, title_overlay, project_path,
              created_at, updated_at, state, model_id, gateway_route_id,
              deleted, last_synced_at
            FROM session
            """,
            """
            DROP TABLE session
            """,
            """
            ALTER TABLE session_v2 RENAME TO session
            """,
            """
            CREATE UNIQUE INDEX ux_session_harness_native
              ON session(harness, native_id) WHERE native_id IS NOT NULL AND deleted = 0
            """,
            """
            CREATE INDEX ix_session_state ON session(state) WHERE deleted = 0
            """,
        ),
    ),
    Migration(
        version=3,
        statements=(
            """
            PRAGMA foreign_keys = OFF
            """,
            """
            CREATE TABLE session_v3 (
              id TEXT PRIMARY KEY,
              harness TEXT NOT NULL,
              native_id TEXT,
              native_title TEXT,
              title_overlay TEXT,
              project_path TEXT NOT NULL,
              created_at INTEGER NOT NULL,
              updated_at INTEGER NOT NULL,
              state TEXT NOT NULL,
              model TEXT,
              gateway_route_id TEXT,
              deleted INTEGER NOT NULL DEFAULT 0,
              last_synced_at INTEGER NOT NULL,
              interaction_mode TEXT
            )
            """,
            """
            INSERT INTO session_v3 (
              id, harness, native_id, native_title, title_overlay, project_path,
              created_at, updated_at, state, model, gateway_route_id,
              deleted, last_synced_at, interaction_mode
            )
            SELECT
              s.id, s.harness, s.native_id, s.native_title, s.title_overlay, s.project_path,
              s.created_at, s.updated_at, s.state, m.name, s.gateway_route_id,
              s.deleted, s.last_synced_at, s.interaction_mode
            FROM session s
            LEFT JOIN model m ON m.id = s.model_id
            """,
            """
            DROP TABLE session
            """,
            """
            ALTER TABLE session_v3 RENAME TO session
            """,
            """
            CREATE UNIQUE INDEX ux_session_harness_native
              ON session(harness, native_id) WHERE native_id IS NOT NULL AND deleted = 0
            """,
            """
            CREATE INDEX ix_session_state ON session(state) WHERE deleted = 0
            """,
            """
            CREATE TABLE gateway_route_new (
              id TEXT PRIMARY KEY,
              provider_name TEXT NOT NULL,
              model_ref TEXT NOT NULL,
              formats TEXT NOT NULL,
              created_at INTEGER NOT NULL
            )
            """,
            """
            INSERT INTO gateway_route_new (id, provider_name, model_ref, formats, created_at)
            SELECT r.id, m.provider, m.model_ref, r.formats, r.created_at
            FROM gateway_route r
            JOIN model m ON m.id = r.model_id
            """,
            """
            DROP TABLE gateway_route
            """,
            """
            ALTER TABLE gateway_route_new RENAME TO gateway_route
            """,
            """
            DROP TABLE model
            """,
            """
            PRAGMA foreign_keys = ON
            """,
        ),
    ),
    Migration(
        version=4,
        statements=(
            """
            ALTER TABLE gateway_route ADD COLUMN reasoning_effort TEXT
            """,
            """
            ALTER TABLE session ADD COLUMN reasoning_effort TEXT
            """,
        ),
    ),
    Migration(
        version=5,
        statements=(
            "ALTER TABLE session ADD COLUMN model_source TEXT NOT NULL DEFAULT 'gateway'"
            " CHECK (model_source IN ('gateway', 'native'))",
        ),
    ),
    Migration(
        version=6,
        statements=(
            "UPDATE session SET model_source = 'native'"
            " WHERE harness IN ('codex', 'claude') AND native_id IS NOT NULL"
            " AND gateway_route_id IS NULL AND model_source = 'gateway'"
            " AND (model IS NULL OR instr(model, '/') = 0)",
        ),
    ),
    Migration(
        version=7,
        statements=(
            """
            CREATE TABLE agent (
              id TEXT PRIMARY KEY,
              parent_session_id TEXT NOT NULL,
              parent_agent_id TEXT,
              session_id TEXT,
              harness TEXT NOT NULL,
              native_id TEXT NOT NULL,
              title TEXT NOT NULL,
              state TEXT NOT NULL,
              created_at INTEGER NOT NULL,
              updated_at INTEGER NOT NULL,
              delegation_id TEXT,
              task_id TEXT,
              transcript_path TEXT
            )
            """,
            "CREATE INDEX ix_agent_parent ON agent(parent_session_id, parent_agent_id)",
        ),
    ),
    Migration(
        version=8,
        statements=(
            "CREATE TABLE agent_classification (session_id TEXT PRIMARY KEY,"
            " harness TEXT NOT NULL, native_id TEXT)",
            "CREATE TABLE agent_cache (session_id TEXT PRIMARY KEY,"
            " revision TEXT NOT NULL, completed_revision TEXT)",
        ),
    ),
    Migration(
        version=9,
        statements=(
            "ALTER TABLE session ADD COLUMN execution_backend TEXT NOT NULL DEFAULT 'host'"
            " CHECK(execution_backend IN ('host', 'docker'))",
            "ALTER TABLE session ADD COLUMN privacy_mode TEXT NOT NULL DEFAULT 'none'"
            " CHECK(privacy_mode IN ('none', 'surrogate'))",
            "ALTER TABLE session ADD COLUMN privacy_scope_id TEXT",
            "ALTER TABLE session ADD COLUMN execution_context TEXT",
            "ALTER TABLE session ADD COLUMN policy_revision INTEGER NOT NULL DEFAULT 1",
            "ALTER TABLE gateway_route ADD COLUMN execution_backend TEXT NOT NULL DEFAULT 'host'"
            " CHECK(execution_backend IN ('host', 'docker'))",
            "ALTER TABLE gateway_route ADD COLUMN privacy_mode TEXT NOT NULL DEFAULT 'none'"
            " CHECK(privacy_mode IN ('none', 'surrogate'))",
            "ALTER TABLE gateway_route ADD COLUMN privacy_scope_id TEXT",
            "CREATE TABLE privacy_scope (id TEXT PRIMARY KEY, version INTEGER NOT NULL,"
            " revision INTEGER NOT NULL, wrapped_key BLOB NOT NULL, payload BLOB NOT NULL,"
            " created_at INTEGER NOT NULL, updated_at INTEGER NOT NULL)",
            "CREATE INDEX ix_session_privacy_scope ON session(privacy_scope_id)",
            "CREATE INDEX ix_route_privacy_scope ON gateway_route(privacy_scope_id)",
            "CREATE TABLE execution_generation (session_id TEXT NOT NULL,"
            " generation INTEGER NOT NULL, owner TEXT NOT NULL, phase TEXT NOT NULL,"
            " container_id TEXT, revision INTEGER NOT NULL, reason TEXT, context TEXT NOT NULL,"
            " created_at INTEGER NOT NULL, updated_at INTEGER NOT NULL,"
            " PRIMARY KEY(session_id, generation))",
            "CREATE TABLE native_auth_binding (id TEXT PRIMARY KEY, harness TEXT NOT NULL,"
            " method TEXT NOT NULL, environment TEXT NOT NULL, credential_ref TEXT NOT NULL,"
            " revision INTEGER NOT NULL, state TEXT NOT NULL, owner TEXT,"
            " updated_at INTEGER NOT NULL)",
        ),
    ),
    Migration(
        version=10,
        statements=(
            "CREATE TABLE session_purge (session_id TEXT PRIMARY KEY,"
            " native_id TEXT, gateway_route_id TEXT, privacy_scope_id TEXT,"
            " execution_context TEXT)",
        ),
    ),
    Migration(
        version=11,
        statements=(
            "ALTER TABLE session ADD COLUMN parent_native_id TEXT",
            "ALTER TABLE session ADD COLUMN parent_session_id TEXT",
            "CREATE INDEX ix_session_parent ON session(parent_session_id)",
        ),
    ),
    Migration(
        version=12,
        statements=("DROP TABLE IF EXISTS native_auth_binding",),
    ),
    Migration(version=13, statements=USAGE_STATEMENTS),
    Migration(
        version=14,
        statements=(
            "CREATE TABLE usage_history_stage (source_key TEXT NOT NULL, event_key TEXT NOT NULL,"
            " session_id TEXT NOT NULL, payload TEXT NOT NULL, PRIMARY KEY(source_key,event_key))",
            "CREATE INDEX ix_usage_stage_session ON usage_history_stage(session_id)",
            "CREATE TABLE usage_catalog_state (source TEXT PRIMARY KEY, payload TEXT NOT NULL)",
        ),
    ),
    Migration(version=15, statements=("ALTER TABLE session ADD COLUMN worktree TEXT",)),
    Migration(
        version=16,
        statements=(
            *(f"DROP TABLE IF EXISTS {table}" for table in USAGE_TABLES),
            "CREATE INDEX ix_session_updated ON session(updated_at DESC) WHERE deleted=0",
            "CREATE INDEX ix_session_project_updated ON session(project_path,updated_at DESC)"
            " WHERE deleted=0",
            "CREATE INDEX ix_session_sync ON session(harness,id)"
            " WHERE native_id IS NOT NULL AND execution_backend='host'",
        ),
    ),
    Migration(
        version=17,
        statements=(
            "CREATE TABLE gateway_route_history (id INTEGER PRIMARY KEY, route_id TEXT NOT NULL,"
            " effective_from INTEGER NOT NULL, provider_name TEXT NOT NULL,"
            " provider_kind TEXT NOT NULL, model_ref TEXT NOT NULL)",
            "CREATE INDEX ix_gateway_route_history ON gateway_route_history(route_id,"
            " effective_from,id)",
        ),
    ),
    Migration(
        version=18,
        statements=(
            "CREATE TABLE conversation_status (target TEXT PRIMARY KEY,"
            " summary TEXT NOT NULL, observed TEXT NOT NULL)",
            "CREATE TABLE conversation_source (target TEXT NOT NULL,source TEXT NOT NULL,"
            " checkpoint TEXT NOT NULL,PRIMARY KEY(target,source))",
            "CREATE TABLE conversation_completion (target TEXT NOT NULL,event_key TEXT NOT NULL,"
            " revision INTEGER,PRIMARY KEY(target,event_key))",
        ),
    ),
    Migration(
        version=19,
        statements=(
            "ALTER TABLE session ADD COLUMN privacy_override INTEGER NOT NULL DEFAULT 0",
        ),
    ),
)


async def apply_migrations(connection: aiosqlite.Connection) -> None:
    cursor = await connection.execute("PRAGMA user_version")
    row = await cursor.fetchone()
    current = row[0] if row is not None else 0
    if current > LATEST_VERSION:
        raise MigrationError(f"database schema {current} is newer than supported {LATEST_VERSION}")
    for migration in MIGRATIONS:
        if migration.version <= current:
            continue
        try:
            await connection.execute("BEGIN")
            for statement in migration.statements:
                await connection.execute(statement)
            await connection.execute(f"PRAGMA user_version = {migration.version}")
            await connection.commit()
        except sqlite3.Error as error:
            await connection.rollback()
            raise MigrationError(f"migration {migration.version} failed: {error}") from error
