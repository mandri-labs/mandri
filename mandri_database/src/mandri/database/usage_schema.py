USAGE_STATEMENTS = (
    "CREATE TABLE usage_revision (id INTEGER PRIMARY KEY CHECK(id=1), revision INTEGER NOT NULL)",
    "INSERT INTO usage_revision VALUES (1,0)",
    "CREATE TABLE usage_observation (source TEXT NOT NULL, source_key TEXT NOT NULL,"
    " session_id TEXT, payload TEXT NOT NULL, PRIMARY KEY(source,source_key))",
    "CREATE TABLE usage_fact (fact_key TEXT PRIMARY KEY, session_id TEXT, root_session_id TEXT,"
    " project_path TEXT, model TEXT, occurred_at INTEGER, payload TEXT NOT NULL,"
    " usd_equivalent TEXT, price_id TEXT)",
    "CREATE INDEX ix_usage_time ON usage_fact(occurred_at)",
    "CREATE INDEX ix_usage_session ON usage_fact(session_id,occurred_at)",
    "CREATE INDEX ix_usage_project ON usage_fact(project_path,occurred_at)",
    "CREATE INDEX ix_usage_root ON usage_fact(root_session_id,occurred_at)",
    "CREATE TABLE usage_baseline (source TEXT NOT NULL, epoch TEXT NOT NULL,"
    " session_id TEXT, payload TEXT NOT NULL, PRIMARY KEY(source,epoch))",
    "CREATE TABLE usage_suppression (session_id TEXT PRIMARY KEY)",
    "CREATE TABLE usage_source_suppression (identity TEXT PRIMARY KEY)",
    "CREATE TABLE usage_cursor (source_key TEXT PRIMARY KEY, session_id TEXT NOT NULL,"
    " payload TEXT NOT NULL)",
    "CREATE INDEX ix_usage_cursor_session ON usage_cursor(session_id)",
    "CREATE TABLE usage_gap (source_key TEXT NOT NULL, event_key TEXT NOT NULL,"
    " session_id TEXT NOT NULL, reason TEXT NOT NULL, PRIMARY KEY(source_key,event_key))",
    "CREATE TABLE usage_account (account_id TEXT PRIMARY KEY, observed_at INTEGER NOT NULL,"
    " payload TEXT NOT NULL)",
    "CREATE TABLE usage_price (price_id TEXT PRIMARY KEY, payload TEXT NOT NULL)",
)
