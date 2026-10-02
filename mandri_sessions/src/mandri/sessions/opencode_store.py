import sqlite3


def session_tables(connection: sqlite3.Connection) -> list[str]:
    names = {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
            " AND name IN ('session', 'session_v2')"
        )
    }
    return [name for name in ("session", "session_v2") if name in names]


def message_table(connection: sqlite3.Connection, session_id: str) -> str:
    if (
        "session_v2" in session_tables(connection)
        and connection.execute("SELECT 1 FROM session_v2 WHERE id = ?", (session_id,)).fetchone()
    ):
        return "session_message"
    return "message"
