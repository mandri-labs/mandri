from pathlib import Path
from typing import Any

type SqlParams = tuple[Any, ...] | dict[str, Any]


class DatabasePort:
    async def connect(self, db_path: Path | str) -> None:
        raise NotImplementedError

    async def close(self) -> None:
        raise NotImplementedError

    async def migrate(self) -> None:
        raise NotImplementedError

    async def execute(self, sql: str, params: SqlParams = ()) -> None:
        raise NotImplementedError

    async def fetch_all(self, sql: str, params: SqlParams = ()) -> list[dict[str, Any]]:
        raise NotImplementedError

    async def fetch_one(self, sql: str, params: SqlParams = ()) -> dict[str, Any] | None:
        raise NotImplementedError
