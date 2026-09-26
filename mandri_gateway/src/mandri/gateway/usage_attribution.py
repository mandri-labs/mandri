from dataclasses import dataclass

from mandri.core.ports.database import DatabasePort


@dataclass(frozen=True)
class UsageAttribution:
    root_session_id: str | None = None
    project_path: str | None = None
    harness: str | None = None


async def route_attribution(db: DatabasePort, route_id: str) -> UsageAttribution:
    bindings = await db.fetch_all(
        "SELECT id, parent_session_id, project_path, harness"
        " FROM session WHERE gateway_route_id = ?",
        (route_id,),
    )
    if not bindings:
        return UsageAttribution()
    roots: set[str] = set()
    projects: set[str] = set()
    harnesses: set[str] = set()
    for binding in bindings:
        row = binding
        seen: set[str] = set()
        harnesses.add(str(row["harness"]))
        while True:
            identity = str(row["id"])
            if identity in seen or len(seen) >= 64:
                return UsageAttribution()
            seen.add(identity)
            projects.add(str(row["project_path"]))
            parent = row.get("parent_session_id")
            if parent is None:
                roots.add(identity)
                break
            ancestor = await db.fetch_one(
                "SELECT id, parent_session_id, project_path, harness FROM session WHERE id = ?",
                (str(parent),),
            )
            if ancestor is None:
                return UsageAttribution()
            row = ancestor
    if len(roots) != 1:
        return UsageAttribution()
    root = next(iter(roots))
    descendants = await db.fetch_all(
        "WITH RECURSIVE tree(id) AS (SELECT id FROM session WHERE id = ?"
        " UNION SELECT s.id FROM session s JOIN tree t ON s.parent_session_id = t.id)"
        " SELECT project_path, harness FROM session WHERE id IN (SELECT id FROM tree)",
        (root,),
    )
    projects.update(str(row["project_path"]) for row in descendants)
    harnesses.update(str(row["harness"]) for row in descendants)
    return UsageAttribution(
        root_session_id=root,
        project_path=next(iter(projects)) if len(projects) == 1 else None,
        harness=next(iter(harnesses)) if len(harnesses) == 1 else None,
    )
