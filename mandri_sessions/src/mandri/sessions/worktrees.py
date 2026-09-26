import asyncio
import dataclasses
import json
from collections.abc import Awaitable
from pathlib import Path
from weakref import WeakValueDictionary

from mandri.core.ids import SessionId
from mandri.core.ports.database import DatabasePort
from mandri.core.types.execution import ProtectionError
from mandri.core.types.worktree_integration import IntegrationPreview, IntegrationStrategy
from mandri.core.types.worktrees import Worktree
from mandri.sessions import worktree_git, worktree_integration
from mandri.sessions.errors import SessionNotFoundError


async def complete_operation[T](operation: Awaitable[T]) -> T:
    task = asyncio.ensure_future(operation)
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        await task
        raise


class SessionWorktrees:
    def __init__(self, db: DatabasePort, root: Path) -> None:
        self._db = db
        self._root = root
        self._lock = asyncio.Lock()
        self._leases: WeakValueDictionary[str, asyncio.Lock] = WeakValueDictionary()

    def lease(self, session_id: str) -> asyncio.Lock:
        return self._leases.setdefault(session_id, asyncio.Lock())

    async def _save(self, session_id: SessionId, worktree: Worktree) -> None:
        row = await self._db.fetch_one(
            "UPDATE session SET worktree = ?, project_path = ?"
            " WHERE id = ? AND deleted = 0 RETURNING id",
            (
                json.dumps(dataclasses.asdict(worktree)),
                str(Path(worktree.path) / worktree.relative_path),
                str(session_id),
            ),
        )
        if row is None:
            raise SessionNotFoundError(f"unknown session {session_id}")

    async def prepare(self, session_id: SessionId, cwd: str, name: str | None) -> Worktree:
        async with self._lock:
            return await complete_operation(self._prepare(session_id, cwd, name))

    async def _prepare(self, session_id: SessionId, cwd: str, name: str | None) -> Worktree:
        worktree = await asyncio.to_thread(
            worktree_git.plan, cwd, self._root, str(session_id), name
        )
        await self._save(session_id, worktree)
        await asyncio.to_thread(
            worktree_git.git,
            worktree.repository,
            "worktree",
            "add",
            "--detach",
            worktree.path,
            worktree.base_commit,
        )
        while True:
            if not await asyncio.to_thread(worktree_git.name_taken, worktree, worktree.id):
                result = await asyncio.to_thread(
                    worktree_git.git, worktree.path, "checkout", "-b", worktree.branch, check=False
                )
                if result.returncode == 0:
                    break
                if not await asyncio.to_thread(worktree_git.name_taken, worktree, worktree.id):
                    raise ProtectionError("worktree_unavailable", result.stderr.strip())
            worktree = dataclasses.replace(worktree, id=worktree_git.random_id())
            await self._save(session_id, worktree)
        worktree = dataclasses.replace(worktree, state="ready")
        await self._save(session_id, worktree)
        return worktree

    async def rename(self, session_id: SessionId, name: str) -> Worktree:
        async with self._lock:
            return await complete_operation(self._rename(session_id, name))

    async def _rename(self, session_id: SessionId, name: str) -> Worktree:
        worktree = await self.get(session_id)
        if worktree is None:
            raise ProtectionError("worktree_missing", "This session does not use a worktree")
        if (
            worktree.state != "ready"
            or worktree.pending_id is not None
            or worktree.pending_integration is not None
        ):
            raise ProtectionError("worktree_unavailable", "Worktree recovery is pending")
        await asyncio.to_thread(worktree_git.validate, worktree)
        name = worktree_git.validate_id(name)
        if name == worktree.id:
            return worktree
        while True:
            await self._save(session_id, dataclasses.replace(worktree, pending_id=name))
            result = await asyncio.to_thread(
                worktree_git.git,
                worktree.repository,
                "branch",
                "-m",
                "--",
                worktree.branch,
                name,
                check=False,
            )
            if result.returncode == 0:
                break
            if not await asyncio.to_thread(worktree_git.name_taken, worktree, name):
                raise ProtectionError("worktree_unavailable", result.stderr.strip())
            name = worktree_git.random_id()
        updated = dataclasses.replace(worktree, id=name)
        try:
            await self._save(session_id, updated)
        except BaseException:
            await asyncio.to_thread(
                worktree_git.git,
                worktree.repository,
                "branch",
                "-m",
                "--",
                updated.branch,
                worktree.branch,
            )
            raise
        return updated

    async def get(self, session_id: SessionId) -> Worktree | None:
        row = await self._db.fetch_one(
            "SELECT worktree FROM session WHERE id = ?", (str(session_id),)
        )
        return Worktree(**json.loads(row["worktree"])) if row and row.get("worktree") else None

    async def remove(
        self, session_id: SessionId, *, discard: bool = False, clear: bool = True
    ) -> None:
        async with self._lock:
            await complete_operation(self._remove(session_id, discard, clear))

    async def _remove(self, session_id: SessionId, discard: bool, clear: bool) -> None:
        worktree = await self.get(session_id)
        if worktree is None or worktree.state == "closed":
            return
        if worktree.pending_integration is not None:
            raise ProtectionError("worktree_unavailable", "Integration recovery is pending")
        explicit_discard = discard
        if not discard and worktree.integrated_commit:
            discard = await asyncio.to_thread(worktree_integration.can_clean, worktree)
        owns_branch = await asyncio.to_thread(worktree_git.check_removal, worktree, discard)
        if owns_branch:
            worktree = dataclasses.replace(worktree, state="removing", discard=explicit_discard)
            await self._save(session_id, worktree)
        await asyncio.to_thread(worktree_git.remove, worktree, discard)
        await asyncio.to_thread(
            worktree_git.git,
            worktree.repository,
            "update-ref",
            "-d",
            worktree_integration.ref(worktree),
        )
        if not clear:
            return
        await self._db.execute(
            "UPDATE session SET worktree = NULL, project_path = ? WHERE id = ?",
            (worktree.source_path, str(session_id)),
        )

    async def validate(self, session_id: SessionId) -> None:
        worktree = await self.get(session_id)
        if worktree is not None:
            if worktree.state == "closed":
                raise ProtectionError("worktree_closed", "This worktree was integrated and cleaned")
            if (
                worktree.state != "ready"
                or worktree.pending_id is not None
                or worktree.pending_integration is not None
            ):
                raise ProtectionError("worktree_unavailable", "Worktree recovery is pending")
            await asyncio.to_thread(worktree_git.validate, worktree)

    async def recover(self) -> None:
        rows = await self._db.fetch_all(
            "SELECT id, worktree FROM session WHERE worktree IS NOT NULL"
        )
        for row in rows:
            worktree = Worktree(**json.loads(row["worktree"]))
            if worktree.pending_integration is not None:
                try:
                    worktree = await self._recover_integration(SessionId(str(row["id"])), worktree)
                except ProtectionError:
                    continue
            if worktree.state == "closing":
                try:
                    await self.finish(SessionId(str(row["id"])))
                except ProtectionError:
                    continue
            if worktree.pending_id is not None:
                renamed = await asyncio.to_thread(
                    worktree_git.branch_exists, worktree, worktree.pending_id
                ) and not await asyncio.to_thread(worktree_git.branch_exists, worktree, worktree.id)
                selected = worktree.pending_id if renamed else worktree.id
                await self._save(
                    SessionId(str(row["id"])),
                    dataclasses.replace(worktree, id=selected, pending_id=None),
                )
            if worktree.state in {"preparing", "removing"}:
                try:
                    await self.remove(SessionId(str(row["id"])), discard=worktree.discard)
                    await self._db.execute(
                        "UPDATE session SET deleted = 1 WHERE id = ?", (str(row["id"]),)
                    )
                except ProtectionError:
                    continue

    async def preview(
        self,
        session_id: SessionId,
        target: str | None,
        strategy: IntegrationStrategy,
    ) -> IntegrationPreview:
        async with self._lock:
            await self.validate(session_id)
            worktree = await self._required(session_id)
            return await asyncio.to_thread(worktree_integration.preview, worktree, target, strategy)

    async def _required(self, session_id: SessionId) -> Worktree:
        worktree = await self.get(session_id)
        if worktree is None:
            raise ProtectionError("worktree_missing", "This session does not use a worktree")
        return worktree

    async def integrate(
        self,
        session_id: SessionId,
        target: str,
        strategy: IntegrationStrategy,
        token: str,
        message: str,
    ) -> Worktree:
        async with self._lock:
            return await complete_operation(
                self._integrate(session_id, target, strategy, token, message)
            )

    async def _integrate(
        self,
        session_id: SessionId,
        target: str,
        strategy: IntegrationStrategy,
        token: str,
        message: str,
    ) -> Worktree:
        await self.validate(session_id)
        worktree = await self._required(session_id)
        plan = await asyncio.to_thread(
            worktree_integration.reviewed, worktree, target, strategy, token
        )
        commit = await asyncio.to_thread(
            worktree_integration.create_commit, worktree, plan, message
        )
        index_tree = await asyncio.to_thread(worktree_git.git, worktree.path, "write-tree")
        pending = {
            "integrated_target": target,
            "integrated_commit": commit,
            "integrated_head": plan.source_head,
            "integrated_tree": plan.source_tree,
            "integrated_index": index_tree.stdout.strip(),
        }
        prepared = dataclasses.replace(worktree, pending_integration=pending)
        await self._save(session_id, prepared)
        try:
            await asyncio.to_thread(worktree_integration.apply, worktree, plan, commit)
        finally:
            await self._recover_integration(session_id, prepared)
        return await self._required(session_id)

    async def _recover_integration(self, session_id: SessionId, worktree: Worktree) -> Worktree:
        values = worktree.pending_integration or {}
        candidate = dataclasses.replace(
            worktree,
            integrated_target=values.get("integrated_target"),
            integrated_commit=values.get("integrated_commit"),
            integrated_head=values.get("integrated_head"),
            integrated_tree=values.get("integrated_tree"),
            integrated_index=values.get("integrated_index"),
            pending_integration=None,
        )
        if await asyncio.to_thread(worktree_integration.integrated, candidate):
            snapshot = await asyncio.to_thread(
                worktree_integration.temporary_commit,
                candidate,
                candidate.integrated_head or "",
                candidate.integrated_tree or "",
            )
            await asyncio.to_thread(
                worktree_git.git,
                worktree.repository,
                "update-ref",
                worktree_integration.ref(worktree),
                snapshot,
            )
            worktree = candidate
        else:
            worktree = dataclasses.replace(worktree, pending_integration=None)
        await self._save(session_id, worktree)
        return worktree

    async def resolve(
        self,
        session_id: SessionId,
        target: str,
        strategy: IntegrationStrategy,
        token: str,
    ) -> None:
        async with self._lock:
            await complete_operation(self._resolve(session_id, target, strategy, token))

    async def _resolve(
        self,
        session_id: SessionId,
        target: str,
        strategy: IntegrationStrategy,
        token: str,
    ) -> None:
        await self.validate(session_id)
        worktree = await self._required(session_id)
        plan = await asyncio.to_thread(
            worktree_integration.reviewed, worktree, target, strategy, token
        )
        await asyncio.to_thread(worktree_integration.prepare_resolution, worktree, plan)

    async def finish(self, session_id: SessionId) -> Worktree:
        async with self._lock:
            return await complete_operation(self._finish(session_id))

    async def _finish(self, session_id: SessionId) -> Worktree:
        worktree = await self._required(session_id)
        if worktree.state == "closed":
            return worktree
        if worktree.state not in {"ready", "closing"} or worktree.pending_integration:
            raise ProtectionError("worktree_unavailable", "Worktree recovery is pending")
        registered = await asyncio.to_thread(worktree_git.registered, worktree)
        if registered:
            if not await asyncio.to_thread(worktree_integration.can_clean, worktree):
                raise ProtectionError(
                    "worktree_has_changes", "The worktree contains unintegrated changes"
                )
            worktree = dataclasses.replace(worktree, state="closing")
            await self._save(session_id, worktree)
            await asyncio.to_thread(worktree_git.remove, worktree, True)
        elif worktree.state != "closing" or Path(worktree.path).exists():
            raise ProtectionError("worktree_missing", "The session worktree is unavailable")
        elif not await asyncio.to_thread(worktree_integration.can_clean, worktree):
            raise ProtectionError("worktree_has_changes", "The integration destination changed")
        elif await asyncio.to_thread(worktree_git.branch_exists, worktree, worktree.branch):
            head = await asyncio.to_thread(
                worktree_git.git, worktree.repository, "rev-parse", f"refs/heads/{worktree.branch}"
            )
            if head.stdout.strip() != worktree.integrated_head:
                raise ProtectionError("worktree_has_changes", "The worktree branch changed")
            await asyncio.to_thread(
                worktree_git.git, worktree.repository, "branch", "-D", "--", worktree.branch
            )
        await asyncio.to_thread(
            worktree_git.git,
            worktree.repository,
            "update-ref",
            "-d",
            worktree_integration.ref(worktree),
        )
        worktree = dataclasses.replace(worktree, state="closed")
        await self._save(session_id, worktree)
        return worktree
