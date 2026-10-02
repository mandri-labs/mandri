import contextlib
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from mandri.core.types.execution import ProtectionError
from mandri.core.types.worktrees import Worktree
from mandri.sessions.worktree_git import environment, git
from mandri.sessions.worktree_projection import (
    DestinationPlan,
    file_fingerprint,
    ignored_collisions,
    index_path,
    tree_entries,
    worktree_tree,
)


def directory(worktree: Worktree) -> Path:
    common = Path(
        git(
            worktree.repository, "rev-parse", "--path-format=absolute", "--git-common-dir"
        ).stdout.strip()
    )
    result = common / "mandri-integrations" / hashlib.sha256(worktree.path.encode()).hexdigest()
    if not result.resolve().is_relative_to(common.resolve()):
        raise ProtectionError("worktree_unavailable", "Integration journal location changed")
    return result


def durable_write(path: Path, data: bytes) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def save(folder: Path, journal: dict[str, Any]) -> None:
    durable_write(folder / "journal.json", json.dumps(journal, sort_keys=True).encode())


class RefTransaction:
    def __init__(self, path: str, branch: str, old: str, new: str | None) -> None:
        self._process = subprocess.Popen(
            ["git", "-C", path, "update-ref", "--stdin", "-m", "Mandri worktree integration"],
            env=environment(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self._committed = False
        try:
            self._send("start\n", "start: ok")
            instruction = f"update {branch} {new} {old}" if new else f"verify {branch} {old}"
            self._send(instruction + "\nprepare\n", "prepare: ok")
        except BaseException:
            self.close()
            raise

    def _send(self, command: str, expected: str) -> None:
        process = self._process
        if process.stdin is None or process.stdout is None:
            raise ProtectionError(
                "worktree_unavailable", "Git reference transaction is unavailable"
            )
        process.stdin.write(command)
        process.stdin.flush()
        if process.stdout.readline().strip() != expected:
            process.wait(timeout=120)
            raise ProtectionError(
                "worktree_target_changed", "The destination branch could not be locked"
            )

    def commit(self) -> None:
        self._send("commit\n", "commit: ok")
        self._committed = True

    def close(self) -> None:
        process = self._process
        try:
            if not self._committed and process.poll() is None:
                self._send("abort\n", "abort: ok")
        finally:
            if process.stdin is not None:
                process.stdin.close()
            process.wait(timeout=120)
            if process.stdout is not None:
                process.stdout.close()
            if process.stderr is not None:
                process.stderr.close()


@contextlib.contextmanager
def reference(path: str, branch: str, old: str, new: str | None) -> Iterator[RefTransaction]:
    transaction = RefTransaction(path, branch, old, new)
    try:
        yield transaction
    finally:
        transaction.close()


@contextlib.contextmanager
def locked_index(path: str, folder: Path, journal: dict[str, Any]) -> Iterator[Path]:
    lock = index_path(path).with_name("index.lock")
    try:
        descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise ProtectionError("worktree_git_busy", "The destination index is locked") from error
    inode = os.fstat(descriptor).st_ino
    os.close(descriptor)
    journal.update(lock_inode=inode, lock_pid=os.getpid())
    save(folder, journal)
    try:
        yield lock
    finally:
        if lock.exists() and lock.stat().st_ino == inode:
            lock.unlink()


def transition(path: str, before: str, after: str) -> None:
    with tempfile.TemporaryDirectory(prefix="mandri-destination-write-") as folder:
        index = str(Path(folder) / "index")
        git(path, "read-tree", before, index=index)
        git(path, "update-index", "--refresh", index=index)
        git(path, "read-tree", "-n", "-m", "-u", before, after, index=index)
        git(path, "read-tree", "-m", "-u", before, after, index=index)


def prepare(
    worktree: Worktree, plan: DestinationPlan, commit: str, old: str
) -> tuple[Path, dict[str, Any]]:
    path = plan.path
    if path is None:
        raise ProtectionError("worktree_unavailable", "The destination checkout is unavailable")
    folder = directory(worktree)
    folder.mkdir(parents=True, exist_ok=True, mode=0o700)
    if (folder / "journal.json").exists():
        raise ProtectionError(
            "worktree_recovery_pending", "A previous integration requires recovery"
        )
    original = index_path(path).read_bytes()
    if hashlib.sha256(original).hexdigest() != plan.index_fingerprint:
        raise ProtectionError("worktree_target_changed", "The destination index changed")
    durable_write(folder / "before-index", original)
    shutil.copyfile(folder / "before-index", folder / "after-index")
    git(path, "read-tree", plan.result_index, index=str(folder / "after-index"))
    after = (folder / "after-index").read_bytes()
    durable_write(folder / "after-index", after)
    refs = []
    for name, tree in {
        "before-index": plan.index_tree,
        "before-worktree": plan.worktree_tree,
        "after-index": plan.result_index,
        "after-worktree": plan.result_worktree,
        "commit": commit,
    }.items():
        ref = "refs/mandri/transactions/" + folder.name + "/" + name
        git(worktree.repository, "update-ref", ref, tree)
        refs.append(ref)
    journal = {
        "path": path,
        "branch": plan.branch,
        "old": old,
        "commit": commit,
        "before_index": plan.index_tree,
        "before_tree": plan.worktree_tree,
        "after_tree": plan.result_worktree,
        "before_files": plan.before_files,
        "after_files": plan.after_files,
        "refs": refs,
    }
    save(folder, journal)
    return folder, journal


def install_index(path: str, lock: Path, data: bytes) -> None:
    with lock.open("wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(lock, index_path(path))


def apply(worktree: Worktree, plan: DestinationPlan, commit: str, old: str) -> None:
    folder, journal = prepare(worktree, plan, commit, old)
    path = journal["path"]
    with (
        reference(path, journal["branch"], old, commit) as transaction,
        locked_index(path, folder, journal) as lock,
    ):
        if git(path, "symbolic-ref", "-q", "HEAD").stdout.strip() != journal["branch"]:
            raise ProtectionError("worktree_target_changed", "The destination checkout changed")
        if (
            index_path(path).read_bytes() != (folder / "before-index").read_bytes()
            or ignored_collisions(path, plan.before_files)
            or any(
                file_fingerprint(path, name) != value for name, value in plan.before_files.items()
            )
        ):
            raise ProtectionError(
                "worktree_target_changed", "The destination changed during integration"
            )
        transition(path, plan.worktree_tree, plan.result_worktree)
        if any(file_fingerprint(path, name) != value for name, value in plan.after_files.items()):
            raise ProtectionError(
                "worktree_target_changed", "The destination changed during integration"
            )
        install_index(path, lock, (folder / "after-index").read_bytes())
        transaction.commit()


def recover(worktree: Worktree) -> None:
    folder = directory(worktree)
    if not (folder / "journal.json").exists():
        return
    journal = json.loads((folder / "journal.json").read_text())
    path = journal["path"]
    head = git(path, "rev-parse", "HEAD").stdout.strip()
    before_index, after_index = (
        (folder / "before-index").read_bytes(),
        (folder / "after-index").read_bytes(),
    )
    if (
        git(path, "symbolic-ref", "-q", "HEAD").stdout.strip() != journal["branch"]
        or head not in {journal["old"], journal["commit"]}
        or index_path(path).read_bytes() not in {before_index, after_index}
    ):
        raise ProtectionError(
            "worktree_recovery_pending", "The destination changed after interrupted integration"
        )
    for name, value in journal["before_files"].items():
        if file_fingerprint(path, name) not in {value, journal["after_files"][name]}:
            raise ProtectionError(
                "worktree_recovery_pending", "Destination edits require manual integration recovery"
            )
    lock = index_path(path).with_name("index.lock")
    if lock.exists() and lock.stat().st_ino == journal.get("lock_inode"):
        pid = journal.get("lock_pid")
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            if lock.read_bytes() in {b"", after_index}:
                lock.unlink()
    with (
        reference(path, journal["branch"], head, None) as transaction,
        locked_index(path, folder, journal) as locked,
    ):
        if (
            git(path, "symbolic-ref", "-q", "HEAD").stdout.strip() != journal["branch"]
            or git(path, "rev-parse", "HEAD").stdout.strip() != head
            or index_path(path).read_bytes() not in {before_index, after_index}
            or any(
                file_fingerprint(path, name) not in {value, journal["after_files"][name]}
                for name, value in journal["before_files"].items()
            )
        ):
            raise ProtectionError(
                "worktree_recovery_pending", "The destination changed during integration recovery"
            )
        current_index = str(folder / "recovery-index")
        if ignored_collisions(path, journal["before_files"]):
            raise ProtectionError(
                "worktree_recovery_pending",
                "Ignored destination files prevent integration recovery",
            )
        current_tree = worktree_tree(path, journal["before_index"])
        git(path, "read-tree", current_tree, index=current_index)
        desired_tree = (
            journal["after_tree"] if head == journal["commit"] else journal["before_tree"]
        )
        entries = tree_entries(path, desired_tree)
        changes = []
        for name in journal["before_files"]:
            mode, oid = entries.get(name, ("0", "0" * 40))
            changes.append(f"{mode} {oid}\t{name}\0")
        git(path, "update-index", "-z", "--index-info", input="".join(changes), index=current_index)
        result = git(path, "write-tree", index=current_index).stdout.strip()
        transition(path, current_tree, result)
        install_index(path, locked, after_index if head == journal["commit"] else before_index)
        transaction.commit()


def discard(worktree: Worktree) -> None:
    folder = directory(worktree)
    journal_path = folder / "journal.json"
    if not journal_path.exists():
        return
    journal = json.loads(journal_path.read_text())
    journal_path.unlink()
    for ref in journal["refs"]:
        git(worktree.repository, "update-ref", "-d", ref)
    for name in ("before-index", "after-index", "recovery-index"):
        (folder / name).unlink(missing_ok=True)
    folder.rmdir()
