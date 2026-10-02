import dataclasses
import hashlib
import os
import re
import stat
import subprocess
import tempfile
from pathlib import Path

from mandri.core.types.execution import ProtectionError
from mandri.sessions.worktree_git import environment, git


@dataclasses.dataclass(frozen=True)
class DestinationPlan:
    path: str | None
    branch: str = ""
    head: str = ""
    index_tree: str = ""
    worktree_tree: str = ""
    result_index: str = ""
    result_worktree: str = ""
    index_fingerprint: str = ""
    conflicts: list[str] = dataclasses.field(default_factory=list)
    before_files: dict[str, str] = dataclasses.field(default_factory=dict)
    after_files: dict[str, str] = dataclasses.field(default_factory=dict)

    @property
    def dirty(self) -> bool:
        return bool(
            self.path and (self.head != self.index_tree or self.index_tree != self.worktree_tree)
        )

    def fingerprint(self) -> dict[str, object]:
        return dataclasses.asdict(self)


def index_path(path: str) -> Path:
    return Path(
        git(path, "rev-parse", "--path-format=absolute", "--git-path", "index").stdout.strip()
    )


def file_fingerprint(path: str, name: str) -> str:
    file = Path(path) / name
    try:
        mode = file.lstat().st_mode
    except FileNotFoundError:
        return "missing"
    if stat.S_ISLNK(mode):
        kind, data = "symlink", os.fsencode(os.readlink(file))
    elif stat.S_ISREG(mode):
        kind, data = "executable" if mode & stat.S_IXUSR else "file", file.read_bytes()
    else:
        return "directory" if stat.S_ISDIR(mode) else "unsupported"
    return kind + ":" + hashlib.sha256(data).hexdigest()


def tree_entries(path: str, tree: str) -> dict[str, tuple[str, str]]:
    entries = {}
    for item in git(path, "ls-tree", "-r", "-z", tree).stdout.split("\0"):
        if item:
            metadata, name = item.split("\t", 1)
            mode, _, oid = metadata.split()
            entries[name] = (mode, oid)
    return entries


def ignored_collisions(path: str, changed: dict[str, str]) -> list[str]:
    ignored = git(
        path, "ls-files", "--others", "--ignored", "--exclude-standard", "-z"
    ).stdout.split("\0")
    return [
        name
        for name in ignored
        if name
        and any(
            name == entry or name.startswith(entry + "/") or entry.startswith(name + "/")
            for entry in changed
        )
    ]


def tree_file_fingerprint(path: str, tree: str, name: str, entry: tuple[str, str] | None) -> str:
    if entry is None:
        return "missing"
    mode, oid = entry
    args = ["git", "-c", f"core.hooksPath={os.devnull}", "-C", path, "cat-file"]
    env = environment()
    env["GIT_ATTR_SOURCE"] = tree
    if mode == "120000":
        args.extend(["blob", oid])
        kind = "symlink"
    else:
        args.extend(["--filters", f"--path={name}", oid])
        kind = "executable" if mode == "100755" else "file"
    result = subprocess.run(args, env=env, capture_output=True, timeout=120)
    if result.returncode:
        raise ProtectionError("worktree_unavailable", "Unable to read destination file content")
    return kind + ":" + hashlib.sha256(result.stdout).hexdigest()


def supported(path: str) -> None:
    sparse = git(path, "config", "--bool", "core.sparseCheckout", check=False).stdout.strip()
    flags = git(path, "ls-files", "-v", "-z").stdout.split("\0")
    debug = git(path, "ls-files", "--debug").stdout
    special = any(
        int(value, 16) & 0x20000000 for value in re.findall(r"flags: ([0-9a-fA-F]+)", debug)
    )
    split = git(path, "rev-parse", "--shared-index-path").stdout.strip()
    if (
        sparse == "true"
        or split
        or special
        or any(item and (item[0] == "S" or item[0].islower()) for item in flags)
    ):
        raise ProtectionError(
            "worktree_target_unsupported", "Sparse or special destination indexes are not supported"
        )
    for item in git(path, "ls-files", "--stage", "-z").stdout.split("\0"):
        if item.startswith("160000 "):
            child = Path(path) / item.split("\t", 1)[1]
            if (child / ".git").exists() and git(
                child, "status", "--porcelain", "--untracked-files=all", "--ignore-submodules=none"
            ).stdout:
                raise ProtectionError(
                    "worktree_target_unsupported", "The destination has local submodule changes"
                )


def worktree_tree(path: str, index_tree: str) -> str:
    with tempfile.TemporaryDirectory(prefix="mandri-destination-index-") as folder:
        index = str(Path(folder) / "index")
        git(path, "read-tree", index_tree, index=index)
        git(path, "add", "--all", "--", ".", index=index)
        return git(path, "write-tree", index=index).stdout.strip()


def merge(path: str, base: str, ours: str, theirs: str) -> tuple[str, list[str]]:
    result = git(
        path,
        "merge-tree",
        "--write-tree",
        "--name-only",
        "-z",
        f"--merge-base={base}",
        ours,
        theirs,
        check=False,
    )
    if result.returncode not in (0, 1):
        raise ProtectionError("worktree_unavailable", result.stderr.strip())
    parts = result.stdout.split("\0")
    return parts[0], parts[1 : parts.index("", 1)] if result.returncode else []


def project(path: str | None, result: str, *, source_conflicts: bool = False) -> DestinationPlan:
    if path is None:
        return DestinationPlan(None)
    supported(path)
    branch = git(path, "symbolic-ref", "-q", "HEAD").stdout.strip()
    head = git(path, "rev-parse", "HEAD^{tree}").stdout.strip()
    index_data = index_path(path).read_bytes()
    with tempfile.TemporaryDirectory(prefix="mandri-destination-index-") as folder:
        index_file = Path(folder) / "index"
        index_file.write_bytes(index_data)
        index_tree = git(path, "write-tree", index=str(index_file)).stdout.strip()
    tree = worktree_tree(path, index_tree)
    fingerprint = hashlib.sha256(index_data).hexdigest()
    if source_conflicts:
        return DestinationPlan(path, branch, head, index_tree, tree, index_fingerprint=fingerprint)
    result_index, conflicts = merge(path, head, index_tree, result)
    result_worktree = ""
    if not conflicts:
        result_worktree, conflicts = merge(path, index_tree, tree, result_index)
    incoming = tree_entries(path, result)
    original = tree_entries(path, head)
    for item in git(path, "ls-files", "--others", "--exclude-standard", "-z").stdout.split("\0"):
        if item and any(
            (name == item or name.startswith(item + "/") or item.startswith(name + "/"))
            and name not in original
            for name in incoming
        ):
            conflicts.append(item)
    before_files, after_files = {}, {}
    if result_worktree and not conflicts:
        before = tree_entries(path, tree)
        after = tree_entries(path, result_worktree)
        for name in sorted(before.keys() | after.keys()):
            if before.get(name) != after.get(name):
                if any(
                    entry and entry[0] == "160000" for entry in (before.get(name), after.get(name))
                ):
                    raise ProtectionError(
                        "worktree_target_unsupported",
                        "Destination submodule updates are not supported",
                    )
                before_files[name] = file_fingerprint(path, name)
                if before.get(name) is not None and before_files[name] != tree_file_fingerprint(
                    path, tree, name, before[name]
                ):
                    raise ProtectionError(
                        "worktree_target_unsupported",
                        "Destination Git filters cannot restore the original file bytes and mode",
                    )
                after_files[name] = tree_file_fingerprint(
                    path, result_worktree, name, after.get(name)
                )
        conflicts = ignored_collisions(path, before_files)
        if conflicts:
            before_files.update({name: file_fingerprint(path, name) for name in conflicts})
            return DestinationPlan(
                path,
                branch,
                head,
                index_tree,
                tree,
                result_index,
                result_worktree,
                fingerprint,
                sorted(set(conflicts)),
                before_files,
                after_files,
            )
        with tempfile.TemporaryDirectory(prefix="mandri-destination-check-") as folder:
            index = str(Path(folder) / "index")
            git(path, "read-tree", tree, index=index)
            git(path, "update-index", "--refresh", index=index)
            check = git(
                path, "read-tree", "-n", "-m", "-u", tree, result_worktree, index=index, check=False
            )
            if check.returncode:
                raise ProtectionError(
                    "worktree_target_changed", "The destination changed during review"
                )
    return DestinationPlan(
        path,
        branch,
        head,
        index_tree,
        tree,
        result_index,
        result_worktree,
        fingerprint,
        sorted(set(conflicts)),
        before_files,
        after_files,
    )
