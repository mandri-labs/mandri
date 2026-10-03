import hashlib
import json
import tempfile
from pathlib import Path

from mandri.core.types.execution import ProtectionError
from mandri.core.types.worktree_integration import IntegrationPreview, IntegrationStrategy
from mandri.core.types.worktrees import Worktree
from mandri.sessions import worktree_projection, worktree_transaction
from mandri.sessions.worktree_git import branch_exists, commit_tree, git, registered, validate


def fail(code: str, message: str) -> None:
    raise ProtectionError(code, message)


def ref(worktree: Worktree) -> str:
    return "refs/mandri/worktrees/" + hashlib.sha256(worktree.path.encode()).hexdigest()


def branches(worktree: Worktree) -> tuple[list[str], str | None]:
    names = git(
        worktree.repository, "for-each-ref", "--format=%(refname:strip=2)", "refs/heads"
    ).stdout.splitlines()
    names = [name for name in names if name != worktree.branch]
    base_branch = worktree.base_ref.removeprefix("refs/heads/")
    if worktree.base_ref.startswith("refs/heads/") and base_branch in names:
        return names, base_branch
    remotes = git(worktree.repository, "remote").stdout.splitlines()
    preferred = "origin" if "origin" in remotes else remotes[0] if len(remotes) == 1 else None
    if preferred:
        head = git(
            worktree.repository, "symbolic-ref", "-q", f"refs/remotes/{preferred}/HEAD", check=False
        ).stdout.strip()
        candidate = head.removeprefix(f"refs/remotes/{preferred}/")
        if candidate in names:
            return names, candidate
        if head:
            tracking = git(
                worktree.repository,
                "for-each-ref",
                "--format=%(refname:strip=2)%00%(upstream)",
                "refs/heads",
            ).stdout.splitlines()
            matches = [
                name
                for entry in tracking
                for name, upstream in [entry.split("\0", 1)]
                if upstream == head and name in names
            ]
            return names, matches[0] if len(matches) == 1 else None
    return names, names[0] if len(names) == 1 else None


def checkout(worktree: Worktree, target: str) -> str | None:
    entries = git(worktree.repository, "worktree", "list", "--porcelain", "-z").stdout
    for entry in entries.split("\0\0"):
        fields = entry.split("\0")
        if f"branch refs/heads/{target}" in fields:
            return fields[0].removeprefix("worktree ")
    return None


def idle_git(path: str) -> None:
    for name in ("MERGE_HEAD", "CHERRY_PICK_HEAD", "REVERT_HEAD", "rebase-merge", "rebase-apply"):
        location = git(
            path, "rev-parse", "--path-format=absolute", "--git-path", name
        ).stdout.strip()
        if Path(location).exists():
            fail("worktree_git_busy", "Finish or abort the current Git operation first")
    if git(path, "ls-files", "--unmerged").stdout:
        fail("worktree_git_busy", "Resolve and stage the conflicted files first")


def snapshot(worktree: Worktree) -> tuple[str, str, str]:
    validate(worktree)
    idle_git(worktree.path)
    current = git(worktree.path, "symbolic-ref", "-q", "HEAD", check=False).stdout.strip()
    if current != f"refs/heads/{worktree.branch}":
        fail("worktree_unavailable", "The session branch changed")
    head = git(worktree.path, "rev-parse", "HEAD").stdout.strip()
    with tempfile.TemporaryDirectory(prefix="mandri-index-") as directory:
        index_file = Path(directory) / "index"
        index_file.write_bytes(worktree_projection.index_path(worktree.path).read_bytes())
        index = str(index_file)
        index_tree = git(worktree.path, "write-tree", index=index).stdout.strip()
        index_file.unlink()
        git(worktree.path, "read-tree", index_tree, index=index)
        git(worktree.path, "add", "--all", "--", ".", index=index)
        tree = git(worktree.path, "write-tree", index=index).stdout.strip()
    for entry in git(worktree.path, "ls-files", "--stage", "-z").stdout.split("\0"):
        if not entry.startswith("160000 "):
            continue
        submodule = Path(worktree.path) / entry.split("\t", 1)[1]
        if (submodule / ".git").exists() and git(
            submodule, "status", "--porcelain", "--untracked-files=all", "--ignore-submodules=none"
        ).stdout:
            fail("worktree_submodule_changes", "Commit submodule changes before integration")
    return head, tree, index_tree


def temporary_commit(worktree: Worktree, head: str, tree: str) -> str:
    return git(
        worktree.path,
        "-c",
        "user.name=Mandri",
        "-c",
        "user.email=mandri@localhost",
        "commit-tree",
        tree,
        "-p",
        head,
        "-m",
        "Worktree integration preview",
    ).stdout.strip()


def _preview(
    worktree: Worktree,
    target: str | None,
    strategy: IntegrationStrategy,
) -> tuple[IntegrationPreview, worktree_projection.DestinationPlan]:
    names, default = branches(worktree)
    target = target or default
    if not target:
        return IntegrationPreview(
            branches=names, default_branch=default, strategy=strategy
        ), worktree_projection.DestinationPlan(None)
    if target not in names:
        fail("worktree_invalid_target", "Select an existing local destination branch")
    head, tree, index_tree = snapshot(worktree)
    target_head = git(worktree.repository, "rev-parse", f"refs/heads/{target}").stdout.strip()
    source = temporary_commit(worktree, head, tree)
    base: list[str] = []
    if worktree.integrated_target == target and worktree.integrated_commit:
        if git(
            worktree.repository,
            "merge-base",
            "--is-ancestor",
            worktree.integrated_commit,
            target_head,
            check=False,
        ).returncode:
            fail("worktree_target_changed", "The previous integration is no longer on this branch")
        base = [f"--merge-base={worktree.integrated_tree}"]
    result = git(
        worktree.repository,
        "merge-tree",
        "--write-tree",
        "--name-only",
        "-z",
        *base,
        target_head,
        source,
        check=False,
    )
    if result.returncode not in (0, 1):
        fail("worktree_unavailable", result.stderr.strip())
    fields = result.stdout.split("\0")
    result_tree = fields[0]
    conflicts = fields[1 : fields.index("", 1)] if result.returncode else []
    location = checkout(worktree, target)
    target_error = None
    try:
        if location:
            idle_git(location)
            if worktree_projection.index_path(location).with_name("index.lock").exists():
                fail("worktree_git_busy", "The destination index is locked")
            if git(location, "rev-parse", "HEAD").stdout.strip() != target_head:
                fail("worktree_target_changed", "The destination checkout changed")
        destination = worktree_projection.project(
            location, result_tree, source_conflicts=bool(conflicts)
        )
        dirty = destination.dirty
    except ProtectionError as error:
        if not location or error.code not in {"worktree_target_unsupported", "worktree_git_busy"}:
            raise
        target_error = error.code
        destination = worktree_projection.DestinationPlan(
            location,
            git(location, "symbolic-ref", "-q", "HEAD").stdout.strip(),
            target_head,
            index_fingerprint=hashlib.sha256(
                worktree_projection.index_path(location).read_bytes()
            ).hexdigest(),
        )
        dirty = bool(git(location, "status", "--porcelain", "--untracked-files=all").stdout)
    fingerprint = [
        target,
        strategy,
        head,
        tree,
        index_tree,
        target_head,
        conflicts if conflicts else result_tree,
        destination.fingerprint(),
        target_error,
    ]
    token = hashlib.sha256(json.dumps(fingerprint).encode()).hexdigest()
    diff_base = git(worktree.repository, "merge-base", head, target_head).stdout.strip()
    diff = git(
        worktree.repository,
        "diff",
        "--no-ext-diff",
        "--no-textconv",
        target_head if not conflicts else diff_base,
        result_tree if not conflicts else tree,
        "--",
    ).stdout
    files = (
        git(
            worktree.repository,
            "diff",
            "--name-only",
            "-z",
            target_head if not conflicts else diff_base,
            result_tree if not conflicts else tree,
            "--",
        )
        .stdout.rstrip("\0")
        .split("\0")
    )
    return IntegrationPreview(
        branches=names,
        default_branch=default,
        target=target,
        token=token,
        diff=diff,
        files=[name for name in files if name],
        conflicts=conflicts,
        target_dirty=dirty,
        target_conflicts=destination.conflicts,
        target_path=location,
        target_error=target_error,
        source_head=head,
        source_tree=tree,
        target_head=target_head,
        result_tree=result_tree,
        strategy=strategy,
    ), destination


def preview(
    worktree: Worktree, target: str | None, strategy: IntegrationStrategy
) -> IntegrationPreview:
    return _preview(worktree, target, strategy)[0]


def reviewed(
    worktree: Worktree,
    target: str,
    strategy: IntegrationStrategy,
    token: str,
) -> IntegrationPreview:
    current = preview(worktree, target, strategy)
    if current.token != token:
        fail("worktree_preview_changed", "Changes detected since review; refresh the preview")
    return current


def create_commit(worktree: Worktree, plan: IntegrationPreview, message: str) -> str:
    if not message.strip():
        fail("worktree_commit_message", "Enter a commit message")
    if plan.conflicts:
        fail("worktree_conflicts", "Resolve conflicts in the session worktree first")
    if plan.target_conflicts:
        fail(
            "worktree_target_conflicts",
            "Destination local changes conflict: " + ", ".join(plan.target_conflicts),
        )
    if not plan.files:
        fail("worktree_no_changes", "There are no changes to integrate")
    if plan.target_error:
        fail(
            plan.target_error,
            "Choose another destination branch or resolve its Git state",
        )
    parents = [plan.target_head]
    if plan.strategy == "merge":
        source = plan.source_head
        head_tree = git(worktree.path, "rev-parse", f"{source}^{{tree}}").stdout.strip()
        if plan.source_tree != head_tree:
            source = commit_tree(worktree, plan.source_tree, [source], message)
        parents.append(source)
    return commit_tree(worktree, plan.result_tree, parents, message)


def apply(worktree: Worktree, plan: IntegrationPreview, commit: str) -> None:
    current, destination = _preview(worktree, plan.target or "", plan.strategy)
    if current.token != plan.token:
        fail("worktree_preview_changed", "Changes detected since review; refresh the preview")
    if destination.path:
        worktree_transaction.apply(worktree, destination, commit, plan.target_head)
    else:
        git(
            worktree.repository,
            "update-ref",
            "-m",
            "Mandri worktree integration",
            f"refs/heads/{plan.target}",
            commit,
            plan.target_head,
        )


def integrated(worktree: Worktree) -> bool:
    return bool(
        worktree.integrated_commit
        and worktree.integrated_target
        and branch_exists(worktree, worktree.integrated_target)
        and git(
            worktree.repository,
            "merge-base",
            "--is-ancestor",
            worktree.integrated_commit,
            f"refs/heads/{worktree.integrated_target}",
            check=False,
        ).returncode
        == 0
    )


def has_ignored_files(worktree: Worktree) -> bool:
    return bool(
        git(worktree.path, "ls-files", "--others", "--ignored", "--exclude-standard", "-z").stdout
    )


def can_clean(worktree: Worktree, *, discard_ignored: bool = False) -> bool:
    if not integrated(worktree):
        return False
    if not registered(worktree):
        if worktree.state not in {"removing", "closing"} or Path(worktree.path).exists():
            return False
        if not branch_exists(worktree, worktree.branch):
            return True
        return (
            git(worktree.repository, "rev-parse", f"refs/heads/{worktree.branch}").stdout.strip()
            == worktree.integrated_head
        )
    head, tree, index_tree = snapshot(worktree)
    head_tree = git(worktree.path, "rev-parse", f"{head}^{{tree}}").stdout.strip()
    if index_tree not in {head_tree, tree}:
        return False
    return (discard_ignored or not has_ignored_files(worktree)) and (head, tree, index_tree) == (
        worktree.integrated_head,
        worktree.integrated_tree,
        worktree.integrated_index,
    )


def prepare_resolution(worktree: Worktree, plan: IntegrationPreview) -> None:
    if not plan.conflicts:
        fail("worktree_no_changes", "There are no conflicts to prepare")
    reviewed(worktree, plan.target or "", plan.strategy, plan.token or "")
    head_tree = git(worktree.path, "rev-parse", "HEAD^{tree}").stdout.strip()
    _, _, index_tree = snapshot(worktree)
    if index_tree not in {head_tree, plan.source_tree}:
        fail("worktree_has_changes", "Commit staged changes before preparing conflict resolution")
    if head_tree != plan.source_tree:
        message = "Save work before resolving integration conflicts"
        commit = commit_tree(worktree, plan.source_tree, [plan.source_head], message)
        reviewed(worktree, plan.target or "", plan.strategy, plan.token or "")
        git(worktree.path, "add", "--all", "--", ".")
        git(
            worktree.path,
            "update-ref",
            "-m",
            "commit: " + message,
            "HEAD",
            commit,
            plan.source_head,
        )
    result = git(worktree.path, "merge", "--no-commit", "--no-ff", plan.target_head, check=False)
    if result.returncode and not git(worktree.path, "ls-files", "--unmerged").stdout:
        fail("worktree_unavailable", result.stderr.strip())
