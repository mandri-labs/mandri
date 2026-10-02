import os
import re
import secrets
import subprocess
from pathlib import Path

from mandri.core.types.execution import ProtectionError
from mandri.core.types.worktrees import Worktree

_ADJECTIVES = (
    "calm",
    "bright",
    "quiet",
    "swift",
    "gentle",
    "bold",
    "keen",
    "clear",
    "lucky",
    "happy",
    "merry",
    "nimble",
    "brave",
    "lively",
    "kind",
    "noble",
)
_COLORS = (
    "amber",
    "silver",
    "coral",
    "azure",
    "golden",
    "jade",
    "indigo",
    "violet",
    "copper",
    "ivory",
    "ruby",
    "olive",
    "teal",
    "pearl",
    "crimson",
    "saffron",
)
_NOUNS = (
    "otter",
    "falcon",
    "cedar",
    "meadow",
    "river",
    "panda",
    "willow",
    "heron",
    "badger",
    "robin",
    "maple",
    "orchid",
    "fox",
    "island",
    "comet",
    "finch",
)


def random_id() -> str:
    return "-".join(secrets.choice(words) for words in (_ADJECTIVES, _COLORS, _NOUNS))


def environment(index: str | None = None) -> dict[str, str]:
    env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["GIT_OPTIONAL_LOCKS"] = "0"
    if index is not None:
        env["GIT_INDEX_FILE"] = index
    return env


def git(
    cwd: str | Path,
    *args: str,
    check: bool = True,
    index: str | None = None,
    input: str | None = None,
) -> subprocess.CompletedProcess[str]:
    env = environment(index)
    try:
        result = subprocess.run(
            ["git", "-c", f"core.hooksPath={os.devnull}", "-C", str(cwd), *args],
            env=env,
            capture_output=True,
            text=True,
            input=input,
            timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise ProtectionError("worktree_unavailable", "Git worktree operation failed") from error
    if check and result.returncode:
        raise ProtectionError(
            "worktree_unavailable", result.stderr.strip() or "Git operation failed"
        )
    return result


def validate_id(value: str) -> str:
    value = value.strip()
    if (
        not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,99}", value)
        or ".." in value
        or "//" in value
        or "@{" in value
        or any(part.startswith(".") or part.endswith((".", ".lock")) for part in value.split("/"))
        or value.endswith("/")
    ):
        raise ProtectionError(
            "worktree_invalid_id", "Use a valid Git branch name (1-100 characters)"
        )
    return value


def plan(cwd: str, root: Path, token: str, name: str | None) -> Worktree:
    selected_id = validate_id(name) if name else random_id()
    try:
        source = Path(cwd).resolve(strict=True)
    except OSError as error:
        raise ProtectionError(
            "worktree_repository_required", "Select an existing Git repository"
        ) from error
    result = git(source, "rev-parse", "--show-toplevel", check=False)
    if result.returncode:
        raise ProtectionError("worktree_repository_required", "Select a Git repository")
    checkout = Path(result.stdout.strip()).resolve()
    listing = git(source, "worktree", "list", "--porcelain", "-z").stdout
    repository = Path(listing.split("\0", 1)[0].removeprefix("worktree ")).resolve()
    if root.resolve().is_relative_to(repository) or root.resolve().is_relative_to(checkout):
        raise ProtectionError(
            "worktree_unavailable", "Worktree storage must be outside the repository"
        )
    commit = git(source, "rev-parse", "--verify", "HEAD", check=False)
    if commit.returncode:
        raise ProtectionError(
            "worktree_repository_required", "The repository needs an initial commit"
        )
    ref = git(source, "symbolic-ref", "-q", "HEAD", check=False).stdout.strip()
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    return Worktree(
        id=selected_id,
        path=str(root.resolve() / f"{selected_id.replace('/', '-')}-{token}"),
        repository=str(repository),
        source_path=str(source),
        base_ref=ref or commit.stdout.strip(),
        base_commit=commit.stdout.strip(),
        relative_path=str(source.relative_to(checkout)),
    )


def registered(worktree: Worktree) -> bool:
    listing = git(worktree.repository, "worktree", "list", "--porcelain", "-z").stdout
    expected = Path(worktree.path).resolve()
    return any(
        Path(entry.removeprefix("worktree ")).resolve() == expected
        for entry in listing.split("\0")
        if entry.startswith("worktree ")
    )


def validate(worktree: Worktree) -> None:
    path = Path(worktree.path)
    if path.is_symlink() or not path.is_dir() or not registered(worktree):
        raise ProtectionError("worktree_missing", "The session worktree is unavailable")
    actual = git(path, "rev-parse", "--show-toplevel").stdout.strip()
    if Path(actual).resolve() != path:
        raise ProtectionError("worktree_missing", "The session worktree identity changed")


def branch_exists(worktree: Worktree, name: str) -> bool:
    return (
        git(
            worktree.repository,
            "show-ref",
            "--verify",
            "--quiet",
            f"refs/heads/{name}",
            check=False,
        ).returncode
        == 0
    )


def name_taken(worktree: Worktree, name: str) -> bool:
    ref = f"refs/heads/{name}"
    refs = git(
        worktree.repository, "for-each-ref", "--format=%(refname)", "refs/heads"
    ).stdout.splitlines()
    return any(
        ref == item or ref.startswith(item + "/") or item.startswith(ref + "/") for item in refs
    )


def check_removal(worktree: Worktree, discard: bool) -> bool:
    owns_branch = worktree.state != "preparing"
    if registered(worktree):
        validate(worktree)
        current = git(worktree.path, "symbolic-ref", "-q", "HEAD", check=False).stdout.strip()
        owns_branch = owns_branch or current == f"refs/heads/{worktree.branch}"
        if not discard:
            dirty = git(
                worktree.path, "status", "--porcelain", "--untracked-files=all", "--ignored"
            )
            merged = git(
                worktree.repository,
                "merge-base",
                "--is-ancestor",
                git(worktree.path, "rev-parse", "HEAD").stdout.strip(),
                worktree.base_ref,
                check=False,
            )
            if dirty.stdout or merged.returncode:
                raise ProtectionError(
                    "worktree_has_changes", "Integrate or explicitly discard the worktree changes"
                )
    elif Path(worktree.path).exists():
        raise ProtectionError("worktree_missing", "Unregistered worktree files require recovery")
    if (
        owns_branch
        and branch_exists(worktree, worktree.id)
        and not discard
        and git(
            worktree.repository,
            "merge-base",
            "--is-ancestor",
            f"refs/heads/{worktree.branch}",
            worktree.base_ref,
            check=False,
        ).returncode
    ):
        raise ProtectionError("worktree_has_changes", "The worktree branch has unmerged commits")
    return owns_branch


def remove(worktree: Worktree, discard: bool) -> None:
    owns_branch = check_removal(worktree, discard)
    if registered(worktree):
        git(worktree.repository, "worktree", "remove", "--force", worktree.path)
    if owns_branch and branch_exists(worktree, worktree.id):
        git(worktree.repository, "branch", "-D", "--", worktree.branch)
