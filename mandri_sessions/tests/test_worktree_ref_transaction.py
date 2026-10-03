import pytest
from mandri.sessions.worktree_git import git
from mandri.sessions.worktree_transaction import reference

from mandri_sessions.tests.test_worktrees import repository

__all__ = ["repository"]


@pytest.mark.parametrize("commit", [False, True])
def test_transaction_locks_unicode_branch_and_commits_only_on_request(repository, commit):
    old = git(repository, "rev-parse", "HEAD").stdout.strip()
    git(repository, "commit", "--allow-empty", "-m", "Candidate")
    new = git(repository, "rev-parse", "HEAD").stdout.strip()
    branch = "refs/heads/integration-é"
    git(repository, "update-ref", branch, old)
    with reference(str(repository), branch, old, new) as transaction:
        assert git(repository, "rev-parse", branch).stdout.strip() == old
        assert git(repository, "update-ref", branch, new, old, check=False).returncode != 0
        if commit:
            transaction.commit()
    assert git(repository, "rev-parse", branch).stdout.strip() == (new if commit else old)
    git(repository, "update-ref", branch, old)
