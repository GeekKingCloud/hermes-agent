"""The unattended orphan-branch pass reclaims finished subagents' merged branches, and only those.

Real git repos, no mocks. Children come from ``tools.subagent_worktree`` itself, so the branch
namespace under test is the one isolation actually creates.
"""

import shutil
import subprocess

import pytest

from hermes_cli import worktree_ops
from tools import subagent_worktree as sw


def _git(args, cwd):
    result = subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=str(cwd),
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, f"git {args}: {result.stderr}"
    return result.stdout.strip()


def _branches(repo):
    return set(_git(["branch", "--format=%(refname:short)"], repo).split())


@pytest.fixture
def repo(tmp_path):
    origin = tmp_path / "origin.git"
    _git(["init", "-q", "--bare", "-b", "main", str(origin)], tmp_path)
    clone = tmp_path / "repo"
    _git(["init", "-q", "-b", "main", str(clone)], tmp_path)
    (clone / "README.md").write_text("seed\n", encoding="utf-8")
    _git(["add", "-A"], clone)
    _git(["commit", "-qm", "seed"], clone)
    _git(["remote", "add", "origin", str(origin)], clone)
    _git(["push", "-q", "-u", "origin", "main"], clone)
    return clone


def _child(repo, name, *, commit=True):
    """A delegated child's worktree, optionally with one commit; returns (info, commit sha)."""
    info = sw.create_subagent_worktree(str(repo), name)
    assert info and info["branch"].startswith(sw.SUBAGENT_BRANCH_PREFIX)
    sha = None
    if commit:
        (repo / ".worktrees" / f"subagent-{name}" / f"{name}.txt").write_text(name + "\n", encoding="utf-8")
        _git(["add", "-A"], info["path"])
        _git(["commit", "-qm", f"work {name}"], info["path"])
        sha = _git(["rev-parse", "HEAD"], info["path"])
    return info, sha


def _leak(info):
    """The tree vanishes without ``git worktree remove`` (its registration goes stale), then the
    child finishes: the leak the pass must reclaim."""
    shutil.rmtree(info["path"])
    sw.finalize_subagent_worktree(info)


def _registered(repo):
    return [ln for ln in _git(["worktree", "list", "--porcelain"], repo).splitlines() if ln.startswith("worktree ")]


def test_leaked_merged_subagent_branch_is_pruned_with_its_stale_registration(repo):
    info, _sha = _child(repo, "merged")
    _leak(info)
    assert len(_registered(repo)) == 2  # the stale registration still pins the branch
    _git(["merge", "-q", "--ff-only", info["branch"]], repo)
    _git(["push", "-q", "origin", "main"], repo)

    worktree_ops._prune_orphaned_branches(str(repo))

    assert info["branch"] not in _branches(repo)
    assert len(_registered(repo)) == 1


def test_leaked_squash_merged_subagent_branch_is_pruned(repo):
    info, sha = _child(repo, "squashed")
    _leak(info)
    (repo / "other.txt").write_text("trunk moved on\n", encoding="utf-8")
    _git(["add", "other.txt"], repo)
    _git(["commit", "-qm", "trunk"], repo)
    _git(["cherry-pick", sha], repo)  # same patch, new commit: what a squash/rebase merge leaves
    _git(["push", "-q", "origin", "main"], repo)

    worktree_ops._prune_orphaned_branches(str(repo))

    assert info["branch"] not in _branches(repo)


def test_unmerged_subagent_branch_is_kept(repo):
    info, sha = _child(repo, "unmerged")
    _leak(info)

    worktree_ops._prune_orphaned_branches(str(repo))

    assert info["branch"] in _branches(repo)
    assert _git(["rev-parse", info["branch"]], repo) == sha


def test_branch_held_only_by_a_disposable_session_branch_is_kept(repo):
    """A ``hermes/hermes-*`` branch is deleted unchecked once its tree goes, so it is no proof of merge."""
    info, sha = _child(repo, "insession")
    _leak(info)
    session = repo / ".worktrees" / "hermes-session"
    _git(["worktree", "add", "-q", "-b", "hermes/hermes-session", str(session), "HEAD"], repo)
    _git(["merge", "-q", "--ff-only", info["branch"]], session)  # merged only into the unpushed session branch

    worktree_ops._prune_orphaned_branches(str(repo))

    assert _git(["rev-parse", info["branch"]], repo) == sha


def test_live_child_branch_is_kept_even_without_a_tree(repo):
    info, _sha = _child(repo, "live", commit=False)  # no commits: fully merged by ancestry
    worktree_ops._prune_orphaned_branches(str(repo))
    assert info["branch"] in _branches(repo)  # registered worktree

    shutil.rmtree(info["path"])  # tree and registration gone, child not finalized: still live
    worktree_ops._prune_orphaned_branches(str(repo))
    assert info["branch"] in _branches(repo)

    sw.finalize_subagent_worktree(info)  # finished: now it goes
    worktree_ops._prune_orphaned_branches(str(repo))
    assert info["branch"] not in _branches(repo)
