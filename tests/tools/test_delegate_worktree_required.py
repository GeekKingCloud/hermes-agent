"""delegation.worktree_isolation: required never runs a child in the parent's checkout."""

import subprocess
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tools import delegate_tool, terminal_tool


def _git(repo, *args):
    subprocess.run(["git", "-c", "user.name=Test", "-c", "user.email=test@example.invalid", *args],
                   cwd=repo, check=True, capture_output=True)


def _delegate(tmp_path, monkeypatch, mode, workspace="repo"):
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))
    monkeypatch.setattr(delegate_tool, "_load_config", lambda: {"worktree_isolation": mode})
    parent_dir = tmp_path / "parent"
    parent_dir.mkdir()
    if workspace in ("repo", "unborn"):
        _git(parent_dir, "init", "-q")
    if workspace == "repo":
        _git(parent_dir, "commit", "--allow-empty", "-qm", "seed")

    parent = SimpleNamespace(_current_task_id="wt-parent", _active_children=[], _active_children_lock=threading.Lock())
    terminal_tool.record_session_cwd("wt-parent", str(parent_dir))
    child_cwds = []

    def converse(**kwargs):
        cwd = terminal_tool.get_session_cwd(kwargs["task_id"])
        child_cwds.append(cwd)
        (Path(cwd) / "child-edit.txt").write_text("edit")
        return {"final_response": "done", "completed": True, "api_calls": 1, "messages": []}

    child = SimpleNamespace(_subagent_id="wt-child", _delegate_role="leaf", close=Mock(),
                            run_conversation=Mock(side_effect=converse))
    parent._active_children.append(child)
    try:
        entry = delegate_tool._run_single_child(0, "edit a file", child=child, parent_agent=parent)
    finally:
        terminal_tool.clear_session_cwd("wt-child")
        terminal_tool.clear_session_cwd("wt-parent")
    return entry, child, parent_dir, child_cwds


@pytest.mark.parametrize("cause", ["not_git", "unborn_head", "non_local_backend", "cwd_bind_fails"])
def test_required_refuses_child_without_a_worktree(tmp_path, monkeypatch, cause):
    workspace = {"not_git": "plain", "unborn_head": "unborn"}.get(cause, "repo")
    if cause == "non_local_backend":
        monkeypatch.setenv("TERMINAL_ENV", "docker")
    if cause == "cwd_bind_fails":
        record = terminal_tool.record_session_cwd

        def record_or_fail(key, cwd):
            if cwd and "/.worktrees/" in cwd:
                raise OSError("bind failed")
            record(key, cwd)

        monkeypatch.setattr(terminal_tool, "record_session_cwd", record_or_fail)

    entry, child, parent_dir, _ = _delegate(tmp_path, monkeypatch, "required", workspace)

    child.run_conversation.assert_not_called()
    assert entry["status"] == "error"
    assert "'required'" in entry["error"] and "not run" in entry["error"]
    assert not (parent_dir / "child-edit.txt").exists()


def test_required_runs_child_in_its_own_worktree(tmp_path, monkeypatch):
    entry, _, parent_dir, child_cwds = _delegate(tmp_path, monkeypatch, "required")

    assert entry["status"] == "completed"
    assert child_cwds == [entry["worktree"]["path"]]
    assert Path(child_cwds[0]).parent == parent_dir / ".worktrees"
    assert not (parent_dir / "child-edit.txt").exists()


def test_best_effort_still_falls_back_to_the_shared_workspace(tmp_path, monkeypatch):
    entry, _, parent_dir, child_cwds = _delegate(tmp_path, monkeypatch, True, "plain")

    assert entry["status"] == "completed"
    assert child_cwds == [str(parent_dir)]
    assert "worktree" not in entry
