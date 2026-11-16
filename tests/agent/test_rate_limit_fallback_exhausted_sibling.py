"""A 429 falls back on the first attempt when the pool's only other entry cannot serve the model.

Regression for the shape seen in production: an Anthropic pool with one usable
credential and one credential benched for a day. The usable one gets a 429. The old hinge
(``has_available() and len(entries()) > 1``) counted the benched entry as somewhere to rotate
to, so ``route_classified_error`` fell through to the ``Retry-After`` sleep (capped at 600 s)
instead of activating the fallback chain, and every new user turn restarted that sleep.

Drives ``route_classified_error`` with a real ``CredentialPool`` on a temp auth store and a
stub agent, at retry_count 0 (the first attempt).
"""
import json
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agent.error_classifier import ClassifiedError, FailoverReason
from agent.turn_recovery import route_classified_error
from agent.turn_retry_state import TurnRetryState

KEY_A = "sk-ant...000a"
KEY_B = "sk-ant...000b"
MODEL_X = "claude-opus-4-5"
MODEL_Y = "claude-sonnet-4-5"


def _entry(entry_id: str, key: str, priority: int, **extra) -> dict:
    row = {"id": entry_id, "label": entry_id, "auth_type": "api_key", "priority": priority,
           "source": "manual", "access_token": key}
    row.update(extra)
    return row


def _exhausted_for_a_day() -> dict:
    now = time.time()
    return {"last_status": "exhausted", "last_status_at": now, "last_error_code": 429,
            "last_error_reason": "rate_limit", "last_error_reset_at": now + 20 * 3600}


@pytest.fixture
def auth_store(tmp_path, monkeypatch):
    root = tmp_path / "hermes-root"
    root.mkdir()
    (tmp_path / "fakehome").mkdir()
    monkeypatch.setenv("HOME", str(tmp_path / "fakehome"))
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "fakehome"))
    for var in ("ANTHROPIC_TOKEN", "ANTHROPIC_API_KEY", "CLAUDE_CODE_OAUTH_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("HERMES_HOME", str(root))
    import hermes_constants
    hermes_constants._default_hermes_root_memo = None  # type: ignore[attr-defined]
    # Fallback activation itself is out of scope: only whether the hinge asks for it.
    import agent.conversation_loop as loop
    monkeypatch.setattr(loop, "_arm_fallback_restart", lambda agent, api_messages, prompt, _retry: prompt)
    return root / "auth.json"


def _pool(auth_store, entries):
    auth_store.write_text(json.dumps({"credential_pool": {"anthropic": entries}}))
    from agent.credential_pool import load_pool
    return load_pool("anthropic")


def _route(pool, *, model=MODEL_X):
    agent = SimpleNamespace(
        provider="anthropic", model=model, log_prefix="", compression_enabled=True,
        _credential_pool=pool, api_key=KEY_A, _credential_pool_entry_id="seat-a",
        _fallback_index=0, _fallback_chain=[{"provider": "openai-codex", "model": "gpt-5.5"}],
        _buffer_diagnostic_status=MagicMock(), _try_activate_fallback=MagicMock(return_value=True),
    )
    classified = ClassifiedError(reason=FailoverReason.rate_limit, status_code=429, provider="anthropic",
                                 model=model, message="This request would exceed your account's rate limit.")
    api_error = Exception("HTTP 429 rate_limit_error")
    api_error.status_code = 429
    verdict = route_classified_error(
        agent, api_error, classified, TurnRetryState(), error_msg=str(api_error), error_context={},
        recovered_with_pool=False, base_url="https://api.anthropic.com", model=model,
        messages=[{"role": "user", "content": "hi"}], api_messages=[], system_message="",
        active_system_prompt="", conversation_history=[], retry_count=0, max_retries=3,
        compression_attempts=0, max_compression_attempts=3, api_call_count=1, effective_task_id="t",
    )
    return verdict, agent._try_activate_fallback


def test_exhausted_sibling_falls_back_on_the_first_429(auth_store):
    pool = _pool(auth_store, [_entry("seat-a", KEY_A, 0), _entry("seat-b", KEY_B, 1, **_exhausted_for_a_day())])
    # What the old hinge saw: something available, more than one entry -> "wait for rotation".
    assert pool.has_available(model=MODEL_X) is True and len(pool.entries()) == 2

    verdict, fallback = _route(pool)

    fallback.assert_called_once()
    assert verdict.action == "break"


def test_two_usable_entries_still_rotate_instead_of_falling_back(auth_store):
    pool = _pool(auth_store, [_entry("seat-a", KEY_A, 0), _entry("seat-b", KEY_B, 1)])

    verdict, fallback = _route(pool)

    fallback.assert_not_called()
    assert verdict.action == "fallthrough"


def test_single_entry_pool_falls_back_as_before(auth_store):
    pool = _pool(auth_store, [_entry("seat-a", KEY_A, 0)])

    verdict, fallback = _route(pool)

    fallback.assert_called_once()
    assert verdict.action == "break"


def test_model_cooldown_on_the_other_entry_counts_as_unusable(auth_store):
    cooled = {"model_cooldowns": {MODEL_X: time.time() + 3600}}
    pool = _pool(auth_store, [_entry("seat-a", KEY_A, 0), _entry("seat-b", KEY_B, 1, **cooled)])

    verdict, fallback = _route(pool, model=MODEL_X)
    fallback.assert_called_once()
    assert verdict.action == "break"

    # The cooldown is per model: another model can still rotate onto seat-b.
    verdict, fallback = _route(pool, model=MODEL_Y)
    fallback.assert_not_called()
    assert verdict.action == "fallthrough"
