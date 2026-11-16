"""An inbound Telegram download that fails through an outage waits for polling to recover.

The final fetch attempt failing with a transient error (502 Bad Gateway, TimedOut) parks the
message instead of dispatching a "could not be downloaded" turn. Polling's recovery re-fetches it
and runs the one real media turn; expiry or an exhausted re-fetch budget gives exactly one notice.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from telegram.error import BadRequest, NetworkError, TimedOut

from gateway.config import PlatformConfig
from plugins.platforms.telegram import telegram_media_fetch, telegram_media_pending as pending
from plugins.platforms.telegram.adapter import TelegramAdapter
from tests.gateway.test_telegram_documents import _make_file_obj, _make_message, _make_update


@pytest.fixture
def adapter(tmp_path, monkeypatch):
    monkeypatch.setattr("gateway.platforms.base.AUDIO_CACHE_DIR", tmp_path / "audio_cache")
    monkeypatch.setattr("gateway.platforms.base.DOCUMENT_CACHE_DIR", tmp_path / "doc_cache")
    monkeypatch.setattr("gateway.platforms.base.IMAGE_CACHE_DIR", tmp_path / "img_cache")
    monkeypatch.setattr(telegram_media_fetch, "FETCH_RETRY_DELAYS", (0.0, 0.0))
    pending.reset_for_tests()
    a = TelegramAdapter(PlatformConfig(enabled=True, token="123:fake-token"))
    a._is_user_authorized_from_message = lambda *_a, **_k: True
    a._should_process_message = lambda *_a, **_k: True
    a.handle_message = AsyncMock()
    a._polling_progress_accepting = True  # a live polling generation, as after connect()
    yield a
    pending.reset_for_tests()


def _voice_message(outcomes):
    """A voice note whose getFile raises each exception in ``outcomes`` once, then succeeds."""
    script = list(outcomes)

    async def get_file(*_a, **_k):
        if script:
            raise script.pop(0)
        return _make_file_obj(b"OggS" + b"\0" * 200)
    msg = _make_message()
    msg.voice = MagicMock(file_size=200)
    msg.voice.get_file = AsyncMock(side_effect=get_file)
    msg.reply_to_message = msg.quote = None
    return msg


def _outage(n):
    return [NetworkError("Bad Gateway") if i % 2 else TimedOut("Timed out") for i in range(n)]


async def _recover(adapter):
    """Polling's next accepted getUpdates after the outage, then let the re-fetch finish."""
    adapter._polling_progress_event = asyncio.Event()  # a new generation: health pending
    assert adapter._record_polling_progress(adapter._polling_generation)
    await _settle()


async def _settle():
    for q in list(pending._queues.values()):
        while q.drain is not None and not q.drain.done():
            await asyncio.wait_for(asyncio.shield(q.drain), 5)
    await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_outage_then_recovery_runs_one_real_turn(adapter):
    msg = _voice_message(_outage(3))  # the whole in-call retry window fails, as on Thrive
    await adapter._handle_media_message(_make_update(msg), MagicMock())
    adapter.handle_message.assert_not_awaited()  # no placeholder turn
    msg.reply_text.assert_not_awaited()  # no "couldn't download" notice yet
    assert pending.pending_count(adapter) == 1

    await _recover(adapter)
    adapter.handle_message.assert_awaited_once()
    event = adapter.handle_message.await_args.args[0]
    assert event.media_urls and event.media_types == ["audio/ogg"]
    assert "could not be downloaded" not in (event.text or "")
    assert "Telegram outage delayed its attachment" in event.text
    msg.reply_text.assert_not_awaited()
    assert pending.pending_count(adapter) == 0

    # Later healthy polling and another recovery find nothing: no second turn.
    await _recover(adapter)
    assert adapter._record_polling_progress(adapter._polling_generation)
    await _settle()
    adapter.handle_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_expiry_sends_one_notice_and_one_turn(adapter, monkeypatch):
    monkeypatch.setattr(pending, "PENDING_TTL_SECONDS", 0.05)
    msg = _voice_message(_outage(100))  # Telegram never comes back inside the window
    adapter._polling_progress_event = asyncio.Event()  # polling stays down
    await adapter._handle_media_message(_make_update(msg), MagicMock())
    assert pending.pending_count(adapter) == 1
    await asyncio.sleep(0.1)  # the expiry timer fires and tries once more
    await _settle()
    msg.reply_text.assert_awaited_once()
    adapter.handle_message.assert_awaited_once()
    event = adapter.handle_message.await_args.args[0]
    assert "could not be downloaded" in event.text and not event.media_urls
    assert pending.pending_count(adapter) == 0
    await _recover(adapter)
    msg.reply_text.assert_awaited_once()
    adapter.handle_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_refetch_budget_ends_in_one_notice(adapter):
    msg = _voice_message(_outage(100))
    await adapter._handle_media_message(_make_update(msg), MagicMock())
    for _ in range(pending.MAX_REFETCHES + 2):  # flapping polling: every reconnect re-fetches
        await _recover(adapter)
    msg.reply_text.assert_awaited_once()
    adapter.handle_message.assert_awaited_once()
    assert "could not be downloaded" in adapter.handle_message.await_args.args[0].text
    assert pending.pending_count(adapter) == 0


@pytest.mark.asyncio
async def test_healthy_polling_refetches_on_the_backoff(adapter, monkeypatch):
    """getUpdates keeps working while getFile fails: the backoff timer re-fetches, long before expiry."""
    monkeypatch.setattr(pending, "RETRY_AFTER_SECONDS", 0.05)
    msg = _voice_message(_outage(3))
    adapter._polling_progress_event.set()  # polling never dropped
    await adapter._handle_media_message(_make_update(msg), MagicMock())
    assert adapter._record_polling_progress(adapter._polling_generation)
    await _settle()
    adapter.handle_message.assert_not_awaited()  # steady progress alone does not re-fetch
    await asyncio.sleep(0.1)
    await _settle()
    adapter.handle_message.assert_awaited_once()
    assert adapter.handle_message.await_args.args[0].media_urls
    msg.reply_text.assert_not_awaited()
    assert pending.pending_count(adapter) == 0


@pytest.mark.asyncio
async def test_backoff_waits_while_polling_is_down(adapter, monkeypatch):
    """With polling down, the backoff timer does not spend re-fetches; the recovery does the work."""
    monkeypatch.setattr(pending, "RETRY_AFTER_SECONDS", 0.02)
    msg = _voice_message(_outage(3))
    adapter._polling_progress_event = asyncio.Event()  # polling is down
    await adapter._handle_media_message(_make_update(msg), MagicMock())
    await asyncio.sleep(0.1)
    await _settle()
    adapter.handle_message.assert_not_awaited()
    assert pending._queues[pending._bot_key(adapter)].entries[0].refetches == 0
    await _recover(adapter)
    adapter.handle_message.assert_awaited_once()
    assert adapter.handle_message.await_args.args[0].media_urls


@pytest.mark.asyncio
async def test_an_expiry_during_a_running_drain_waits_without_spinning(adapter, monkeypatch):
    """B expires while A's last attempt is still fetching: B's timer re-checks each second, not every loop turn."""
    monkeypatch.setattr(pending, "PENDING_TTL_SECONDS", 0.2)
    monkeypatch.setattr(pending, "RETRY_AFTER_SECONDS", 60.0)
    calls = []
    on_timer = pending._on_timer
    monkeypatch.setattr(pending, "_on_timer", lambda *a: (calls.append(a), on_timer(*a)))
    adapter._polling_progress_event = asyncio.Event()  # polling stays down
    first = [True]

    async def slow_get_file(*_a, **_k):
        if first[0]:  # the original fetch fails at once; the expiry replay hangs in getFile
            raise TimedOut("Timed out")
        await asyncio.sleep(0.3)
        raise TimedOut("Timed out")
    a = _voice_message([])
    a.voice.get_file = AsyncMock(side_effect=slow_get_file)
    await adapter._handle_media_message(_make_update(a), MagicMock())
    first[0] = False
    await asyncio.sleep(0.05)
    b = _voice_message(_outage(100))
    await adapter._handle_media_message(_make_update(b), MagicMock())
    await asyncio.sleep(1.6)  # A's expiry replay (~0.9 s) runs while B's deadline passes
    await _settle()
    await asyncio.sleep(1.1)
    await _settle()
    assert len(calls) < 10
    a.reply_text.assert_awaited_once()
    b.reply_text.assert_awaited_once()
    assert adapter.handle_message.await_count == 2
    assert pending.pending_count(adapter) == 0


@pytest.mark.asyncio
async def test_concurrent_recoveries_replay_once(adapter):
    msg = _voice_message(_outage(3))
    await adapter._handle_media_message(_make_update(msg), MagicMock())
    adapter._polling_progress_event = asyncio.Event()
    adapter._record_polling_progress(adapter._polling_generation)
    adapter._polling_progress_event = asyncio.Event()
    adapter._record_polling_progress(adapter._polling_generation)  # second recovery while the first drains
    await _settle()
    adapter.handle_message.assert_awaited_once()
    assert adapter.handle_message.await_args.args[0].media_urls
    msg.reply_text.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["refusal", "webhook", "album"])
async def test_non_outage_failures_keep_the_immediate_notice(adapter, case):
    msg = _voice_message([BadRequest("Wrong file_id")] if case == "refusal" else _outage(100))
    if case == "webhook":
        adapter._webhook_mode = True
    if case == "album":  # an album item joins its album's turn, as before
        msg.media_group_id = "g1"
        adapter._queue_media_group_event = AsyncMock()
    await adapter._handle_media_message(_make_update(msg), MagicMock())
    msg.reply_text.assert_awaited_once()
    dispatched = adapter._queue_media_group_event if case == "album" else adapter.handle_message
    dispatched.assert_awaited_once()
    assert "could not be downloaded" in dispatched.await_args.args[-1].text
    assert pending.pending_count(adapter) == 0


@pytest.mark.asyncio
async def test_direct_cache_call_outside_the_handler_is_not_parked(adapter):
    """Only _handle_media_message can park: a direct caller still gets the note on its event."""
    from gateway.platforms.event import MessageEvent, MessageType
    msg = _voice_message(_outage(100))
    event = MessageEvent(text="", message_type=MessageType.VOICE)
    await adapter._cache_inbound_av(msg, event, msg.voice, "voice message", "voice", ".ogg", "audio/ogg")
    assert "could not be downloaded" in event.text
    assert pending.pending_count(adapter) == 0


@pytest.mark.asyncio
async def test_local_cache_write_error_still_notifies_at_once(adapter, monkeypatch):
    """Only a getFile/download failure waits; a disk error after a good download notifies now."""
    monkeypatch.setattr("plugins.platforms.telegram.adapter.cache_audio_from_bytes_async",
                        AsyncMock(side_effect=OSError(28, "No space left on device")))
    msg = _voice_message([])
    await adapter._handle_media_message(_make_update(msg), MagicMock())
    msg.reply_text.assert_awaited_once()
    adapter.handle_message.assert_awaited_once()
    assert pending.pending_count(adapter) == 0


def test_parked_update_rebinds_to_a_rebuilt_adapters_bot():
    """A reconnect-watcher rebuild closes the old Bot; the replay downloads through the live one."""
    from types import SimpleNamespace
    from telegram import Bot, Update
    old, live = Bot("123:fake-token"), Bot("123:fake-token")
    update = Update.de_json({"update_id": 5, "message": {
        "message_id": 9, "date": 0, "chat": {"id": 1, "type": "private"},
        "from": {"id": 1, "is_bot": False, "first_name": "I"},
        "voice": {"file_id": "F", "file_unique_id": "U", "duration": 3}}}, old)
    rebound = pending._rebind(SimpleNamespace(_bot=live), update)
    assert rebound.message.voice.get_bot() is live and rebound.message.voice.file_id == "F"
    assert pending._rebind(SimpleNamespace(_bot=old), update) is update
