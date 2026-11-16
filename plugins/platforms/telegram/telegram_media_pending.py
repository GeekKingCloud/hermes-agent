"""Inbound Telegram media whose download failed during a Telegram outage waits for the recovery.

``fetch_telegram_file`` retries one fetch for about 45 s. A Bot API outage (502 Bad Gateway,
timeouts) lasts longer; the message was then dispatched at once as a "could not be downloaded" turn
and the user had to send the file again. Now a message whose final attempt failed with a transient
error (``NetworkError``/``TimedOut``/5xx, a bare transport error or a pool timeout) is parked here
instead of dispatched. It is fetched again when polling reports healthy again (the first accepted
getUpdates of a polling generation: "polling recovered"), or by its backoff timer
(``RETRY_AFTER_SECONDS``, doubling to ``RETRY_AFTER_MAX_SECONDS``) while polling stays healthy and
downloads still fail, and then runs as the normal media turn. Only when ``MAX_REFETCHES`` re-fetches
fail, or the entry reaches ``PENDING_TTL_SECONDS`` (the timer makes one last attempt then), does the
message take the existing failure path: one "could not be downloaded" notice and one turn carrying
the note. Only a failure raised inside ``fetch_telegram_file`` parks; a local cache-write error
still notifies at once. The message is dispatched once: an entry leaves
the list before it is replayed, and only that replay can put it back.

Scope: a polling-mode, non-album message handled by ``_handle_media_message`` (photo, voice, audio,
video, video note, document). Webhook mode, album items, observed group media, replied-to media and
stickers keep their immediate paths. The list is process memory, keyed by bot, so it survives the
reconnect watcher's adapter rebuild. A gateway exit inside the window loses it: the update is
already receipted and Telegram does not redeliver it. ``log_unserved`` names that at disconnect.
"""

from __future__ import annotations

import asyncio
import contextvars
import functools
import hashlib
import logging
import time
import weakref
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger("plugins.platforms.telegram.adapter")

# How long a parked message waits for Telegram to recover before the failure notice.
PENDING_TTL_SECONDS = 15 * 60
# Re-fetches per message (each a full fetch with its own retries); bounds a flapping connection.
MAX_REFETCHES = 8
# Parked messages per bot; beyond it a failure takes the immediate notice path, as before.
MAX_PENDING_PER_BOT = 32
# While polling is healthy but downloads fail (getFile-only fault), re-fetch after this, doubling.
RETRY_AFTER_SECONDS = 60.0
RETRY_AFTER_MAX_SECONDS = 240.0


class MediaFetchDeferred(Exception):
    """Raised by the failure path instead of notifying; the media handler parks the update."""

    def __init__(self, kind: str):
        super().__init__(kind)
        self.kind = kind


@dataclass
class PendingMedia:
    update: Any
    kind: str
    parked_at: float
    deadline: float
    refetches: int = 0
    timer: Optional[asyncio.TimerHandle] = field(default=None, repr=False)


@dataclass
class _BotQueue:
    entries: list = field(default_factory=list)
    adapter: Any = None  # weakref to the newest adapter for this bot
    drain: Optional[asyncio.Task] = None
    again: bool = False  # a recovery arrived while a drain ran: drain everything once more


_queues: dict[str, _BotQueue] = {}
# Set by the media handler wrapper: only its own failure path may park.
_handling: contextvars.ContextVar[bool] = contextvars.ContextVar("telegram_media_handling", default=False)
# The entry the current task is replaying, so the failure path re-parks or finishes that same entry.
_replaying: contextvars.ContextVar[Optional[PendingMedia]] = contextvars.ContextVar(
    "telegram_media_replay", default=None)


def _bot_key(adapter: Any) -> str:
    token = str(getattr(getattr(adapter, "config", None), "token", "") or "")
    return hashlib.sha256(token.encode()).hexdigest()[:16] if token else f"adapter-{id(adapter)}"


def _queue(adapter: Any) -> _BotQueue:
    q = _queues.setdefault(_bot_key(adapter), _BotQueue())
    q.adapter = weakref.ref(adapter)
    return q


def pending_count(adapter: Any) -> int:
    q = _queues.get(_bot_key(adapter))
    return len(q.entries) if q else 0


def _raised_by_fetch(exc: BaseException) -> bool:
    """The error came out of ``fetch_telegram_file`` (getFile/download), not a later local step."""
    from plugins.platforms.telegram.telegram_media_fetch import fetch_telegram_file
    code = getattr(fetch_telegram_file, "__code__", None)
    tb = exc.__traceback__
    while tb is not None:
        if tb.tb_frame.f_code is code:
            return True
        tb = tb.tb_next
    return False


def is_deferrable_fetch_error(exc: Exception) -> bool:
    """A failure a later attempt can cure: the fetch retry's transport set, plus a pool timeout."""
    from plugins.platforms.telegram.adapter import TelegramAdapter as tg
    from plugins.platforms.telegram.telegram_media_fetch import is_retryable_fetch_error
    if tg._is_bad_request_error(exc) or tg._looks_like_auth_error(exc):
        return False
    return is_retryable_fetch_error(exc) or tg._looks_like_pool_timeout(exc)


def should_defer(adapter: Any, msg: Any, exc: Exception) -> bool:
    """Whether this failed download waits for Telegram's recovery instead of notifying now."""
    if not _handling.get() or getattr(adapter, "_webhook_mode", False) or getattr(msg, "media_group_id", None):
        return False
    if not is_deferrable_fetch_error(exc) or not _raised_by_fetch(exc):
        return False
    entry = _replaying.get()
    if entry is not None:  # a replay keeps waiting while it has re-fetches and time left
        return entry.refetches < MAX_REFETCHES and time.monotonic() < entry.deadline
    return pending_count(adapter) < MAX_PENDING_PER_BOT


def _park(adapter: Any, update: Any, kind: str) -> None:
    q = _queue(adapter)
    now = time.monotonic()
    entry = _replaying.get() or PendingMedia(update=update, kind=kind, parked_at=now,
                                             deadline=now + PENDING_TTL_SECONDS)
    q.entries.append(entry)
    _arm(_bot_key(adapter), entry, min(RETRY_AFTER_SECONDS * 2 ** entry.refetches, RETRY_AFTER_MAX_SECONDS))
    logger.warning(
        "[Telegram] %s: download deferred until Telegram recovers (re-fetches %d/%d, %d pending, expires in %ds)",
        kind, entry.refetches, MAX_REFETCHES, len(q.entries), int(entry.deadline - now))


def deferring_media_fetch(handler):
    """Decorate ``_handle_media_message``: a deferred download parks the update; nothing dispatches."""
    @functools.wraps(handler)
    async def wrapper(adapter, update, context):
        token = _handling.set(True)
        try:
            return await handler(adapter, update, context)
        except MediaFetchDeferred as deferred:
            adapter._accept_update()  # handed to the pending list: served, not unhandled
            _park(adapter, update, deferred.kind)
            return None
        finally:
            _handling.reset(token)
    return wrapper


def on_polling_progress(adapter: Any, *, recovered: bool) -> None:
    """Accepted getUpdates progress: on the recovery transition, re-fetch every parked message."""
    q = _queues.get(_bot_key(adapter))
    if q is None or not q.entries:
        return
    q.adapter = weakref.ref(adapter)
    if recovered:
        _start_drain(q, None)


def _arm(key: str, entry: PendingMedia, delay: float) -> None:
    """(Re)arm the entry's timer: its backoff re-fetch, never later than its deadline."""
    if entry.timer is not None:
        entry.timer.cancel()
    delay = max(0.0, min(delay, entry.deadline - time.monotonic()))
    entry.timer = asyncio.get_running_loop().call_later(delay, _on_timer, key, entry)


def _on_timer(key: str, entry: PendingMedia) -> None:
    """Backoff or deadline: re-fetch this entry if polling is healthy or it is due to expire."""
    entry.timer = None
    q = _queues.get(key)
    if q is None or entry not in q.entries:
        return
    adapter = q.adapter() if q.adapter is not None else None
    expired = time.monotonic() >= entry.deadline
    if q.drain is not None and not q.drain.done():
        # A drain is running; look again shortly. Not via _arm: its deadline cap is 0 for an expired
        # entry, which would re-fire every loop iteration for as long as the drain's replay takes.
        entry.timer = asyncio.get_running_loop().call_later(1.0, _on_timer, key, entry)
    elif expired or (adapter is not None and adapter._polling_progress_event.is_set()):
        _start_drain(q, entry)
    else:  # polling is down: its recovery re-fetches; keep the budget for that
        _arm(key, entry, min(RETRY_AFTER_SECONDS * 2 ** entry.refetches, RETRY_AFTER_MAX_SECONDS))


def _start_drain(q: _BotQueue, only: Optional[PendingMedia]) -> None:
    if q.drain is not None and not q.drain.done():
        q.again = True  # a recovery during a drain: drain everything once more
        return
    adapter = q.adapter() if q.adapter is not None else None
    if adapter is None or getattr(adapter, "_teardown_started", False):
        return  # the bot's next live adapter drains it on its first polling progress
    q.drain = asyncio.get_running_loop().create_task(_drain(adapter, q, only))


async def _drain(adapter: Any, q: _BotQueue, only: Optional[PendingMedia]) -> None:
    """Replay parked entries oldest first, one at a time; a re-parked entry waits for the next trigger."""
    # This task copied its starter's context; none of that is an update this drain is handling.
    _handling.set(False)
    _replaying.set(None)
    if (admission := getattr(adapter, "_update_admission", None)) is not None:
        admission.set(None)
    while True:
        q.again = False
        for entry in [e for e in q.entries if only is None or e is only]:
            if getattr(adapter, "_teardown_started", False):
                return
            if entry not in q.entries:
                continue
            q.entries.remove(entry)
            try:
                await _replay(adapter, entry)
            except asyncio.CancelledError:
                if entry not in q.entries:  # cancelled mid-replay (teardown): keep it for the next adapter
                    q.entries.insert(0, entry)
                raise
            except Exception:
                logger.warning("[Telegram] %s: deferred media replay failed", entry.kind, exc_info=True)
            if entry not in q.entries and entry.timer is not None:  # finished: its timer is moot
                entry.timer.cancel()
                entry.timer = None
        if not q.again or not q.entries:
            return
        only = None


def _rebind(adapter: Any, update: Any) -> Any:
    """The parked update bound to the live bot: a rebuilt adapter shut the old bot's client down."""
    bot = getattr(adapter, "_bot", None)
    try:
        from telegram import Update
        if bot is not None and isinstance(update, Update) and update.get_bot() is not bot:
            return Update.de_json(update.to_dict(), bot)
    except Exception:
        logger.debug("[Telegram] Could not rebind a parked update to the live bot", exc_info=True)
    return update


async def _replay(adapter: Any, entry: PendingMedia) -> None:
    """Run the parked message through the media handler again; its failure path decides what follows."""
    entry.refetches += 1
    logger.info("[Telegram] %s: re-fetching deferred download (attempt %d/%d, parked %ds ago)",
                entry.kind, entry.refetches, MAX_REFETCHES, int(time.monotonic() - entry.parked_at))
    token = _replaying.set(entry)
    try:
        await adapter._handle_media_message(_rebind(adapter, entry.update), None)
    finally:
        _replaying.reset(token)


def replay_note() -> Optional[str]:
    """The agent-visible note for a replayed message (fetched or finally failed), else ``None``."""
    entry = _replaying.get()
    if entry is None:
        return None
    minutes = max(1, round((time.monotonic() - entry.parked_at) / 60))
    return f"[This message was sent about {minutes} min ago; a Telegram outage delayed its attachment.]"


def log_unserved(adapter: Any) -> None:
    """At disconnect: what a gateway exit would lose (a replacement adapter still re-fetches it)."""
    if n := pending_count(adapter):
        logger.warning("[Telegram] %d inbound media download(s) still deferred at disconnect; a replacement "
                       "adapter re-fetches them, a gateway exit loses them", n)


def reset_for_tests() -> None:
    for q in _queues.values():
        for e in q.entries:
            if e.timer is not None:
                e.timer.cancel()
        if q.drain is not None:
            q.drain.cancel()
    _queues.clear()
