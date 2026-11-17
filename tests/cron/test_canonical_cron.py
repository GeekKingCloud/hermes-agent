"""Agent cron must enter the owner before importing any agent-side storage."""
import asyncio
from contextlib import asynccontextmanager

from cron import scheduler


def test_agent_cron_is_transport_only_and_refuses_without_owner(monkeypatch):
    from hermes_cli import gateway_client
    calls = []

    @asynccontextmanager
    async def unavailable():
        calls.append('connect')
        raise gateway_client.GatewayClientError('owner_unavailable')
        yield

    monkeypatch.setattr(gateway_client, 'connect_gateway', unavailable)
    monkeypatch.setattr(scheduler, '_prepare_job_prompt', lambda *a: (_ for _ in ()).throw(
        AssertionError('agent prompt preparation belongs to the owner')))
    result = scheduler.run_job({'id': 'job', 'prompt': 'test'}, execution_id='fire')
    assert calls == ['connect']
    assert result[0] is False and 'owner_unavailable' in result[3]


def test_owner_ticker_uses_direct_admission_without_loopback(monkeypatch):
    from gateway import session_cron
    from types import SimpleNamespace
    from hermes_constants import get_hermes_home
    import threading

    async def probe():
        authority = SimpleNamespace(db=SimpleNamespace(db_path=str(get_hermes_home() / 'state.db')))
        session_cron.bind_owner(authority)
        calls = []

        async def request(owner, operation, params, actor=None):
            assert owner is authority
            calls.append((operation, threading.get_ident()))
            if operation == 'submit':
                return {'session_id': 'cron-fire', 'admission_id': 'admit'}
            return {'status': 'terminal', 'result': [True, 'document', 'answer', None]}

        monkeypatch.setattr(session_cron, 'operation', request)
        # A nested job must not inherit its parent's execution authorization.
        token = session_cron._execution.set((authority, 'parent-session', 'parent-job', 'parent-fire'))
        try:
            result = await asyncio.to_thread(scheduler.run_job, {'id': 'job'}, execution_id='fire')
        finally:
            session_cron._execution.reset(token)
        assert result == (True, 'document', 'answer', None)
        assert [op for op, _ in calls] == ['submit', 'status']
        assert all(tid == threading.get_ident() for _, tid in calls)
        session_cron.unbind_owner(authority)

    asyncio.run(probe())


def test_lost_cron_ack_retains_identity_instead_of_reporting_failure(tmp_path, monkeypatch):
    from cron.scheduler_authority import run_canonical_job, CronExecutionUnknown
    from hermes_cli import gateway_client
    from hermes_constants import get_hermes_home
    import json
    import pytest

    class Peer:
        async def rpc(self, method, **params):
            raise gateway_client.GatewayClientError('Gateway disconnected')

    @asynccontextmanager
    async def connected():
        yield Peer()

    monkeypatch.setattr(gateway_client, 'connect_gateway', connected)
    with pytest.raises(CronExecutionUnknown):
        run_canonical_job({'id': 'job'}, execution_id='fire')
    records = list((get_hermes_home() / 'cron' / 'admissions').glob('*.json'))
    assert len(records) == 1
    assert json.loads(records[0].read_text())['params']['request_id'] == 'fire'


def _observe_scripted(monkeypatch, answer, *, inactivity='600'):
    """Drive the observer against an owner whose ``cron.status``/``cron.recover`` answers are
    scripted by ``answer(n, now)`` (n counts both), on a fake clock and without real backoff
    sleeps. ``calls`` also records each dial."""
    from types import SimpleNamespace
    from cron import scheduler_authority
    from hermes_cli import gateway_client

    now = [0.0]
    calls = []

    class Peer:
        async def rpc(self, method, **params):
            calls.append(method)
            if method == 'cron.submit':
                return {'session_id': 'cron-fire', 'admission_id': 'admit'}
            return answer(calls.count('cron.status') + calls.count('cron.recover'), now)

    @asynccontextmanager
    async def connected():
        calls.append('dial')
        yield Peer()

    real_sleep = asyncio.sleep

    async def no_wait(delay):
        await real_sleep(0)

    monkeypatch.setenv('HERMES_CRON_TIMEOUT', inactivity)
    monkeypatch.setattr(gateway_client, 'connect_gateway', connected)
    monkeypatch.setattr(scheduler_authority, 'time', SimpleNamespace(monotonic=lambda: now[0]), raising=False)
    monkeypatch.setattr(asyncio, 'sleep', no_wait)
    return calls


def test_socket_cut_by_a_restart_wait_reconnects_and_recovers_the_run(monkeypatch):
    """A restart's after-turn wait cuts the firer's socket (1013) while the run it waits for goes
    on: the firer dials again, recovers the same admission and books its real result."""
    from cron.scheduler_authority import run_canonical_job
    from hermes_cli import gateway_client

    def answer(n, now):
        if n == 1:
            raise gateway_client.GatewayClientError(gateway_client.DISCONNECTED)
        if n == 2:  # the recover on the new socket
            return {'status': 'started', 'session_id': 'cron-fire', 'admission_id': 'admit'}
        return {'status': 'terminal', 'result': [True, 'document', 'answer', None]}

    calls = _observe_scripted(monkeypatch, answer)
    assert run_canonical_job({'id': 'job'}, execution_id='fire') == (True, 'document', 'answer', None)
    assert calls == ['dial', 'cron.submit', 'cron.status', 'dial', 'cron.recover', 'cron.status']


def test_owner_gone_past_inactivity_budget_stays_unverified(monkeypatch):
    from cron.scheduler_authority import run_canonical_job, CronExecutionUnknown
    from hermes_cli import gateway_client
    import pytest

    def answer(n, now):
        now[0] += 30
        raise gateway_client.GatewayClientError(gateway_client.DISCONNECTED)

    calls = _observe_scripted(monkeypatch, answer, inactivity='90')
    with pytest.raises(CronExecutionUnknown):
        run_canonical_job({'id': 'job'}, execution_id='fire')
    assert calls == ['dial', 'cron.submit', 'cron.status', 'dial', 'cron.recover', 'dial', 'cron.recover']


def test_owner_refusal_during_status_poll_is_not_retried(monkeypatch):
    from cron.scheduler_authority import run_canonical_job, CronExecutionUnknown
    from hermes_cli import gateway_client
    import pytest

    def answer(n, now):
        raise gateway_client.GatewayRPCError('permission_denied')

    calls = _observe_scripted(monkeypatch, answer)
    with pytest.raises(CronExecutionUnknown):
        run_canonical_job({'id': 'job'}, execution_id='fire')
    assert calls == ['dial', 'cron.submit', 'cron.status']
