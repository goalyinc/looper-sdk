"""Cancellation and creation identity regressions."""

import asyncio

import httpx
import pytest

from looper_sdk import (
    CreationError,
    CreationTimeout,
    RunStateLost,
    SessionClosed,
    SessionCloseTimeout,
    SessionExpired,
)
from looper_sdk.proto.v1 import runs_pb2

from .test_lifecycle import CONFIG, Api, connect
from .test_transport import _failure


@pytest.fixture(autouse=True)
def immediate_shutdown(monkeypatch):
    monkeypatch.setattr("looper_sdk.transport.CLOSE_GRACE_S", 0, raising=False)


def test_timeout_cancellation_during_cleanup_reaches_the_outer_timeout():
    async def run():
        api = Api()

        async def handler(request):
            if request.url.path.endswith(":close"):
                await asyncio.sleep(0.05)
            return await api(request)

        service = await connect(handler)
        try:
            with pytest.raises(TimeoutError):
                async with asyncio.timeout(0.01):
                    async with service:
                        raise ValueError("body failed before timeout")
        finally:
            await service.close()

    asyncio.run(run())


def test_session_shutdown_is_an_ordinary_failure_for_an_uncancelled_caller():
    async def run():
        service = await connect(Api())
        entered = asyncio.Event()

        async def work():
            entered.set()
            await asyncio.Event().wait()

        async def caller():
            with pytest.raises(SessionClosed):
                await service.session.run(work)
            caller = asyncio.current_task()
            assert caller is not None and caller.cancelling() == 0

        task = asyncio.create_task(caller())
        await entered.wait()
        await service.close()
        await task

    asyncio.run(run())


def test_caller_cancellation_returns_promptly_while_owned_work_drains():
    async def run():
        service = await connect(Api())
        entered, release = asyncio.Event(), asyncio.Event()

        async def work():
            entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                await release.wait()

        task = asyncio.create_task(service.session.run(work))
        await entered.wait()
        try:
            task.cancel()
            await asyncio.sleep(0.01)
            assert task.done()
            with pytest.raises(asyncio.CancelledError):
                await task
        finally:
            task.cancel()
            release.set()
            await asyncio.gather(task, return_exceptions=True)
            await service.close()

    asyncio.run(run())


def test_close_deadline_does_not_block_the_server_close_or_lose_owned_tasks():
    async def run():
        api = Api()
        service = await connect(api)
        entered, release = asyncio.Event(), asyncio.Event()

        async def work():
            entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                await release.wait()

        task = asyncio.create_task(service.session.run(work))
        await entered.wait()
        try:
            with pytest.raises(SessionCloseTimeout) as raised:
                await service.close(timeout_s=0.02)
            assert type(raised.value).__name__ == "SessionCloseTimeout"
            assert raised.value.retryable
            assert any(request.url.path.endswith(":close") for request in api.calls)
            assert service.session.transport.client.is_closed
            assert service.session._closing is not None
            assert not service.session._closing.done()
        finally:
            task.cancel()
            release.set()
            await asyncio.gather(task, return_exceptions=True)
            await service.close()

    asyncio.run(run())


def test_cancelling_close_does_not_wait_for_cleanup_or_replace_cancellation():
    async def run():
        api = Api()
        closing, release = asyncio.Event(), asyncio.Event()

        async def handler(request):
            if request.url.path.endswith(":close"):
                closing.set()
                await release.wait()
            return await api(request)

        service = await connect(handler)
        task = asyncio.create_task(service.close())
        await closing.wait()
        try:
            task.cancel()
            await asyncio.sleep(0.01)
            assert task.done()
            with pytest.raises(asyncio.CancelledError):
                await task
        finally:
            task.cancel()
            release.set()
            await asyncio.gather(task, return_exceptions=True)
            await service.close()

    asyncio.run(run())


def test_expired_heartbeat_never_changes_the_creation_key_scope():
    async def run():
        api = Api()

        async def handler(request):
            if request.url.path.endswith(":heartbeat"):
                return _failure(409, SessionExpired("expired"))
            return await api(request)

        service = await connect(handler)
        try:
            original = service.session.id
            with pytest.raises(SessionExpired):
                await service.session._heartbeat()
            assert service.session.id == original
            assert sum(request.url.path == "/v1/sessions" for request in api.calls) == 1
        finally:
            await service.close()

    asyncio.run(run())


def test_public_session_reference_and_identity_cannot_be_rebound():
    async def run():
        async with await connect(Api()) as service:
            with pytest.raises(AttributeError):
                setattr(service, "session", object())
            with pytest.raises(AttributeError):
                setattr(service.session, "id", "another-session")

    asyncio.run(run())


def test_sdk_errors_expose_retry_semantics_as_data():
    assert SessionClosed("closed").retryable is False
    assert CreationTimeout(key="k", session_id="s", run_id=None).retryable is True


@pytest.mark.parametrize("stage", ["submission", "readiness"])
def test_creation_failure_chains_the_original_typed_error(stage):
    import looper_sdk

    async def run():
        api = Api((runs_pb2.Run.FAILED,))

        async def handler(request):
            if stage == "submission" and request.url.path == "/v1/runs":
                return _failure(503, looper_sdk.Unavailable("not admitted", retry_after_s=1000))
            return await api(request)

        async with await connect(handler) as service:
            with pytest.raises(RuntimeError) as raised:
                await service.create_training_client(CONFIG, key="creation")
            error = raised.value
            assert isinstance(error, looper_sdk.CreationError)
            expected = looper_sdk.Unavailable if stage == "submission" else RunStateLost
            assert isinstance(error.__cause__, expected)
            assert error.__cause__ is error.cause
            assert error.__cause__.__traceback__ is not None
            assert error.retryable == (stage == "submission")
            assert (error.key, error.session_id) == ("creation", "s")
            assert error.run_id == (None if stage == "submission" else "r")

    asyncio.run(run())


def test_a_late_result_cannot_revive_a_call_after_session_closure():
    async def run():
        service = await connect(Api())
        entered, release = asyncio.Event(), asyncio.Event()

        async def work():
            entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                await release.wait()
            return "late result"

        task = asyncio.create_task(service.session.run(work))
        await entered.wait()
        try:
            with pytest.raises(SessionCloseTimeout):
                await service.close(timeout_s=0.01)
            release.set()
            with pytest.raises(SessionClosed):
                await task
        finally:
            release.set()
            await service.close()

    asyncio.run(run())


def test_caller_cancellation_wins_when_session_shutdown_races_with_it():
    async def run():
        service = await connect(Api())
        entered = asyncio.Event()

        async def work():
            entered.set()
            await asyncio.Event().wait()

        caller = asyncio.create_task(service.session.run(work))
        await entered.wait()
        close = asyncio.create_task(service.close())
        caller.cancel()
        with pytest.raises(asyncio.CancelledError):
            await caller
        await close

    asyncio.run(run())


def test_caller_cancellation_is_not_replaced_by_a_cleanup_failure():
    async def run():
        api = Api()
        closing, release = asyncio.Event(), asyncio.Event()

        async def handler(request):
            if request.url.path.endswith(":close"):
                closing.set()
                await release.wait()
            return await api(request)

        class BrokenClose(httpx.MockTransport):
            async def aclose(self):
                raise OSError("cleanup failed")

        from looper_sdk import ServiceClient

        service = await ServiceClient.Config(base_url="http://looper", project="p").build(
            ServiceClient.Runtime(transport=BrokenClose(handler))
        )
        task = asyncio.create_task(service.close())
        await closing.wait()
        task.cancel()
        try:
            with pytest.raises(asyncio.CancelledError):
                await task
        finally:
            release.set()
            with pytest.raises(OSError, match="cleanup failed"):
                await service.close()

    asyncio.run(run())


def test_retrying_creation_after_expiry_cannot_allocate_another_run():
    async def run():
        api = Api()
        api.block = True
        expired = False

        async def handler(request):
            if expired and request.url.path in ("/v1/runs", "/v1/sessions/s:heartbeat"):
                return _failure(409, SessionExpired("expired"))
            return await api(request)

        async with await connect(handler) as service:
            with pytest.raises(CreationTimeout) as first:
                await service.create_training_client(CONFIG, key="stable", timeout_s=0.01)
            expired = True
            with pytest.raises(SessionExpired):
                await service.session._heartbeat()
            with pytest.raises(CreationError) as retry:
                await service.create_training_client(CONFIG, key=first.value.key)
            assert isinstance(retry.value.__cause__, SessionExpired)
            assert not retry.value.retryable
            assert first.value.session_id == service.session.id
            assert sum(request.url.path == "/v1/runs" for request in api.calls) == 1
            assert sum(request.url.path == "/v1/sessions" for request in api.calls) == 1

    asyncio.run(run())


def test_a_future_sampling_handle_shares_the_session_lifetime():
    from looper_sdk.transport import session_operation

    class SamplingHandle:
        def __init__(self, session):
            self._session = session
            self.entered = asyncio.Event()

        @session_operation
        async def sample(self):
            self.entered.set()
            await asyncio.Event().wait()

    async def run():
        service = await connect(Api())
        sampler = SamplingHandle(service.session)
        handled = asyncio.Event()

        async def sample():
            try:
                await sampler.sample()
            except SessionClosed:
                handled.set()

        async with asyncio.TaskGroup() as group:
            group.create_task(sample())
            await sampler.entered.wait()
            await service.close()
        assert handled.is_set()
        with pytest.raises(SessionClosed):
            await sampler.sample()

    asyncio.run(run())


@pytest.mark.parametrize("phase", ["http", "retry", "submit_lock"])
def test_close_interrupts_the_whole_training_call(phase):
    from looper_sdk import Unavailable

    async def run():
        api = Api()
        entered = asyncio.Event()

        async def handler(request):
            if request.url.path.endswith(":optimStep"):
                entered.set()
                if phase == "retry":
                    return _failure(503, Unavailable("temporarily unavailable", retry_after_s=60))
                await asyncio.Event().wait()
            return await api(request)

        service = await connect(handler)
        client = await service.create_training_client(CONFIG)
        if phase == "submit_lock":
            await client._submitting.acquire()
        task = asyncio.create_task(client.optim_step())
        try:
            if phase == "submit_lock":
                await asyncio.sleep(0)
                await asyncio.sleep(0)
            else:
                await entered.wait()
            await service.close()
            with pytest.raises(SessionClosed):
                await task
            assert not task.cancelled()
            if phase == "submit_lock":
                assert not entered.is_set()
        finally:
            if client._submitting.locked():
                client._submitting.release()
            await service.close()

    asyncio.run(run())


def test_heartbeat_shutdown_is_not_logged_as_a_failed_heartbeat(caplog):
    async def run():
        service = await connect(Api())
        service.session.heartbeat_interval_s = 0
        await service.close()
        assert "missed a heartbeat" not in caplog.text

    asyncio.run(run())


def test_close_keeps_heartbeats_alive_until_pending_creation_finishes(monkeypatch):
    monkeypatch.setattr("looper_sdk.transport.CLOSE_GRACE_S", 1, raising=False)

    async def run():
        from looper_sdk.proto.v1 import sessions_pb2

        from .test_lifecycle import answer

        api = Api()
        entered, release, heartbeat = asyncio.Event(), asyncio.Event(), asyncio.Event()

        async def handler(request):
            if request.url.path == "/v1/sessions":
                return answer(
                    sessions_pb2.OpenSessionResponse(
                        session=sessions_pb2.Session(id="s", project="p", heartbeat_interval_s=0.01)
                    )
                )
            if request.url.path.endswith(":heartbeat"):
                heartbeat.set()
                return answer(sessions_pb2.HeartbeatSessionResponse())
            if request.url.path == "/v1/runs":
                entered.set()
                await release.wait()
            return await api(request)

        service = await connect(handler)
        pending = asyncio.create_task(service.create_training_client(CONFIG))
        await entered.wait()
        closing = asyncio.create_task(service.close())
        try:
            await asyncio.wait_for(heartbeat.wait(), 0.2)
            assert not closing.done()
            assert not any(r.url.path.endswith(":close") for r in api.calls)
            with pytest.raises(SessionClosed):
                await service.create_training_client(CONFIG)
            release.set()
            client = await asyncio.wait_for(pending, 0.2)
            assert client.run_id == "r"
            await closing
            with pytest.raises(SessionClosed):
                await client.status()
        finally:
            release.set()
            await asyncio.gather(pending, closing, return_exceptions=True)
            await service.close()

    asyncio.run(run())


def test_close_cancels_unfinished_calls_only_after_its_grace_period(monkeypatch):
    monkeypatch.setattr("looper_sdk.transport.CLOSE_GRACE_S", 0.05, raising=False)

    async def run():
        api = Api()
        service = await connect(api)
        entered = asyncio.Event()

        async def work():
            entered.set()
            await asyncio.Event().wait()

        pending = asyncio.create_task(service.session.run(work))
        await entered.wait()
        closing = asyncio.create_task(service.close())
        await asyncio.sleep(0.01)
        assert not pending.done()
        assert not any(r.url.path.endswith(":close") for r in api.calls)
        await closing
        with pytest.raises(SessionClosed):
            await pending
        assert any(r.url.path.endswith(":close") for r in api.calls)

    asyncio.run(run())


def test_a_short_close_wait_does_not_cut_short_the_grace_period(monkeypatch):
    monkeypatch.setattr("looper_sdk.transport.CLOSE_GRACE_S", 1)

    async def run():
        service = await connect(Api())
        entered, release = asyncio.Event(), asyncio.Event()

        async def work():
            entered.set()
            await release.wait()
            return "finished"

        pending = asyncio.create_task(service.session.run(work))
        await entered.wait()
        try:
            with pytest.raises(SessionCloseTimeout):
                await service.close(timeout_s=0.01)
            assert not pending.done()
            release.set()
            assert await pending == "finished"
        finally:
            release.set()
            await asyncio.gather(pending, return_exceptions=True)
            await service.close()

    asyncio.run(run())
