"""SDK readiness and explicit closure, without a server."""

import asyncio
import logging

import httpx
import pytest

from looper_sdk import (
    CreationError,
    CreationTimeout,
    CrossEntropy,
    Full,
    ServiceClient,
    SessionClosed,
    TrainingConfig,
)
from looper_sdk.proto.v1 import operations_pb2, runs_pb2, sessions_pb2, values_pb2
from looper_sdk.transport import PROTOBUF


@pytest.fixture(autouse=True)
def immediate_shutdown(monkeypatch):
    monkeypatch.setattr("looper_sdk.transport.CLOSE_GRACE_S", 0, raising=False)


CONFIG = TrainingConfig(model="fake/tiny", kind=Full(), tokens_per_step=64, max_seq_len=16)


def answer(message):
    return httpx.Response(
        200, content=message.SerializeToString(), headers={"content-type": PROTOBUF}
    )


class Api:
    def __init__(self, states=(runs_pb2.Run.ACTIVE,)):
        self.states = iter(states)
        self.calls = []
        self.created = asyncio.Event()
        self.polled = asyncio.Event()
        self.block = False
        self.waiting_reason = values_pb2.WAITING_REASON_UNSPECIFIED

    async def __call__(self, request):
        self.calls.append(request)
        path = request.url.path
        if path == "/v1/sessions":
            return answer(
                sessions_pb2.OpenSessionResponse(
                    session=sessions_pb2.Session(id="s", project="p", heartbeat_interval_s=3600)
                )
            )
        if path.endswith(":close"):
            return answer(sessions_pb2.CloseSessionResponse())
        run = runs_pb2.Run(id="r", writer_session_id="s", state=runs_pb2.Run.QUEUED)
        if path == "/v1/runs":
            self.created.set()
            return answer(runs_pb2.CreateRunResponse(run=run))
        if path == "/v1/runs/r":
            self.polled.set()
            if self.block:
                await asyncio.Event().wait()
            run.state = next(self.states, runs_pb2.Run.STARTING)
            if run.state == runs_pb2.Run.QUEUED:
                run.waiting_reason = self.waiting_reason
            return answer(runs_pb2.GetRunResponse(run=run))
        if path.endswith(":await"):
            self.polled.set()
            await asyncio.Event().wait()
        return answer(operations_pb2.GetOperationResponse())


async def connect(api):
    return await ServiceClient.Config(base_url="http://looper", project="p").build(
        ServiceClient.Runtime(transport=httpx.MockTransport(api))
    )


def test_creation_waits_for_the_run_to_be_active():
    async def run():
        api = Api((runs_pb2.Run.STARTING, runs_pb2.Run.ACTIVE))
        async with await connect(api) as service:
            client = await service.create_training_client(CONFIG)
            assert (await client.status()).state.value != "queued"
            assert sum(r.method == "GET" for r in api.calls) >= 2

    asyncio.run(run())


def test_a_run_held_back_by_its_orgs_gpu_limit_warns_once(caplog, monkeypatch):
    monkeypatch.setattr("looper_sdk.training.READINESS_POLL_S", 0.01)

    async def run():
        api = Api((runs_pb2.Run.QUEUED,) * 3 + (runs_pb2.Run.ACTIVE,))
        api.waiting_reason = values_pb2.WAITING_REASON_ORG_GPU_LIMIT
        async with await connect(api) as service:
            client = await service.create_training_client(CONFIG)
            assert (await client.status()).waiting_reason is None  # STARTING after ACTIVE

    asyncio.run(run())
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert warnings == ["Run r is waiting: its organization is at its GPU limit"]


def test_close_cancels_pending_creation_and_rejects_new_work():
    async def run():
        api = Api()
        api.block = True
        service = await connect(api)
        pending = asyncio.create_task(service.create_training_client(CONFIG))
        await api.created.wait()
        await asyncio.sleep(0)
        await service.close()
        assert pending.done()
        with pytest.raises(SessionClosed):
            await pending

        before = len(api.calls)
        with pytest.raises(SessionClosed):
            await service.create_training_client(CONFIG)
        assert len(api.calls) == before
        assert not any(r.method == "DELETE" for r in api.calls)

    asyncio.run(run())


def test_closed_session_invalidates_existing_clients_and_operation_waits():
    async def run():
        api = Api()
        service = await connect(api)
        client = await service.create_training_client(CONFIG)
        from looper_sdk.operations import Operation

        operation = Operation(
            operations_pb2.Operation(id="o", kind=operations_pb2.Operation.OPTIM_STEP),
            session=service.session,
            adopt_epoch=lambda epoch: None,
        )
        api.polled.clear()
        pending = asyncio.create_task(operation.result())
        await api.polled.wait()
        await asyncio.gather(service.close(), service.close())
        with pytest.raises(SessionClosed):
            await pending

        with pytest.raises(SessionClosed):
            await client.status()
        with pytest.raises(SessionClosed):
            await operation.state()
        assert sum(r.url.path.endswith(":close") for r in api.calls) == 1

    asyncio.run(run())


@pytest.mark.parametrize(
    "call",
    [
        lambda client: client.forward_backward([], CrossEntropy()),
        lambda client: client.optim_step(),
        lambda client: client.save_state("state"),
        lambda client: client.save_weights("weights"),
        lambda client: client.publish_weights(),
        lambda client: client.create_sampling_client(max_seq_len=16, max_concurrent_sequences=1),
    ],
    ids=[
        "forward_backward",
        "optim_step",
        "save_state",
        "save_weights",
        "publish_weights",
        "create_sampling_client",
    ],
)
def test_every_training_call_after_close_raises_session_closed(call):
    async def run():
        api = Api()
        service = await connect(api)
        client = await service.create_training_client(CONFIG)
        await service.close()
        before = len(api.calls)
        with pytest.raises(SessionClosed):
            await call(client)
        assert len(api.calls) == before

    asyncio.run(run())


def test_creation_timeout_exposes_the_accepted_run():
    async def run():
        api = Api()
        api.block = True
        async with await connect(api) as service:
            with pytest.raises(CreationTimeout) as raised:
                await service.create_training_client(CONFIG, key="creation", timeout_s=0.03)
            assert raised.value.run_id == "r"
            assert raised.value.key == "creation"
            assert raised.value.session_id == "s"

    asyncio.run(run())


@pytest.mark.parametrize(
    "state, error",
    [
        (runs_pb2.Run.FAILED, "RunStateLost"),
        (runs_pb2.Run.DELETED, "Cancelled"),
        (runs_pb2.Run.SUSPENDED, "RunSuspended"),
    ],
)
def test_creation_refuses_terminal_states(state, error):
    import looper_sdk

    async def run():
        async with await connect(Api((state,))) as service:
            with pytest.raises(CreationError) as raised:
                await service.create_training_client(CONFIG)
            assert isinstance(raised.value.__cause__, getattr(looper_sdk, error))

    asyncio.run(run())


def test_cancelled_close_leaves_cleanup_owned_and_context_cannot_reenter():
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
        task.cancel()
        await asyncio.sleep(0)
        assert task.done()
        with pytest.raises(asyncio.CancelledError):
            await task
        release.set()
        await service.close()
        assert service.session.transport.client.is_closed

        with pytest.raises(SessionClosed):
            async with service:
                pytest.fail("a closed service was entered")

    asyncio.run(run())


def test_close_interrupts_creation_between_readiness_requests():
    async def run():
        api = Api((runs_pb2.Run.STARTING,))
        service = await connect(api)
        task = asyncio.create_task(service.create_training_client(CONFIG))
        await api.polled.wait()
        await service.close()
        with pytest.raises(SessionClosed):
            await task
        assert sum(r.method == "GET" for r in api.calls) == 1

    asyncio.run(run())


def test_body_exception_does_not_swallow_cancellation_during_cleanup():
    async def run():
        api = Api()
        closing, release = asyncio.Event(), asyncio.Event()
        original = ValueError("loop failed")

        async def handler(request):
            if request.url.path.endswith(":close"):
                closing.set()
                await release.wait()
            return await api(request)

        service = await connect(handler)

        async def scoped():
            async with service:
                raise original

        task = asyncio.create_task(scoped())
        await closing.wait()
        task.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        await service.close()
        assert service.session.transport.client.is_closed

    asyncio.run(run())


def test_context_body_exception_survives_transport_cleanup_failure():
    class BrokenClose(httpx.MockTransport):
        async def aclose(self):
            raise OSError("transport cleanup failed")

    async def run():
        service = await ServiceClient.Config(base_url="http://looper", project="p").build(
            ServiceClient.Runtime(transport=BrokenClose(Api()))
        )
        original = ValueError("loop failed")
        with pytest.raises(ValueError) as raised:
            async with service:
                raise original
        assert raised.value is original
        assert any("OSError" in note for note in original.__notes__)

    asyncio.run(run())
