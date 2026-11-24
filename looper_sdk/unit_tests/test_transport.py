"""The transport's retries and errors, and the session's lifecycle, against
httpx's mock transport. Synchronous tests, each running its own event loop:
looper-isolated installs pytest alone."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable

import httpx
import pytest

from looper_sdk import wire
from looper_sdk.errors import (
    CapacityExhausted,
    LooperError,
    Unavailable,
    WriterSuperseded,
)
from looper_sdk.operations import Operation
from looper_sdk.proto.v1 import operations_pb2, sessions_pb2
from looper_sdk.service import ServiceClient
from looper_sdk.transport import PROTOBUF, Transport, open_session

type Handler = Callable[[httpx.Request], httpx.Response]


def _failure(status: int, error: LooperError) -> httpx.Response:
    content = wire.error_to_proto(error).SerializeToString()
    return httpx.Response(status, content=content, headers={"content-type": PROTOBUF})


def _session(identity: str) -> httpx.Response:
    session = sessions_pb2.Session(id=identity, project="p", heartbeat_interval_s=3600)
    content = sessions_pb2.OpenSessionResponse(session=session).SerializeToString()
    return httpx.Response(200, content=content, headers={"content-type": PROTOBUF})


def _transport(handler: Handler, *, retry_timeout_s: float = 5.0) -> Transport:
    client = httpx.AsyncClient(base_url="http://looper", transport=httpx.MockTransport(handler))
    return Transport(client, retry_timeout_s=retry_timeout_s)


def _open(handler: Handler, *, retry_timeout_s: float = 5.0) -> sessions_pb2.OpenSessionResponse:
    async def call() -> sessions_pb2.OpenSessionResponse:
        transport = _transport(handler, retry_timeout_s=retry_timeout_s)
        try:
            return await transport.call(
                "POST",
                "/v1/sessions",
                sessions_pb2.OpenSessionRequest(project="p"),
                sessions_pb2.OpenSessionResponse,
            )
        finally:
            await transport.close()

    return asyncio.run(call())


def test_a_retryable_failure_or_a_lost_answer_is_sent_again_until_answered() -> None:
    bodies: list[bytes] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(request.content)
        if len(bodies) == 1:
            raise httpx.ReadTimeout("the answer was lost", request=request)
        if len(bodies) == 2:
            return _failure(503, Unavailable("the coordinator restarts"))
        return _session("s-1")

    assert _open(handler).session.id == "s-1"
    assert len(bodies) == 3 and len(set(bodies)) == 1


def test_a_failure_no_retry_fixes_raises_its_typed_error_at_once() -> None:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return _failure(409, WriterSuperseded("another session took the run over"))

    with pytest.raises(WriterSuperseded, match="took the run over"):
        _open(handler)
    assert len(calls) == 1


@pytest.mark.parametrize(
    ("status", "error", "attempts"),
    [(502, Unavailable, 3), (404, LooperError, 1)],
)
def test_an_answer_not_from_looper_counts_as_unavailable_only_from_a_5xx(
    status: int, error: type[LooperError], attempts: int
) -> None:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if len(calls) < 3:
            return httpx.Response(status, text="<html>proxy</html>")
        return _session("s-1")

    if error is Unavailable:
        assert _open(handler).session.id == "s-1"
    else:
        with pytest.raises(LooperError, match="not looper.v1's answer") as raised:
            _open(handler)
        assert type(raised.value) is LooperError and not raised.value.retryable
    assert len(calls) == attempts


def test_retries_stop_once_the_next_would_pass_retry_timeout_s() -> None:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return _failure(429, CapacityExhausted("full", retry_after_s=60.0))

    with pytest.raises(CapacityExhausted):
        _open(handler, retry_timeout_s=1.0)
    assert len(calls) == 1


def test_session_capacity_warns_before_retry_and_only_once_per_open(caplog) -> None:
    calls = 0
    message = "Your organization has reached its limit of 10 open service sessions; close one"

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls > 1:
            assert any(message in record.getMessage() for record in caplog.records)
        if calls <= 3:
            return _failure(429, CapacityExhausted(message, retry_after_s=0))
        return _session("s-1")

    assert _open(handler).session.id == "s-1"
    warnings = [record for record in caplog.records if record.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert message in warnings[0].getMessage()
    assert calls == 4


def test_session_capacity_warns_even_when_retries_are_disabled(caplog) -> None:
    message = "Your organization has reached its limit of 10 open service sessions; close one"
    with pytest.raises(CapacityExhausted, match="limit of 10"):
        _open(lambda request: _failure(429, CapacityExhausted(message)), retry_timeout_s=0)
    assert any(
        record.levelno == logging.WARNING and message in record.getMessage()
        for record in caplog.records
    )


def test_a_result_times_out_while_the_api_is_unreachable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/sessions":
            return _session("s-1")
        if request.url.path.endswith(":close"):
            return httpx.Response(200, content=b"", headers={"content-type": PROTOBUF})
        raise httpx.ConnectError("refused", request=request)

    async def awaiting() -> None:
        transport = _transport(handler, retry_timeout_s=300.0)
        message = operations_pb2.Operation(id="op-1", kind=operations_pb2.Operation.OPTIM_STEP)
        session = await open_session(transport, "p")
        operation = Operation(message, session=session, adopt_epoch=lambda epoch: None)
        try:
            await operation.result(timeout_s=0.3)
        finally:
            await session.close()

    with pytest.raises(TimeoutError, match="op-1 did not resolve in 0.3 s"):
        asyncio.run(asyncio.wait_for(awaiting(), timeout=5))


@pytest.mark.parametrize("exit_kind", ["normal", "exception", "cancelled", "explicit"])
def test_a_service_closes_its_session_and_resources(exit_kind: str) -> None:
    closed: list[str] = []
    error = ValueError("the training loop failed")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/sessions":
            return _session("s-1")
        closed.append(request.url.path)
        content = sessions_pb2.CloseSessionResponse().SerializeToString()
        return httpx.Response(200, content=content, headers={"content-type": PROTOBUF})

    async def use_service() -> None:
        tasks_before = asyncio.all_tasks()
        service = await ServiceClient.Config(base_url="http://looper", project="p").build(
            ServiceClient.Runtime(transport=httpx.MockTransport(handler))
        )
        background_tasks = asyncio.all_tasks() - tasks_before

        async def scoped() -> None:
            async with service as entered:
                assert entered is service
                if exit_kind == "exception":
                    raise error
                if exit_kind == "cancelled":
                    task = asyncio.current_task()
                    assert task is not None
                    task.cancel()
                    await asyncio.sleep(0)

        try:
            if exit_kind == "explicit":
                await service.close()
            elif exit_kind == "exception":
                with pytest.raises(ValueError) as raised:
                    await scoped()
                assert raised.value is error
            elif exit_kind == "cancelled":
                with pytest.raises(asyncio.CancelledError):
                    await asyncio.create_task(scoped())
            else:
                await scoped()
            assert closed == ["/v1/sessions/s-1:close"]
            assert service.session.transport.client.is_closed
            assert background_tasks and all(task.done() for task in background_tasks)
        finally:
            if not service.session.transport.client.is_closed:
                await service.close()

    asyncio.run(use_service())


@pytest.mark.parametrize(
    ("fields", "match"),
    [
        ({"base_url": "localhost:8080"}, "http"),
        ({"request_timeout_s": 30.0}, "every long await times out"),
        ({"retry_timeout_s": -1.0}, "retry_timeout_s"),
        ({"display_name": ""}, "display_name"),
        ({"display_name": "n" * 256}, "display_name"),
        ({"display_name": "my run"}, "display_name"),
        ({"display_name": "v1.2"}, "display_name"),
        ({"display_name": "café"}, "display_name"),
    ],
)
def test_a_service_config_refuses_what_cannot_work(fields: dict[str, object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        ServiceClient.Config(**{"base_url": "http://looper", "project": "p", **fields})  # type: ignore[arg-type]


@pytest.mark.parametrize("display_name", [None, "sweep_lr-3e4", "n" * 255])
def test_a_service_opens_its_session_under_its_display_name(display_name: str | None) -> None:
    sent: list[sessions_pb2.OpenSessionRequest] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/sessions":
            sent.append(sessions_pb2.OpenSessionRequest.FromString(request.content))
            return _session("s-1")
        content = sessions_pb2.CloseSessionResponse().SerializeToString()
        return httpx.Response(200, content=content, headers={"content-type": PROTOBUF})

    async def use_service() -> None:
        config = ServiceClient.Config(
            base_url="http://looper", project="p", display_name=display_name
        )
        service = await config.build(ServiceClient.Runtime(transport=httpx.MockTransport(handler)))
        await service.close()

    asyncio.run(use_service())
    assert [request.display_name for request in sent] == [display_name or ""]
