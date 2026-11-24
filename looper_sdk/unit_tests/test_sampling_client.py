"""The SDK's standalone sampler, against a
scripted API: creation's readiness wait and its errors, one generated key per
``sample()`` kept across lost answers, bounded result polls that a local
timeout leaves running, and a failed operation's stored error."""

from __future__ import annotations

import asyncio
import dataclasses
from uuid import NAMESPACE_URL, uuid5

import httpx
import pytest

from looper_sdk import (
    CheckpointNotReady,
    CreationError,
    CreationTimeout,
    QueueTimeout,
    SamplerConfig,
    SampleResult,
    SamplerState,
    SampleSequence,
    SamplingParams,
    ServiceClient,
    wire,
)
from looper_sdk.errors import (
    Internal,
    InvalidArgument,
    LooperError,
    ResultExpired,
    SamplerFailed,
    SessionClosed,
    Unavailable,
)
from looper_sdk.operations import OperationState
from looper_sdk.proto.v1 import sampling_pb2, sessions_pb2
from looper_sdk.sampling import FinishReason, SamplingUsage
from looper_sdk.transport import PROTOBUF
from looper_sdk.values import Version

CONFIG = SamplerConfig(checkpoint_name="after-sft", max_seq_len=64, max_concurrent_sequences=4)
PARAMS = SamplingParams(max_tokens=4)
RESULT = SampleResult(
    sequences=(
        SampleSequence(token_ids=(5, 6), logprobs=(-0.5, -0.25), finish_reason=FinishReason.LENGTH),
    ),
    model="Qwen/Qwen3.5-9B",
    checkpoint_name="after-sft",
)


@pytest.fixture(autouse=True)
def immediate_shutdown(monkeypatch):
    monkeypatch.setattr("looper_sdk.transport.CLOSE_GRACE_S", 0, raising=False)
    monkeypatch.setattr("looper_sdk.sampling_client.READINESS_POLL_S", 0.01)


def answer(message, status: int = 200):
    return httpx.Response(
        status, content=message.SerializeToString(), headers={"content-type": PROTOBUF}
    )


def sampler(state=sampling_pb2.Sampler.READY) -> sampling_pb2.Sampler:
    return sampling_pb2.Sampler(
        id="m",
        config=wire.sampler_config_to_proto(CONFIG),
        state=state,
        model="Qwen/Qwen3.5-9B",
        version=wire.version_to_proto(Version(run_id="r", epoch=0, step=10)),
    )


class Api:
    """Scripted ``/v1`` answers: the sampler's states in turn, each sample's
    receipt, and each await's answer."""

    def __init__(self, states=(sampling_pb2.Sampler.READY,)) -> None:
        self.states = list(states)
        self.calls: list[httpx.Request] = []
        self.creation_refusal: LooperError | None = None
        self.lose_sample_answers = 0
        self.awaits: list = []  # each await's operation, in turn; then running
        self.read: sampling_pb2.SampleOperation | None = None  # a GET's answer; else running
        self.keys: list[str] = []

    async def __call__(self, request: httpx.Request) -> httpx.Response:
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
        if path == "/v1/samplers":
            if self.creation_refusal is not None:
                return answer(wire.error_to_proto(self.creation_refusal), 409)
            return answer(
                sampling_pb2.CreateSamplerResponse(sampler=sampler(sampling_pb2.Sampler.STARTING))
            )
        if path == "/v1/samplers/m":
            state = self.states.pop(0) if len(self.states) > 1 else self.states[0]
            return answer(sampling_pb2.GetSamplerResponse(sampler=sampler(state)))
        if path.endswith(":stop"):
            return answer(
                sampling_pb2.StopSamplerResponse(sampler=sampler(sampling_pb2.Sampler.STOPPING))
            )
        if path.endswith(":sample"):
            body = sampling_pb2.SubmitSampleRequest.FromString(request.content)
            self.keys.append(body.key)
            if self.lose_sample_answers:
                self.lose_sample_answers -= 1
                raise httpx.ReadError("the answer was lost")
            operation = sampling_pb2.SampleOperation(
                id=f"op-{body.key}", key=body.key, sampler_id="m",
                state=sampling_pb2.SampleOperation.QUEUED,
            )  # fmt: skip
            return answer(sampling_pb2.SubmitSampleResponse(operation=operation))
        if path.endswith(":await"):
            operation = self.awaits.pop(0) if self.awaits else None
            if operation is None:
                await asyncio.sleep(0.01)  # a long poll that ends unresolved
                operation = sampling_pb2.SampleOperation(
                    id="op", state=sampling_pb2.SampleOperation.RUNNING
                )
            return answer(sampling_pb2.AwaitSampleOperationResponse(operation=operation))
        if "/operations/" in path:
            operation = self.read or sampling_pb2.SampleOperation(
                id="op", state=sampling_pb2.SampleOperation.RUNNING
            )
            return answer(sampling_pb2.GetSampleOperationResponse(operation=operation))
        raise AssertionError(f"unscripted {request.method} {path}")


async def connect(api: Api) -> ServiceClient:
    return await ServiceClient.Config(
        base_url="http://looper", project="p", retry_timeout_s=5
    ).build(ServiceClient.Runtime(transport=httpx.MockTransport(api)))


def test_creation_waits_until_the_sampler_is_ready() -> None:
    async def run() -> None:
        api = Api(states=(sampling_pb2.Sampler.STARTING, sampling_pb2.Sampler.READY))
        async with await connect(api) as service:
            client = await service.create_sampling_client(CONFIG, key="c1")
            assert (client.sampler_id, client.checkpoint_name) == ("m", "after-sft")
            status = await client.status()
            assert status.state is SamplerState.READY and status.config == CONFIG
        body = sampling_pb2.CreateSamplerRequest.FromString(api.calls[1].content)
        assert (body.session_id, body.key) == ("s", "c1")

    asyncio.run(run())


def test_a_checkpoint_still_uploading_is_raised_at_once_not_retried() -> None:
    async def run() -> None:
        api = Api()
        api.creation_refusal = CheckpointNotReady("after-sft is still uploading")
        async with await connect(api) as service:
            with pytest.raises(CreationError) as caught:
                await service.create_sampling_client(CONFIG, key="c1")
        assert isinstance(caught.value.cause, CheckpointNotReady)
        assert caught.value.key == "c1" and caught.value.run_id is None
        assert sum(r.url.path == "/v1/samplers" for r in api.calls) == 1

    asyncio.run(run())


def test_a_creation_that_outlasts_its_timeout_names_what_was_accepted() -> None:
    async def run() -> None:
        api = Api(states=(sampling_pb2.Sampler.STARTING,))
        async with await connect(api) as service:
            with pytest.raises(CreationTimeout) as caught:
                await service.create_sampling_client(CONFIG, key="c1", timeout_s=0.1)
        assert (caught.value.key, caught.value.session_id) == ("c1", "s")
        assert caught.value.run_id == "m"  # the accepted sampler

    asyncio.run(run())


def test_a_failed_sampler_fails_its_creation() -> None:
    async def run() -> None:
        api = Api(states=(sampling_pb2.Sampler.STARTING, sampling_pb2.Sampler.FAILED))
        async with await connect(api) as service:
            with pytest.raises(CreationError):
                await service.create_sampling_client(CONFIG)

    asyncio.run(run())


def test_a_lost_acceptance_is_resent_under_its_one_key() -> None:
    async def run() -> None:
        api = Api()
        async with await connect(api) as service:
            client = await service.create_sampling_client(CONFIG)
            api.lose_sample_answers = 1
            first = await client.sample([1, 2, 3], params=PARAMS, n=2)
            second = await client.sample([1, 2, 3], params=PARAMS, n=2)
        assert api.keys[0] == api.keys[1] == first.key  # the resend
        assert second.key != first.key  # another call, another request

    asyncio.run(run())


def test_a_request_the_sampler_cannot_serve_is_refused_before_sending() -> None:
    async def run() -> None:
        api = Api()
        async with await connect(api) as service:
            client = await service.create_sampling_client(CONFIG)
            with pytest.raises(InvalidArgument, match="max_concurrent_sequences"):
                await client.sample([1], params=PARAMS, n=5)
            with pytest.raises(InvalidArgument):
                await client.sample([], params=PARAMS)
        assert not api.keys

    asyncio.run(run())


def test_result_polls_until_done_and_raises_a_failed_operation_s_error() -> None:
    async def run() -> None:
        api = Api()
        async with await connect(api) as service:
            client = await service.create_sampling_client(CONFIG)
            done = await client.sample([1, 2], params=PARAMS)
            running = sampling_pb2.SampleOperation(
                id=done.id, state=sampling_pb2.SampleOperation.RUNNING
            )
            succeeded = sampling_pb2.SampleOperation(
                id=done.id,
                state=sampling_pb2.SampleOperation.SUCCEEDED,
                result=wire.sample_result_to_proto(RESULT),
            )
            api.awaits = [running, running, succeeded]
            assert await done.result() == RESULT

            expired = await client.sample([3], params=PARAMS)
            api.awaits = [
                sampling_pb2.SampleOperation(
                    id=expired.id,
                    state=sampling_pb2.SampleOperation.FAILED,
                    error=wire.error_to_proto(QueueTimeout("waited too long")),
                )
            ]
            with pytest.raises(QueueTimeout):
                await expired.result()
            # Polling again raises the same stored error; nothing is resent.
            api.awaits = [api_failed(expired.id)]
            with pytest.raises(QueueTimeout):
                await expired.result()
        assert len(api.keys) == 2

    asyncio.run(run())


def test_usage_is_none_until_success_then_the_counts_read_now() -> None:
    async def run() -> None:
        api = Api()
        async with await connect(api) as service:
            client = await service.create_sampling_client(CONFIG)
            done = await client.sample([1, 2], params=PARAMS)
            assert await done.usage() is None
            usage = SamplingUsage(num_prompt_tokens=2, num_generated_tokens=5, num_cached_tokens=1)
            api.read = sampling_pb2.SampleOperation(
                id=done.id,
                state=sampling_pb2.SampleOperation.SUCCEEDED,
                usage=wire.sampling_usage_to_proto(usage),
            )
            assert await done.state() is OperationState.SUCCEEDED
            assert await done.usage() == usage

    asyncio.run(run())


def api_failed(operation_id: str) -> sampling_pb2.SampleOperation:
    return sampling_pb2.SampleOperation(
        id=operation_id,
        state=sampling_pb2.SampleOperation.FAILED,
        error=wire.error_to_proto(QueueTimeout("waited too long")),
    )


def test_a_local_timeout_ends_only_the_wait() -> None:
    async def run() -> None:
        api = Api()
        async with await connect(api) as service:
            client = await service.create_sampling_client(CONFIG)
            operation = await client.sample([1], params=PARAMS)
            with pytest.raises(TimeoutError):
                await operation.result(timeout_s=0.05)
            assert (await operation.state()).value == "running"  # still there
        assert len(api.keys) == 1

    asyncio.run(run())


def test_sampling_calls_have_their_own_connection_pool() -> None:
    async def run() -> None:
        api = Api()
        async with await connect(api) as service:
            client = await service.create_sampling_client(CONFIG)
            assert client.transport is not service.session.transport

    asyncio.run(run())


def test_terminal_unavailable_is_returned_as_an_operation_envelope() -> None:
    async def run() -> None:
        api = Api()
        async with await connect(api) as service:
            client = await service.create_sampling_client(CONFIG)
            operation = await client.sample([1], params=PARAMS)
            api.awaits = [
                sampling_pb2.SampleOperation(
                    id=operation.id,
                    state=sampling_pb2.SampleOperation.FAILED,
                    error=wire.error_to_proto(Unavailable("drained")),
                )
            ]
            terminal = await operation._await_terminal()
            assert terminal.state == sampling_pb2.SampleOperation.FAILED
            assert isinstance(wire.error_from_proto(terminal.error), Unavailable)
            assert len(api.keys) == 1

    asyncio.run(run())


def test_http_unavailable_is_not_a_terminal_generation_failure() -> None:
    async def run() -> None:
        api = Api()

        async def unavailable(request):
            if request.url.path.endswith(":await"):
                return answer(wire.error_to_proto(Unavailable("API offline")), 503)
            return await api(request)

        async with await ServiceClient.Config(
            base_url="http://looper", project="p", retry_timeout_s=0
        ).build(ServiceClient.Runtime(transport=httpx.MockTransport(unavailable))) as service:
            client = await service.create_sampling_client(CONFIG)
            operation = await client.sample([1], params=PARAMS)
            with pytest.raises(Unavailable):
                await operation.result()
            assert len(api.keys) == 1

    asyncio.run(run())


class RetryApi(Api):
    """Idempotent receipts and persistent per-operation outcomes over scripted HTTP."""

    def __init__(self, *outcomes: LooperError | SampleResult) -> None:
        super().__init__()
        self.outcomes = outcomes
        self.operations: dict[str, sampling_pb2.SampleOperation] = {}
        self.submissions: list[sampling_pb2.SubmitSampleRequest] = []

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith(":sample"):
            body = sampling_pb2.SubmitSampleRequest.FromString(request.content)
            self.submissions.append(body)
            receipt = await super().__call__(request)
            operation = sampling_pb2.SubmitSampleResponse.FromString(receipt.content).operation
            if operation.id not in self.operations:
                outcome = self.outcomes[min(len(self.operations), len(self.outcomes) - 1)]
                if isinstance(outcome, LooperError):
                    operation.state = sampling_pb2.SampleOperation.FAILED
                    operation.error.CopyFrom(wire.error_to_proto(outcome))
                else:
                    operation.state = sampling_pb2.SampleOperation.SUCCEEDED
                    operation.result.CopyFrom(wire.sample_result_to_proto(outcome))
                    operation.usage.CopyFrom(
                        wire.sampling_usage_to_proto(
                            SamplingUsage(
                                num_prompt_tokens=1,
                                num_generated_tokens=2,
                                num_cached_tokens=0,
                            )
                        )
                    )
                self.operations[operation.id] = operation
            return receipt
        if "/operations/" in path:
            self.calls.append(request)
            operation_id = path.split("/operations/")[1].removesuffix(":await")
            operation = self.operations[operation_id]
            if path.endswith(":await"):
                return answer(sampling_pb2.AwaitSampleOperationResponse(operation=operation))
            return answer(sampling_pb2.GetSampleOperationResponse(operation=operation))
        return await super().__call__(request)


def test_a_drain_failure_replaces_the_request_with_the_original_inputs() -> None:
    async def run() -> None:
        version = Version(run_id="new-policy", epoch=0, step=7)
        result = dataclasses.replace(
            RESULT, sequences=(dataclasses.replace(RESULT.sequences[0], version=version),)
        )
        api = RetryApi(Unavailable("drained"), result)
        async with await connect(api) as service:
            client = await service.create_sampling_client(CONFIG)
            prompt = [1]
            operation = await client.sample(prompt, params=PARAMS, key="original")
            original_id, original_key = operation.id, operation.key
            assert await operation.state() is OperationState.FAILED
            assert await operation.usage() is None
            assert api.keys == ["original"]  # reads do not generate
            prompt.append(2)
            assert await operation.result(timeout_s=5) == result
            assert operation.id != original_id
            assert operation.key != original_key
            assert api.operations[original_id].state == sampling_pb2.SampleOperation.FAILED
            assert all(s.request == api.submissions[0].request for s in api.submissions)
            assert list(api.submissions[-1].request.prompt) == [1]
            assert await operation.state() is OperationState.SUCCEEDED
            assert await operation.usage() == SamplingUsage(
                num_prompt_tokens=1,
                num_generated_tokens=2,
                num_cached_tokens=0,
            )
            assert await operation.result() == result
            assert len(api.keys) == 2

    asyncio.run(run())


@pytest.mark.parametrize(
    "error",
    [
        Unavailable("permanent", retryable=False),
        Internal("engine failed", retryable=False),
        QueueTimeout("expired"),
        SamplerFailed("lost"),
        ResultExpired("reclaimed"),
    ],
)
def test_non_retryable_terminal_errors_never_replace_the_operation(error) -> None:
    async def run() -> None:
        api = RetryApi(error)
        async with await connect(api) as service:
            client = await service.create_sampling_client(CONFIG)
            operation = await client.sample([1], params=PARAMS)
            with pytest.raises(type(error)):
                await operation.result()
            assert len(api.keys) == 1

    asyncio.run(run())


@pytest.mark.parametrize(
    "error",
    [
        Unavailable("temporary"),
        Internal("temporary"),
        InvalidArgument("recoverable", retryable=True),
        LooperError("unknown code", retryable=True),
    ],
)
def test_the_retryable_flag_alone_authorizes_replacement(error) -> None:
    async def run() -> None:
        api = RetryApi(error, RESULT)
        async with await connect(api) as service:
            client = await service.create_sampling_client(CONFIG)
            operation = await client.sample([1], params=PARAMS)
            assert await operation.result() == RESULT
            assert len(api.operations) == 2

    asyncio.run(run())


def test_an_unavailable_error_without_a_retryable_wire_flag_is_not_retried() -> None:
    async def run() -> None:
        api = RetryApi(Unavailable("missing flag"), RESULT)
        async with await connect(api) as service:
            client = await service.create_sampling_client(CONFIG)
            operation = await client.sample([1], params=PARAMS)
            api.operations[operation.id].error.ClearField("retryable")
            with pytest.raises(Unavailable) as caught:
                await operation.result()
            assert caught.value.retryable is False
            assert len(api.operations) == 1

    asyncio.run(run())


@pytest.mark.parametrize(
    "error",
    [Unavailable("cancelled"), Internal("cancelled"), LooperError("cancelled", retryable=True)],
)
def test_cancelled_operations_are_not_retried_even_with_a_retryable_error(error) -> None:
    async def run() -> None:
        api = RetryApi(error)
        async with await connect(api) as service:
            client = await service.create_sampling_client(CONFIG)
            operation = await client.sample([1], params=PARAMS)
            api.operations[operation.id].state = sampling_pb2.SampleOperation.CANCELLED
            with pytest.raises(type(error)):
                await operation.result()
            assert len(api.keys) == 1

    asyncio.run(run())


@pytest.mark.parametrize("cap", [1, 2, 3])
def test_the_attempt_cap_includes_the_first_and_survives_repeated_waits(cap) -> None:
    async def run() -> None:
        api = RetryApi(Unavailable("drained"))
        async with await connect(api) as service:
            client = await service.create_sampling_client(CONFIG)
            operation = await client.sample([1], params=PARAMS, max_attempts=cap)
            for _ in range(2):
                with pytest.raises(Unavailable) as caught:
                    await operation.result()
                assert caught.value.retryable
            assert len(api.operations) == cap

    asyncio.run(run())


@pytest.mark.parametrize("cap", [0, -1, 1.5, True, "3"])
def test_an_invalid_attempt_cap_is_refused_before_submission(cap) -> None:
    async def run() -> None:
        api = RetryApi(RESULT)
        async with await connect(api) as service:
            client = await service.create_sampling_client(CONFIG)
            with pytest.raises(InvalidArgument, match="max_attempts"):
                await client.sample([1], params=PARAMS, max_attempts=cap)
            assert not api.keys

    asyncio.run(run())


@pytest.mark.parametrize("separate_handles", [False, True])
def test_concurrent_waiters_converge_on_one_replacement(separate_handles) -> None:
    async def run() -> None:
        api = RetryApi(Unavailable("drained"), RESULT)
        async with await connect(api) as service:
            client = await service.create_sampling_client(CONFIG)
            first = await client.sample([1], params=PARAMS, key="same")
            second = (
                await client.sample([1], params=PARAMS, key="same") if separate_handles else first
            )
            assert await asyncio.gather(first.result(), second.result()) == [RESULT, RESULT]
            assert len(api.operations) == 2
            assert (first.id, first.key) == (second.id, second.key)
            expected = uuid5(NAMESPACE_URL, "looper:sampling-retry:m:op-same:1").hex
            assert first.key == expected

    asyncio.run(run())


@pytest.mark.parametrize("interruption", ["lost_receipt", "cancelled_receipt", "timeout_receipt"])
def test_uncertain_replacement_submission_resumes_with_the_same_key(interruption) -> None:
    async def run() -> None:
        api = RetryApi(Unavailable("drained"), RESULT)
        accepted, hold = asyncio.Event(), asyncio.Event()
        interrupted = False

        async def transport(request):
            nonlocal interrupted
            receipt = await api(request)
            if (
                request.url.path.endswith(":sample")
                and len(api.operations) == 2
                and not interrupted
            ):
                interrupted = True
                accepted.set()
                if interruption == "lost_receipt":
                    raise httpx.ReadError("accepted, response lost")
                await hold.wait()
            return receipt

        async with await ServiceClient.Config(
            base_url="http://looper",
            project="p",
            retry_timeout_s=0,
        ).build(ServiceClient.Runtime(transport=httpx.MockTransport(transport))) as service:
            client = await service.create_sampling_client(CONFIG)
            operation = await client.sample([1], params=PARAMS)
            timeout = 0.5 if interruption == "timeout_receipt" else None
            waiting = asyncio.create_task(operation.result(timeout_s=timeout))
            await asyncio.wait_for(accepted.wait(), 5)
            if interruption == "cancelled_receipt":
                waiting.cancel()
            error = {
                "lost_receipt": Unavailable,
                "cancelled_receipt": asyncio.CancelledError,
                "timeout_receipt": TimeoutError,
            }[interruption]
            with pytest.raises(error):
                await waiting
            hold.set()
            assert await operation.result(timeout_s=5) == RESULT
            assert len(api.operations) == 2
            assert api.keys[-1] == api.keys[-2]
            assert operation.key == api.keys[-1]

    asyncio.run(run())


def test_a_lost_replacement_response_is_resent_by_transport_under_the_same_key() -> None:
    async def run() -> None:
        api = RetryApi(Unavailable("drained"), RESULT)
        lost = False

        async def transport(request):
            nonlocal lost
            response = await api(request)
            if request.url.path.endswith(":sample") and len(api.operations) == 2 and not lost:
                lost = True
                raise httpx.ReadError("receipt lost")
            return response

        async with await ServiceClient.Config(base_url="http://looper", project="p").build(
            ServiceClient.Runtime(transport=httpx.MockTransport(transport))
        ) as service:
            client = await service.create_sampling_client(CONFIG)
            operation = await client.sample([1], params=PARAMS)
            assert await operation.result() == RESULT
            assert len(api.operations) == 2
            assert api.keys[-1] == api.keys[-2]

    asyncio.run(run())


def test_result_expiration_after_success_does_not_regenerate() -> None:
    async def run() -> None:
        api = RetryApi(Unavailable("drained"), RESULT)
        expired = False

        async def transport(request):
            if request.url.path.endswith(":await") and expired:
                return answer(wire.error_to_proto(ResultExpired("reclaimed")), 410)
            return await api(request)

        async with await ServiceClient.Config(base_url="http://looper", project="p").build(
            ServiceClient.Runtime(transport=httpx.MockTransport(transport))
        ) as service:
            client = await service.create_sampling_client(CONFIG)
            operation = await client.sample([1], params=PARAMS)
            assert await operation.result() == RESULT
            expired = True
            with pytest.raises(ResultExpired):
                await operation.result()
            assert len(api.operations) == 2

    asyncio.run(run())


def test_result_timeout_includes_lock_acquisition_and_leaves_the_other_waiter_running() -> None:
    async def run() -> None:
        api = RetryApi(Unavailable("drained"), RESULT)
        entered, release = asyncio.Event(), asyncio.Event()

        async def transport(request):
            if request.url.path.endswith(":await") and not release.is_set():
                entered.set()
                await release.wait()
            return await api(request)

        async with await ServiceClient.Config(base_url="http://looper", project="p").build(
            ServiceClient.Runtime(transport=httpx.MockTransport(transport))
        ) as service:
            client = await service.create_sampling_client(CONFIG)
            operation = await client.sample([1], params=PARAMS)
            first = asyncio.create_task(operation.result())
            await entered.wait()
            with pytest.raises(TimeoutError):
                await operation.result(timeout_s=0.02)
            release.set()
            assert await first == RESULT
            assert len(api.operations) == 2

    asyncio.run(run())


@pytest.mark.parametrize("interruption", ["timeout", "session_close"])
def test_a_wait_interrupted_during_backoff_does_not_submit(interruption, monkeypatch) -> None:
    async def run() -> None:
        api = RetryApi(Unavailable("drained", retry_after_s=3600), RESULT)
        failed = asyncio.Event()

        async def transport(request):
            response = await api(request)
            if request.url.path.endswith(":await"):
                failed.set()
            return response

        monkeypatch.setattr("looper_sdk.transport.CLOSE_GRACE_S", 0)
        async with await ServiceClient.Config(base_url="http://looper", project="p").build(
            ServiceClient.Runtime(transport=httpx.MockTransport(transport))
        ) as service:
            client = await service.create_sampling_client(CONFIG)
            operation = await client.sample([1], params=PARAMS)
            waiting = asyncio.create_task(operation.result(timeout_s=0.05))
            await failed.wait()
            if interruption == "session_close":
                await service.close()
            with pytest.raises(TimeoutError if interruption == "timeout" else SessionClosed):
                await waiting
            assert len(api.operations) == 1

    asyncio.run(run())


def test_replaying_the_original_key_recovers_the_same_replacement_chain() -> None:
    async def run() -> None:
        api = RetryApi(Unavailable("first drain"), Unavailable("second drain"), RESULT)
        async with await connect(api) as service:
            client = await service.create_sampling_client(CONFIG)
            first = await client.sample([1], params=PARAMS, key="original")
            original_id = first.id
            assert await first.result() == RESULT
            replay = await client.sample([1], params=PARAMS, key="original")
            assert replay.id == original_id
            assert await replay.result() == RESULT
            assert (replay.id, replay.key) == (first.id, first.key)
            assert first.key == uuid5(NAMESPACE_URL, "looper:sampling-retry:m:op-original:2").hex
            assert len(api.operations) == 3

    asyncio.run(run())


@pytest.mark.parametrize(
    "hint,expected", [(0, 0), (0.25, 0.25), (-1, 0.1), (float("nan"), 0.1), (float("inf"), 0.1)]
)
def test_retry_hints_override_backoff_only_when_finite_and_nonnegative(hint, expected, monkeypatch):
    async def run() -> None:
        sleeps = []
        sleep = asyncio.sleep

        async def observed(delay):
            if delay < 3600:
                sleeps.append(delay)
                await sleep(0)
            else:
                await sleep(delay)

        monkeypatch.setattr("looper_sdk.sampling_client.asyncio.sleep", observed)
        api = RetryApi(Unavailable("drained", retry_after_s=hint), RESULT)
        async with await connect(api) as service:
            client = await service.create_sampling_client(CONFIG)
            operation = await client.sample([1], params=PARAMS)
            assert await operation.result() == RESULT
            assert sleeps == [expected]

    asyncio.run(run())
