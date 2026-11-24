"""A standalone sampler's client: made by
``ServiceClient.create_sampling_client``, owned by the program's session.

``sample()`` returns an operation once the service has accepted the request
into the sampler's queue; requests run concurrently in acceptance order and
may finish in any order. Each call makes one key, and the transport resends
the request under it while an answer is lost, so a lost answer never makes a
second request; a new call is a new request. ``result()`` waits through
bounded long polls; its timeout ends only the wait. A terminal failed
operation marked retryable permits a replacement under a deterministic fresh
key, up to ``max_attempts`` including the original. Other failures are raised unchanged.
The handle follows the latest accepted operation and keeps retry bookkeeping private.

Sampling has an HTTP connection pool of its own, so long result polls never
hold the connections the session's heartbeats and training calls use.
"""

from __future__ import annotations

import asyncio
import logging
import math
from collections.abc import Sequence
from uuid import NAMESPACE_URL, uuid4, uuid5

from looper_sdk import wire
from looper_sdk._checks import _count
from looper_sdk.errors import SamplerFailed
from looper_sdk.operations import LONG_POLL_S, OperationState
from looper_sdk.proto.v1 import sampling_pb2
from looper_sdk.sampling import (
    SamplerConfig,
    SampleRequest,
    SampleResult,
    SamplerState,
    SamplerStatus,
    SamplingParams,
    SamplingUsage,
)
from looper_sdk.transport import FIRST_WAIT_S, LAST_WAIT_S, Session, Transport, session_operation

__all__ = ["SamplingClient", "SamplingOperation"]

# Seconds between a starting sampler's status reads.
READINESS_POLL_S = 2.0
_ENDED = (SamplerState.FAILED, SamplerState.STOPPING, SamplerState.STOPPED)
log = logging.getLogger(__name__)


class SamplingClient:
    """One sampler of the program's session, made by
    ``ServiceClient.create_sampling_client`` or, for a run's own sampler,
    ``TrainingClient.create_sampling_client``.

    Attributes:
        sampler_id: The service's name for it.
        config: What it was created with: its source and sizes.
        checkpoint_name: What it serves, as it was created.
    """

    def __init__(
        self, message: sampling_pb2.Sampler, *, session: Session, transport: Transport
    ) -> None:
        self.sampler_id = message.id
        self.config: SamplerConfig = wire.sampler_config_from_proto(message.config)
        self.checkpoint_name = self.config.checkpoint_name
        self._session, self.transport = session, transport

    def __repr__(self) -> str:
        return (
            f"SamplingClient(sampler_id={self.sampler_id!r}, checkpoint={self.checkpoint_name!r})"
        )

    def _path(self, suffix: str = "") -> str:
        return f"/v1/samplers/{self.sampler_id}{suffix}"

    @session_operation
    async def status(self) -> SamplerStatus:
        """What the service last observed: not proof the sampler is alive now."""
        answer = await self.transport.call(
            "GET",
            self._path(f"?session_id={self._session.id}"),
            None,
            sampling_pb2.GetSamplerResponse,
        )
        return wire.sampler_status_from_proto(answer.sampler)

    async def wait_until_ready(self, *, timeout_s: float | None = None) -> None:
        """Return once the sampler is ready: its model converted, loaded and
        serving. ``SamplerFailed`` if it ended instead; TimeoutError once
        ``timeout_s`` passed, which leaves it starting."""
        if timeout_s is not None and (not math.isfinite(timeout_s) or timeout_s < 0):
            raise ValueError("timeout_s must be finite and nonnegative, or None")
        async with asyncio.timeout(timeout_s):
            while True:
                status = await self.status()
                if status.state is SamplerState.READY:
                    return
                if status.state in _ENDED:
                    if status.error is not None:
                        raise status.error
                    raise SamplerFailed(f"Sampler {self.sampler_id} is {status.state.value}")
                await asyncio.sleep(READINESS_POLL_S)

    @session_operation
    async def sample(
        self,
        prompt: Sequence[int],
        *,
        params: SamplingParams,
        n: int = 1,
        key: str | None = None,
        max_attempts: int = 3,
    ) -> SamplingOperation:
        """Queue ``n`` completions of ``prompt`` and return the operation once
        the service has accepted it. A request this sampler could never serve
        is refused here, before anything is sent. ``result()`` replaces terminal
        retryable failures up to ``max_attempts`` including this acceptance;
        1 disables replacement. Each replacement can sample newer weights."""
        max_attempts = _count("max_attempts", max_attempts, minimum=1)
        request = SampleRequest(prompt=prompt, params=params, n=n)
        request.check(self.config)
        key = str(uuid4()) if key is None else key
        receipt = await self._submit_sample(request, key)
        return SamplingOperation(receipt, client=self, request=request, max_attempts=max_attempts)

    async def _submit_sample(
        self, request: SampleRequest, key: str
    ) -> sampling_pb2.SampleOperation:
        message = sampling_pb2.SubmitSampleRequest(
            session_id=self._session.id, key=key, request=wire.sample_request_to_proto(request)
        )
        answer = await self.transport.call(
            "POST", self._path(":sample"), message, sampling_pb2.SubmitSampleResponse
        )
        return answer.operation

    @session_operation
    async def stop(self) -> None:
        """Stop the sampler: its unfinished requests are cancelled and its
        GPUs released. Completed results stay readable."""
        await self.transport.call(
            "POST",
            self._path(":stop"),
            sampling_pb2.StopSamplerRequest(session_id=self._session.id),
            sampling_pb2.StopSamplerResponse,
        )


class SamplingOperation:
    """Sampling that may replace a retryable failed operation.

    Attributes:
        id: The latest accepted server operation; can change during result().
        key: Its acceptance key, which changes with its operation ID.
    """

    def __init__(
        self,
        message: sampling_pb2.SampleOperation,
        *,
        client: SamplingClient,
        request: SampleRequest,
        max_attempts: int,
    ) -> None:
        self._initial_id = message.id
        self._id, self._key = message.id, message.key
        self._attempt_count = 1
        self._client, self._session = client, client._session
        self._request, self._max_attempts = request, max_attempts
        self._result_lock = asyncio.Lock()

    @property
    def id(self) -> str:
        return self._id

    @property
    def key(self) -> str:
        return self._key

    def __repr__(self) -> str:
        return f"SamplingOperation(id={self.id!r}, key={self.key!r})"

    def _path(self, suffix: str = "") -> str:
        return self._client._path(f"/operations/{self.id}{suffix}")

    @session_operation
    async def state(self) -> OperationState:
        """The latest known operation's state; initiates no replacement."""
        operation = await self._read()
        return OperationState(sampling_pb2.SampleOperation.State.Name(operation.state).lower())

    @session_operation
    async def usage(self) -> SamplingUsage | None:
        """The latest known operation's tokens, read now: None until it
        succeeded, or for one recorded before Looper counted tokens. Kept
        after its result's bytes expire."""
        operation = await self._read()
        if not operation.HasField("usage"):
            return None
        return wire.sampling_usage_from_proto(operation.usage)

    async def _read(self) -> sampling_pb2.SampleOperation:
        answer = await self._client.transport.call(
            "GET",
            self._path(f"?session_id={self._session.id}"),
            None,
            sampling_pb2.GetSampleOperationResponse,
        )
        return answer.operation

    @session_operation
    async def result(self, *, timeout_s: float | None = None) -> SampleResult:
        """Wait, replacing terminal retryable failures within the attempt cap.

        The timeout covers lock acquisition, backoff, submissions and all
        attempts for this wait. Timeout or cancellation leaves remote work
        intact; a later wait resumes without resetting the cap. Other errors
        propagate unchanged. Replacements may produce different samples.
        """
        try:
            async with asyncio.timeout(timeout_s):
                async with self._result_lock:
                    return await self._resolved()
        except TimeoutError:
            raise TimeoutError(
                f"Sampling operation {self.id} did not finish in {timeout_s} s"
            ) from None

    async def _resolved(self) -> SampleResult:
        while True:
            operation = await self._await_terminal()
            if operation.state == sampling_pb2.SampleOperation.SUCCEEDED:
                return wire.sample_result_from_proto(operation.result)
            error = wire.error_from_proto(operation.error)
            if (
                operation.state != sampling_pb2.SampleOperation.FAILED
                or not error.retryable
                or self._attempt_count >= self._max_attempts
            ):
                raise error
            index = self._attempt_count
            key = uuid5(
                NAMESPACE_URL,
                f"looper:sampling-retry:{self._client.sampler_id}:{self._initial_id}:{index}",
            ).hex
            delay = min(FIRST_WAIT_S * 2 ** min(index - 1, 16), LAST_WAIT_S)
            hint = error.retry_after_s
            if hint is not None and math.isfinite(hint) and hint >= 0:
                delay = hint
            log.info(
                "Sampler %s: operation %s failed retryably; submitting attempt %d/%d "
                "with key %s in %g s",
                self._client.sampler_id,
                self.id,
                index + 1,
                self._max_attempts,
                key,
                delay,
            )
            await asyncio.sleep(delay)
            receipt = await self._client._submit_sample(self._request, key)
            self._id, self._key = receipt.id, receipt.key
            self._attempt_count += 1

    async def _await_terminal(self) -> sampling_pb2.SampleOperation:
        """Read an outcome; HTTP failures propagate separately from stored errors."""
        session = self._session
        while True:
            answer = await self._client.transport.call(
                "POST",
                self._path(":await"),
                sampling_pb2.AwaitSampleOperationRequest(
                    session_id=session.id, timeout_s=LONG_POLL_S
                ),
                sampling_pb2.AwaitSampleOperationResponse,
            )
            operation = answer.operation
            if operation.state in (
                sampling_pb2.SampleOperation.SUCCEEDED,
                sampling_pb2.SampleOperation.FAILED,
                sampling_pb2.SampleOperation.CANCELLED,
            ):
                return operation
