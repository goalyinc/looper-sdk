"""``TrainingClient``: one run, from its writer.

- **Order.** A client sends its run's submits one at a time, each once the
  previous one is accepted, so acceptance order is the program's order. A
  submit the caller cancels may still be accepted, after a later one.
- **Keys.** Each submit carries a key, a UUID unless the caller names one.
  A resend after a lost answer carries the same key, so the service returns
  the operation it accepted instead of accepting another.
- **An expired session.** ``SESSION_EXPIRED`` is terminal: open a new
  service and run.
- **Unsure submits.** A submit whose retries ran out may have been accepted.
  Its error names its key: a resend under it, or ``find_operation``, settles
  it, where a new submit would apply the batch twice.
- **Epochs.** Submits carry the epoch the client last saw. A
  ``RunRolledBack``, from a submit or an awaited operation, moves the client
  to the run's new epoch once it is raised; the caller decides what to submit
  again.

A run's sampler is made here too, and its weights published to it:
``create_sampling_client`` makes a sampler of the session whose source is
this run, and ``publish_weights`` is an ordered operation that moves the
run's current step into it.
"""

from __future__ import annotations

import asyncio
import dataclasses
import enum
import math
from collections.abc import Awaitable, Callable, Sequence
from typing import Any
from urllib.parse import urlencode
from uuid import uuid4

from looper_sdk import wire
from looper_sdk.checkpoints import StateCheckpoint, WeightsCheckpoint
from looper_sdk.errors import (
    Cancelled,
    LooperError,
    RunRolledBack,
    RunStateLost,
    RunSuspended,
)
from looper_sdk.losses import Loss
from looper_sdk.operations import Operation
from looper_sdk.optimizers import check_learning_rate
from looper_sdk.proto.v1 import operations_pb2, runs_pb2
from looper_sdk.results import ForwardBackwardResult, PublishResult, StepResult
from looper_sdk.sampling import RUN_PREFIX, SamplerConfig
from looper_sdk.sampling_client import SamplingClient
from looper_sdk.transport import Session, session_operation
from looper_sdk.values import Datum

READINESS_POLL_S = 0.5


class RunState(enum.StrEnum):
    """Where a run is."""

    QUEUED = "queued"
    STARTING = "starting"
    ACTIVE = "active"
    RECOVERING = "recovering"
    SUSPENDED = "suspended"
    RESUMING = "resuming"
    FAILED = "failed"
    DELETED = "deleted"


_READINESS_FAILURES = {
    RunState.FAILED: RunStateLost,
    RunState.DELETED: Cancelled,
    RunState.SUSPENDED: RunSuspended,
}


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class RunStatus:
    """A run, as the service holds it now.

    Attributes:
        state: Where the run is.
        epoch: The run's epoch; a rollback starts the next one.
        step: Completed optimizer steps.
        durable_step: The newest step a restore point or a state checkpoint
            covers.
    """

    state: RunState
    epoch: int
    step: int
    durable_step: int


class TrainingClient:
    """A run's writer, made by ``ServiceClient.create_training_client``.

    Every submit (``forward_backward``, ``optim_step``, ``save_state``,
    ``save_weights``, ``publish_weights``) returns an ``Operation`` once the
    service has accepted it, in the order the program submitted it; its
    ``result()`` waits for what it computed. Each submit carries a ``key``, a
    new UUID unless the caller names one: a resend under the same key, in
    the same epoch, returns the operation already accepted instead of a
    second one, and ``find_operation`` finds it after a lost answer.
    ``after`` names operations this one runs only after; if one of them
    fails, this one fails with ``DependencyFailed`` without running. A
    ``RunRolledBack`` moves the client to the run's new epoch, whose keys
    are new.

    Attributes:
        run_id: The run's id.
        epoch: The run's epoch as this client last saw it, which its submits
            carry.
    """

    def __init__(
        self,
        run: runs_pb2.Run,
        *,
        session: Session,
        create_sampler: Callable[..., Awaitable[SamplingClient]],
    ):
        self.run_id, self.epoch = run.id, run.epoch
        self._session = session
        # The service client's create_sampling_client: a run's sampler is a
        # sampler of the session, made on its sampling connections.
        self._create_sampler = create_sampler
        self._submitting = asyncio.Lock()

    def __repr__(self) -> str:
        return f"TrainingClient(run_id={self.run_id!r}, epoch={self.epoch})"

    def _adopt_epoch(self, epoch: int) -> None:
        # An error recorded at an older rollback may arrive after a newer one.
        self.epoch = max(self.epoch, epoch)

    def _operation(self, message: operations_pb2.Operation) -> Operation[Any]:
        return Operation(message, session=self._session, adopt_epoch=self._adopt_epoch)

    async def _send(
        self,
        verb: str,
        request: runs_pb2.ForwardBackwardRequest
        | runs_pb2.OptimStepRequest
        | runs_pb2.SaveCheckpointRequest
        | runs_pb2.PublishWeightsRequest
        | runs_pb2.SaveWeightsRequest,
        response: type[
            runs_pb2.ForwardBackwardResponse
            | runs_pb2.OptimStepResponse
            | runs_pb2.SaveCheckpointResponse
            | runs_pb2.PublishWeightsResponse
            | runs_pb2.SaveWeightsResponse
        ],
        key: str,
        after: Sequence[Operation[Any]],
    ) -> operations_pb2.Operation:
        request.submission.CopyFrom(
            operations_pb2.Submission(
                session_id=self._session.id,
                writer_generation=1,
                epoch=self.epoch,
                key=key,
                after=[operation.id for operation in after],
            )
        )
        try:
            answer = await self._session.transport.call(
                "POST", f"/v1/runs/{self.run_id}:{verb}", request, response
            )
        except RunRolledBack as exc:
            self._adopt_epoch(exc.epoch)
            raise
        return answer.operation

    async def _submit(
        self,
        verb: str,
        request: runs_pb2.ForwardBackwardRequest
        | runs_pb2.OptimStepRequest
        | runs_pb2.SaveCheckpointRequest
        | runs_pb2.PublishWeightsRequest
        | runs_pb2.SaveWeightsRequest,
        response: type[
            runs_pb2.ForwardBackwardResponse
            | runs_pb2.OptimStepResponse
            | runs_pb2.SaveCheckpointResponse
            | runs_pb2.PublishWeightsResponse
            | runs_pb2.SaveWeightsResponse
        ],
        key: str | None,
        after: Sequence[Operation[Any]],
    ) -> Operation[Any]:
        key = str(uuid4()) if key is None else key
        async with self._submitting:
            try:
                message = await self._send(verb, request, response, key, after)
            except LooperError as exc:
                exc.key = key
                raise
        return self._operation(message)

    @session_operation
    async def forward_backward(
        self,
        data: Sequence[Datum],
        loss: Loss,
        *,
        key: str | None = None,
        after: Sequence[Operation[Any]] = (),
    ) -> Operation[ForwardBackwardResult]:
        """Accepts a forward and backward pass over the data, whose gradients
        accumulate until the next ``optim_step``."""
        request = runs_pb2.ForwardBackwardRequest(
            run_id=self.run_id,
            data=[wire.datum_to_proto(datum) for datum in data],
            loss=wire.loss_to_proto(loss),
        )
        return await self._submit(
            "forwardBackward", request, runs_pb2.ForwardBackwardResponse, key, after
        )

    @session_operation
    async def optim_step(
        self,
        *,
        learning_rate: float | None = None,
        key: str | None = None,
        after: Sequence[Operation[Any]] = (),
    ) -> Operation[StepResult]:
        """Accepts an optimizer step over the accumulated gradients: the run's
        optimizer, dividing by the targets, at ``learning_rate`` if named,
        for this step alone, else at its own."""
        request = runs_pb2.OptimStepRequest(run_id=self.run_id)
        if learning_rate is not None:
            request.learning_rate = check_learning_rate(learning_rate)
        return await self._submit("optimStep", request, runs_pb2.OptimStepResponse, key, after)

    @session_operation
    async def save_state(
        self,
        name: str,
        *,
        ttl_s: int | None = None,
        key: str | None = None,
        after: Sequence[Operation[Any]] = (),
    ) -> Operation[StateCheckpoint]:
        """Accept a save of the run's training state, its weights with its
        optimizer's, at its current step; a sampler converts it at launch.
        Await ``result()`` separately: it answers once the step is saved,
        while its upload may still be going on (``wait_checkpoint``).

        The name belongs to the project and replaces its previous target only
        on successful completion. A wait timeout never cancels this operation.
        """
        request = runs_pb2.SaveCheckpointRequest(run_id=self.run_id, name=name)
        if ttl_s is not None:
            request.ttl_s = ttl_s
        return await self._submit(
            "saveCheckpoint", request, runs_pb2.SaveCheckpointResponse, key, after
        )

    @session_operation
    async def publish_weights(
        self, *, key: str | None = None, after: Sequence[Operation[Any]] = ()
    ) -> Operation[PublishResult]:
        """Accepts a publish of the run's current step into the run's
        sampler, which serves it from then on. It sits at a
        step boundary, as a save does: ``GradientsPending`` with gradients
        accumulated and not applied. A loop submits it with its step and
        need not await it; its result names the version now served.
        ``SamplerFailed`` if the run's sampler failed, or did not take it."""
        request = runs_pb2.PublishWeightsRequest(run_id=self.run_id)
        return await self._submit(
            "publishWeights", request, runs_pb2.PublishWeightsResponse, key, after
        )

    async def create_sampling_client(
        self,
        *,
        max_seq_len: int,
        max_concurrent_sequences: int,
        key: str | None = None,
        timeout_s: float | None = None,
    ) -> SamplingClient:
        """Create this run's sampler and wait until it is linked to the
        run's trainer: a sampler of the session, as
        ``ServiceClient.create_sampling_client`` makes, whose source is this
        run. It serves nothing until the run's first ``publish_weights``,
        and then each version the run publishes. A run has one at a time;
        a failed or stopped one is replaced by creating another. The
        deadline and the key work as the service client's do."""
        config = SamplerConfig(
            checkpoint_name=f"{RUN_PREFIX}{self.run_id}",
            max_seq_len=max_seq_len,
            max_concurrent_sequences=max_concurrent_sequences,
        )
        return await self._create_sampler(config, key=key, timeout_s=timeout_s)

    @session_operation
    async def save_weights(
        self,
        name: str,
        *,
        ttl_s: int | None = None,
        key: str | None = None,
        after: Sequence[Operation[Any]] = (),
    ) -> Operation[WeightsCheckpoint]:
        """Accept an export of the weights alone, for sampling:
        Hugging Face safetensors, or PEFT's layout for LoRA, which a sampler
        loads as they are. It holds no optimizer state.

        The name belongs to the project and replaces its previous target only
        on successful completion. A wait timeout never cancels this operation.
        """
        request = runs_pb2.SaveWeightsRequest(run_id=self.run_id, name=name)
        if ttl_s is not None:
            request.ttl_s = ttl_s
        return await self._submit("saveWeights", request, runs_pb2.SaveWeightsResponse, key, after)

    @session_operation
    async def find_operation(self, key: str, *, epoch: int | None = None) -> Operation[Any] | None:
        """The operation the key names in the epoch, the run's current one by
        default; how a submit whose answer was lost is found."""
        query = {"key": key} if epoch is None else {"key": key, "epoch": str(epoch)}
        answer = await self._session.transport.call(
            "GET",
            f"/v1/runs/{self.run_id}/operations?{urlencode(query)}",
            None,
            operations_pb2.FindOperationResponse,
        )
        return self._operation(answer.operation) if answer.HasField("operation") else None

    @session_operation
    async def wait_until_ready(self, *, timeout_s: float | None = None) -> None:
        """Wait until the run is active and takes work. ``RunStateLost``,
        ``Cancelled`` or ``RunSuspended`` if it failed, was deleted or is
        suspended instead. Waiting starts and resumes nothing."""
        if timeout_s is not None and (not math.isfinite(timeout_s) or timeout_s < 0):
            raise ValueError("timeout_s must be finite and nonnegative, or None")
        async with asyncio.timeout(timeout_s):
            while True:
                status = await self.status()
                if status.state is RunState.ACTIVE:
                    self._adopt_epoch(status.epoch)
                    return
                if error := _READINESS_FAILURES.get(status.state):
                    raise error(f"Run {self.run_id} is {status.state.value}")
                await asyncio.sleep(READINESS_POLL_S)

    @session_operation
    async def status(self) -> RunStatus:
        """The run as the service holds it now: its state, epoch and steps."""
        answer = await self._session.transport.call(
            "GET", f"/v1/runs/{self.run_id}", None, runs_pb2.GetRunResponse
        )
        run = answer.run
        return RunStatus(
            state=RunState(runs_pb2.Run.State.Name(run.state).lower()),
            epoch=run.epoch,
            step=run.step,
            durable_step=run.durable_step,
        )

    @session_operation
    async def delete(self) -> None:
        """Deletes the run: its queued operations are cancelled, a running one
        finishes, and its GPUs are released."""
        await self._session.transport.call(
            "DELETE", f"/v1/runs/{self.run_id}", None, runs_pb2.DeleteRunResponse
        )
