"""An accepted operation: its receipt at once,
and what it computed once it has run.

Awaiting a submit returns the receipt, once the operation is durable and
ordered; awaiting ``result()`` returns what it computed. ``result()`` waits
through ``queued``, ``running`` and ``outcome_unknown``, one long poll at a
time, and raises the operation's typed error if it failed or was cancelled.
A ``RunRolledBack`` moves its client to the run's new epoch first.
"""

from __future__ import annotations

import asyncio
import enum
from collections.abc import Callable
from typing import Any

from looper_sdk import wire
from looper_sdk.errors import RunRolledBack
from looper_sdk.proto.v1 import operations_pb2
from looper_sdk.transport import Session, session_operation

# The longest one await asks the API to wait, which caps it at 30 s.
LONG_POLL_S = 30.0
_RESULTS: dict[str, Callable[[operations_pb2.Operation], Any]] = {
    "forward_backward": lambda message: wire.forward_backward_result_from_proto(
        message.forward_backward
    ),
    "optim_step": lambda message: wire.step_result_from_proto(message.optim_step),
    "save_checkpoint": lambda message: wire.checkpoint_from_proto(message.save_checkpoint),
    "publish_weights": lambda message: wire.publish_result_from_proto(message.publish_weights),
    "save_weights": lambda message: wire.checkpoint_from_proto(message.save_weights),
}
_FAILED = (operations_pb2.Operation.FAILED, operations_pb2.Operation.CANCELLED)


class OperationState(enum.StrEnum):
    """Where an operation is."""

    QUEUED = "queued"
    RUNNING = "running"
    OUTCOME_UNKNOWN = "outcome_unknown"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


def _state(message: operations_pb2.Operation) -> OperationState:
    return OperationState(operations_pb2.Operation.State.Name(message.state).lower())


class Operation[T]:
    """An accepted operation of a run, made by its run's client.

    Attributes:
        id: The service's name for it.
        seq: Its place in its run's order.
        key: Its idempotency key, unique in its epoch of the run.
        epoch: The run's epoch it was accepted in.
        kind: ``forward_backward``, ``optim_step``, ``save_checkpoint``
            (``save_state``'s), ``save_weights`` or ``publish_weights``.
    """

    def __init__(
        self,
        message: operations_pb2.Operation,
        *,
        session: Session,
        adopt_epoch: Callable[[int], None],
    ):
        self.id, self.seq = message.id, message.seq
        self.key, self.epoch = message.key, message.epoch
        self.kind = operations_pb2.Operation.Kind.Name(message.kind).lower()
        self._session, self._adopt_epoch = session, adopt_epoch

    def __repr__(self) -> str:
        return f"Operation(id={self.id!r}, seq={self.seq}, key={self.key!r}, kind={self.kind})"

    @session_operation
    async def result(self, *, timeout_s: float | None = None) -> T:
        """What it computed, once it has run; TimeoutError once ``timeout_s``
        has passed, resends to an unreachable API included. ``state()`` reads
        where it is without waiting."""
        try:
            async with asyncio.timeout(timeout_s):
                return await self._resolved()
        except TimeoutError:
            raise TimeoutError(f"Operation {self.id} did not resolve in {timeout_s} s") from None

    async def _resolved(self) -> T:
        while True:
            answer = await self._session.transport.call(
                "POST",
                f"/v1/operations/{self.id}:await",
                operations_pb2.AwaitOperationRequest(timeout_s=LONG_POLL_S),
                operations_pb2.AwaitOperationResponse,
            )
            message = answer.operation
            if message.state == operations_pb2.Operation.SUCCEEDED:
                return _RESULTS[self.kind](message)
            if message.state in _FAILED:
                error = wire.error_from_proto(message.error)
                if isinstance(error, RunRolledBack):
                    self._adopt_epoch(error.epoch)
                raise error

    @session_operation
    async def state(self) -> OperationState:
        """Where the operation is now, read without waiting."""
        answer = await self._session.transport.call(
            "GET", f"/v1/operations/{self.id}", None, operations_pb2.GetOperationResponse
        )
        return _state(answer.operation)
