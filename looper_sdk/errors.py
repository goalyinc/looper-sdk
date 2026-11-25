"""The typed errors of ``looper.v1``.

Every registered code has a class here. An error from the service carries its
category and whether a retry may succeed as data, so a caller routes on
``retryable`` without parsing the message; each class's defaults are the
service's own, for errors raised without the service's word. A code the service adds
later arrives as a plain ``LooperError``.

``InvalidArgument`` is also a ``ValueError``: the SDK's values raise it when
constructed from invalid fields, before anything is sent.
"""

from __future__ import annotations

import enum
from typing import ClassVar


class ErrorCode(enum.StrEnum):
    """The registered codes, named as on the wire."""

    INVALID_ARGUMENT = "INVALID_ARGUMENT"
    UNAUTHENTICATED = "UNAUTHENTICATED"
    SPEND_REFUSED = "SPEND_REFUSED"
    PERMISSION_DENIED = "PERMISSION_DENIED"
    QUOTA_EXCEEDED = "QUOTA_EXCEEDED"
    NOT_FOUND = "NOT_FOUND"
    IDEMPOTENCY_CONFLICT = "IDEMPOTENCY_CONFLICT"
    WRITER_SUPERSEDED = "WRITER_SUPERSEDED"
    SESSION_EXPIRED = "SESSION_EXPIRED"
    DEPENDENCY_FAILED = "DEPENDENCY_FAILED"
    RUN_ROLLED_BACK = "RUN_ROLLED_BACK"
    GRADIENTS_PENDING = "GRADIENTS_PENDING"
    VERSION_SUPERSEDED = "VERSION_SUPERSEDED"
    RUN_SUSPENDED = "RUN_SUSPENDED"
    CANCELLED = "CANCELLED"
    RUN_STATE_LOST = "RUN_STATE_LOST"
    PAYLOAD_TOO_LARGE = "PAYLOAD_TOO_LARGE"
    INCOMPLETE_ACCUMULATION = "INCOMPLETE_ACCUMULATION"
    EMPTY_STEP = "EMPTY_STEP"
    CAPACITY_EXHAUSTED = "CAPACITY_EXHAUSTED"
    RUN_NOT_READY = "RUN_NOT_READY"
    UNAVAILABLE = "UNAVAILABLE"
    INTERNAL = "INTERNAL"
    CHECKPOINT_NOT_READY = "CHECKPOINT_NOT_READY"
    QUEUE_TIMEOUT = "QUEUE_TIMEOUT"
    SAMPLER_FAILED = "SAMPLER_FAILED"
    RESULT_EXPIRED = "RESULT_EXPIRED"
    CHECKPOINT_NAME_TAKEN = "CHECKPOINT_NAME_TAKEN"


class Category(enum.StrEnum):
    """Whose failure it is: the caller's request, or the service."""

    USER = "user"
    SERVER = "server"


class LooperError(Exception):
    """A failure of a ``looper.v1`` call.

    Attributes:
        message: For people; never parsed.
        retryable: The same request may succeed later or elsewhere.
        category: Whose failure it is.
        retry_after_s: How long to wait before a retry, when the service
            advertises it; None otherwise.
        key: The idempotency key of the submit that raised it, set by the
            SDK's clients. Whether a submit that failed retryably was
            accepted is unknown: resending under this key, or
            ``find_operation(key)``, settles it. None otherwise.
    """

    code: ClassVar[ErrorCode | None] = None
    default_retryable: ClassVar[bool] = False
    default_category: ClassVar[Category] = Category.USER

    def __init__(
        self,
        message: str,
        *,
        retryable: bool | None = None,
        category: Category | None = None,
        retry_after_s: float | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.retryable = self.default_retryable if retryable is None else retryable
        self.category = self.default_category if category is None else category
        self.retry_after_s = retry_after_s
        self.key: str | None = None


class InvalidArgument(LooperError, ValueError):
    """A refused value or request: a datum, an option, or a call the run cannot take."""

    code = ErrorCode.INVALID_ARGUMENT


class Unauthenticated(LooperError):
    """No credential, one that is not an API key, or a key the service does
    not know, has revoked or let expire. A key works only in the
    deployment that issued it."""

    code = ErrorCode.UNAUTHENTICATED


class SpendRefused(LooperError):
    """Billing refuses resource creation or compute; retry after restoring eligibility."""

    code = ErrorCode.SPEND_REFUSED


class PermissionDenied(LooperError):
    """The caller may not make this call, such as a submit to a run another
    session writes."""

    code = ErrorCode.PERMISSION_DENIED


class QuotaExceeded(LooperError):
    """A save over the project's stored bytes, or a run or sampler that needs more
    GPUs than its organization may use. Retry a save once space is freed; a run or
    sampler that large is refused again until the organization's limit is raised."""

    code = ErrorCode.QUOTA_EXCEEDED


class NotFound(LooperError):
    """No such session, run, operation, sampler or checkpoint in the
    caller's organization."""

    code = ErrorCode.NOT_FOUND


class IdempotencyConflict(LooperError):
    """A key reused, in the same epoch of the run, with another payload."""

    code = ErrorCode.IDEMPOTENCY_CONFLICT


class WriterSuperseded(LooperError):
    """Another session took the run over."""

    code = ErrorCode.WRITER_SUPERSEDED


class SessionExpired(LooperError):
    """The submitting session missed its heartbeats."""

    code = ErrorCode.SESSION_EXPIRED


class DependencyFailed(LooperError):
    """An operation named in ``after`` failed or was cancelled, so this one
    failed at its turn, without running."""

    code = ErrorCode.DEPENDENCY_FAILED


class RunRolledBack(LooperError):
    """The run was restored to an earlier step.

    Attributes:
        epoch: The run's new epoch, which the client adopts.
        restored_step: The step the run went back to.
        restored_seq: The last seq the restored state covers.
        first_lost_seq: The first seq lost; the range is empty when
            ``last_lost_seq < first_lost_seq``.
        last_lost_seq: The last seq lost.
    """

    code = ErrorCode.RUN_ROLLED_BACK
    default_category = Category.SERVER

    def __init__(
        self,
        message: str,
        *,
        epoch: int,
        restored_step: int,
        restored_seq: int,
        first_lost_seq: int,
        last_lost_seq: int,
        retryable: bool | None = None,
        category: Category | None = None,
        retry_after_s: float | None = None,
    ) -> None:
        super().__init__(
            message, retryable=retryable, category=category, retry_after_s=retry_after_s
        )
        self.epoch = epoch
        self.restored_step = restored_step
        self.restored_seq = restored_seq
        self.first_lost_seq = first_lost_seq
        self.last_lost_seq = last_lost_seq


class GradientsPending(LooperError):
    """A save, export or publish after a forward_backward that no optim_step
    follows yet: each sits at a step boundary."""

    code = ErrorCode.GRADIENTS_PENDING


class VersionSuperseded(LooperError):
    """A pinned version of a run's weights is no longer served."""

    code = ErrorCode.VERSION_SUPERSEDED


class RunSuspended(LooperError):
    """The run is suspended, so it takes no work until it resumes."""

    code = ErrorCode.RUN_SUSPENDED


class Cancelled(LooperError):
    """The operation's run was deleted before it ran."""

    code = ErrorCode.CANCELLED


class RunStateLost(LooperError):
    """The run's state is gone: no restore point loads. The message names
    the run's newest ``state`` checkpoint, if any."""

    code = ErrorCode.RUN_STATE_LOST
    default_category = Category.SERVER


class PayloadTooLarge(LooperError):
    """Over a bound of the service; a retry cannot fix it."""

    code = ErrorCode.PAYLOAD_TOO_LARGE


class IncompleteAccumulation(LooperError):
    """An optim_step over gradients that include a failed forward_backward."""

    code = ErrorCode.INCOMPLETE_ACCUMULATION


class EmptyStep(LooperError):
    """An optim_step with no target of nonzero weight accumulated."""

    code = ErrorCode.EMPTY_STEP


class CapacityExhausted(LooperError):
    """An admission limit of the service: no capacity now. Retry after
    ``retry_after_s`` when it is set."""

    code = ErrorCode.CAPACITY_EXHAUSTED
    default_retryable = True
    default_category = Category.SERVER


class RunNotReady(LooperError):
    """A call the run cannot take while it is queued, starting, resuming or
    recovering."""

    code = ErrorCode.RUN_NOT_READY
    default_retryable = True
    default_category = Category.SERVER


class Unavailable(LooperError):
    """The service cannot answer now, as during a restart or a failover.
    Retryable, though the outage need not be short."""

    code = ErrorCode.UNAVAILABLE
    default_retryable = True
    default_category = Category.SERVER


class Internal(LooperError):
    """Our fault. Retryable for idempotent requests, as every ordered call is."""

    code = ErrorCode.INTERNAL
    default_retryable = True
    default_category = Category.SERVER


class CheckpointNotReady(LooperError):
    """A sampler's checkpoint name binds no ready checkpoint yet, only one
    still uploading or being exported, or a checkpoint to delete is still
    being exported. An export in flight cannot be deleted, as it would publish
    anyway. Not retried automatically; create the sampler, or delete the
    checkpoint, again once it is ``READY_TO_USE``."""

    code = ErrorCode.CHECKPOINT_NOT_READY


class QueueTimeout(LooperError):
    """A sampling request waited in its sampler's queue past the service's
    queue deadline and was never sent to the engine.
    Sampling again is a new request; the caller's own wait timing out is a
    plain ``TimeoutError`` instead."""

    code = ErrorCode.QUEUE_TIMEOUT
    default_category = Category.SERVER


class SamplerFailed(LooperError):
    """The sampler ended for good: it failed or stopped answering, the
    service restarted, or its run lost its trainer. Nothing on it can succeed
    again; create another sampler."""

    code = ErrorCode.SAMPLER_FAILED
    default_category = Category.SERVER


class ResultExpired(LooperError):
    """The computation succeeded, but its original raw result was reclaimed."""

    code = ErrorCode.RESULT_EXPIRED


class CheckpointNameTaken(LooperError):
    """An export's name, `<source>-hf`, is held by a live checkpoint that is not
    an export of its source, or reserved by another source's export in flight.
    Not retried; free the name first."""

    code = ErrorCode.CHECKPOINT_NAME_TAKEN


ERRORS: dict[ErrorCode, type[LooperError]] = {
    cls.code: cls
    for cls in (
        InvalidArgument,
        Unauthenticated,
        SpendRefused,
        PermissionDenied,
        QuotaExceeded,
        NotFound,
        IdempotencyConflict,
        WriterSuperseded,
        SessionExpired,
        DependencyFailed,
        RunRolledBack,
        GradientsPending,
        VersionSuperseded,
        RunSuspended,
        Cancelled,
        RunStateLost,
        PayloadTooLarge,
        IncompleteAccumulation,
        EmptyStep,
        CapacityExhausted,
        RunNotReady,
        Unavailable,
        Internal,
        CheckpointNotReady,
        QueueTimeout,
        SamplerFailed,
        ResultExpired,
        CheckpointNameTaken,
    )
    if cls.code is not None
}


class SessionClosed(RuntimeError):
    """A local handle is closed; retrying on that service cannot succeed."""

    retryable: ClassVar[bool] = False


class SessionCloseTimeout(TimeoutError):
    """Cleanup continues; another close wait on the same service may finish."""

    retryable: ClassVar[bool] = True


class CreationError(RuntimeError):
    """A creation's service failure, with its recovery context.

    Attributes:
        cause: The original typed service error, also the exception's cause.
        retryable: The service failure's retry semantics, unchanged.
        key: The creation key, scoped to the originating session.
        session_id: The service's permanent session identity.
        run_id: The accepted run, or sampler for a sampling client, if its
            receipt was received.
    """

    def __init__(self, cause: LooperError, *, key: str, session_id: str, run_id: str | None):
        super().__init__(str(cause))
        self.cause, self.retryable = cause, cause.retryable
        self.key, self.session_id, self.run_id = key, session_id, run_id


class CheckpointFailed(RuntimeError):
    """A save succeeded, but its checkpoint's upload did not finish, or an
    export did not, so it is not usable. Waiting again cannot
    change it; a new save can. So can exporting again, unless ``failure``
    says a retry will not help or the source expired; if the name was
    taken, delete the checkpoint that holds it first.

    Attributes:
        checkpoint_id: The failed checkpoint.
        failure: Why, for people.
    """

    retryable: ClassVar[bool] = False

    def __init__(self, *, checkpoint_id: str, failure: str):
        super().__init__(f"Checkpoint {checkpoint_id} failed: {failure}")
        self.checkpoint_id, self.failure = checkpoint_id, failure


class CreationTimeout(TimeoutError):
    """Creation stopped waiting; the backend run may still exist.

    Attributes:
        retryable: The same creation may be retried on the same live service
            under its original key; another session is a different creation.
        key: Creation key to reuse in the originating session.
        session_id: The service's permanent session identity.
        run_id: The accepted run, or sampler for a sampling client, if its
            receipt was received.
    """

    retryable: ClassVar[bool] = True

    def __init__(self, *, key: str, session_id: str, run_id: str | None):
        super().__init__(f"Client creation timed out: key={key}, id={run_id}")
        self.key, self.session_id, self.run_id = key, session_id, run_id
