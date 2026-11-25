# Looper SDK reference: errors

Every failure the SDK raises. An error from the service carries its code, whose fault it is, and whether the same request may succeed later, as data: route on `retryable`, never on the message.

Generated from the docstrings of goaly-looper 0.1.0b5. Import every name from `looper_sdk`. The reference has three pages: [clients](api-reference.md), [values](api-values.md) and [errors](api-errors.md).

## Errors at a glance

| Error | Is a | Code | Retryable | Category | Raised when |
|---|---|---|---|---|---|
| [`LooperError`](#loopererror) | `Exception` | — | no | `user` | A failure of a `looper.v1` call. |
| `InvalidArgument` | `LooperError`, `ValueError` | `INVALID_ARGUMENT` | no | `user` | A refused value or request: a datum, an option, or a call the run cannot take. |
| `Unauthenticated` | `LooperError` | `UNAUTHENTICATED` | no | `user` | No credential, one that is not an API key, or a key the service does not know, has revoked or let expire. |
| `SpendRefused` | `LooperError` | `SPEND_REFUSED` | no | `user` | Billing refuses resource creation or compute; retry after restoring eligibility. |
| `PermissionDenied` | `LooperError` | `PERMISSION_DENIED` | no | `user` | The caller may not make this call, such as a submit to a run another session writes. |
| `QuotaExceeded` | `LooperError` | `QUOTA_EXCEEDED` | no | `user` | A save over the project's stored bytes; retry once space is freed. |
| `NotFound` | `LooperError` | `NOT_FOUND` | no | `user` | No such session, run, operation, sampler or checkpoint in the caller's organization. |
| `IdempotencyConflict` | `LooperError` | `IDEMPOTENCY_CONFLICT` | no | `user` | A key reused, in the same epoch of the run, with another payload. |
| `WriterSuperseded` | `LooperError` | `WRITER_SUPERSEDED` | no | `user` | Another session took the run over. |
| `SessionExpired` | `LooperError` | `SESSION_EXPIRED` | no | `user` | The submitting session missed its heartbeats. |
| `DependencyFailed` | `LooperError` | `DEPENDENCY_FAILED` | no | `user` | An operation named in `after` failed or was cancelled, so this one failed at its turn, without running. |
| [`RunRolledBack`](#runrolledback) | `LooperError` | `RUN_ROLLED_BACK` | no | `server` | The run was restored to an earlier step. |
| `GradientsPending` | `LooperError` | `GRADIENTS_PENDING` | no | `user` | A save, export or publish after a forward_backward that no optim_step follows yet: each sits at a step boundary. |
| `VersionSuperseded` | `LooperError` | `VERSION_SUPERSEDED` | no | `user` | A pinned version of a run's weights is no longer served. |
| `RunSuspended` | `LooperError` | `RUN_SUSPENDED` | no | `user` | The run is suspended, so it takes no work until it resumes. |
| `Cancelled` | `LooperError` | `CANCELLED` | no | `user` | The operation's run was deleted before it ran. |
| `RunStateLost` | `LooperError` | `RUN_STATE_LOST` | no | `server` | The run's state is gone: no restore point loads. |
| `PayloadTooLarge` | `LooperError` | `PAYLOAD_TOO_LARGE` | no | `user` | Over a bound of the service; a retry cannot fix it. |
| `IncompleteAccumulation` | `LooperError` | `INCOMPLETE_ACCUMULATION` | no | `user` | An optim_step over gradients that include a failed forward_backward. |
| `EmptyStep` | `LooperError` | `EMPTY_STEP` | no | `user` | An optim_step with no target of nonzero weight accumulated. |
| `CapacityExhausted` | `LooperError` | `CAPACITY_EXHAUSTED` | yes | `server` | An admission limit of the service: no capacity now. |
| `RunNotReady` | `LooperError` | `RUN_NOT_READY` | yes | `server` | A call the run cannot take while it is queued, starting, resuming or recovering. |
| `Unavailable` | `LooperError` | `UNAVAILABLE` | yes | `server` | The service cannot answer now, as during a restart or a failover. |
| `Internal` | `LooperError` | `INTERNAL` | yes | `server` | Our fault. |
| `CheckpointNotReady` | `LooperError` | `CHECKPOINT_NOT_READY` | no | `user` | A sampler's checkpoint name binds no ready checkpoint yet, only one still uploading or being exported, or a checkpoint to delete is still being exported. |
| `QueueTimeout` | `LooperError` | `QUEUE_TIMEOUT` | no | `server` | A sampling request waited in its sampler's queue past the service's queue deadline and was never sent to the engine. |
| `SamplerFailed` | `LooperError` | `SAMPLER_FAILED` | no | `server` | The sampler ended for good: it failed or stopped answering, the service restarted, or its run lost its trainer. |
| `ResultExpired` | `LooperError` | `RESULT_EXPIRED` | no | `user` | The computation succeeded, but its original raw result was reclaimed. |
| `CheckpointNameTaken` | `LooperError` | `CHECKPOINT_NAME_TAKEN` | no | `user` | An export's name, `<source>-hf`, is held by a live checkpoint that is not an export of its source, or reserved by another source's export in flight. |
| [`CheckpointFailed`](#checkpointfailed) | `RuntimeError` | — | no | — | A save succeeded, but its checkpoint's upload did not finish, or an export did not, so it is not usable. |
| [`CreationError`](#creationerror) | `RuntimeError` | — | per instance | — | A creation's service failure, with its recovery context. |
| [`CreationTimeout`](#creationtimeout) | `TimeoutError` | — | yes | — | Creation stopped waiting; the backend run may still exist. |
| `SessionClosed` | `RuntimeError` | — | no | — | A local handle is closed; retrying on that service cannot succeed. |
| `SessionCloseTimeout` | `TimeoutError` | — | yes | — | Cleanup continues; another close wait on the same service may finish. |

## ErrorCode

The registered codes, named as on the wire.

| Member | Value |
|---|---|
| `INVALID_ARGUMENT` | `'INVALID_ARGUMENT'` |
| `UNAUTHENTICATED` | `'UNAUTHENTICATED'` |
| `SPEND_REFUSED` | `'SPEND_REFUSED'` |
| `PERMISSION_DENIED` | `'PERMISSION_DENIED'` |
| `QUOTA_EXCEEDED` | `'QUOTA_EXCEEDED'` |
| `NOT_FOUND` | `'NOT_FOUND'` |
| `IDEMPOTENCY_CONFLICT` | `'IDEMPOTENCY_CONFLICT'` |
| `WRITER_SUPERSEDED` | `'WRITER_SUPERSEDED'` |
| `SESSION_EXPIRED` | `'SESSION_EXPIRED'` |
| `DEPENDENCY_FAILED` | `'DEPENDENCY_FAILED'` |
| `RUN_ROLLED_BACK` | `'RUN_ROLLED_BACK'` |
| `GRADIENTS_PENDING` | `'GRADIENTS_PENDING'` |
| `VERSION_SUPERSEDED` | `'VERSION_SUPERSEDED'` |
| `RUN_SUSPENDED` | `'RUN_SUSPENDED'` |
| `CANCELLED` | `'CANCELLED'` |
| `RUN_STATE_LOST` | `'RUN_STATE_LOST'` |
| `PAYLOAD_TOO_LARGE` | `'PAYLOAD_TOO_LARGE'` |
| `INCOMPLETE_ACCUMULATION` | `'INCOMPLETE_ACCUMULATION'` |
| `EMPTY_STEP` | `'EMPTY_STEP'` |
| `CAPACITY_EXHAUSTED` | `'CAPACITY_EXHAUSTED'` |
| `RUN_NOT_READY` | `'RUN_NOT_READY'` |
| `UNAVAILABLE` | `'UNAVAILABLE'` |
| `INTERNAL` | `'INTERNAL'` |
| `CHECKPOINT_NOT_READY` | `'CHECKPOINT_NOT_READY'` |
| `QUEUE_TIMEOUT` | `'QUEUE_TIMEOUT'` |
| `SAMPLER_FAILED` | `'SAMPLER_FAILED'` |
| `RESULT_EXPIRED` | `'RESULT_EXPIRED'` |
| `CHECKPOINT_NAME_TAKEN` | `'CHECKPOINT_NAME_TAKEN'` |

## Category

Whose failure it is: the caller's request, or the service.

| Member | Value |
|---|---|
| `USER` | `'user'` |
| `SERVER` | `'server'` |

## LooperError

A failure of a `looper.v1` call.

| Attribute | Description |
|---|---|
| `message` | For people; never parsed. |
| `retryable` | The same request may succeed later or elsewhere. |
| `category` | Whose failure it is. |
| `retry_after_s` | How long to wait before a retry, when the service advertises it; None otherwise. |
| `key` | The idempotency key of the submit that raised it, set by the SDK's clients. Whether a submit that failed retryably was accepted is unknown: resending under this key, or `find_operation(key)`, settles it. None otherwise. |

## RunRolledBack

The run was restored to an earlier step.

| Attribute | Description |
|---|---|
| `message` | For people; never parsed. |
| `retryable` | The same request may succeed later or elsewhere. |
| `category` | Whose failure it is. |
| `retry_after_s` | How long to wait before a retry, when the service advertises it; None otherwise. |
| `key` | The idempotency key of the submit that raised it, set by the SDK's clients. Whether a submit that failed retryably was accepted is unknown: resending under this key, or `find_operation(key)`, settles it. None otherwise. |
| `epoch` | The run's new epoch, which the client adopts. |
| `restored_step` | The step the run went back to. |
| `restored_seq` | The last seq the restored state covers. |
| `first_lost_seq` | The first seq lost; the range is empty when `last_lost_seq < first_lost_seq`. |
| `last_lost_seq` | The last seq lost. |

## CheckpointFailed

A save succeeded, but its checkpoint's upload did not finish, or an
export did not, so it is not usable. Waiting again cannot
change it; a new save can. So can exporting again, unless `failure`
says a retry will not help or the source expired; if the name was
taken, delete the checkpoint that holds it first.

| Attribute | Description |
|---|---|
| `checkpoint_id` | The failed checkpoint. |
| `failure` | Why, for people. |

## CreationError

A creation's service failure, with its recovery context.

| Attribute | Description |
|---|---|
| `cause` | The original typed service error, also the exception's cause. |
| `retryable` | The service failure's retry semantics, unchanged. |
| `key` | The creation key, scoped to the originating session. |
| `session_id` | The service's permanent session identity. |
| `run_id` | The accepted run, or sampler for a sampling client, if its receipt was received. |

## CreationTimeout

Creation stopped waiting; the backend run may still exist.

| Attribute | Description |
|---|---|
| `retryable` | The same creation may be retried on the same live service under its original key; another session is a different creation. |
| `key` | Creation key to reuse in the originating session. |
| `session_id` | The service's permanent session identity. |
| `run_id` | The accepted run, or sampler for a sampling client, if its receipt was received. |
