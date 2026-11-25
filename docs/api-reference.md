# Looper SDK reference

The clients a program holds: its session, each run it trains, each operation it submits, and each sampler it samples.

Generated from the docstrings of goaly-looper 0.1.0b5. Import every name from `looper_sdk`. The reference has three pages: [clients](api-reference.md), [values](api-values.md) and [errors](api-errors.md).

## ServiceClient

A program's session with Looper, built from `ServiceClient.Config`.

It creates the runs a program trains, each with a `TrainingClient`,
and the samplers it samples, each with a `SamplingClient`, and reads
the project's checkpoints. Everything it creates belongs to its session:
closing it, with `close()` or by leaving `async with`, retires the
session's runs and samplers, while their checkpoints stay.

### ServiceClient.Config

Where the service is, who calls it, and how long a call may take.

```python
ServiceClient.Config(
    *,
    base_url: str = 'https://api.looper.goaly.ai',
    project: str,
    display_name: str | None = None,
    api_key: str | None = None,
    request_timeout_s: float = 60.0,
    retry_timeout_s: float = 300.0,
)
```

| Field | Type | Default | Description |
|---|---|---|---|
| `base_url` | `str` | `'https://api.looper.goaly.ai'` | The API root without `/v1`: production by default, `https://dev.api.looper.goaly.ai` for dev. |
| `project` | `str` | required | The project the session opens in; its runs belong to it. |
| `display_name` | `str \| None` | `None` | A label for the session on the console: 1 to 255 ASCII letters, digits, `_` or `-`. Not unique, and fixed once the session opens. None for no label. |
| `api_key` | `str \| None` | `None` | The API key every request sends as `Authorization: Bearer`. None reads `$LOOPER_API_KEY` when the client is built; with neither, no credential is sent and `build()` raises `Unauthenticated`. Kept out of the config's repr, so a logged config never shows it. |
| `request_timeout_s` | `float` | `60.0` | One HTTP request's deadline. It must exceed the 30 s an await waits at the API. |
| `retry_timeout_s` | `float` | `300.0` | How long a call keeps resending a retryable failure, such as the API's restart, before raising it. |

### ServiceClient.Config.build

```python
async def build(runtime: ServiceClient.Runtime | None = None) -> ServiceClient
```

Opens the program's session.

### ServiceClient.Runtime

What a client is built over, besides its config.

```python
ServiceClient.Runtime(
    *,
    transport: httpx.AsyncBaseTransport | None = None,
    storage_transport: httpx.AsyncBaseTransport | None = None,
)
```

| Field | Type | Default | Description |
|---|---|---|---|
| `transport` | `httpx.AsyncBaseTransport \| None` | `None` | The HTTP transport, such as one in front of an API in the caller's process; None for httpx's own. |
| `storage_transport` | `httpx.AsyncBaseTransport \| None` | `None` | The transport that downloads take to storage's signed URLs, such as a test's storage; None for httpx's own. |

### ServiceClient.create_training_client

```python
async def create_training_client(
    config: TrainingConfig,
    *,
    key: str | None = None,
    timeout_s: float | None = None,
) -> TrainingClient
```

Create a run and wait until its training node is ready.

The optional deadline covers submission, retries and readiness. A timeout
leaves accepted backend work intact; its error identifies the creation.

### ServiceClient.create_sampling_client

```python
async def create_sampling_client(
    config: SamplerConfig,
    *,
    key: str | None = None,
    timeout_s: float | None = None,
) -> SamplingClient
```

Create a sampler and wait until it serves: its
source resolved and pinned, its GPUs started, a checkpoint converted
and its model loaded. As `create_training_client`: the deadline
covers submission, retries and readiness, and a timeout leaves the
accepted sampler starting, its error naming it; resending under the
same key answers the same sampler. A refusal, such as
`CheckpointNotReady` for a name still uploading or still being
exported, raises at once.

### ServiceClient.get_checkpoint

```python
async def get_checkpoint(checkpoint_id: str) -> Checkpoint
```

The project's checkpoint of that id, in whatever stage it is.
`NotFound` if the project has none.

### ServiceClient.wait_checkpoint

```python
async def wait_checkpoint(
    checkpoint_id: str,
    *,
    timeout_s: float | None = None,
) -> Checkpoint
```

The checkpoint once it is `READY_TO_USE`: uploaded, named and
loadable. `CheckpointFailed` once it failed, and
TimeoutError once `timeout_s` has passed; the upload goes on.

### ServiceClient.get_checkpoint_by_name

```python
async def get_checkpoint_by_name(name: str) -> Checkpoint
```

The checkpoint the project's `name` binds: the last one that
reached `READY_TO_USE` under it. `NotFound` if the
name binds none, as before the first one is ready.

### ServiceClient.list_checkpoints

```python
async def list_checkpoints(
    *,
    limit: int = 100,
    after_name: str | None = None,
) -> tuple[Checkpoint, ...]
```

The project's named checkpoints, in name order: up to `limit`,
1 to 1000, of those named after `after_name`, or from the first.
Pass the last name returned as `after_name` to read the next page.

### ServiceClient.delete_checkpoint

```python
async def delete_checkpoint(checkpoint_id: str) -> None
```

Deletes the checkpoint's name; its files are reclaimed later. An id
the project does not name deletes nothing: a `FAILED` checkpoint
holds no name, so `get_checkpoint` still reads it. A checkpoint still
being exported raises `CheckpointNotReady`: delete it once it is
`READY_TO_USE`.

### ServiceClient.download_checkpoint

```python
async def download_checkpoint(
    checkpoint_id: str,
    directory: str | os.PathLike[str],
    *,
    verify: bool = False,
    parallel: int = 8,
    observer: DownloadObserver | None = None,
) -> DownloadResult
```

Downloads a ready weights checkpoint's files into `directory`.
Run it again after a stop: it continues from the bytes on disk.
`verify` checks finished files again; `observer` follows
progress. `DownloadError` means the directory or a file needs the
user.

### ServiceClient.close

```python
async def close(*, timeout_s: float | None = 70.0) -> None
```

Closes the session. Its backend runs and samplers are retired
asynchronously.

## DEFAULT_BASE_URL

The API root a `ServiceClient.Config` uses unless it names another.

```python
DEFAULT_BASE_URL = 'https://api.looper.goaly.ai'
```

## TrainingClient

A run's writer, made by `ServiceClient.create_training_client`.

Every submit (`forward_backward`, `optim_step`, `save_state`,
`save_weights`, `publish_weights`) returns an `Operation` once the
service has accepted it, in the order the program submitted it; its
`result()` waits for what it computed. Each submit carries a `key`, a
new UUID unless the caller names one: a resend under the same key, in
the same epoch, returns the operation already accepted instead of a
second one, and `find_operation` finds it after a lost answer.
`after` names operations this one runs only after; if one of them
fails, this one fails with `DependencyFailed` without running. A
`RunRolledBack` moves the client to the run's new epoch, whose keys
are new.

| Attribute | Description |
|---|---|
| `run_id` | The run's id. |
| `epoch` | The run's epoch as this client last saw it, which its submits carry. |

### TrainingClient.forward_backward

```python
async def forward_backward(
    data: Sequence[Datum],
    loss: Loss,
    *,
    key: str | None = None,
    after: Sequence[Operation[Any]] = (),
) -> Operation[ForwardBackwardResult]
```

Accepts a forward and backward pass over the data, whose gradients
accumulate until the next `optim_step`.

### TrainingClient.optim_step

```python
async def optim_step(
    *,
    learning_rate: float | None = None,
    key: str | None = None,
    after: Sequence[Operation[Any]] = (),
) -> Operation[StepResult]
```

Accepts an optimizer step over the accumulated gradients: the run's
optimizer, dividing by the targets, at `learning_rate` if named,
for this step alone, else at its own.

### TrainingClient.save_state

```python
async def save_state(
    name: str,
    *,
    ttl_s: int | None = None,
    key: str | None = None,
    after: Sequence[Operation[Any]] = (),
) -> Operation[StateCheckpoint]
```

Accept a save of the run's training state, its weights with its
optimizer's, at its current step; a sampler converts it at launch.
Await `result()` separately: it answers once the step is saved,
while its upload may still be going on (`wait_checkpoint`).

The name belongs to the project and replaces its previous target only
on successful completion. A wait timeout never cancels this operation.

### TrainingClient.publish_weights

```python
async def publish_weights(
    *,
    key: str | None = None,
    after: Sequence[Operation[Any]] = (),
) -> Operation[PublishResult]
```

Accepts a publish of the run's current step into the run's
sampler, which serves it from then on. It sits at a
step boundary, as a save does: `GradientsPending` with gradients
accumulated and not applied. A loop submits it with its step and
need not await it; its result names the version now served.
`SamplerFailed` if the run's sampler failed, or did not take it.

### TrainingClient.create_sampling_client

```python
async def create_sampling_client(
    *,
    max_seq_len: int,
    max_concurrent_sequences: int,
    key: str | None = None,
    timeout_s: float | None = None,
) -> SamplingClient
```

Create this run's sampler and wait until it is linked to the
run's trainer: a sampler of the session, as
`ServiceClient.create_sampling_client` makes, whose source is this
run. It serves nothing until the run's first `publish_weights`,
and then each version the run publishes. A run has one at a time;
a failed or stopped one is replaced by creating another. The
deadline and the key work as the service client's do.

### TrainingClient.save_weights

```python
async def save_weights(
    name: str,
    *,
    ttl_s: int | None = None,
    key: str | None = None,
    after: Sequence[Operation[Any]] = (),
) -> Operation[WeightsCheckpoint]
```

Accept an export of the weights alone, for sampling:
Hugging Face safetensors, or PEFT's layout for LoRA, which a sampler
loads as they are. It holds no optimizer state.

The name belongs to the project and replaces its previous target only
on successful completion. A wait timeout never cancels this operation.

### TrainingClient.find_operation

```python
async def find_operation(key: str, *, epoch: int | None = None) -> Operation[Any] | None
```

The operation the key names in the epoch, the run's current one by
default; how a submit whose answer was lost is found.

### TrainingClient.wait_until_ready

```python
async def wait_until_ready(*, timeout_s: float | None = None) -> None
```

Wait until the run is active and takes work. `RunStateLost`,
`Cancelled` or `RunSuspended` if it failed, was deleted or is
suspended instead. Waiting starts and resumes nothing. While its
organization's GPU limit holds it back, it logs a warning each time
that reason appears.

### TrainingClient.status

```python
async def status() -> RunStatus
```

The run as the service holds it now: its state, epoch and steps.

### TrainingClient.delete

```python
async def delete() -> None
```

Deletes the run: its queued operations are cancelled, a running one
finishes, and its GPUs are released.

## RunStatus

A run, as the service holds it now.

```python
RunStatus(
    *,
    state: RunState,
    epoch: int,
    step: int,
    durable_step: int,
    waiting_reason: Literal['org_gpu_limit'] | None = None,
)
```

| Field | Type | Default | Description |
|---|---|---|---|
| `state` | `RunState` | required | Where the run is. |
| `epoch` | `int` | required | The run's epoch; a rollback starts the next one. |
| `step` | `int` | required | Completed optimizer steps. |
| `durable_step` | `int` | required | The newest step a restore point or a state checkpoint covers. |
| `waiting_reason` | `Literal['org_gpu_limit'] \| None` | `None` | Why a queued run has no trainer yet: "org_gpu_limit" while its organization's GPUs in use leave no room for it; None otherwise. |

## RunState

Where a run is.

| Member | Value |
|---|---|
| `QUEUED` | `'queued'` |
| `STARTING` | `'starting'` |
| `ACTIVE` | `'active'` |
| `RECOVERING` | `'recovering'` |
| `SUSPENDED` | `'suspended'` |
| `RESUMING` | `'resuming'` |
| `FAILED` | `'failed'` |
| `DELETED` | `'deleted'` |

## Operation

An accepted operation of a run, made by its run's client.

| Attribute | Description |
|---|---|
| `id` | The service's name for it. |
| `seq` | Its place in its run's order. |
| `key` | Its idempotency key, unique in its epoch of the run. |
| `epoch` | The run's epoch it was accepted in. |
| `kind` | `forward_backward`, `optim_step`, `save_checkpoint` (`save_state`'s), `save_weights` or `publish_weights`. |

### Operation.result

```python
async def result(*, timeout_s: float | None = None) -> T
```

What it computed, once it has run; TimeoutError once `timeout_s`
has passed, resends to an unreachable API included. `state()` reads
where it is without waiting.

### Operation.state

```python
async def state() -> OperationState
```

Where the operation is now, read without waiting.

## OperationState

Where an operation is.

| Member | Value |
|---|---|
| `QUEUED` | `'queued'` |
| `RUNNING` | `'running'` |
| `OUTCOME_UNKNOWN` | `'outcome_unknown'` |
| `SUCCEEDED` | `'succeeded'` |
| `FAILED` | `'failed'` |
| `CANCELLED` | `'cancelled'` |

## SamplingClient

One sampler of the program's session, made by
`ServiceClient.create_sampling_client` or, for a run's own sampler,
`TrainingClient.create_sampling_client`.

| Attribute | Description |
|---|---|
| `sampler_id` | The service's name for it. |
| `config` | What it was created with: its source and sizes. |
| `checkpoint_name` | What it serves, as it was created. |

### SamplingClient.status

```python
async def status() -> SamplerStatus
```

What the service last observed: not proof the sampler is alive now.

### SamplingClient.wait_until_ready

```python
async def wait_until_ready(*, timeout_s: float | None = None) -> None
```

Return once the sampler is ready: its model converted, loaded and
serving. `SamplerFailed` if it ended instead; TimeoutError once
`timeout_s` passed, which leaves it starting. While its organization's
GPU limit holds it back, it logs a warning each time that reason
appears.

### SamplingClient.sample

```python
async def sample(
    prompt: Sequence[int],
    *,
    params: SamplingParams,
    n: int = 1,
    key: str | None = None,
    max_attempts: int = 3,
) -> SamplingOperation
```

Queue `n` completions of `prompt` and return the operation once
the service has accepted it. A request this sampler could never serve
is refused here, before anything is sent. `result()` replaces terminal
retryable failures up to `max_attempts` including this acceptance;
1 disables replacement. Each replacement can sample newer weights.

### SamplingClient.stop

```python
async def stop() -> None
```

Stop the sampler: its unfinished requests are cancelled and its
GPUs released. Completed results stay readable.

## SamplingOperation

Sampling that may replace a retryable failed operation.

| Attribute | Description |
|---|---|
| `id` | The latest accepted server operation; can change during result(). |
| `key` | Its acceptance key, which changes with its operation ID. |

### SamplingOperation.state

```python
async def state() -> OperationState
```

The latest known operation's state; initiates no replacement.

### SamplingOperation.usage

```python
async def usage() -> SamplingUsage | None
```

The latest known operation's tokens, read now: None until it
succeeded, or for one recorded before Looper counted tokens. Kept
after its result's bytes expire.

### SamplingOperation.result

```python
async def result(*, timeout_s: float | None = None) -> SampleResult
```

Wait, replacing terminal retryable failures within the attempt cap.

The timeout covers lock acquisition, backoff, submissions and all
attempts for this wait. Timeout or cancellation leaves remote work
intact; a later wait resumes without resetting the cap. Other errors
propagate unchanged. Replacements may produce different samples.

## SamplerStatus

What the service last observed of a sampler: not proof that it is
alive now.

```python
SamplerStatus(
    *,
    id: str,
    config: SamplerConfig,
    state: SamplerState,
    model: str,
    version: Version | None,
    created_at: datetime,
    error: LooperError | None = None,
    usage: SamplingUsage | None = None,
    waiting_reason: Literal['org_gpu_limit'] | None = None,
)
```

| Field | Type | Default | Description |
|---|---|---|---|
| `id` | `str` | required | The service's name for it. |
| `config` | `SamplerConfig` | required | What it was created with. |
| `state` | `SamplerState` | required | Where it is. |
| `model` | `str` | required | The catalog model it serves, a checkpoint's base included. |
| `version` | `Version \| None` | required | The trained weights it serves, as its checkpoint recorded them when the sampler was created; None for a base reference. A run's sampler: the version its run last published to it, and None before the first. |
| `created_at` | `datetime` | required | When the service accepted it. |
| `error` | `LooperError \| None` | `None` | Why it failed, once it has; None otherwise. |
| `usage` | `SamplingUsage \| None` | `None` | Its succeeded requests' usage, summed when the status was read; None when one of them was recorded before Looper counted tokens. |
| `waiting_reason` | `Literal['org_gpu_limit'] \| None` | `None` | Why a starting sampler has not started: "org_gpu_limit" while its organization's GPUs in use leave no room for it; None otherwise, and always None for a run's sampler, which waits for its run. |

## SamplerState

Where a sampler is. `failed` and `stopped` are
final: a failed sampler is never relaunched; create another.

| Member | Value |
|---|---|
| `STARTING` | `'starting'` |
| `READY` | `'ready'` |
| `FAILED` | `'failed'` |
| `STOPPING` | `'stopping'` |
| `STOPPED` | `'stopped'` |
