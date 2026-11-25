# Looper SDK reference: values

The values a program builds and the results it reads. Each is frozen and checks itself when it is made: an invalid one raises `InvalidArgument`, before anything is sent.

Generated from the docstrings of goaly-looper 0.1.0b6. Import every name from `looper_sdk`. The reference has three pages: [clients](api-reference.md), [values](api-values.md) and [errors](api-errors.md).

## TrainingConfig

What a run trains, and how much. Looper sizes the run's
GPUs from the three sizing fields, which are fixed for its life.

```python
TrainingConfig(
    *,
    model: str,
    kind: Lora | Full,
    tokens_per_step: int,
    max_seq_len: int,
    sampled_tokens_per_step: int = 0,
    optimizer: Optimizer | None = None,
)
```

| Field | Type | Default | Description |
|---|---|---|---|
| `model` | `str` | required | A model of the catalog, by its Hugging Face name. |
| `kind` | `Lora \| Full` | required | LoRA factors over the frozen base, or every parameter. |
| `tokens_per_step` | `int` | required | The tokens one optimizer step trains, over its forward_backward calls. A larger batch still runs, through gradient accumulation; this sets only the step time. |
| `max_seq_len` | `int` | required | The longest sequence the run trains or samples. |
| `sampled_tokens_per_step` | `int` | `0` | Unused, and 0 by default: a run's sampler is sized when `TrainingClient.create_sampling_client` creates it. |
| `optimizer` | `Optimizer \| None` | `None` | What every `optim_step` steps, with its clipping. None trains with the model's default AdamW for the run's kind, at a constant rate. |

## Lora

Low-rank factors over the frozen base.

```python
Lora(
    *,
    rank: int = 32,
    train_attn: bool = True,
    train_mlp: bool = True,
    train_unembed: bool = True,
)
```

| Field | Type | Default | Description |
|---|---|---|---|
| `rank` | `int` | `32` | The factors' rank; their product is scaled by 32 / rank. |
| `train_attn` | `bool` | `True` | Factors on the attention projections, linear attention's included. |
| `train_mlp` | `bool` | `True` | Factors on the MLPs, and an MoE layer's experts. |
| `train_unembed` | `bool` | `True` | Factors on the LM head. |

## Full

Every parameter trains.

```python
Full()
```

## ModelSpec

A saved model's identity and trainable parameters.

```python
ModelSpec(*, model: str, kind: Lora | Full)
```

| Field | Type | Default | Description |
|---|---|---|---|
| `model` | `str` | required | The source run's catalog model name, not a resolved revision. |
| `kind` | `Lora \| Full` | required | Full parameters or the exact LoRA rank and target switches. |

## AdamW

AdamW over every trainable parameter: fp32 moments, bias correction
counted over the run's own steps.

```python
AdamW(
    *,
    learning_rate: float,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-08,
    weight_decay: float = 0.0,
    max_grad_norm: float | None = None,
)
```

| Field | Type | Default | Description |
|---|---|---|---|
| `learning_rate` | `float` | required | The rate every step trains at, above 0. |
| `betas` | `tuple[float, float]` | `(0.9, 0.999)` | The first and second moments' decay rates, each in [0, 1). |
| `eps` | `float` | `1e-08` | Added to the second moment's root, above 0. |
| `weight_decay` | `float` | `0.0` | Decoupled weight decay, at least 0. |
| `max_grad_norm` | `float \| None` | `None` | The global gradient norm a step clips to before the update, above 0. None measures the norm without clipping. |

## Muon

Muon over the model's matrices, and AdamW over every other parameter,
at one learning rate and one weight decay: each matrix's update is
orthogonalized and scaled to AdamW's update size. Its profile names the
matrices; one that names none refuses Muon when the run is created.

```python
Muon(
    *,
    learning_rate: float,
    weight_decay: float = 0.1,
    momentum: float = 0.95,
    nesterov: bool = True,
    ns_steps: int = 5,
    betas: tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-08,
    max_grad_norm: float | None = None,
)
```

| Field | Type | Default | Description |
|---|---|---|---|
| `learning_rate` | `float` | required | Both halves' rate, above 0, every step trains at. |
| `weight_decay` | `float` | `0.1` | Both halves' decoupled weight decay, at least 0. |
| `momentum` | `float` | `0.95` | The decay rate of Muon's momentum, in [0, 1). |
| `nesterov` | `bool` | `True` | Whether Muon steps along its momentum's Nesterov look-ahead rather than the momentum itself. |
| `ns_steps` | `int` | `5` | Newton–Schulz iterations orthogonalizing each update, at least 1. |
| `betas` | `tuple[float, float]` | `(0.9, 0.95)` | The AdamW half's moment decay rates, each in [0, 1). |
| `eps` | `float` | `1e-08` | Added to the AdamW half's second-moment root, above 0. |
| `max_grad_norm` | `float \| None` | `None` | The global gradient norm a step clips to before the update, above 0. None measures the norm without clipping. |

## Optimizer

Any optimizer a run may name in `TrainingConfig.optimizer`.

```python
Optimizer = looper_sdk.optimizers.AdamW | looper_sdk.optimizers.Muon
```

## Datum

One sequence: each token after the first is a target,
predicted from the tokens before it.

```python
Datum(
    *,
    tokens: Sequence[int],
    weights: Sequence[float],
    columns: Mapping[str, Sequence[float]] = {},
    datum_id: str | None = None,
    reward: float | None = None,
    group_id: str | None = None,
    group_size: int | None = None,
)
```

| Field | Type | Default | Description |
|---|---|---|---|
| `tokens` | `Sequence[int]` | required | Token ids, `N` of them. |
| `weights` | `Sequence[float]` | required | One per token, 0 or 1: whether the term for predicting `tokens[i]` counts. `weights[0]` is 0, since nothing predicts the first token, and at least one other is 1. A weight of 0 masks its token, which is still attended as context. No loss weighs its targets, so no other weight is taken. |
| `columns` | `Mapping[str, Sequence[float]]` | `{}` | Per-token floats by name, aligned to `tokens` as `weights` is: `logprobs`, the sampler's record of each token, for `Grpo` and `Dapo`. Finite wherever the weight is 1; float32 on the wire. |
| `datum_id` | `str \| None` | `None` | Echoed in this datum's results; unique within a call. |
| `reward` | `float \| None` | `None` | The datum's score: finite, float64 on the wire. `reward`, `group_id` and `group_size` come together or not at all, on exactly the datums of a loss that compares groups, which forms their advantages from the group's rewards. |
| `group_id` | `str \| None` | `None` | The group of samples the datum is compared within. |
| `group_size` | `int \| None` | `None` | How many datums the group has, 1 or more. A group is sent whole within one call. |

## CrossEntropy

SFT: −Σ log π_θ(y), the targets imitated. Reads no column and compares
no groups.

```python
CrossEntropy()
```

| Class attribute | Value | Description |
|---|---|---|
| `reads` | `frozenset()` | The columns its datums carry. |
| `min_group_size` | `None` | None: it compares no groups, so its datums carry no reward and no group. |

## Grpo

GRPO: each datum's reward normalized within its group, trained with
PPO's clipped surrogate, and with `kl_weight` above 0 a penalty on the
policy drifting from the sampler. `clip_high > clip_low` is DAPO's
clip-higher.

```python
Grpo(
    *,
    clip_low: float = 0.2,
    clip_high: float = 0.2,
    kl_weight: float = 0.0,
    scale_by_std: bool = True,
)
```

| Field | Type | Default | Description |
|---|---|---|---|
| `clip_low` | `float` | `0.2` | The ratio's floor, as an offset below 1: in (0, 1). |
| `clip_high` | `float` | `0.2` | The ratio's ceiling, as an offset above 1: above 0. |
| `kl_weight` | `float` | `0.0` | The penalty's weight, 0 or above: the k3 estimate of KL(π_sampler ‖ π_θ), Σ(r − 1 − log r), added to the surrogate. 0, the default, adds none. |
| `scale_by_std` | `bool` | `True` | Whether a datum's advantage, its reward less its group's mean, is divided by the group's standard deviation. False leaves it undivided, as Dr. GRPO does. |

| Class attribute | Value | Description |
|---|---|---|
| `reads` | `frozenset({'logprobs'})` | The columns its datums carry: the sampler's `logprobs`. The trainer forms the advantages, so a datum never carries them. |
| `min_group_size` | `2` | The fewest datums a group may have: a mean needs something to compare against. |

## Dapo

DAPO: GRPO's advantages and loss with the clip's upper bound at 0.28.

```python
Dapo(
    *,
    clip_low: float = 0.2,
    clip_high: float = 0.28,
    kl_weight: float = 0.0,
    scale_by_std: bool = True,
)
```

| Field | Type | Default | Description |
|---|---|---|---|
| `clip_low` | `float` | `0.2` | The ratio's floor, as an offset below 1: in (0, 1). |
| `clip_high` | `float` | `0.28` | The ratio's ceiling, as an offset above 1: above 0. |
| `kl_weight` | `float` | `0.0` | The penalty's weight, 0 or above: the k3 estimate of KL(π_sampler ‖ π_θ), Σ(r − 1 − log r), added to the surrogate. 0, the default, adds none. |
| `scale_by_std` | `bool` | `True` | Whether a datum's advantage, its reward less its group's mean, is divided by the group's standard deviation. False leaves it undivided, as Dr. GRPO does. |

| Class attribute | Value | Description |
|---|---|---|
| `reads` | `frozenset({'logprobs'})` | The columns its datums carry: the sampler's `logprobs`. The trainer forms the advantages, so a datum never carries them. |
| `min_group_size` | `2` | The fewest datums a group may have: a mean needs something to compare against. |

## Loss

Any loss a `forward_backward` call may name.

```python
Loss = looper_sdk.losses.CrossEntropy | looper_sdk.losses.Grpo | looper_sdk.losses.Dapo
```

## ForwardBackwardResult

What a forward_backward counted from its request. The loss and its
metrics come with the step.

```python
ForwardBackwardResult(
    *,
    weighted_targets: int,
    data: Sequence[DatumResult],
    num_tokens: int | None,
)
```

| Field | Type | Default | Description |
|---|---|---|---|
| `weighted_targets` | `int` | required | The targets, the tokens of weight 1. |
| `data` | `Sequence[DatumResult]` | required | One entry per datum. |
| `num_tokens` | `int \| None` | required | Every token of every datum, the work billed as trained; None in a result recorded before Looper counted it. |

## DatumResult

One datum's results, in the order of the call's data.

```python
DatumResult(*, datum_id: str | None)
```

| Field | Type | Default | Description |
|---|---|---|---|
| `datum_id` | `str \| None` | required | The datum's own, echoed. |

## StepResult

Both halves of an optim_step's reduction.

```python
StepResult(
    *,
    step: int,
    version: Version,
    loss_sum: float,
    denominator: float,
    weighted_targets: int,
    weight_sum: float,
    metrics: Mapping[str, float],
    grad_norm: float,
    clipped: bool,
    lr: float,
    num_tokens: int | None,
)
```

| Field | Type | Default | Description |
|---|---|---|---|
| `step` | `int` | required | Completed optimizer steps, this one included. |
| `version` | `Version` | required | The weights it made. |
| `loss_sum` | `float` | required | The losses accumulated since the last step, summed. |
| `denominator` | `float` | required | What the gradients were divided by. |
| `weighted_targets` | `int` | required | The targets with nonzero weight accumulated. |
| `weight_sum` | `float` | required | The targets' weights accumulated, summed; with weights of 0 or 1, the same count as `weighted_targets`. |
| `metrics` | `Mapping[str, float]` | required | The loss's own diagnostics, by name, as the trainer reported them. |
| `grad_norm` | `float` | required | The global gradient norm before clipping. |
| `clipped` | `bool` | required | Whether clipping scaled the gradients. |
| `lr` | `float` | required | The learning rate the step applied. |
| `num_tokens` | `int \| None` | required | The `num_tokens` of the forward_backwards whose gradients the step applied, summed; None when one of them has none. |

## PublishResult

What a publish_weights moved.

```python
PublishResult(*, version: Version, sampler_id: str)
```

| Field | Type | Default | Description |
|---|---|---|---|
| `version` | `Version` | required | The version the run's sampler now serves. |
| `sampler_id` | `str` | required | The sampler it reached. |

## Version

A run's weights after some optimizer steps.

```python
Version(*, run_id: str, epoch: int, step: int)
```

| Field | Type | Default | Description |
|---|---|---|---|
| `run_id` | `str` | required | The run that trained them. |
| `epoch` | `int` | required | The run's epoch they were made in; it counts the run's rollbacks. |
| `step` | `int` | required | The optimizer steps the run had completed. |

## Checkpoint

A checkpoint observed at save or registry lookup time: a
`StateCheckpoint` or a `WeightsCheckpoint`, which `kind` names.

```python
Checkpoint(
    *,
    name: str,
    expires_at: datetime | None,
    id: str,
    operation_id: str | None,
    project: str,
    principal: str,
    source_run_id: str,
    version: Version,
    model_spec: ModelSpec,
    size_bytes: int,
    created_at: datetime,
    state: CheckpointState,
    failure: str | None = None,
)
```

| Field | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | required | The name it binds once `READY_TO_USE`; until then, the name its save or export asked for, which still resolves to its previous target. |
| `expires_at` | `datetime \| None` | required | Set once `READY_TO_USE`, when its TTL starts. An export takes no TTL of its own but its source's `expires_at`, copied at its first request; a retry after it failed keeps that copy. |
| `id` | `str` | required | The service's name for it, which `get_checkpoint` and `wait_checkpoint` take. |
| `operation_id` | `str \| None` | required | The save operation that made it; None for a checkpoint no save made, such as an exported one. |
| `project` | `str` | required | The project it belongs to. |
| `principal` | `str` | required | Creating session's user or service identity; an export's is its source's. |
| `source_run_id` | `str` | required | The run it was saved from. |
| `version` | `Version` | required | Exact source run, epoch and committed optimizer step. |
| `model_spec` | `ModelSpec` | required | Source run model name and exact trainable parameters. |
| `size_bytes` | `int` | required | Physical directory size, known once `READY_TO_USE`, and 0 before; aliases do not duplicate bytes. |
| `created_at` | `datetime` | required | Time of registry publication once `READY_TO_USE`; before, the time its save succeeded or, for an export, the time it was requested, or last requested again after it failed. With timezone. |
| `state` | `CheckpointState` | required | Where it is; see `CheckpointState`. |
| `failure` | `str \| None` | `None` | Why a `FAILED` checkpoint failed, for people; None otherwise. |

## StateCheckpoint

Training state, from `save_state`: the weights with the optimizer's
state, which a sampler converts at launch.

Its fields are `Checkpoint`'s.

| Class attribute | Value | Description |
|---|---|---|
| `kind` | `'state'` | `state` or `weights`: which save made it, or `weights` for an export of a state checkpoint. |

## WeightsCheckpoint

The weights alone, from `save_weights` or exported from a
full-parameter `StateCheckpoint`: Hugging Face safetensors, or PEFT's
layout for LoRA, which a sampler loads as they are. It holds no
optimizer state.

Its fields are `Checkpoint`'s.

| Class attribute | Value | Description |
|---|---|---|
| `kind` | `'weights'` | `state` or `weights`: which save made it, or `weights` for an export of a state checkpoint. |

## CheckpointState

Where a checkpoint is. A save succeeds once its step is
on the run's disk, and its checkpoint is `UPLOADING`; only a
`READY_TO_USE` one is uploaded, named and loadable.

| Member | Value |
|---|---|
| `UPLOADING` | `'uploading'` |
| `READY_TO_USE` | `'ready_to_use'` |
| `FAILED` | `'failed'` |

## SamplerConfig

What a sampler serves, and how much.

```python
SamplerConfig(*, checkpoint_name: str, max_seq_len: int, max_concurrent_sequences: int)
```

| Field | Type | Default | Description |
|---|---|---|---|
| `checkpoint_name` | `str` | required | A published checkpoint of the session's project, by its name, or `base:<model>` for a catalog model's base weights. Resolved once, when the sampler is created: a later publication under the same name does not change what it serves. Or `run:<run_id>` for a run of the session, as `TrainingClient.create_sampling_client` writes it: the sampler serves each version the run publishes to it. |
| `max_seq_len` | `int` | required | The longest prompt plus completion one sample takes, in tokens. |
| `max_concurrent_sequences` | `int` | required | The sequences the sampler generates at once; a request for `n` completions takes `n` of them. |

### SamplerConfig.base_model

Property: `str \| None`.

The catalog model a `base:` reference names; None for a checkpoint.

### SamplerConfig.run_id

Property: `str \| None`.

The run a `run:` reference names; None for any other source.

## SamplingParams

How each completion is drawn.

```python
SamplingParams(
    *,
    max_tokens: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
    stop_token_ids: Sequence[int] = (),
)
```

| Field | Type | Default | Description |
|---|---|---|---|
| `max_tokens` | `int` | required | The most tokens each completion generates. |
| `temperature` | `float` | `1.0` | 0 is greedy. |
| `top_p` | `float` | `1.0` | Nucleus sampling's mass, in (0, 1]. |
| `stop_token_ids` | `Sequence[int]` | `()` | Token ids that end a completion. |

## SampleRequest

`n` completions of one prompt: what `SamplingClient.sample` sends.

```python
SampleRequest(*, prompt: Sequence[int], params: SamplingParams, n: int = 1)
```

| Field | Type | Default | Description |
|---|---|---|---|
| `prompt` | `Sequence[int]` | required | Token ids, never text; at least one. |
| `params` | `SamplingParams` | required | How each completion is drawn. |
| `n` | `int` | `1` | Completions wanted; each takes one of the sampler's `max_concurrent_sequences` while it generates. |

### SampleRequest.check

```python
def check(config: SamplerConfig) -> None
```

InvalidArgument unless the sampler `config` describes can serve
it: an `n` it could never fit would wait forever.

## SampleResult

A request's completions, `n` of them, and what generated them.

```python
SampleResult(
    *,
    sequences: Sequence[SampleSequence],
    model: str,
    checkpoint_name: str,
    version: Version | None = None,
)
```

| Field | Type | Default | Description |
|---|---|---|---|
| `sequences` | `Sequence[SampleSequence]` | required | The completions, one per `n`. |
| `model` | `str` | required | The catalog model the sampler serves. |
| `checkpoint_name` | `str` | required | As the sampler was created with it. |
| `version` | `Version \| None` | `None` | The trained weights it serves; None for a base reference, and for a run's sampler, whose sequences each name their own. |

## SampleSequence

One completion.

```python
SampleSequence(
    *,
    token_ids: Sequence[int],
    logprobs: Sequence[float],
    finish_reason: FinishReason,
    cached_tokens: int = 0,
    version: Version | None = None,
)
```

| Field | Type | Default | Description |
|---|---|---|---|
| `token_ids` | `Sequence[int]` | required | The tokens generated. |
| `logprobs` | `Sequence[float]` | required | The engine's log probability of each chosen token, one per token; not a normalized distribution over candidates. |
| `finish_reason` | `FinishReason` | required | Why it ended. |
| `cached_tokens` | `int` | `0` | Prompt tokens the engine's prefix cache served. |
| `version` | `Version \| None` | `None` | The trained weights that generated it. From a run's sampler, the oldest version that generated one of its tokens: a publish during the generation lands under it. None for a base reference. |

## SamplingUsage

The tokens of a request's engine work, or of a sampler's requests
summed.

```python
SamplingUsage(
    *,
    num_prompt_tokens: int,
    num_generated_tokens: int,
    num_cached_tokens: int,
)
```

| Field | Type | Default | Description |
|---|---|---|---|
| `num_prompt_tokens` | `int` | required | The prompt's tokens, counted once per request whatever its `n`: the engine computes a prompt once and its `n` sequences share it. |
| `num_generated_tokens` | `int` | required | Every token generated, over all `n` sequences. |
| `num_cached_tokens` | `int` | required | The prompt tokens the engine's prefix cache served, the most any of the request's sequences reported; at most `num_prompt_tokens`. For `n` above 1 the engine first computes the prompt once, then reports it cached for each copy, so this can count prompt tokens that request computed. |

## FinishReason

Why a completion ended: `STOP` at a stop token or the model's end
of text, `LENGTH` at `max_tokens`.

| Member | Value |
|---|---|
| `STOP` | `'stop'` |
| `LENGTH` | `'length'` |

## DownloadResult

A finished download: every file complete and correct.

```python
DownloadResult(*, directory: Path, files: int, size_bytes: int)
```

| Field | Type | Default | Description |
|---|---|---|---|
| `directory` | `Path` | required | Where the files are, as a model directory. |
| `files` | `int` | required | How many files the checkpoint has. |
| `size_bytes` | `int` | required | The checkpoint's bytes, all on disk. |

## DownloadPlan

What a download found on disk before its first byte.

```python
DownloadPlan(*, files: int, size_bytes: int, on_disk_bytes: int)
```

| Field | Type | Default | Description |
|---|---|---|---|
| `files` | `int` | required | The checkpoint's files. |
| `size_bytes` | `int` | required | The checkpoint's bytes. |
| `on_disk_bytes` | `int` | required | Bytes already on disk, finished or partial, which the download keeps. |

## DownloadObserver

What a download tells its caller as it goes, such as a progress bar.
None of its methods ever receives a URL.

### DownloadObserver.started

```python
def started(plan: DownloadPlan) -> None
```

Once, before the first byte: what is to come and what is on disk.

### DownloadObserver.advanced

```python
def advanced(size_bytes: int) -> None
```

Bytes written; negative when a file starts again from byte 0.

### DownloadObserver.finished

```python
def finished(path: str) -> None
```

A file is complete and correct, with its final name.

### DownloadObserver.notice

```python
def notice(message: str) -> None
```

A message for people: a lost connection, a file downloaded again.
