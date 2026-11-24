"""The SDK's values, losses, results and errors as ``looper.v1``'s messages,
and back.

The one module of the SDK that maps between the two. The clients encode
through it, and Looper's API decodes requests through it, so decoding
a request *is* validating it: each ``*_from_proto`` builds the typed value,
which refuses what is invalid with ``InvalidArgument``, and refuses a field
the wire requires but the message lacks.
"""

from __future__ import annotations

from datetime import UTC

from google.protobuf.message import Message

from looper_sdk.checkpoints import CHECKPOINT_KINDS, Checkpoint, CheckpointState
from looper_sdk.errors import (
    ERRORS,
    Category,
    ErrorCode,
    InvalidArgument,
    LooperError,
    RunRolledBack,
)
from looper_sdk.losses import CrossEntropy, Dapo, Grpo, Loss
from looper_sdk.optimizers import AdamW, Muon, Optimizer
from looper_sdk.proto.v1 import (
    checkpoints_pb2,
    errors_pb2,
    losses_pb2,
    operations_pb2,
    sampling_pb2,
    values_pb2,
)
from looper_sdk.results import DatumResult, ForwardBackwardResult, PublishResult, StepResult
from looper_sdk.sampling import (
    FinishReason,
    SamplerConfig,
    SampleRequest,
    SampleResult,
    SamplerState,
    SamplerStatus,
    SampleSequence,
    SamplingParams,
    SamplingUsage,
)
from looper_sdk.values import Datum, Full, Lora, ModelSpec, TrainingConfig, Version


def _required(message: Message, field: str, where: str):
    if not message.HasField(field):
        raise InvalidArgument(f"{where}.{field} is required")
    return getattr(message, field)


def _arm(message: Message, oneof: str, where: str) -> str:
    arm = message.WhichOneof(oneof)
    if arm is None:
        raise InvalidArgument(f"{where}.{oneof} is required")
    return arm


def version_to_proto(version: Version) -> values_pb2.Version:
    return values_pb2.Version(run_id=version.run_id, epoch=version.epoch, step=version.step)


def version_from_proto(message: values_pb2.Version) -> Version:
    return Version(run_id=message.run_id, epoch=message.epoch, step=message.step)


def _kind_to_proto(
    kind: Lora | Full, message: values_pb2.TrainingConfig | values_pb2.ModelSpec
) -> None:
    if isinstance(kind, Lora):
        message.lora.CopyFrom(
            values_pb2.TrainingConfig.Lora(
                rank=kind.rank,
                train_attn=kind.train_attn,
                train_mlp=kind.train_mlp,
                train_unembed=kind.train_unembed,
            )
        )
    else:
        message.full.SetInParent()


def _kind_from_proto(message: values_pb2.TrainingConfig | values_pb2.ModelSpec) -> Lora | Full:
    where = message.DESCRIPTOR.name
    if _arm(message, "kind", where) == "lora":
        lora = message.lora
        return Lora(
            rank=lora.rank,
            train_attn=_required(lora, "train_attn", where + ".lora"),
            train_mlp=_required(lora, "train_mlp", where + ".lora"),
            train_unembed=_required(lora, "train_unembed", where + ".lora"),
        )
    return Full()


def model_spec_to_proto(value: ModelSpec) -> values_pb2.ModelSpec:
    message = values_pb2.ModelSpec(model=value.model)
    _kind_to_proto(value.kind, message)
    return message


def model_spec_from_proto(message: values_pb2.ModelSpec) -> ModelSpec:
    return ModelSpec(model=message.model, kind=_kind_from_proto(message))


def training_config_to_proto(config: TrainingConfig) -> values_pb2.TrainingConfig:
    message = values_pb2.TrainingConfig(
        model=config.model,
        tokens_per_step=config.tokens_per_step,
        max_seq_len=config.max_seq_len,
        sampled_tokens_per_step=config.sampled_tokens_per_step,
    )
    _kind_to_proto(config.kind, message)
    if config.optimizer is not None:
        message.optimizer.CopyFrom(optimizer_to_proto(config.optimizer))
    return message


def training_config_from_proto(message: values_pb2.TrainingConfig) -> TrainingConfig:
    return TrainingConfig(
        model=message.model,
        kind=_kind_from_proto(message),
        tokens_per_step=message.tokens_per_step,
        max_seq_len=message.max_seq_len,
        sampled_tokens_per_step=message.sampled_tokens_per_step,
        optimizer=(
            optimizer_from_proto(message.optimizer) if message.HasField("optimizer") else None
        ),
    )


def optimizer_to_proto(optimizer: Optimizer) -> values_pb2.Optimizer:
    if isinstance(optimizer, Muon):
        return values_pb2.Optimizer(
            muon=values_pb2.Optimizer.Muon(
                learning_rate=optimizer.learning_rate,
                weight_decay=optimizer.weight_decay,
                momentum=optimizer.momentum,
                nesterov=optimizer.nesterov,
                ns_steps=optimizer.ns_steps,
                beta1=optimizer.betas[0],
                beta2=optimizer.betas[1],
                eps=optimizer.eps,
                max_grad_norm=optimizer.max_grad_norm,
            )
        )
    return values_pb2.Optimizer(
        adamw=values_pb2.Optimizer.AdamW(
            learning_rate=optimizer.learning_rate,
            beta1=optimizer.betas[0],
            beta2=optimizer.betas[1],
            eps=optimizer.eps,
            weight_decay=optimizer.weight_decay,
            max_grad_norm=optimizer.max_grad_norm,
        )
    )


def optimizer_from_proto(message: values_pb2.Optimizer) -> Optimizer:
    if _arm(message, "kind", "Optimizer") == "muon":
        muon, where = message.muon, "Optimizer.muon"
        return Muon(
            learning_rate=muon.learning_rate,
            weight_decay=_required(muon, "weight_decay", where),
            momentum=_required(muon, "momentum", where),
            nesterov=_required(muon, "nesterov", where),
            ns_steps=muon.ns_steps,
            betas=(_required(muon, "beta1", where), _required(muon, "beta2", where)),
            eps=muon.eps,
            max_grad_norm=muon.max_grad_norm if muon.HasField("max_grad_norm") else None,
        )
    adamw, where = message.adamw, "Optimizer.adamw"
    return AdamW(
        learning_rate=adamw.learning_rate,
        betas=(_required(adamw, "beta1", where), _required(adamw, "beta2", where)),
        eps=adamw.eps,
        weight_decay=_required(adamw, "weight_decay", where),
        max_grad_norm=adamw.max_grad_norm if adamw.HasField("max_grad_norm") else None,
    )


def datum_to_proto(datum: Datum) -> values_pb2.Datum:
    # None leaves an optional field unset.
    message = values_pb2.Datum(
        tokens=datum.tokens,
        weights=datum.weights,
        datum_id=datum.datum_id,
        reward=datum.reward,
        group_id=datum.group_id,
        group_size=datum.group_size,
    )
    for name, values in datum.columns.items():
        message.columns[name].values.extend(values)
    return message


def datum_from_proto(message: values_pb2.Datum) -> Datum:
    def given(field: str):
        return getattr(message, field) if message.HasField(field) else None

    return Datum(
        tokens=tuple(message.tokens),
        weights=tuple(message.weights),
        columns={name: tuple(column.values) for name, column in message.columns.items()},
        datum_id=given("datum_id"),
        reward=given("reward"),
        group_id=given("group_id"),
        group_size=given("group_size"),
    )


def loss_to_proto(loss: Loss) -> losses_pb2.Loss:
    message = losses_pb2.Loss()
    if isinstance(loss, CrossEntropy):
        message.cross_entropy.SetInParent()
    else:
        # By exact type: a Dapo is a Grpo, and keeps its own arm.
        arm = message.dapo if type(loss) is Dapo else message.grpo
        arm.clip_low = loss.clip_low
        arm.clip_high = loss.clip_high
        arm.kl_weight = loss.kl_weight
        arm.scale_by_std = loss.scale_by_std
    return message


def loss_from_proto(message: losses_pb2.Loss) -> Loss:
    arm = _arm(message, "loss", "Loss")
    if arm == "cross_entropy":
        return CrossEntropy()
    settings = getattr(message, arm)
    cls = Dapo if arm == "dapo" else Grpo
    return cls(
        clip_low=_required(settings, "clip_low", f"Loss.{arm}"),
        clip_high=_required(settings, "clip_high", f"Loss.{arm}"),
        # Unset is no penalty and GRPO's scaling: a loss sent before the
        # fields decodes as it was.
        kl_weight=settings.kl_weight,
        scale_by_std=settings.scale_by_std if settings.HasField("scale_by_std") else True,
    )


def _datum_results_to_proto(data) -> list[operations_pb2.DatumResult]:
    return [
        operations_pb2.DatumResult(datum_id=d.datum_id)
        if d.datum_id is not None
        else operations_pb2.DatumResult()
        for d in data
    ]


def _datum_results_from_proto(data) -> tuple[DatumResult, ...]:
    return tuple(DatumResult(datum_id=d.datum_id if d.HasField("datum_id") else None) for d in data)


def forward_backward_result_to_proto(
    result: ForwardBackwardResult,
) -> operations_pb2.ForwardBackwardResult:
    return operations_pb2.ForwardBackwardResult(
        weighted_targets=result.weighted_targets,
        data=_datum_results_to_proto(result.data),
        num_tokens=result.num_tokens,
    )


def forward_backward_result_from_proto(
    message: operations_pb2.ForwardBackwardResult,
) -> ForwardBackwardResult:
    return ForwardBackwardResult(
        weighted_targets=message.weighted_targets,
        data=_datum_results_from_proto(message.data),
        num_tokens=message.num_tokens if message.HasField("num_tokens") else None,
    )


def step_result_to_proto(result: StepResult) -> operations_pb2.StepResult:
    return operations_pb2.StepResult(
        step=result.step,
        version=version_to_proto(result.version),
        loss_sum=result.loss_sum,
        denominator=result.denominator,
        weighted_targets=result.weighted_targets,
        weight_sum=result.weight_sum,
        metrics=dict(result.metrics),
        grad_norm=result.grad_norm,
        clipped=result.clipped,
        lr=result.lr,
        num_tokens=result.num_tokens,
    )


def step_result_from_proto(message: operations_pb2.StepResult) -> StepResult:
    return StepResult(
        step=message.step,
        version=version_from_proto(message.version),
        loss_sum=message.loss_sum,
        denominator=message.denominator,
        weighted_targets=message.weighted_targets,
        weight_sum=message.weight_sum,
        metrics=dict(message.metrics),
        grad_norm=message.grad_norm,
        clipped=message.clipped,
        lr=message.lr,
        num_tokens=message.num_tokens if message.HasField("num_tokens") else None,
    )


def publish_result_to_proto(result: PublishResult) -> operations_pb2.PublishResult:
    return operations_pb2.PublishResult(
        version=version_to_proto(result.version), sampler_id=result.sampler_id
    )


def publish_result_from_proto(message: operations_pb2.PublishResult) -> PublishResult:
    return PublishResult(
        version=version_from_proto(_required(message, "version", "PublishResult")),
        sampler_id=message.sampler_id,
    )


_CATEGORIES = {
    Category.USER: errors_pb2.Error.USER,
    Category.SERVER: errors_pb2.Error.SERVER,
}


def error_to_proto(error: LooperError) -> errors_pb2.Error:
    message = errors_pb2.Error(
        code=error.code.value if error.code else "CODE_UNSPECIFIED",
        category=_CATEGORIES[error.category],
        retryable=error.retryable,
        message=error.message,
    )
    if error.retry_after_s is not None:
        message.retry_after_s = error.retry_after_s
    if isinstance(error, RunRolledBack):
        message.run_rolled_back.CopyFrom(
            errors_pb2.RunRolledBack(
                epoch=error.epoch,
                restored_step=error.restored_step,
                restored_seq=error.restored_seq,
                first_lost_seq=error.first_lost_seq,
                last_lost_seq=error.last_lost_seq,
            )
        )
    return message


def error_from_proto(message: errors_pb2.Error) -> LooperError:
    """The typed error a message names. A code this SDK does not know, or a
    rollback without its details, arrives as a plain LooperError."""
    category = {v: k for k, v in _CATEGORIES.items()}.get(message.category)
    common = {
        "retryable": message.retryable,
        "category": category,
        "retry_after_s": message.retry_after_s if message.HasField("retry_after_s") else None,
    }
    try:
        code = ErrorCode(errors_pb2.Error.Code.Name(message.code))
    except ValueError:
        return LooperError(f"code {message.code}: {message.message}", **common)
    if code is ErrorCode.RUN_ROLLED_BACK:
        if not message.HasField("run_rolled_back"):
            return LooperError(f"RUN_ROLLED_BACK without its details: {message.message}", **common)
        details = message.run_rolled_back
        return RunRolledBack(
            message.message,
            epoch=details.epoch,
            restored_step=details.restored_step,
            restored_seq=details.restored_seq,
            first_lost_seq=details.first_lost_seq,
            last_lost_seq=details.last_lost_seq,
            **common,
        )
    return ERRORS[code](message.message, **common)


_CHECKPOINT_STATES = {
    CheckpointState.UPLOADING: checkpoints_pb2.Checkpoint.UPLOADING,
    CheckpointState.READY_TO_USE: checkpoints_pb2.Checkpoint.READY_TO_USE,
    CheckpointState.FAILED: checkpoints_pb2.Checkpoint.FAILED,
}
# A server older than the stages sends none, and published only usable ones.
_CHECKPOINT_STATES_FROM = {v: k for k, v in _CHECKPOINT_STATES.items()} | {
    checkpoints_pb2.Checkpoint.STATE_UNSPECIFIED: CheckpointState.READY_TO_USE
}


_CHECKPOINT_KINDS = {
    "state": checkpoints_pb2.Checkpoint.STATE,
    "weights": checkpoints_pb2.Checkpoint.WEIGHTS,
}
# A server older than the kinds sends none, and saved only state.
_CHECKPOINT_KINDS_FROM = {v: k for k, v in _CHECKPOINT_KINDS.items()} | {
    checkpoints_pb2.Checkpoint.KIND_UNSPECIFIED: "state"
}


def checkpoint_to_proto(value: Checkpoint) -> checkpoints_pb2.Checkpoint:
    message = checkpoints_pb2.Checkpoint(
        name=value.name,
        id=value.id,
        operation_id=value.operation_id or "",
        project=value.project,
        principal=value.principal,
        source_run_id=value.source_run_id,
        version=version_to_proto(value.version),
        model_spec=model_spec_to_proto(value.model_spec),
        size_bytes=value.size_bytes,
        state=_CHECKPOINT_STATES[value.state],
        failure=value.failure or "",
        kind=_CHECKPOINT_KINDS[value.kind],
    )
    message.created_at.FromDatetime(value.created_at)
    if value.expires_at is not None:
        message.expires_at.FromDatetime(value.expires_at)
    return message


def checkpoint_from_proto(message: checkpoints_pb2.Checkpoint) -> Checkpoint:
    _required(message, "version", "Checkpoint")
    _required(message, "created_at", "Checkpoint")
    state = _CHECKPOINT_STATES_FROM.get(message.state)
    if state is None:
        raise InvalidArgument(f"Checkpoint.state {message.state} is not one this SDK knows")
    kind = _CHECKPOINT_KINDS_FROM.get(message.kind)
    if kind is None:
        raise InvalidArgument(f"Checkpoint.kind {message.kind} is not one this SDK knows")
    return CHECKPOINT_KINDS[kind](
        name=message.name,
        expires_at=message.expires_at.ToDatetime(tzinfo=UTC)
        if message.HasField("expires_at")
        else None,
        id=message.id,
        operation_id=message.operation_id or None,
        project=message.project,
        principal=message.principal,
        source_run_id=message.source_run_id,
        version=version_from_proto(message.version),
        model_spec=model_spec_from_proto(_required(message, "model_spec", "Checkpoint")),
        size_bytes=message.size_bytes,
        created_at=message.created_at.ToDatetime(tzinfo=UTC),
        state=state,
        failure=message.failure or None,
    )


_FINISH = {
    FinishReason.STOP: sampling_pb2.SampleSequence.STOP,
    FinishReason.LENGTH: sampling_pb2.SampleSequence.LENGTH,
}


def sample_request_to_proto(request: SampleRequest) -> sampling_pb2.SampleRequest:
    params = request.params
    return sampling_pb2.SampleRequest(
        prompt=list(request.prompt),
        params=sampling_pb2.SamplingParams(
            max_tokens=params.max_tokens,
            temperature=params.temperature,
            top_p=params.top_p,
            stop_token_ids=list(params.stop_token_ids),
        ),
        n=request.n,
    )


def sample_request_from_proto(message: sampling_pb2.SampleRequest) -> SampleRequest:
    params = _required(message, "params", "SampleRequest")
    return SampleRequest(
        prompt=tuple(message.prompt),
        params=SamplingParams(
            max_tokens=params.max_tokens,
            temperature=params.temperature,
            top_p=params.top_p,
            stop_token_ids=tuple(params.stop_token_ids),
        ),
        n=message.n,
    )


def sample_result_to_proto(result: SampleResult) -> sampling_pb2.SampleResult:
    message = sampling_pb2.SampleResult(
        sequences=[
            sampling_pb2.SampleSequence(
                token_ids=list(s.token_ids),
                logprobs=list(s.logprobs),
                finish_reason=_FINISH[s.finish_reason],
                cached_tokens=s.cached_tokens,
                version=None if s.version is None else version_to_proto(s.version),
            )
            for s in result.sequences
        ],
        model=result.model,
        checkpoint_name=result.checkpoint_name,
    )
    if result.version is not None:
        message.version.CopyFrom(version_to_proto(result.version))
    return message


def sample_result_from_proto(message: sampling_pb2.SampleResult) -> SampleResult:
    finish = {v: k for k, v in _FINISH.items()}
    sequences = []
    for s in message.sequences:
        if s.finish_reason not in finish:
            raise InvalidArgument("SampleSequence.finish_reason is unspecified")
        sequences.append(
            SampleSequence(
                token_ids=tuple(s.token_ids),
                logprobs=tuple(s.logprobs),
                finish_reason=finish[s.finish_reason],
                cached_tokens=s.cached_tokens,
                version=version_from_proto(s.version) if s.HasField("version") else None,
            )
        )
    return SampleResult(
        sequences=tuple(sequences),
        model=message.model,
        checkpoint_name=message.checkpoint_name,
        version=version_from_proto(message.version) if message.HasField("version") else None,
    )


def sampler_config_to_proto(config: SamplerConfig) -> sampling_pb2.SamplerConfig:
    return sampling_pb2.SamplerConfig(
        checkpoint_name=config.checkpoint_name,
        max_seq_len=config.max_seq_len,
        max_concurrent_sequences=config.max_concurrent_sequences,
    )


def sampler_config_from_proto(message: sampling_pb2.SamplerConfig) -> SamplerConfig:
    return SamplerConfig(
        checkpoint_name=message.checkpoint_name,
        max_seq_len=message.max_seq_len,
        max_concurrent_sequences=message.max_concurrent_sequences,
    )


def sampling_usage_to_proto(usage: SamplingUsage) -> sampling_pb2.SamplingUsage:
    return sampling_pb2.SamplingUsage(
        num_prompt_tokens=usage.num_prompt_tokens,
        num_generated_tokens=usage.num_generated_tokens,
        num_cached_tokens=usage.num_cached_tokens,
    )


def sampling_usage_from_proto(message: sampling_pb2.SamplingUsage) -> SamplingUsage:
    return SamplingUsage(
        num_prompt_tokens=message.num_prompt_tokens,
        num_generated_tokens=message.num_generated_tokens,
        num_cached_tokens=message.num_cached_tokens,
    )


def sampler_status_to_proto(status: SamplerStatus) -> sampling_pb2.Sampler:
    message = sampling_pb2.Sampler(
        id=status.id,
        config=sampler_config_to_proto(status.config),
        state=status.state.name,
        model=status.model,
    )
    message.created_at.FromDatetime(status.created_at)
    if status.version is not None:
        message.version.CopyFrom(version_to_proto(status.version))
    if status.error is not None:
        message.error.CopyFrom(error_to_proto(status.error))
    if status.usage is not None:
        message.usage.CopyFrom(sampling_usage_to_proto(status.usage))
    return message


def sampler_status_from_proto(message: sampling_pb2.Sampler) -> SamplerStatus:
    return SamplerStatus(
        id=message.id,
        config=sampler_config_from_proto(_required(message, "config", "Sampler")),
        state=SamplerState(sampling_pb2.Sampler.State.Name(message.state).lower()),
        model=message.model,
        version=version_from_proto(message.version) if message.HasField("version") else None,
        created_at=message.created_at.ToDatetime(tzinfo=UTC),
        error=error_from_proto(message.error) if message.HasField("error") else None,
        usage=sampling_usage_from_proto(message.usage) if message.HasField("usage") else None,
    )
