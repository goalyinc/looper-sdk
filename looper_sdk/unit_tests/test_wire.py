"""The values travel as looper.v1's messages and come back equal, through
binary protobuf and through protobuf's JSON mapping; decoding refuses what
the wire requires but a message lacks."""

from __future__ import annotations

import dataclasses
import re
from typing import Any

import pytest
from google.protobuf import json_format

from looper_sdk import (
    AdamW,
    CapacityExhausted,
    Category,
    CheckpointNameTaken,
    CrossEntropy,
    Dapo,
    Datum,
    DatumResult,
    ErrorCode,
    ForwardBackwardResult,
    Full,
    Grpo,
    InvalidArgument,
    LooperError,
    Lora,
    Muon,
    PublishResult,
    RunRolledBack,
    StepResult,
    TrainingConfig,
    Version,
    wire,
)
from looper_sdk.errors import ERRORS
from looper_sdk.proto.v1 import errors_pb2, losses_pb2, operations_pb2, values_pb2


def through_both_encodings(message):
    """The message after a binary and a JSON round trip, checked equal."""
    as_binary = type(message).FromString(message.SerializeToString())
    as_json = json_format.Parse(json_format.MessageToJson(message), type(message)())
    assert as_binary == message == as_json
    return as_json


VALUES = [
    (Version(run_id="run-1", epoch=2, step=30), wire.version_to_proto, wire.version_from_proto),
    (
        TrainingConfig(
            model="Qwen/Qwen3.5-9B",
            kind=Full(),
            tokens_per_step=131_072,
            max_seq_len=4_096,
            sampled_tokens_per_step=524_288,
        ),
        wire.training_config_to_proto,
        wire.training_config_from_proto,
    ),
    (
        TrainingConfig(
            model="Qwen/Qwen3.5-9B",
            kind=Lora(rank=8, train_unembed=False),
            tokens_per_step=8_192,
            max_seq_len=8_192,
            optimizer=AdamW(learning_rate=5e-4, max_grad_norm=1.0),
        ),
        wire.training_config_to_proto,
        wire.training_config_from_proto,
    ),
    # Every setting that may be 0, at 0.
    (
        AdamW(
            learning_rate=1e-5,
            betas=(0.0, 0.0),
            weight_decay=0.0,
        ),
        wire.optimizer_to_proto,
        wire.optimizer_from_proto,
    ),
    (
        AdamW(
            learning_rate=2e-5,
            betas=(0.9, 0.95),
            eps=1e-12,
            weight_decay=0.1,
        ),
        wire.optimizer_to_proto,
        wire.optimizer_from_proto,
    ),
    (Muon(learning_rate=3e-4), wire.optimizer_to_proto, wire.optimizer_from_proto),
    # Every Muon setting that may be 0 or False, so.
    (
        Muon(
            learning_rate=2e-4,
            weight_decay=0.0,
            momentum=0.0,
            nesterov=False,
            ns_steps=1,
            betas=(0.0, 0.0),
            max_grad_norm=0.5,
        ),
        wire.optimizer_to_proto,
        wire.optimizer_from_proto,
    ),
    (Datum(tokens=[5, 6], weights=[0, 1]), wire.datum_to_proto, wire.datum_from_proto),
    (
        Datum(
            tokens=[1, 2, 248_000],
            weights=[0, 1, 0],
            columns={"logprobs": [0, -0.25, -2]},
            datum_id="d-1",
            reward=-0.5,
            group_id="prompt-7",
            group_size=8,
        ),
        wire.datum_to_proto,
        wire.datum_from_proto,
    ),
    # A reward of 0 is given, not absent.
    (
        Datum(tokens=[5, 6], weights=[0, 1], reward=0, group_id="g", group_size=1),
        wire.datum_to_proto,
        wire.datum_from_proto,
    ),
    (CrossEntropy(), wire.loss_to_proto, wire.loss_from_proto),
    (Grpo(), wire.loss_to_proto, wire.loss_from_proto),
    (Grpo(clip_low=0.1, clip_high=0.28), wire.loss_to_proto, wire.loss_from_proto),
    (Grpo(kl_weight=0.05), wire.loss_to_proto, wire.loss_from_proto),
    (Dapo(), wire.loss_to_proto, wire.loss_from_proto),
    (Dapo(kl_weight=0.1), wire.loss_to_proto, wire.loss_from_proto),
    (Grpo(scale_by_std=False), wire.loss_to_proto, wire.loss_from_proto),
    # A Dapo keeps its own arm, even with Grpo's settings.
    (Dapo(clip_high=0.2), wire.loss_to_proto, wire.loss_from_proto),
    (
        ForwardBackwardResult(
            weighted_targets=8,
            data=[DatumResult(datum_id=None), DatumResult(datum_id="d-2")],
            num_tokens=12,
        ),
        wire.forward_backward_result_to_proto,
        wire.forward_backward_result_from_proto,
    ),
    # A result recorded before Looper counted tokens has none, not 0.
    (
        ForwardBackwardResult(weighted_targets=8, data=[], num_tokens=None),
        wire.forward_backward_result_to_proto,
        wire.forward_backward_result_from_proto,
    ),
    (
        ForwardBackwardResult(weighted_targets=0, data=[], num_tokens=0),
        wire.forward_backward_result_to_proto,
        wire.forward_backward_result_from_proto,
    ),
    (
        StepResult(
            step=31,
            version=Version(run_id="run-1", epoch=2, step=31),
            loss_sum=12.5,
            denominator=8.0,
            weighted_targets=8,
            weight_sum=8.0,
            metrics={"mean_ratio": 1.0},
            grad_norm=0.75,
            clipped=False,
            lr=1e-5,
            num_tokens=24,
        ),
        wire.step_result_to_proto,
        wire.step_result_from_proto,
    ),
    (
        StepResult(
            step=1,
            version=Version(run_id="run-1", epoch=0, step=1),
            loss_sum=1.0,
            denominator=1.0,
            weighted_targets=1,
            weight_sum=1.0,
            metrics={},
            grad_norm=0.5,
            clipped=False,
            lr=1e-5,
            num_tokens=None,
        ),
        wire.step_result_to_proto,
        wire.step_result_from_proto,
    ),
    (
        PublishResult(version=Version(run_id="run-1", epoch=0, step=3), sampler_id="s-1"),
        wire.publish_result_to_proto,
        wire.publish_result_from_proto,
    ),
]


@pytest.mark.parametrize("value, encode, decode", VALUES, ids=lambda v: type(v).__name__)
def test_a_value_comes_back_equal(value, encode, decode) -> None:
    assert decode(through_both_encodings(encode(value))) == value


def test_a_datum_in_json_names_its_columns() -> None:
    datum = Datum(
        tokens=[1, 2],
        weights=[0, 1],
        columns={"logprobs": [0, -1]},
        datum_id="d",
        reward=0.5,
        group_id="g",
        group_size=4,
    )
    assert json_format.MessageToDict(wire.datum_to_proto(datum)) == {
        "tokens": [1, 2],
        "weights": [0, 1],
        "columns": {"logprobs": {"values": [0, -1]}},
        "datumId": "d",
        "reward": 0.5,
        "groupId": "g",
        "groupSize": 4,
    }


def ADAMW(**fields):  # noqa: N802
    """looper.v1's AdamW with every setting, but for ``fields``."""
    settings = dict(learning_rate=1e-5, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0)
    return values_pb2.Optimizer.AdamW(**{**settings, **fields})


def MUON(**fields):  # noqa: N802
    """looper.v1's Muon with every setting, but for ``fields``."""
    settings = dict(
        learning_rate=3e-4,
        weight_decay=0.1,
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
        beta1=0.9,
        beta2=0.95,
        eps=1e-8,
    )
    return values_pb2.Optimizer.Muon(**{**settings, **fields})


@pytest.mark.parametrize(
    "message, decode, says",
    [
        (values_pb2.TrainingConfig(model="m", tokens_per_step=2, max_seq_len=1),
         wire.training_config_from_proto, "TrainingConfig.kind is required"),
        (values_pb2.TrainingConfig(
            model="m", tokens_per_step=2, max_seq_len=1, lora=values_pb2.TrainingConfig.Lora(rank=8)
        ), wire.training_config_from_proto, "TrainingConfig.lora.train_attn is required"),
        (losses_pb2.Loss(), wire.loss_from_proto, "Loss.loss is required"),
        (losses_pb2.Loss(grpo=losses_pb2.Loss.Grpo(clip_high=0.2)), wire.loss_from_proto,
         "Loss.grpo.clip_low is required"),
        (losses_pb2.Loss(dapo=losses_pb2.Loss.Grpo(clip_low=0.2)), wire.loss_from_proto,
         "Loss.dapo.clip_high is required"),
        (losses_pb2.Loss(grpo=losses_pb2.Loss.Grpo(clip_low=0, clip_high=0.2)),
         wire.loss_from_proto, "Grpo.clip_low must be in (0, 1)"),
        (values_pb2.Datum(tokens=[1, 2], weights=[0]), wire.datum_from_proto,
         "has 1 values for 2 tokens"),
        (values_pb2.Datum(tokens=[1, 2]), wire.datum_from_proto, "has 0 values for 2 tokens"),
        (values_pb2.Datum(tokens=[1, 2], weights=[0, 1], reward=1, group_id="g"),
         wire.datum_from_proto, "reward, group_id and group_size come together"),
        (values_pb2.Optimizer(), wire.optimizer_from_proto, "Optimizer.kind is required"),
        (values_pb2.Optimizer(adamw=ADAMW(beta1=None)), wire.optimizer_from_proto,
         "Optimizer.adamw.beta1 is required"),
        (values_pb2.Optimizer(adamw=ADAMW(weight_decay=None)), wire.optimizer_from_proto,
         "Optimizer.adamw.weight_decay is required"),
        (values_pb2.Optimizer(adamw=ADAMW(learning_rate=0)), wire.optimizer_from_proto,
         "AdamW.learning_rate must be above 0"),
        (values_pb2.Optimizer(muon=MUON(momentum=None)), wire.optimizer_from_proto,
         "Optimizer.muon.momentum is required"),
        (values_pb2.Optimizer(muon=MUON(nesterov=None)), wire.optimizer_from_proto,
         "Optimizer.muon.nesterov is required"),
        (values_pb2.Optimizer(muon=MUON(weight_decay=None)), wire.optimizer_from_proto,
         "Optimizer.muon.weight_decay is required"),
        (values_pb2.Optimizer(muon=MUON(beta2=None)), wire.optimizer_from_proto,
         "Optimizer.muon.beta2 is required"),
        (values_pb2.Optimizer(muon=MUON(ns_steps=0)), wire.optimizer_from_proto,
         "Muon.ns_steps must be at least 1"),
        (operations_pb2.PublishResult(sampler_id="s-1"), wire.publish_result_from_proto,
         "PublishResult.version is required"),
    ],
)  # fmt: skip
def test_decoding_refuses_what_the_wire_requires(message, decode, says) -> None:
    with pytest.raises(InvalidArgument, match=re.escape(says)):
        decode(message)


def test_a_loss_sent_before_kl_weight_and_scale_by_std_decodes_as_it_was() -> None:
    earlier = losses_pb2.Loss(dapo=losses_pb2.Loss.Grpo(clip_low=0.2, clip_high=0.28))
    assert wire.loss_from_proto(earlier) == Dapo(kl_weight=0.0, scale_by_std=True)


@pytest.mark.parametrize("code", list(ErrorCode))
def test_every_registered_error_comes_back_as_its_class(code) -> None:
    if code is ErrorCode.RUN_ROLLED_BACK:
        error = RunRolledBack(
            "restored",
            epoch=3,
            restored_step=40,
            restored_seq=120,
            first_lost_seq=121,
            last_lost_seq=125,
        )
    else:
        error = ERRORS[code](f"{code} happened")
    decoded = wire.error_from_proto(through_both_encodings(wire.error_to_proto(error)))
    assert type(decoded) is ERRORS[code]
    assert vars(decoded) == vars(error)


def test_an_error_keeps_the_services_retry_semantics() -> None:
    error = CapacityExhausted("too many runs", retryable=True, retry_after_s=2.5)
    message = wire.error_to_proto(error)
    assert json_format.MessageToDict(message) == {
        "code": "CAPACITY_EXHAUSTED",
        "category": "SERVER",
        "retryable": True,
        "message": "too many runs",
        "retryAfterS": 2.5,
    }
    decoded = wire.error_from_proto(message)
    assert (decoded.retryable, decoded.category, decoded.retry_after_s) == (
        True,
        Category.SERVER,
        2.5,
    )


def test_an_unknown_code_arrives_as_a_plain_looper_error() -> None:
    message = errors_pb2.Error(code=99, category=errors_pb2.Error.USER, message="new")  # pyright: ignore[reportArgumentType]
    decoded = wire.error_from_proto(message)
    assert type(decoded) is LooperError
    assert decoded.message == "code 99: new"
    assert decoded.category is Category.USER


def test_a_rollback_without_its_details_arrives_as_a_plain_looper_error() -> None:
    message = errors_pb2.Error(code=errors_pb2.Error.RUN_ROLLED_BACK, message="restored")
    assert type(wire.error_from_proto(message)) is LooperError


def test_every_class_defaults_to_its_registered_semantics() -> None:
    retryable = {code for code, cls in ERRORS.items() if cls.default_retryable}
    assert retryable == {
        ErrorCode.CAPACITY_EXHAUSTED,
        ErrorCode.RUN_NOT_READY,
        ErrorCode.UNAVAILABLE,
        ErrorCode.INTERNAL,
    }
    assert set(ERRORS) == set(ErrorCode)
    assert [name for name in errors_pb2.Error.Code.keys() if name != "CODE_UNSPECIFIED"] == [
        code.value for code in ErrorCode
    ]


def test_checkpoint_name_taken_is_registered_as_28_and_never_retried() -> None:
    error = CheckpointNameTaken("'foo-hf' is held by an export of another checkpoint")
    assert ERRORS[ErrorCode.CHECKPOINT_NAME_TAKEN] is CheckpointNameTaken
    assert list(ErrorCode)[-1] is ErrorCode.CHECKPOINT_NAME_TAKEN
    assert (error.retryable, error.category) == (False, Category.USER)
    message = wire.error_to_proto(error)
    assert message.code == errors_pb2.Error.CHECKPOINT_NAME_TAKEN == 28
    decoded = wire.error_from_proto(through_both_encodings(message))
    assert type(decoded) is CheckpointNameTaken
    assert (decoded.message, decoded.retryable) == (error.message, False)


def checkpoint(**fields: Any):
    from datetime import UTC, datetime

    from looper_sdk import CheckpointState, ModelSpec, StateCheckpoint

    run = "11111111-1111-1111-1111-111111111111"
    defaults: dict[str, Any] = dict(
        name="latest",
        expires_at=None,
        id="22222222-2222-2222-2222-222222222222",
        operation_id="33333333-3333-3333-3333-333333333333",
        project="research",
        principal="alice",
        source_run_id=run,
        version=Version(run_id=run, epoch=2, step=100),
        model_spec=ModelSpec(model="Qwen/Qwen3.5-9B", kind=Full()),
        size_bytes=20,
        created_at=datetime(2026, 10, 27, tzinfo=UTC),
        state=CheckpointState.READY_TO_USE,
    )
    return StateCheckpoint(**(defaults | fields))


@pytest.mark.parametrize("with_expiry", [False, True])
@pytest.mark.parametrize("kind", [Full(), Lora(rank=7, train_mlp=False, train_unembed=False)])
def test_named_checkpoint_wire_round_trip(with_expiry, kind):
    from datetime import UTC, datetime, timedelta

    from looper_sdk import ModelSpec

    created = datetime(2026, 10, 27, tzinfo=UTC)
    value = checkpoint(
        expires_at=created + timedelta(seconds=60) if with_expiry else None,
        model_spec=ModelSpec(model="Qwen/Qwen3.5-9B", kind=kind),
    )
    assert wire.checkpoint_from_proto(wire.checkpoint_to_proto(value)) == value


@pytest.mark.parametrize(
    "fields",
    [
        {"state": "uploading", "size_bytes": 0},
        {"state": "failed", "size_bytes": 0, "failure": "the run's GPUs were lost"},
    ],
)
def test_a_checkpoint_in_every_stage_crosses_the_wire(fields):
    from looper_sdk import CheckpointState

    value = checkpoint(**(fields | {"state": CheckpointState(fields["state"])}))
    assert wire.checkpoint_from_proto(wire.checkpoint_to_proto(value)) == value


def test_a_checkpoint_with_no_operation_crosses_the_wire_as_an_empty_string():
    value = checkpoint(operation_id=None)
    message = wire.checkpoint_to_proto(value)
    assert message.operation_id == ""
    assert wire.checkpoint_from_proto(through_both_encodings(message)) == value


def test_a_checkpoint_from_a_server_older_than_the_stages_is_ready_to_use():
    from looper_sdk import CheckpointState

    message = wire.checkpoint_to_proto(checkpoint())
    message.ClearField("state")

    assert wire.checkpoint_from_proto(message).state is CheckpointState.READY_TO_USE


def test_a_weights_checkpoint_crosses_the_wire_as_one():
    from looper_sdk import StateCheckpoint, WeightsCheckpoint

    state = checkpoint()
    value = WeightsCheckpoint(**{f.name: getattr(state, f.name) for f in dataclasses.fields(state)})
    answered = wire.checkpoint_from_proto(wire.checkpoint_to_proto(value))

    assert answered == value
    assert not isinstance(answered, StateCheckpoint)


def test_a_checkpoint_from_a_server_older_than_the_kinds_is_a_state_checkpoint():
    from looper_sdk import StateCheckpoint

    message = wire.checkpoint_to_proto(checkpoint())
    message.ClearField("kind")

    assert type(wire.checkpoint_from_proto(message)) is StateCheckpoint


def test_a_checkpoint_kind_the_sdk_does_not_know_is_refused():
    message = wire.checkpoint_to_proto(checkpoint())
    setattr(message, "kind", 99)  # a value no enum member names

    with pytest.raises(InvalidArgument, match="kind 99"):
        wire.checkpoint_from_proto(message)


def test_a_checkpoint_state_the_sdk_does_not_know_is_refused():
    message = wire.checkpoint_to_proto(checkpoint())
    setattr(message, "state", 99)  # a value no enum member names

    with pytest.raises(InvalidArgument, match="state 99"):
        wire.checkpoint_from_proto(message)


@pytest.mark.parametrize(
    "fields",
    [
        {"state": "failed"},
        {"state": "ready_to_use", "failure": "lost"},
        {"state": "ready"},
    ],
)
def test_a_checkpoint_states_its_failure_exactly_when_it_failed(fields):
    with pytest.raises(InvalidArgument):
        checkpoint(**fields)


@pytest.mark.parametrize("kind", [Full(), Lora(rank=8, train_mlp=False, train_unembed=False)])
def test_model_spec_wire_preserves_trainable_parameters(kind):
    from looper_sdk import ModelSpec

    spec = ModelSpec(model="Qwen/Qwen3.5-9B", kind=kind)
    assert wire.model_spec_from_proto(wire.model_spec_to_proto(spec)) == spec


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"model": "Qwen/Qwen3.5-9B"},
        {
            "model": "Qwen/Qwen3.5-9B",
            "lora": {"rank": 0, "train_attn": True, "train_mlp": False, "train_unembed": False},
        },
        {"model": "Qwen/Qwen3.5-9B", "lora": {"rank": 8}},
    ],
)
def test_model_spec_wire_refuses_incomplete_or_invalid_identity(payload):
    from google.protobuf.json_format import ParseDict

    from looper_sdk.proto.v1 import values_pb2

    with pytest.raises(InvalidArgument):
        wire.model_spec_from_proto(ParseDict(payload, values_pb2.ModelSpec()))


def test_checkpoint_wire_requires_model_spec():
    from datetime import UTC, datetime

    from looper_sdk.proto.v1 import checkpoints_pb2

    message = checkpoints_pb2.Checkpoint()
    message.version.CopyFrom(wire.version_to_proto(Version(run_id="run", epoch=0, step=1)))
    message.created_at.FromDatetime(datetime.now(UTC))
    with pytest.raises(InvalidArgument, match="model_spec"):
        wire.checkpoint_from_proto(message)


def test_checkpoint_binary_response_is_readable_without_model_spec_field():
    from google.protobuf import descriptor_pb2, descriptor_pool, json_format, message_factory

    from looper_sdk import ModelSpec
    from looper_sdk.proto.v1 import checkpoints_pb2

    pool = descriptor_pool.DescriptorPool()

    def add_dependencies(descriptor):
        for dependency in descriptor.dependencies:
            add_dependencies(dependency)
        try:
            pool.FindFileByName(descriptor.name)
        except KeyError:
            pool.AddSerializedFile(descriptor.serialized_pb)

    for dependency in checkpoints_pb2.DESCRIPTOR.dependencies:
        add_dependencies(dependency)
    old_file = descriptor_pb2.FileDescriptorProto.FromString(
        checkpoints_pb2.DESCRIPTOR.serialized_pb
    )
    checkpoint = old_file.message_type[0]
    kept = [field for field in checkpoint.field if field.number != 11]
    del checkpoint.field[:]
    checkpoint.field.extend(kept)
    pool.AddSerializedFile(old_file.SerializeToString())
    old_class = message_factory.GetMessageClass(pool.FindMessageTypeByName("looper.v1.Checkpoint"))
    current = checkpoints_pb2.Checkpoint(
        id="saved-id",
        name="latest",
        source_run_id="run-id",
        model_spec=wire.model_spec_to_proto(ModelSpec(model="Qwen/Qwen3.5-9B", kind=Full())),
    )
    old = old_class.FromString(current.SerializeToString())
    assert json_format.MessageToDict(old, preserving_proto_field_name=True) == {
        "id": "saved-id",
        "name": "latest",
        "source_run_id": "run-id",
    }
