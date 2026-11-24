import datetime

from google.protobuf import timestamp_pb2 as _timestamp_pb2
from looper_sdk.proto.v1 import errors_pb2 as _errors_pb2
from looper_sdk.proto.v1 import checkpoints_pb2 as _checkpoints_pb2
from looper_sdk.proto.v1 import values_pb2 as _values_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Submission(_message.Message):
    __slots__ = ("session_id", "writer_generation", "epoch", "key", "after")
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    WRITER_GENERATION_FIELD_NUMBER: _ClassVar[int]
    EPOCH_FIELD_NUMBER: _ClassVar[int]
    KEY_FIELD_NUMBER: _ClassVar[int]
    AFTER_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    writer_generation: int
    epoch: int
    key: str
    after: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, session_id: _Optional[str] = ..., writer_generation: _Optional[int] = ..., epoch: _Optional[int] = ..., key: _Optional[str] = ..., after: _Optional[_Iterable[str]] = ...) -> None: ...

class DatumResult(_message.Message):
    __slots__ = ("datum_id",)
    DATUM_ID_FIELD_NUMBER: _ClassVar[int]
    datum_id: str
    def __init__(self, datum_id: _Optional[str] = ...) -> None: ...

class ForwardBackwardResult(_message.Message):
    __slots__ = ("weighted_targets", "data", "num_tokens")
    WEIGHTED_TARGETS_FIELD_NUMBER: _ClassVar[int]
    DATA_FIELD_NUMBER: _ClassVar[int]
    NUM_TOKENS_FIELD_NUMBER: _ClassVar[int]
    weighted_targets: int
    data: _containers.RepeatedCompositeFieldContainer[DatumResult]
    num_tokens: int
    def __init__(self, weighted_targets: _Optional[int] = ..., data: _Optional[_Iterable[_Union[DatumResult, _Mapping]]] = ..., num_tokens: _Optional[int] = ...) -> None: ...

class StepResult(_message.Message):
    __slots__ = ("step", "version", "loss_sum", "denominator", "weighted_targets", "weight_sum", "metrics", "grad_norm", "clipped", "lr", "num_tokens")
    class MetricsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: float
        def __init__(self, key: _Optional[str] = ..., value: _Optional[float] = ...) -> None: ...
    STEP_FIELD_NUMBER: _ClassVar[int]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    LOSS_SUM_FIELD_NUMBER: _ClassVar[int]
    DENOMINATOR_FIELD_NUMBER: _ClassVar[int]
    WEIGHTED_TARGETS_FIELD_NUMBER: _ClassVar[int]
    WEIGHT_SUM_FIELD_NUMBER: _ClassVar[int]
    METRICS_FIELD_NUMBER: _ClassVar[int]
    GRAD_NORM_FIELD_NUMBER: _ClassVar[int]
    CLIPPED_FIELD_NUMBER: _ClassVar[int]
    LR_FIELD_NUMBER: _ClassVar[int]
    NUM_TOKENS_FIELD_NUMBER: _ClassVar[int]
    step: int
    version: _values_pb2.Version
    loss_sum: float
    denominator: float
    weighted_targets: int
    weight_sum: float
    metrics: _containers.ScalarMap[str, float]
    grad_norm: float
    clipped: bool
    lr: float
    num_tokens: int
    def __init__(self, step: _Optional[int] = ..., version: _Optional[_Union[_values_pb2.Version, _Mapping]] = ..., loss_sum: _Optional[float] = ..., denominator: _Optional[float] = ..., weighted_targets: _Optional[int] = ..., weight_sum: _Optional[float] = ..., metrics: _Optional[_Mapping[str, float]] = ..., grad_norm: _Optional[float] = ..., clipped: _Optional[bool] = ..., lr: _Optional[float] = ..., num_tokens: _Optional[int] = ...) -> None: ...

class PublishResult(_message.Message):
    __slots__ = ("version", "sampler_id")
    VERSION_FIELD_NUMBER: _ClassVar[int]
    SAMPLER_ID_FIELD_NUMBER: _ClassVar[int]
    version: _values_pb2.Version
    sampler_id: str
    def __init__(self, version: _Optional[_Union[_values_pb2.Version, _Mapping]] = ..., sampler_id: _Optional[str] = ...) -> None: ...

class Operation(_message.Message):
    __slots__ = ("id", "run_id", "seq", "key", "epoch", "kind", "state", "queue_state", "forward_backward", "optim_step", "save_checkpoint", "publish_weights", "save_weights", "error", "durable_step", "rolled_back_by", "accepted_at", "started_at", "finished_at", "raw_data_expired_at")
    class Kind(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        KIND_UNSPECIFIED: _ClassVar[Operation.Kind]
        FORWARD_BACKWARD: _ClassVar[Operation.Kind]
        OPTIM_STEP: _ClassVar[Operation.Kind]
        SAVE_CHECKPOINT: _ClassVar[Operation.Kind]
        PUBLISH_WEIGHTS: _ClassVar[Operation.Kind]
        SAVE_WEIGHTS: _ClassVar[Operation.Kind]
    KIND_UNSPECIFIED: Operation.Kind
    FORWARD_BACKWARD: Operation.Kind
    OPTIM_STEP: Operation.Kind
    SAVE_CHECKPOINT: Operation.Kind
    PUBLISH_WEIGHTS: Operation.Kind
    SAVE_WEIGHTS: Operation.Kind
    class State(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        STATE_UNSPECIFIED: _ClassVar[Operation.State]
        QUEUED: _ClassVar[Operation.State]
        RUNNING: _ClassVar[Operation.State]
        OUTCOME_UNKNOWN: _ClassVar[Operation.State]
        SUCCEEDED: _ClassVar[Operation.State]
        FAILED: _ClassVar[Operation.State]
        CANCELLED: _ClassVar[Operation.State]
    STATE_UNSPECIFIED: Operation.State
    QUEUED: Operation.State
    RUNNING: Operation.State
    OUTCOME_UNKNOWN: Operation.State
    SUCCEEDED: Operation.State
    FAILED: Operation.State
    CANCELLED: Operation.State
    class QueueState(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        QUEUE_STATE_UNSPECIFIED: _ClassVar[Operation.QueueState]
        WAITING_FOR_GPUS: _ClassVar[Operation.QueueState]
        STARTING: _ClassVar[Operation.QueueState]
        RESUMING: _ClassVar[Operation.QueueState]
        RECOVERING: _ClassVar[Operation.QueueState]
        SPEND_BLOCKED: _ClassVar[Operation.QueueState]
        WAITING: _ClassVar[Operation.QueueState]
    QUEUE_STATE_UNSPECIFIED: Operation.QueueState
    WAITING_FOR_GPUS: Operation.QueueState
    STARTING: Operation.QueueState
    RESUMING: Operation.QueueState
    RECOVERING: Operation.QueueState
    SPEND_BLOCKED: Operation.QueueState
    WAITING: Operation.QueueState
    ID_FIELD_NUMBER: _ClassVar[int]
    RUN_ID_FIELD_NUMBER: _ClassVar[int]
    SEQ_FIELD_NUMBER: _ClassVar[int]
    KEY_FIELD_NUMBER: _ClassVar[int]
    EPOCH_FIELD_NUMBER: _ClassVar[int]
    KIND_FIELD_NUMBER: _ClassVar[int]
    STATE_FIELD_NUMBER: _ClassVar[int]
    QUEUE_STATE_FIELD_NUMBER: _ClassVar[int]
    FORWARD_BACKWARD_FIELD_NUMBER: _ClassVar[int]
    OPTIM_STEP_FIELD_NUMBER: _ClassVar[int]
    SAVE_CHECKPOINT_FIELD_NUMBER: _ClassVar[int]
    PUBLISH_WEIGHTS_FIELD_NUMBER: _ClassVar[int]
    SAVE_WEIGHTS_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    DURABLE_STEP_FIELD_NUMBER: _ClassVar[int]
    ROLLED_BACK_BY_FIELD_NUMBER: _ClassVar[int]
    ACCEPTED_AT_FIELD_NUMBER: _ClassVar[int]
    STARTED_AT_FIELD_NUMBER: _ClassVar[int]
    FINISHED_AT_FIELD_NUMBER: _ClassVar[int]
    RAW_DATA_EXPIRED_AT_FIELD_NUMBER: _ClassVar[int]
    id: str
    run_id: str
    seq: int
    key: str
    epoch: int
    kind: Operation.Kind
    state: Operation.State
    queue_state: Operation.QueueState
    forward_backward: ForwardBackwardResult
    optim_step: StepResult
    save_checkpoint: _checkpoints_pb2.Checkpoint
    publish_weights: PublishResult
    save_weights: _checkpoints_pb2.Checkpoint
    error: _errors_pb2.Error
    durable_step: int
    rolled_back_by: str
    accepted_at: _timestamp_pb2.Timestamp
    started_at: _timestamp_pb2.Timestamp
    finished_at: _timestamp_pb2.Timestamp
    raw_data_expired_at: _timestamp_pb2.Timestamp
    def __init__(self, id: _Optional[str] = ..., run_id: _Optional[str] = ..., seq: _Optional[int] = ..., key: _Optional[str] = ..., epoch: _Optional[int] = ..., kind: _Optional[_Union[Operation.Kind, str]] = ..., state: _Optional[_Union[Operation.State, str]] = ..., queue_state: _Optional[_Union[Operation.QueueState, str]] = ..., forward_backward: _Optional[_Union[ForwardBackwardResult, _Mapping]] = ..., optim_step: _Optional[_Union[StepResult, _Mapping]] = ..., save_checkpoint: _Optional[_Union[_checkpoints_pb2.Checkpoint, _Mapping]] = ..., publish_weights: _Optional[_Union[PublishResult, _Mapping]] = ..., save_weights: _Optional[_Union[_checkpoints_pb2.Checkpoint, _Mapping]] = ..., error: _Optional[_Union[_errors_pb2.Error, _Mapping]] = ..., durable_step: _Optional[int] = ..., rolled_back_by: _Optional[str] = ..., accepted_at: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ..., started_at: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ..., finished_at: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ..., raw_data_expired_at: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ...) -> None: ...

class GetOperationRequest(_message.Message):
    __slots__ = ("operation_id",)
    OPERATION_ID_FIELD_NUMBER: _ClassVar[int]
    operation_id: str
    def __init__(self, operation_id: _Optional[str] = ...) -> None: ...

class GetOperationResponse(_message.Message):
    __slots__ = ("operation",)
    OPERATION_FIELD_NUMBER: _ClassVar[int]
    operation: Operation
    def __init__(self, operation: _Optional[_Union[Operation, _Mapping]] = ...) -> None: ...

class AwaitOperationRequest(_message.Message):
    __slots__ = ("operation_id", "timeout_s")
    OPERATION_ID_FIELD_NUMBER: _ClassVar[int]
    TIMEOUT_S_FIELD_NUMBER: _ClassVar[int]
    operation_id: str
    timeout_s: float
    def __init__(self, operation_id: _Optional[str] = ..., timeout_s: _Optional[float] = ...) -> None: ...

class AwaitOperationResponse(_message.Message):
    __slots__ = ("operation",)
    OPERATION_FIELD_NUMBER: _ClassVar[int]
    operation: Operation
    def __init__(self, operation: _Optional[_Union[Operation, _Mapping]] = ...) -> None: ...

class FindOperationRequest(_message.Message):
    __slots__ = ("run_id", "key", "epoch")
    RUN_ID_FIELD_NUMBER: _ClassVar[int]
    KEY_FIELD_NUMBER: _ClassVar[int]
    EPOCH_FIELD_NUMBER: _ClassVar[int]
    run_id: str
    key: str
    epoch: int
    def __init__(self, run_id: _Optional[str] = ..., key: _Optional[str] = ..., epoch: _Optional[int] = ...) -> None: ...

class FindOperationResponse(_message.Message):
    __slots__ = ("operation",)
    OPERATION_FIELD_NUMBER: _ClassVar[int]
    operation: Operation
    def __init__(self, operation: _Optional[_Union[Operation, _Mapping]] = ...) -> None: ...
