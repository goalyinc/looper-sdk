import datetime

from google.protobuf import timestamp_pb2 as _timestamp_pb2
from looper_sdk.proto.v1 import losses_pb2 as _losses_pb2
from looper_sdk.proto.v1 import operations_pb2 as _operations_pb2
from looper_sdk.proto.v1 import values_pb2 as _values_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Run(_message.Message):
    __slots__ = ("id", "project", "config", "state", "epoch", "writer_session_id", "writer_generation", "step", "durable_step", "created_at", "waiting_reason")
    class State(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        STATE_UNSPECIFIED: _ClassVar[Run.State]
        QUEUED: _ClassVar[Run.State]
        STARTING: _ClassVar[Run.State]
        ACTIVE: _ClassVar[Run.State]
        RECOVERING: _ClassVar[Run.State]
        SUSPENDED: _ClassVar[Run.State]
        RESUMING: _ClassVar[Run.State]
        FAILED: _ClassVar[Run.State]
        DELETED: _ClassVar[Run.State]
    STATE_UNSPECIFIED: Run.State
    QUEUED: Run.State
    STARTING: Run.State
    ACTIVE: Run.State
    RECOVERING: Run.State
    SUSPENDED: Run.State
    RESUMING: Run.State
    FAILED: Run.State
    DELETED: Run.State
    ID_FIELD_NUMBER: _ClassVar[int]
    PROJECT_FIELD_NUMBER: _ClassVar[int]
    CONFIG_FIELD_NUMBER: _ClassVar[int]
    STATE_FIELD_NUMBER: _ClassVar[int]
    EPOCH_FIELD_NUMBER: _ClassVar[int]
    WRITER_SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    WRITER_GENERATION_FIELD_NUMBER: _ClassVar[int]
    STEP_FIELD_NUMBER: _ClassVar[int]
    DURABLE_STEP_FIELD_NUMBER: _ClassVar[int]
    CREATED_AT_FIELD_NUMBER: _ClassVar[int]
    WAITING_REASON_FIELD_NUMBER: _ClassVar[int]
    id: str
    project: str
    config: _values_pb2.TrainingConfig
    state: Run.State
    epoch: int
    writer_session_id: str
    writer_generation: int
    step: int
    durable_step: int
    created_at: _timestamp_pb2.Timestamp
    waiting_reason: _values_pb2.WaitingReason
    def __init__(self, id: _Optional[str] = ..., project: _Optional[str] = ..., config: _Optional[_Union[_values_pb2.TrainingConfig, _Mapping]] = ..., state: _Optional[_Union[Run.State, str]] = ..., epoch: _Optional[int] = ..., writer_session_id: _Optional[str] = ..., writer_generation: _Optional[int] = ..., step: _Optional[int] = ..., durable_step: _Optional[int] = ..., created_at: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ..., waiting_reason: _Optional[_Union[_values_pb2.WaitingReason, str]] = ...) -> None: ...

class CreateRunRequest(_message.Message):
    __slots__ = ("session_id", "key", "config")
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    KEY_FIELD_NUMBER: _ClassVar[int]
    CONFIG_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    key: str
    config: _values_pb2.TrainingConfig
    def __init__(self, session_id: _Optional[str] = ..., key: _Optional[str] = ..., config: _Optional[_Union[_values_pb2.TrainingConfig, _Mapping]] = ...) -> None: ...

class CreateRunResponse(_message.Message):
    __slots__ = ("run",)
    RUN_FIELD_NUMBER: _ClassVar[int]
    run: Run
    def __init__(self, run: _Optional[_Union[Run, _Mapping]] = ...) -> None: ...

class GetRunRequest(_message.Message):
    __slots__ = ("run_id",)
    RUN_ID_FIELD_NUMBER: _ClassVar[int]
    run_id: str
    def __init__(self, run_id: _Optional[str] = ...) -> None: ...

class GetRunResponse(_message.Message):
    __slots__ = ("run",)
    RUN_FIELD_NUMBER: _ClassVar[int]
    run: Run
    def __init__(self, run: _Optional[_Union[Run, _Mapping]] = ...) -> None: ...

class DeleteRunRequest(_message.Message):
    __slots__ = ("run_id",)
    RUN_ID_FIELD_NUMBER: _ClassVar[int]
    run_id: str
    def __init__(self, run_id: _Optional[str] = ...) -> None: ...

class DeleteRunResponse(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class ForwardBackwardRequest(_message.Message):
    __slots__ = ("run_id", "submission", "data", "loss")
    RUN_ID_FIELD_NUMBER: _ClassVar[int]
    SUBMISSION_FIELD_NUMBER: _ClassVar[int]
    DATA_FIELD_NUMBER: _ClassVar[int]
    LOSS_FIELD_NUMBER: _ClassVar[int]
    run_id: str
    submission: _operations_pb2.Submission
    data: _containers.RepeatedCompositeFieldContainer[_values_pb2.Datum]
    loss: _losses_pb2.Loss
    def __init__(self, run_id: _Optional[str] = ..., submission: _Optional[_Union[_operations_pb2.Submission, _Mapping]] = ..., data: _Optional[_Iterable[_Union[_values_pb2.Datum, _Mapping]]] = ..., loss: _Optional[_Union[_losses_pb2.Loss, _Mapping]] = ...) -> None: ...

class ForwardBackwardResponse(_message.Message):
    __slots__ = ("operation",)
    OPERATION_FIELD_NUMBER: _ClassVar[int]
    operation: _operations_pb2.Operation
    def __init__(self, operation: _Optional[_Union[_operations_pb2.Operation, _Mapping]] = ...) -> None: ...

class OptimStepRequest(_message.Message):
    __slots__ = ("run_id", "submission", "learning_rate")
    RUN_ID_FIELD_NUMBER: _ClassVar[int]
    SUBMISSION_FIELD_NUMBER: _ClassVar[int]
    LEARNING_RATE_FIELD_NUMBER: _ClassVar[int]
    run_id: str
    submission: _operations_pb2.Submission
    learning_rate: float
    def __init__(self, run_id: _Optional[str] = ..., submission: _Optional[_Union[_operations_pb2.Submission, _Mapping]] = ..., learning_rate: _Optional[float] = ...) -> None: ...

class OptimStepResponse(_message.Message):
    __slots__ = ("operation",)
    OPERATION_FIELD_NUMBER: _ClassVar[int]
    operation: _operations_pb2.Operation
    def __init__(self, operation: _Optional[_Union[_operations_pb2.Operation, _Mapping]] = ...) -> None: ...

class SaveCheckpointRequest(_message.Message):
    __slots__ = ("run_id", "submission", "name", "ttl_s")
    RUN_ID_FIELD_NUMBER: _ClassVar[int]
    SUBMISSION_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    TTL_S_FIELD_NUMBER: _ClassVar[int]
    run_id: str
    submission: _operations_pb2.Submission
    name: str
    ttl_s: int
    def __init__(self, run_id: _Optional[str] = ..., submission: _Optional[_Union[_operations_pb2.Submission, _Mapping]] = ..., name: _Optional[str] = ..., ttl_s: _Optional[int] = ...) -> None: ...

class SaveCheckpointResponse(_message.Message):
    __slots__ = ("operation",)
    OPERATION_FIELD_NUMBER: _ClassVar[int]
    operation: _operations_pb2.Operation
    def __init__(self, operation: _Optional[_Union[_operations_pb2.Operation, _Mapping]] = ...) -> None: ...

class PublishWeightsRequest(_message.Message):
    __slots__ = ("run_id", "submission")
    RUN_ID_FIELD_NUMBER: _ClassVar[int]
    SUBMISSION_FIELD_NUMBER: _ClassVar[int]
    run_id: str
    submission: _operations_pb2.Submission
    def __init__(self, run_id: _Optional[str] = ..., submission: _Optional[_Union[_operations_pb2.Submission, _Mapping]] = ...) -> None: ...

class PublishWeightsResponse(_message.Message):
    __slots__ = ("operation",)
    OPERATION_FIELD_NUMBER: _ClassVar[int]
    operation: _operations_pb2.Operation
    def __init__(self, operation: _Optional[_Union[_operations_pb2.Operation, _Mapping]] = ...) -> None: ...

class SaveWeightsRequest(_message.Message):
    __slots__ = ("run_id", "submission", "name", "ttl_s")
    RUN_ID_FIELD_NUMBER: _ClassVar[int]
    SUBMISSION_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    TTL_S_FIELD_NUMBER: _ClassVar[int]
    run_id: str
    submission: _operations_pb2.Submission
    name: str
    ttl_s: int
    def __init__(self, run_id: _Optional[str] = ..., submission: _Optional[_Union[_operations_pb2.Submission, _Mapping]] = ..., name: _Optional[str] = ..., ttl_s: _Optional[int] = ...) -> None: ...

class SaveWeightsResponse(_message.Message):
    __slots__ = ("operation",)
    OPERATION_FIELD_NUMBER: _ClassVar[int]
    operation: _operations_pb2.Operation
    def __init__(self, operation: _Optional[_Union[_operations_pb2.Operation, _Mapping]] = ...) -> None: ...
