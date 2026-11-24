import datetime

from google.protobuf import timestamp_pb2 as _timestamp_pb2
from looper_sdk.proto.v1 import errors_pb2 as _errors_pb2
from looper_sdk.proto.v1 import values_pb2 as _values_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class SamplingParams(_message.Message):
    __slots__ = ("max_tokens", "temperature", "top_p", "stop_token_ids")
    MAX_TOKENS_FIELD_NUMBER: _ClassVar[int]
    TEMPERATURE_FIELD_NUMBER: _ClassVar[int]
    TOP_P_FIELD_NUMBER: _ClassVar[int]
    STOP_TOKEN_IDS_FIELD_NUMBER: _ClassVar[int]
    max_tokens: int
    temperature: float
    top_p: float
    stop_token_ids: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, max_tokens: _Optional[int] = ..., temperature: _Optional[float] = ..., top_p: _Optional[float] = ..., stop_token_ids: _Optional[_Iterable[int]] = ...) -> None: ...

class SampleRequest(_message.Message):
    __slots__ = ("prompt", "params", "n")
    PROMPT_FIELD_NUMBER: _ClassVar[int]
    PARAMS_FIELD_NUMBER: _ClassVar[int]
    N_FIELD_NUMBER: _ClassVar[int]
    prompt: _containers.RepeatedScalarFieldContainer[int]
    params: SamplingParams
    n: int
    def __init__(self, prompt: _Optional[_Iterable[int]] = ..., params: _Optional[_Union[SamplingParams, _Mapping]] = ..., n: _Optional[int] = ...) -> None: ...

class SampleSequence(_message.Message):
    __slots__ = ("token_ids", "logprobs", "finish_reason", "cached_tokens", "version")
    class FinishReason(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        FINISH_REASON_UNSPECIFIED: _ClassVar[SampleSequence.FinishReason]
        STOP: _ClassVar[SampleSequence.FinishReason]
        LENGTH: _ClassVar[SampleSequence.FinishReason]
    FINISH_REASON_UNSPECIFIED: SampleSequence.FinishReason
    STOP: SampleSequence.FinishReason
    LENGTH: SampleSequence.FinishReason
    TOKEN_IDS_FIELD_NUMBER: _ClassVar[int]
    LOGPROBS_FIELD_NUMBER: _ClassVar[int]
    FINISH_REASON_FIELD_NUMBER: _ClassVar[int]
    CACHED_TOKENS_FIELD_NUMBER: _ClassVar[int]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    token_ids: _containers.RepeatedScalarFieldContainer[int]
    logprobs: _containers.RepeatedScalarFieldContainer[float]
    finish_reason: SampleSequence.FinishReason
    cached_tokens: int
    version: _values_pb2.Version
    def __init__(self, token_ids: _Optional[_Iterable[int]] = ..., logprobs: _Optional[_Iterable[float]] = ..., finish_reason: _Optional[_Union[SampleSequence.FinishReason, str]] = ..., cached_tokens: _Optional[int] = ..., version: _Optional[_Union[_values_pb2.Version, _Mapping]] = ...) -> None: ...

class SamplingUsage(_message.Message):
    __slots__ = ("num_prompt_tokens", "num_generated_tokens", "num_cached_tokens")
    NUM_PROMPT_TOKENS_FIELD_NUMBER: _ClassVar[int]
    NUM_GENERATED_TOKENS_FIELD_NUMBER: _ClassVar[int]
    NUM_CACHED_TOKENS_FIELD_NUMBER: _ClassVar[int]
    num_prompt_tokens: int
    num_generated_tokens: int
    num_cached_tokens: int
    def __init__(self, num_prompt_tokens: _Optional[int] = ..., num_generated_tokens: _Optional[int] = ..., num_cached_tokens: _Optional[int] = ...) -> None: ...

class SampleResult(_message.Message):
    __slots__ = ("sequences", "model", "checkpoint_name", "version")
    SEQUENCES_FIELD_NUMBER: _ClassVar[int]
    MODEL_FIELD_NUMBER: _ClassVar[int]
    CHECKPOINT_NAME_FIELD_NUMBER: _ClassVar[int]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    sequences: _containers.RepeatedCompositeFieldContainer[SampleSequence]
    model: str
    checkpoint_name: str
    version: _values_pb2.Version
    def __init__(self, sequences: _Optional[_Iterable[_Union[SampleSequence, _Mapping]]] = ..., model: _Optional[str] = ..., checkpoint_name: _Optional[str] = ..., version: _Optional[_Union[_values_pb2.Version, _Mapping]] = ...) -> None: ...

class SamplerConfig(_message.Message):
    __slots__ = ("checkpoint_name", "max_seq_len", "max_concurrent_sequences")
    CHECKPOINT_NAME_FIELD_NUMBER: _ClassVar[int]
    MAX_SEQ_LEN_FIELD_NUMBER: _ClassVar[int]
    MAX_CONCURRENT_SEQUENCES_FIELD_NUMBER: _ClassVar[int]
    checkpoint_name: str
    max_seq_len: int
    max_concurrent_sequences: int
    def __init__(self, checkpoint_name: _Optional[str] = ..., max_seq_len: _Optional[int] = ..., max_concurrent_sequences: _Optional[int] = ...) -> None: ...

class Sampler(_message.Message):
    __slots__ = ("id", "config", "state", "model", "version", "created_at", "error", "usage")
    class State(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        STATE_UNSPECIFIED: _ClassVar[Sampler.State]
        STARTING: _ClassVar[Sampler.State]
        READY: _ClassVar[Sampler.State]
        FAILED: _ClassVar[Sampler.State]
        STOPPING: _ClassVar[Sampler.State]
        STOPPED: _ClassVar[Sampler.State]
    STATE_UNSPECIFIED: Sampler.State
    STARTING: Sampler.State
    READY: Sampler.State
    FAILED: Sampler.State
    STOPPING: Sampler.State
    STOPPED: Sampler.State
    ID_FIELD_NUMBER: _ClassVar[int]
    CONFIG_FIELD_NUMBER: _ClassVar[int]
    STATE_FIELD_NUMBER: _ClassVar[int]
    MODEL_FIELD_NUMBER: _ClassVar[int]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    CREATED_AT_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    USAGE_FIELD_NUMBER: _ClassVar[int]
    id: str
    config: SamplerConfig
    state: Sampler.State
    model: str
    version: _values_pb2.Version
    created_at: _timestamp_pb2.Timestamp
    error: _errors_pb2.Error
    usage: SamplingUsage
    def __init__(self, id: _Optional[str] = ..., config: _Optional[_Union[SamplerConfig, _Mapping]] = ..., state: _Optional[_Union[Sampler.State, str]] = ..., model: _Optional[str] = ..., version: _Optional[_Union[_values_pb2.Version, _Mapping]] = ..., created_at: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ..., error: _Optional[_Union[_errors_pb2.Error, _Mapping]] = ..., usage: _Optional[_Union[SamplingUsage, _Mapping]] = ...) -> None: ...

class CreateSamplerRequest(_message.Message):
    __slots__ = ("session_id", "key", "config")
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    KEY_FIELD_NUMBER: _ClassVar[int]
    CONFIG_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    key: str
    config: SamplerConfig
    def __init__(self, session_id: _Optional[str] = ..., key: _Optional[str] = ..., config: _Optional[_Union[SamplerConfig, _Mapping]] = ...) -> None: ...

class CreateSamplerResponse(_message.Message):
    __slots__ = ("sampler",)
    SAMPLER_FIELD_NUMBER: _ClassVar[int]
    sampler: Sampler
    def __init__(self, sampler: _Optional[_Union[Sampler, _Mapping]] = ...) -> None: ...

class GetSamplerResponse(_message.Message):
    __slots__ = ("sampler",)
    SAMPLER_FIELD_NUMBER: _ClassVar[int]
    sampler: Sampler
    def __init__(self, sampler: _Optional[_Union[Sampler, _Mapping]] = ...) -> None: ...

class StopSamplerRequest(_message.Message):
    __slots__ = ("session_id",)
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    def __init__(self, session_id: _Optional[str] = ...) -> None: ...

class StopSamplerResponse(_message.Message):
    __slots__ = ("sampler",)
    SAMPLER_FIELD_NUMBER: _ClassVar[int]
    sampler: Sampler
    def __init__(self, sampler: _Optional[_Union[Sampler, _Mapping]] = ...) -> None: ...

class SampleOperation(_message.Message):
    __slots__ = ("id", "key", "sampler_id", "state", "error", "raw_data_expired_at", "result", "usage")
    class State(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        STATE_UNSPECIFIED: _ClassVar[SampleOperation.State]
        QUEUED: _ClassVar[SampleOperation.State]
        RUNNING: _ClassVar[SampleOperation.State]
        SUCCEEDED: _ClassVar[SampleOperation.State]
        FAILED: _ClassVar[SampleOperation.State]
        CANCELLED: _ClassVar[SampleOperation.State]
    STATE_UNSPECIFIED: SampleOperation.State
    QUEUED: SampleOperation.State
    RUNNING: SampleOperation.State
    SUCCEEDED: SampleOperation.State
    FAILED: SampleOperation.State
    CANCELLED: SampleOperation.State
    ID_FIELD_NUMBER: _ClassVar[int]
    KEY_FIELD_NUMBER: _ClassVar[int]
    SAMPLER_ID_FIELD_NUMBER: _ClassVar[int]
    STATE_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    RAW_DATA_EXPIRED_AT_FIELD_NUMBER: _ClassVar[int]
    RESULT_FIELD_NUMBER: _ClassVar[int]
    USAGE_FIELD_NUMBER: _ClassVar[int]
    id: str
    key: str
    sampler_id: str
    state: SampleOperation.State
    error: _errors_pb2.Error
    raw_data_expired_at: _timestamp_pb2.Timestamp
    result: SampleResult
    usage: SamplingUsage
    def __init__(self, id: _Optional[str] = ..., key: _Optional[str] = ..., sampler_id: _Optional[str] = ..., state: _Optional[_Union[SampleOperation.State, str]] = ..., error: _Optional[_Union[_errors_pb2.Error, _Mapping]] = ..., raw_data_expired_at: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ..., result: _Optional[_Union[SampleResult, _Mapping]] = ..., usage: _Optional[_Union[SamplingUsage, _Mapping]] = ...) -> None: ...

class SubmitSampleRequest(_message.Message):
    __slots__ = ("session_id", "key", "request")
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    KEY_FIELD_NUMBER: _ClassVar[int]
    REQUEST_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    key: str
    request: SampleRequest
    def __init__(self, session_id: _Optional[str] = ..., key: _Optional[str] = ..., request: _Optional[_Union[SampleRequest, _Mapping]] = ...) -> None: ...

class SubmitSampleResponse(_message.Message):
    __slots__ = ("operation",)
    OPERATION_FIELD_NUMBER: _ClassVar[int]
    operation: SampleOperation
    def __init__(self, operation: _Optional[_Union[SampleOperation, _Mapping]] = ...) -> None: ...

class GetSampleOperationResponse(_message.Message):
    __slots__ = ("operation",)
    OPERATION_FIELD_NUMBER: _ClassVar[int]
    operation: SampleOperation
    def __init__(self, operation: _Optional[_Union[SampleOperation, _Mapping]] = ...) -> None: ...

class AwaitSampleOperationRequest(_message.Message):
    __slots__ = ("session_id", "timeout_s")
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    TIMEOUT_S_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    timeout_s: float
    def __init__(self, session_id: _Optional[str] = ..., timeout_s: _Optional[float] = ...) -> None: ...

class AwaitSampleOperationResponse(_message.Message):
    __slots__ = ("operation",)
    OPERATION_FIELD_NUMBER: _ClassVar[int]
    operation: SampleOperation
    def __init__(self, operation: _Optional[_Union[SampleOperation, _Mapping]] = ...) -> None: ...
