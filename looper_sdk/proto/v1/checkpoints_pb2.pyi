import datetime

from google.protobuf import timestamp_pb2 as _timestamp_pb2
from looper_sdk.proto.v1 import values_pb2 as _values_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Checkpoint(_message.Message):
    __slots__ = ("id", "operation_id", "project", "principal", "source_run_id", "version", "size_bytes", "created_at", "name", "expires_at", "model_spec", "state", "failure", "kind")
    class State(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        STATE_UNSPECIFIED: _ClassVar[Checkpoint.State]
        UPLOADING: _ClassVar[Checkpoint.State]
        READY_TO_USE: _ClassVar[Checkpoint.State]
        FAILED: _ClassVar[Checkpoint.State]
    STATE_UNSPECIFIED: Checkpoint.State
    UPLOADING: Checkpoint.State
    READY_TO_USE: Checkpoint.State
    FAILED: Checkpoint.State
    class Kind(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        KIND_UNSPECIFIED: _ClassVar[Checkpoint.Kind]
        STATE: _ClassVar[Checkpoint.Kind]
        WEIGHTS: _ClassVar[Checkpoint.Kind]
    KIND_UNSPECIFIED: Checkpoint.Kind
    STATE: Checkpoint.Kind
    WEIGHTS: Checkpoint.Kind
    ID_FIELD_NUMBER: _ClassVar[int]
    OPERATION_ID_FIELD_NUMBER: _ClassVar[int]
    PROJECT_FIELD_NUMBER: _ClassVar[int]
    PRINCIPAL_FIELD_NUMBER: _ClassVar[int]
    SOURCE_RUN_ID_FIELD_NUMBER: _ClassVar[int]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    SIZE_BYTES_FIELD_NUMBER: _ClassVar[int]
    CREATED_AT_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    EXPIRES_AT_FIELD_NUMBER: _ClassVar[int]
    MODEL_SPEC_FIELD_NUMBER: _ClassVar[int]
    STATE_FIELD_NUMBER: _ClassVar[int]
    FAILURE_FIELD_NUMBER: _ClassVar[int]
    KIND_FIELD_NUMBER: _ClassVar[int]
    id: str
    operation_id: str
    project: str
    principal: str
    source_run_id: str
    version: _values_pb2.Version
    size_bytes: int
    created_at: _timestamp_pb2.Timestamp
    name: str
    expires_at: _timestamp_pb2.Timestamp
    model_spec: _values_pb2.ModelSpec
    state: Checkpoint.State
    failure: str
    kind: Checkpoint.Kind
    def __init__(self, id: _Optional[str] = ..., operation_id: _Optional[str] = ..., project: _Optional[str] = ..., principal: _Optional[str] = ..., source_run_id: _Optional[str] = ..., version: _Optional[_Union[_values_pb2.Version, _Mapping]] = ..., size_bytes: _Optional[int] = ..., created_at: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ..., name: _Optional[str] = ..., expires_at: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ..., model_spec: _Optional[_Union[_values_pb2.ModelSpec, _Mapping]] = ..., state: _Optional[_Union[Checkpoint.State, str]] = ..., failure: _Optional[str] = ..., kind: _Optional[_Union[Checkpoint.Kind, str]] = ...) -> None: ...

class GetCheckpointRequest(_message.Message):
    __slots__ = ("session_id", "checkpoint_id")
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    CHECKPOINT_ID_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    checkpoint_id: str
    def __init__(self, session_id: _Optional[str] = ..., checkpoint_id: _Optional[str] = ...) -> None: ...

class GetCheckpointResponse(_message.Message):
    __slots__ = ("checkpoint",)
    CHECKPOINT_FIELD_NUMBER: _ClassVar[int]
    checkpoint: Checkpoint
    def __init__(self, checkpoint: _Optional[_Union[Checkpoint, _Mapping]] = ...) -> None: ...

class GetCheckpointByNameRequest(_message.Message):
    __slots__ = ("session_id", "name")
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    name: str
    def __init__(self, session_id: _Optional[str] = ..., name: _Optional[str] = ...) -> None: ...

class GetCheckpointByNameResponse(_message.Message):
    __slots__ = ("checkpoint",)
    CHECKPOINT_FIELD_NUMBER: _ClassVar[int]
    checkpoint: Checkpoint
    def __init__(self, checkpoint: _Optional[_Union[Checkpoint, _Mapping]] = ...) -> None: ...

class ListCheckpointsRequest(_message.Message):
    __slots__ = ("session_id", "limit", "after_name")
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    LIMIT_FIELD_NUMBER: _ClassVar[int]
    AFTER_NAME_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    limit: int
    after_name: str
    def __init__(self, session_id: _Optional[str] = ..., limit: _Optional[int] = ..., after_name: _Optional[str] = ...) -> None: ...

class ListCheckpointsResponse(_message.Message):
    __slots__ = ("checkpoints",)
    CHECKPOINTS_FIELD_NUMBER: _ClassVar[int]
    checkpoints: _containers.RepeatedCompositeFieldContainer[Checkpoint]
    def __init__(self, checkpoints: _Optional[_Iterable[_Union[Checkpoint, _Mapping]]] = ...) -> None: ...

class DeleteCheckpointRequest(_message.Message):
    __slots__ = ("session_id", "checkpoint_id")
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    CHECKPOINT_ID_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    checkpoint_id: str
    def __init__(self, session_id: _Optional[str] = ..., checkpoint_id: _Optional[str] = ...) -> None: ...

class DeleteCheckpointResponse(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class GetCheckpointDownloadRequest(_message.Message):
    __slots__ = ("session_id", "checkpoint_id", "paths")
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    CHECKPOINT_ID_FIELD_NUMBER: _ClassVar[int]
    PATHS_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    checkpoint_id: str
    paths: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, session_id: _Optional[str] = ..., checkpoint_id: _Optional[str] = ..., paths: _Optional[_Iterable[str]] = ...) -> None: ...

class DownloadFile(_message.Message):
    __slots__ = ("path", "size_bytes", "crc32c", "url")
    PATH_FIELD_NUMBER: _ClassVar[int]
    SIZE_BYTES_FIELD_NUMBER: _ClassVar[int]
    CRC32C_FIELD_NUMBER: _ClassVar[int]
    URL_FIELD_NUMBER: _ClassVar[int]
    path: str
    size_bytes: int
    crc32c: int
    url: str
    def __init__(self, path: _Optional[str] = ..., size_bytes: _Optional[int] = ..., crc32c: _Optional[int] = ..., url: _Optional[str] = ...) -> None: ...

class GetCheckpointDownloadResponse(_message.Message):
    __slots__ = ("files", "expires_at")
    FILES_FIELD_NUMBER: _ClassVar[int]
    EXPIRES_AT_FIELD_NUMBER: _ClassVar[int]
    files: _containers.RepeatedCompositeFieldContainer[DownloadFile]
    expires_at: _timestamp_pb2.Timestamp
    def __init__(self, files: _Optional[_Iterable[_Union[DownloadFile, _Mapping]]] = ..., expires_at: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ...) -> None: ...
