import datetime

from google.protobuf import timestamp_pb2 as _timestamp_pb2
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Session(_message.Message):
    __slots__ = ("id", "project", "state", "expires_at", "heartbeat_interval_s")
    class State(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        STATE_UNSPECIFIED: _ClassVar[Session.State]
        ACTIVE: _ClassVar[Session.State]
        CLOSED: _ClassVar[Session.State]
        EXPIRED: _ClassVar[Session.State]
    STATE_UNSPECIFIED: Session.State
    ACTIVE: Session.State
    CLOSED: Session.State
    EXPIRED: Session.State
    ID_FIELD_NUMBER: _ClassVar[int]
    PROJECT_FIELD_NUMBER: _ClassVar[int]
    STATE_FIELD_NUMBER: _ClassVar[int]
    EXPIRES_AT_FIELD_NUMBER: _ClassVar[int]
    HEARTBEAT_INTERVAL_S_FIELD_NUMBER: _ClassVar[int]
    id: str
    project: str
    state: Session.State
    expires_at: _timestamp_pb2.Timestamp
    heartbeat_interval_s: float
    def __init__(self, id: _Optional[str] = ..., project: _Optional[str] = ..., state: _Optional[_Union[Session.State, str]] = ..., expires_at: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ..., heartbeat_interval_s: _Optional[float] = ...) -> None: ...

class OpenSessionRequest(_message.Message):
    __slots__ = ("project", "display_name")
    PROJECT_FIELD_NUMBER: _ClassVar[int]
    DISPLAY_NAME_FIELD_NUMBER: _ClassVar[int]
    project: str
    display_name: str
    def __init__(self, project: _Optional[str] = ..., display_name: _Optional[str] = ...) -> None: ...

class OpenSessionResponse(_message.Message):
    __slots__ = ("session",)
    SESSION_FIELD_NUMBER: _ClassVar[int]
    session: Session
    def __init__(self, session: _Optional[_Union[Session, _Mapping]] = ...) -> None: ...

class HeartbeatSessionRequest(_message.Message):
    __slots__ = ("session_id",)
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    def __init__(self, session_id: _Optional[str] = ...) -> None: ...

class HeartbeatSessionResponse(_message.Message):
    __slots__ = ("session",)
    SESSION_FIELD_NUMBER: _ClassVar[int]
    session: Session
    def __init__(self, session: _Optional[_Union[Session, _Mapping]] = ...) -> None: ...

class CloseSessionRequest(_message.Message):
    __slots__ = ("session_id",)
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    def __init__(self, session_id: _Optional[str] = ...) -> None: ...

class CloseSessionResponse(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...
