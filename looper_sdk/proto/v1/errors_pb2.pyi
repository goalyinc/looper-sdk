from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Error(_message.Message):
    __slots__ = ("code", "category", "retryable", "message", "retry_after_s", "run_rolled_back")
    class Code(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        CODE_UNSPECIFIED: _ClassVar[Error.Code]
        INVALID_ARGUMENT: _ClassVar[Error.Code]
        UNAUTHENTICATED: _ClassVar[Error.Code]
        SPEND_REFUSED: _ClassVar[Error.Code]
        PERMISSION_DENIED: _ClassVar[Error.Code]
        QUOTA_EXCEEDED: _ClassVar[Error.Code]
        NOT_FOUND: _ClassVar[Error.Code]
        IDEMPOTENCY_CONFLICT: _ClassVar[Error.Code]
        WRITER_SUPERSEDED: _ClassVar[Error.Code]
        SESSION_EXPIRED: _ClassVar[Error.Code]
        DEPENDENCY_FAILED: _ClassVar[Error.Code]
        RUN_ROLLED_BACK: _ClassVar[Error.Code]
        GRADIENTS_PENDING: _ClassVar[Error.Code]
        VERSION_SUPERSEDED: _ClassVar[Error.Code]
        RUN_SUSPENDED: _ClassVar[Error.Code]
        CANCELLED: _ClassVar[Error.Code]
        RUN_STATE_LOST: _ClassVar[Error.Code]
        PAYLOAD_TOO_LARGE: _ClassVar[Error.Code]
        INCOMPLETE_ACCUMULATION: _ClassVar[Error.Code]
        EMPTY_STEP: _ClassVar[Error.Code]
        CAPACITY_EXHAUSTED: _ClassVar[Error.Code]
        RUN_NOT_READY: _ClassVar[Error.Code]
        UNAVAILABLE: _ClassVar[Error.Code]
        INTERNAL: _ClassVar[Error.Code]
        CHECKPOINT_NOT_READY: _ClassVar[Error.Code]
        QUEUE_TIMEOUT: _ClassVar[Error.Code]
        SAMPLER_FAILED: _ClassVar[Error.Code]
        RESULT_EXPIRED: _ClassVar[Error.Code]
        CHECKPOINT_NAME_TAKEN: _ClassVar[Error.Code]
    CODE_UNSPECIFIED: Error.Code
    INVALID_ARGUMENT: Error.Code
    UNAUTHENTICATED: Error.Code
    SPEND_REFUSED: Error.Code
    PERMISSION_DENIED: Error.Code
    QUOTA_EXCEEDED: Error.Code
    NOT_FOUND: Error.Code
    IDEMPOTENCY_CONFLICT: Error.Code
    WRITER_SUPERSEDED: Error.Code
    SESSION_EXPIRED: Error.Code
    DEPENDENCY_FAILED: Error.Code
    RUN_ROLLED_BACK: Error.Code
    GRADIENTS_PENDING: Error.Code
    VERSION_SUPERSEDED: Error.Code
    RUN_SUSPENDED: Error.Code
    CANCELLED: Error.Code
    RUN_STATE_LOST: Error.Code
    PAYLOAD_TOO_LARGE: Error.Code
    INCOMPLETE_ACCUMULATION: Error.Code
    EMPTY_STEP: Error.Code
    CAPACITY_EXHAUSTED: Error.Code
    RUN_NOT_READY: Error.Code
    UNAVAILABLE: Error.Code
    INTERNAL: Error.Code
    CHECKPOINT_NOT_READY: Error.Code
    QUEUE_TIMEOUT: Error.Code
    SAMPLER_FAILED: Error.Code
    RESULT_EXPIRED: Error.Code
    CHECKPOINT_NAME_TAKEN: Error.Code
    class Category(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        CATEGORY_UNSPECIFIED: _ClassVar[Error.Category]
        USER: _ClassVar[Error.Category]
        SERVER: _ClassVar[Error.Category]
    CATEGORY_UNSPECIFIED: Error.Category
    USER: Error.Category
    SERVER: Error.Category
    CODE_FIELD_NUMBER: _ClassVar[int]
    CATEGORY_FIELD_NUMBER: _ClassVar[int]
    RETRYABLE_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    RETRY_AFTER_S_FIELD_NUMBER: _ClassVar[int]
    RUN_ROLLED_BACK_FIELD_NUMBER: _ClassVar[int]
    code: Error.Code
    category: Error.Category
    retryable: bool
    message: str
    retry_after_s: float
    run_rolled_back: RunRolledBack
    def __init__(self, code: _Optional[_Union[Error.Code, str]] = ..., category: _Optional[_Union[Error.Category, str]] = ..., retryable: _Optional[bool] = ..., message: _Optional[str] = ..., retry_after_s: _Optional[float] = ..., run_rolled_back: _Optional[_Union[RunRolledBack, _Mapping]] = ...) -> None: ...

class RunRolledBack(_message.Message):
    __slots__ = ("epoch", "restored_step", "restored_seq", "first_lost_seq", "last_lost_seq")
    EPOCH_FIELD_NUMBER: _ClassVar[int]
    RESTORED_STEP_FIELD_NUMBER: _ClassVar[int]
    RESTORED_SEQ_FIELD_NUMBER: _ClassVar[int]
    FIRST_LOST_SEQ_FIELD_NUMBER: _ClassVar[int]
    LAST_LOST_SEQ_FIELD_NUMBER: _ClassVar[int]
    epoch: int
    restored_step: int
    restored_seq: int
    first_lost_seq: int
    last_lost_seq: int
    def __init__(self, epoch: _Optional[int] = ..., restored_step: _Optional[int] = ..., restored_seq: _Optional[int] = ..., first_lost_seq: _Optional[int] = ..., last_lost_seq: _Optional[int] = ...) -> None: ...
