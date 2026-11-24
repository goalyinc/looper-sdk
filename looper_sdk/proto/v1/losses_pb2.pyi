from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Loss(_message.Message):
    __slots__ = ("cross_entropy", "grpo", "dapo")
    class CrossEntropy(_message.Message):
        __slots__ = ()
        def __init__(self) -> None: ...
    class Grpo(_message.Message):
        __slots__ = ("clip_low", "clip_high", "kl_weight", "scale_by_std")
        CLIP_LOW_FIELD_NUMBER: _ClassVar[int]
        CLIP_HIGH_FIELD_NUMBER: _ClassVar[int]
        KL_WEIGHT_FIELD_NUMBER: _ClassVar[int]
        SCALE_BY_STD_FIELD_NUMBER: _ClassVar[int]
        clip_low: float
        clip_high: float
        kl_weight: float
        scale_by_std: bool
        def __init__(self, clip_low: _Optional[float] = ..., clip_high: _Optional[float] = ..., kl_weight: _Optional[float] = ..., scale_by_std: _Optional[bool] = ...) -> None: ...
    CROSS_ENTROPY_FIELD_NUMBER: _ClassVar[int]
    GRPO_FIELD_NUMBER: _ClassVar[int]
    DAPO_FIELD_NUMBER: _ClassVar[int]
    cross_entropy: Loss.CrossEntropy
    grpo: Loss.Grpo
    dapo: Loss.Grpo
    def __init__(self, cross_entropy: _Optional[_Union[Loss.CrossEntropy, _Mapping]] = ..., grpo: _Optional[_Union[Loss.Grpo, _Mapping]] = ..., dapo: _Optional[_Union[Loss.Grpo, _Mapping]] = ...) -> None: ...
