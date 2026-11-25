from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class WaitingReason(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    WAITING_REASON_UNSPECIFIED: _ClassVar[WaitingReason]
    WAITING_REASON_ORG_GPU_LIMIT: _ClassVar[WaitingReason]
WAITING_REASON_UNSPECIFIED: WaitingReason
WAITING_REASON_ORG_GPU_LIMIT: WaitingReason

class Version(_message.Message):
    __slots__ = ("run_id", "epoch", "step")
    RUN_ID_FIELD_NUMBER: _ClassVar[int]
    EPOCH_FIELD_NUMBER: _ClassVar[int]
    STEP_FIELD_NUMBER: _ClassVar[int]
    run_id: str
    epoch: int
    step: int
    def __init__(self, run_id: _Optional[str] = ..., epoch: _Optional[int] = ..., step: _Optional[int] = ...) -> None: ...

class TrainingConfig(_message.Message):
    __slots__ = ("model", "lora", "full", "tokens_per_step", "max_seq_len", "sampled_tokens_per_step", "optimizer")
    class Lora(_message.Message):
        __slots__ = ("rank", "train_attn", "train_mlp", "train_unembed")
        RANK_FIELD_NUMBER: _ClassVar[int]
        TRAIN_ATTN_FIELD_NUMBER: _ClassVar[int]
        TRAIN_MLP_FIELD_NUMBER: _ClassVar[int]
        TRAIN_UNEMBED_FIELD_NUMBER: _ClassVar[int]
        rank: int
        train_attn: bool
        train_mlp: bool
        train_unembed: bool
        def __init__(self, rank: _Optional[int] = ..., train_attn: _Optional[bool] = ..., train_mlp: _Optional[bool] = ..., train_unembed: _Optional[bool] = ...) -> None: ...
    class Full(_message.Message):
        __slots__ = ()
        def __init__(self) -> None: ...
    MODEL_FIELD_NUMBER: _ClassVar[int]
    LORA_FIELD_NUMBER: _ClassVar[int]
    FULL_FIELD_NUMBER: _ClassVar[int]
    TOKENS_PER_STEP_FIELD_NUMBER: _ClassVar[int]
    MAX_SEQ_LEN_FIELD_NUMBER: _ClassVar[int]
    SAMPLED_TOKENS_PER_STEP_FIELD_NUMBER: _ClassVar[int]
    OPTIMIZER_FIELD_NUMBER: _ClassVar[int]
    model: str
    lora: TrainingConfig.Lora
    full: TrainingConfig.Full
    tokens_per_step: int
    max_seq_len: int
    sampled_tokens_per_step: int
    optimizer: Optimizer
    def __init__(self, model: _Optional[str] = ..., lora: _Optional[_Union[TrainingConfig.Lora, _Mapping]] = ..., full: _Optional[_Union[TrainingConfig.Full, _Mapping]] = ..., tokens_per_step: _Optional[int] = ..., max_seq_len: _Optional[int] = ..., sampled_tokens_per_step: _Optional[int] = ..., optimizer: _Optional[_Union[Optimizer, _Mapping]] = ...) -> None: ...

class Optimizer(_message.Message):
    __slots__ = ("adamw", "muon")
    class AdamW(_message.Message):
        __slots__ = ("learning_rate", "beta1", "beta2", "eps", "weight_decay", "max_grad_norm")
        LEARNING_RATE_FIELD_NUMBER: _ClassVar[int]
        BETA1_FIELD_NUMBER: _ClassVar[int]
        BETA2_FIELD_NUMBER: _ClassVar[int]
        EPS_FIELD_NUMBER: _ClassVar[int]
        WEIGHT_DECAY_FIELD_NUMBER: _ClassVar[int]
        MAX_GRAD_NORM_FIELD_NUMBER: _ClassVar[int]
        learning_rate: float
        beta1: float
        beta2: float
        eps: float
        weight_decay: float
        max_grad_norm: float
        def __init__(self, learning_rate: _Optional[float] = ..., beta1: _Optional[float] = ..., beta2: _Optional[float] = ..., eps: _Optional[float] = ..., weight_decay: _Optional[float] = ..., max_grad_norm: _Optional[float] = ...) -> None: ...
    class Muon(_message.Message):
        __slots__ = ("learning_rate", "weight_decay", "momentum", "nesterov", "ns_steps", "beta1", "beta2", "eps", "max_grad_norm")
        LEARNING_RATE_FIELD_NUMBER: _ClassVar[int]
        WEIGHT_DECAY_FIELD_NUMBER: _ClassVar[int]
        MOMENTUM_FIELD_NUMBER: _ClassVar[int]
        NESTEROV_FIELD_NUMBER: _ClassVar[int]
        NS_STEPS_FIELD_NUMBER: _ClassVar[int]
        BETA1_FIELD_NUMBER: _ClassVar[int]
        BETA2_FIELD_NUMBER: _ClassVar[int]
        EPS_FIELD_NUMBER: _ClassVar[int]
        MAX_GRAD_NORM_FIELD_NUMBER: _ClassVar[int]
        learning_rate: float
        weight_decay: float
        momentum: float
        nesterov: bool
        ns_steps: int
        beta1: float
        beta2: float
        eps: float
        max_grad_norm: float
        def __init__(self, learning_rate: _Optional[float] = ..., weight_decay: _Optional[float] = ..., momentum: _Optional[float] = ..., nesterov: _Optional[bool] = ..., ns_steps: _Optional[int] = ..., beta1: _Optional[float] = ..., beta2: _Optional[float] = ..., eps: _Optional[float] = ..., max_grad_norm: _Optional[float] = ...) -> None: ...
    ADAMW_FIELD_NUMBER: _ClassVar[int]
    MUON_FIELD_NUMBER: _ClassVar[int]
    adamw: Optimizer.AdamW
    muon: Optimizer.Muon
    def __init__(self, adamw: _Optional[_Union[Optimizer.AdamW, _Mapping]] = ..., muon: _Optional[_Union[Optimizer.Muon, _Mapping]] = ...) -> None: ...

class ModelSpec(_message.Message):
    __slots__ = ("model", "lora", "full")
    MODEL_FIELD_NUMBER: _ClassVar[int]
    LORA_FIELD_NUMBER: _ClassVar[int]
    FULL_FIELD_NUMBER: _ClassVar[int]
    model: str
    lora: TrainingConfig.Lora
    full: TrainingConfig.Full
    def __init__(self, model: _Optional[str] = ..., lora: _Optional[_Union[TrainingConfig.Lora, _Mapping]] = ..., full: _Optional[_Union[TrainingConfig.Full, _Mapping]] = ...) -> None: ...

class Column(_message.Message):
    __slots__ = ("values",)
    VALUES_FIELD_NUMBER: _ClassVar[int]
    values: _containers.RepeatedScalarFieldContainer[float]
    def __init__(self, values: _Optional[_Iterable[float]] = ...) -> None: ...

class Datum(_message.Message):
    __slots__ = ("tokens", "weights", "columns", "datum_id", "reward", "group_id", "group_size")
    class ColumnsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: Column
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[Column, _Mapping]] = ...) -> None: ...
    TOKENS_FIELD_NUMBER: _ClassVar[int]
    WEIGHTS_FIELD_NUMBER: _ClassVar[int]
    COLUMNS_FIELD_NUMBER: _ClassVar[int]
    DATUM_ID_FIELD_NUMBER: _ClassVar[int]
    REWARD_FIELD_NUMBER: _ClassVar[int]
    GROUP_ID_FIELD_NUMBER: _ClassVar[int]
    GROUP_SIZE_FIELD_NUMBER: _ClassVar[int]
    tokens: _containers.RepeatedScalarFieldContainer[int]
    weights: _containers.RepeatedScalarFieldContainer[float]
    columns: _containers.MessageMap[str, Column]
    datum_id: str
    reward: float
    group_id: str
    group_size: int
    def __init__(self, tokens: _Optional[_Iterable[int]] = ..., weights: _Optional[_Iterable[float]] = ..., columns: _Optional[_Mapping[str, Column]] = ..., datum_id: _Optional[str] = ..., reward: _Optional[float] = ..., group_id: _Optional[str] = ..., group_size: _Optional[int] = ...) -> None: ...
