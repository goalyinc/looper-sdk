"""The values a client sends: what a run trains, and its data.

Each is a frozen value that validates itself on construction, so an invalid
one cannot exist; it raises ``InvalidArgument``, a ``ValueError``. The service
validates the same way when it decodes a request, independently of the
client. Checks that need the model, such as the vocabulary or
the context, are the service's.
"""

from __future__ import annotations

import dataclasses
import math
import operator
import types
from collections.abc import Mapping, Sequence

from looper_sdk._checks import _count, _finite, _flag
from looper_sdk.errors import InvalidArgument
from looper_sdk.optimizers import AdamW, Muon, Optimizer

# The per-token columns a datum may carry beside its weights; losses.py says
# which loss reads which. The trainer forms the advantages, so a datum never
# carries them.
COLUMNS = ("logprobs",)
# Token ids travel as int32 (proto/v1/values.proto).
_MAX_TOKEN_ID = 2**31 - 1


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class Version:
    """A run's weights after some optimizer steps.

    Attributes:
        run_id: The run that trained them.
        epoch: The run's epoch they were made in; it counts the run's
            rollbacks.
        step: The optimizer steps the run had completed.
    """

    run_id: str
    epoch: int
    step: int

    def __post_init__(self) -> None:
        if not isinstance(self.run_id, str) or not self.run_id:
            raise InvalidArgument(f"Version.run_id must be a run's id, not {self.run_id!r}")
        object.__setattr__(self, "epoch", _count("Version.epoch", self.epoch, minimum=0))
        object.__setattr__(self, "step", _count("Version.step", self.step, minimum=0))


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class Lora:
    """Low-rank factors over the frozen base.

    Attributes:
        rank: The factors' rank; their product is scaled by 32 / rank.
        train_attn: Factors on the attention projections, linear attention's
            included.
        train_mlp: Factors on the MLPs, and an MoE layer's experts.
        train_unembed: Factors on the LM head.
    """

    rank: int = 32
    train_attn: bool = True
    train_mlp: bool = True
    train_unembed: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "rank", _count("Lora.rank", self.rank, minimum=1))
        switches = [
            _flag(f"Lora.{name}", getattr(self, name))
            for name in ("train_attn", "train_mlp", "train_unembed")
        ]
        if not any(switches):
            raise InvalidArgument("Lora trains nothing: set train_attn, train_mlp or train_unembed")


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class Full:
    """Every parameter trains."""


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class ModelSpec:
    """A saved model's identity and trainable parameters.

    Attributes:
        model: The source run's catalog model name, not a resolved revision.
        kind: Full parameters or the exact LoRA rank and target switches.
    """

    model: str
    kind: Lora | Full

    def __post_init__(self) -> None:
        if not isinstance(self.model, str) or not self.model or self.model != self.model.strip():
            raise InvalidArgument(
                "ModelSpec.model must name a model, such as 'Qwen/Qwen3.5-9B', "
                f"not {self.model!r}"
            )
        if not isinstance(self.kind, Lora | Full):
            raise InvalidArgument(f"ModelSpec.kind must be Lora() or Full(), not {self.kind!r}")


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class TrainingConfig:
    """What a run trains, and how much. Looper sizes the run's
    GPUs from the three sizing fields, which are fixed for its life.

    Attributes:
        model: A model of the catalog, by its Hugging Face name.
        kind: LoRA factors over the frozen base, or every parameter.
        tokens_per_step: The tokens one optimizer step trains, over its
            forward_backward calls. A larger batch still runs, through
            gradient accumulation; this sets only the step time.
        max_seq_len: The longest sequence the run trains or samples.
        sampled_tokens_per_step: Unused, and 0 by default: a run's sampler
            is sized when ``TrainingClient.create_sampling_client`` creates
            it.
        optimizer: What every ``optim_step`` steps, with its clipping. None
            trains with the model's default AdamW for the run's kind, at a
            constant rate.
    """

    model: str
    kind: Lora | Full
    tokens_per_step: int
    max_seq_len: int
    sampled_tokens_per_step: int = 0
    optimizer: Optimizer | None = None

    def __post_init__(self) -> None:
        ModelSpec(model=self.model, kind=self.kind)
        tokens = _count("TrainingConfig.tokens_per_step", self.tokens_per_step, minimum=1)
        longest = _count("TrainingConfig.max_seq_len", self.max_seq_len, minimum=1)
        if longest > tokens:
            raise InvalidArgument(
                f"TrainingConfig.max_seq_len ({longest}) exceeds tokens_per_step ({tokens}): "
                "a step must hold at least one sequence of the longest length"
            )
        sampled = _count(
            "TrainingConfig.sampled_tokens_per_step", self.sampled_tokens_per_step, minimum=0
        )
        if self.optimizer is not None and not isinstance(self.optimizer, AdamW | Muon):
            raise InvalidArgument(
                "TrainingConfig.optimizer must be AdamW(...), Muon(...) or None, "
                f"not {self.optimizer!r}"
            )
        object.__setattr__(self, "tokens_per_step", tokens)
        object.__setattr__(self, "max_seq_len", longest)
        object.__setattr__(self, "sampled_tokens_per_step", sampled)


def _token_ids(name: str, values: object) -> tuple[int, ...]:
    if isinstance(values, str | bytes):
        raise InvalidArgument(f"{name} must be a sequence of token ids")
    if not isinstance(values, Sequence):
        try:
            values = list(values)  # pyright: ignore[reportArgumentType]
        except TypeError:
            raise InvalidArgument(f"{name} must be a sequence of token ids") from None
    ids = []
    for i, value in enumerate(values):
        if isinstance(value, bool):
            raise InvalidArgument(f"{name}[{i}] must be a token id, not {value!r}")
        try:
            token = operator.index(value)
        except TypeError:
            raise InvalidArgument(f"{name}[{i}] must be a token id, not {value!r}") from None
        if not 0 <= token <= _MAX_TOKEN_ID:
            raise InvalidArgument(f"{name}[{i}] = {token} is not a token id")
        ids.append(token)
    return tuple(ids)


def _floats(name: str, values: object) -> tuple[float, ...]:
    if isinstance(values, str | bytes):
        raise InvalidArgument(f"{name} must be a sequence of numbers")
    try:
        return tuple(float(value) for value in values)  # pyright: ignore[reportGeneralTypeIssues]
    except (TypeError, ValueError):
        raise InvalidArgument(f"{name} must be a sequence of numbers") from None


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class Datum:
    """One sequence: each token after the first is a target,
    predicted from the tokens before it.

    Attributes:
        tokens: Token ids, ``N`` of them.
        weights: One per token, 0 or 1: whether the term for predicting
            ``tokens[i]`` counts. ``weights[0]`` is 0, since nothing predicts
            the first token, and at least one other is 1. A weight of 0 masks
            its token, which is still attended as context. No loss weighs
            its targets, so no other weight is taken.
        columns: Per-token floats by name, aligned to ``tokens`` as
            ``weights`` is: ``logprobs``, the sampler's record of each token,
            for ``Grpo`` and ``Dapo``. Finite wherever the weight is 1;
            float32 on the wire.
        datum_id: Echoed in this datum's results; unique within a call.
        reward: The datum's score: finite, float64 on the wire. ``reward``,
            ``group_id`` and ``group_size`` come together or not at all, on
            exactly the datums of a loss that compares groups, which forms
            their advantages from the group's rewards.
        group_id: The group of samples the datum is compared within.
        group_size: How many datums the group has, 1 or more. A group is
            sent whole within one call.
    """

    tokens: Sequence[int]
    weights: Sequence[float]
    columns: Mapping[str, Sequence[float]] = dataclasses.field(default_factory=dict)
    datum_id: str | None = None
    reward: float | None = None
    group_id: str | None = None
    group_size: int | None = None

    def __post_init__(self) -> None:
        tokens = _token_ids("Datum.tokens", self.tokens)
        weights = _floats("Datum.weights", self.weights)
        if not tokens:
            raise InvalidArgument("Datum.tokens is empty")
        if len(weights) != len(tokens):
            raise InvalidArgument(
                f"Datum.weights has {len(weights)} values for {len(tokens)} tokens"
            )
        for i, weight in enumerate(weights):
            if weight not in (0, 1):
                raise InvalidArgument(
                    f"Datum.weights[{i}] = {weight!r}: a weight is 0 or 1, since no loss "
                    "weighs its targets"
                )
        if weights[0] != 0:
            raise InvalidArgument(
                f"Datum.weights[0] is {weights[0]}, not 0: nothing predicts the first token"
            )
        if not any(weights[1:]):
            raise InvalidArgument("Datum.weights masks every target: weigh at least one token")
        if not isinstance(self.columns, Mapping):
            raise InvalidArgument("Datum.columns must map column names to values")
        unknown = sorted(set(self.columns) - set(COLUMNS))
        if unknown:
            raise InvalidArgument(f"Datum.columns has unknown columns {unknown}; known: {COLUMNS}")
        columns = {}
        for name in COLUMNS:
            if name not in self.columns:
                continue
            values = _floats(f"Datum.columns[{name!r}]", self.columns[name])
            if len(values) != len(tokens):
                raise InvalidArgument(
                    f"Datum.columns[{name!r}] has {len(values)} values for {len(tokens)} tokens"
                )
            for i, value in enumerate(values):
                if weights[i] != 0 and not math.isfinite(value):
                    raise InvalidArgument(f"Datum.columns[{name!r}][{i}] = {value!r} is not finite")
            columns[name] = values
        if self.datum_id is not None and (
            not isinstance(self.datum_id, str) or not 1 <= len(self.datum_id) <= 256
        ):
            raise InvalidArgument(
                f"Datum.datum_id must be 1 to 256 characters, not {self.datum_id!r}"
            )
        if not (self.reward is None) == (self.group_id is None) == (self.group_size is None):
            raise InvalidArgument(
                "Datum.reward, group_id and group_size come together: a rewarded datum "
                "belongs to a group of a declared size"
            )
        if self.reward is not None:
            if not isinstance(self.group_id, str) or not self.group_id:
                raise InvalidArgument(f"Datum.group_id must name a group, not {self.group_id!r}")
            object.__setattr__(self, "reward", _finite("Datum.reward", self.reward))
            object.__setattr__(
                self, "group_size", _count("Datum.group_size", self.group_size, minimum=1)
            )
        object.__setattr__(self, "tokens", tokens)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "columns", types.MappingProxyType(columns))
