"""The values validate themselves on construction: every refusal is an
InvalidArgument, which is also a ValueError."""

from __future__ import annotations

import math
import re
from typing import Any

import pytest

from looper_sdk import Datum, Full, InvalidArgument, Lora, TrainingConfig, Version


def config(**overrides) -> TrainingConfig:
    fields: dict[str, Any] = dict(
        model="Qwen/Qwen3.5-9B", kind=Full(), tokens_per_step=8192, max_seq_len=4096
    )
    return TrainingConfig(**(fields | overrides))


def test_a_training_config_keeps_its_fields_and_defaults_to_no_sampler() -> None:
    assert config() == TrainingConfig(
        model="Qwen/Qwen3.5-9B",
        kind=Full(),
        tokens_per_step=8192,
        max_seq_len=4096,
        sampled_tokens_per_step=0,
    )
    assert Lora() == Lora(rank=32, train_attn=True, train_mlp=True, train_unembed=True)


@pytest.mark.parametrize(
    "overrides, says",
    [
        ({"model": ""}, "must name a model"),
        ({"model": " Qwen/Qwen3.5-9B"}, "must name a model"),
        ({"model": 9}, "must name a model"),
        ({"kind": "full"}, "must be Lora() or Full()"),
        ({"tokens_per_step": 0}, "at least 1"),
        ({"tokens_per_step": 8192.0}, "must be an integer"),
        ({"tokens_per_step": True}, "must be an integer"),
        ({"max_seq_len": 0}, "at least 1"),
        ({"max_seq_len": 16384}, "a step must hold at least one sequence"),
        ({"sampled_tokens_per_step": -1}, "at least 0"),
    ],
)
def test_a_training_config_refuses(overrides, says) -> None:
    with pytest.raises(InvalidArgument, match=re.escape(says)):
        config(**overrides)


@pytest.mark.parametrize(
    "fields, says",
    [
        ({"rank": 0}, "at least 1"),
        ({"rank": 8.5}, "must be an integer"),
        ({"train_mlp": 1}, "must be True or False"),
        ({"train_attn": False, "train_mlp": False, "train_unembed": False}, "trains nothing"),
    ],
)
def test_lora_refuses(fields, says) -> None:
    with pytest.raises(InvalidArgument, match=re.escape(says)):
        Lora(**fields)


def test_a_datum_keeps_its_tokens_weights_and_columns_as_tuples() -> None:
    datum = Datum(
        tokens=[1, 2, 3],
        weights=(0, 1, 0),
        columns={"logprobs": [0, -0.5, -1]},
        datum_id="d-1",
        reward=1,
        group_id="prompt-7",
        group_size=8,
    )
    assert datum.tokens == (1, 2, 3)
    assert datum.weights == (0.0, 1.0, 0.0)
    assert dict(datum.columns) == {"logprobs": (0.0, -0.5, -1.0)}
    assert (datum.reward, datum.group_id, datum.group_size) == (1.0, "prompt-7", 8)
    assert datum == Datum(
        tokens=(1, 2, 3),
        weights=[0, 1, 0],
        columns={"logprobs": (0, -0.5, -1)},
        datum_id="d-1",
        reward=1.0,
        group_id="prompt-7",
        group_size=8,
    )
    with pytest.raises(TypeError):
        datum.columns["logprobs"] = (1.0, 1.0, 1.0)  # pyright: ignore[reportIndexIssue]


def test_a_datum_takes_any_sequence_of_integers() -> None:
    assert Datum(tokens=bytearray([1, 2, 3]), weights=(0, 1, 1)).tokens == (1, 2, 3)


def test_a_column_need_not_be_finite_where_its_weight_is_zero() -> None:
    datum = Datum(tokens=[1, 2], weights=[0, 1], columns={"logprobs": [math.nan, -0.5]})
    assert math.isnan(datum.columns["logprobs"][0])


@pytest.mark.parametrize(
    "fields, says",
    [
        ({"tokens": [], "weights": []}, "tokens is empty"),
        ({"weights": [0, 1]}, "has 2 values for 3 tokens"),
        ({"tokens": [1, -1, 3]}, "tokens[1] = -1 is not a token id"),
        ({"tokens": [1, 2, 2**31]}, "tokens[2] = 2147483648 is not a token id"),
        ({"tokens": [1, 2.0, 3]}, "tokens[1] must be a token id"),
        ({"tokens": [1, True, 3]}, "tokens[1] must be a token id"),
        ({"tokens": "abc"}, "must be a sequence of token ids"),
        ({"tokens": 3}, "must be a sequence of token ids"),
        ({"weights": ["a", 1, 1]}, "must be a sequence of numbers"),
        ({"weights": [0, math.nan, 1]}, "weights[1] = nan: a weight is 0 or 1"),
        ({"weights": [0, 0.5, 1]}, "weights[1] = 0.5: a weight is 0 or 1"),
        ({"weights": [0, 1, -1]}, "weights[2] = -1.0: a weight is 0 or 1"),
        ({"weights": [1, 1, 1]}, "weights[0] is 1.0, not 0"),
        ({"weights": [0, 0, 0]}, "masks every target"),
        ({"columns": [("logprobs", [0, 0, 0])]}, "must map column names"),
        ({"columns": {"weights": [0, 1, 1]}}, "unknown columns ['weights']"),
        ({"columns": {"mask": [1, 1, 1]}}, "unknown columns ['mask']"),
        ({"columns": {"logprobs": [0, 0]}}, "has 2 values for 3 tokens"),
        ({"columns": {"logprobs": [0, "a", 0]}}, "must be a sequence of numbers"),
        ({"columns": {"logprobs": [0, math.inf, 0]}}, "['logprobs'][1] = inf is not finite"),
        ({"columns": {"advantages": [0, 0, 1]}}, "unknown columns ['advantages']"),
        ({"datum_id": ""}, "1 to 256 characters"),
        ({"datum_id": "x" * 257}, "1 to 256 characters"),
        ({"datum_id": 7}, "1 to 256 characters"),
        ({"reward": 1.0}, "reward, group_id and group_size come together"),
        ({"group_id": "a", "group_size": 2}, "reward, group_id and group_size come together"),
        ({"reward": math.nan, "group_id": "a", "group_size": 2}, "Datum.reward must be finite"),
        ({"reward": "1", "group_id": "a", "group_size": 2}, "Datum.reward must be a number"),
        ({"reward": 1, "group_id": "", "group_size": 2}, "Datum.group_id must name a group"),
        ({"reward": 1, "group_id": 7, "group_size": 2}, "Datum.group_id must name a group"),
        ({"reward": 1, "group_id": "a", "group_size": 0}, "Datum.group_size must be at least 1"),
        ({"reward": 1, "group_id": "a", "group_size": 2.0}, "Datum.group_size must be an integer"),
    ],
)
def test_a_datum_refuses(fields, says) -> None:
    with pytest.raises(InvalidArgument, match=re.escape(says)):
        Datum(**({"tokens": [1, 2, 3], "weights": [0, 1, 1]} | fields))


@pytest.mark.parametrize(
    "fields",
    [{"run_id": ""}, {"run_id": 3}, {"epoch": -1}, {"step": -1}, {"step": 1.0}],
)
def test_a_version_refuses(fields) -> None:
    with pytest.raises(InvalidArgument):
        Version(**({"run_id": "run-1", "epoch": 0, "step": 0} | fields))


def test_every_refusal_is_a_value_error() -> None:
    with pytest.raises(ValueError):
        Version(run_id="", epoch=0, step=0)


@pytest.mark.parametrize("kind", [Full(), Lora(rank=8, train_mlp=False, train_unembed=False)])
def test_model_spec_keeps_exact_trainable_parameters(kind):
    from dataclasses import FrozenInstanceError

    from looper_sdk import ModelSpec

    spec = ModelSpec(model="Qwen/Qwen3.5-9B", kind=kind)
    assert spec.model == "Qwen/Qwen3.5-9B"
    assert spec.kind == kind
    with pytest.raises(FrozenInstanceError):
        setattr(spec, "model", "other")


@pytest.mark.parametrize(
    "fields",
    [
        {"model": ""},
        {"model": " Qwen/Qwen3.5-9B"},
        {"model": 9},
        {"kind": "full"},
        {"kind": None},
    ],
)
def test_model_spec_refuses_invalid_identity(fields):
    from looper_sdk import ModelSpec

    with pytest.raises(InvalidArgument):
        ModelSpec(**({"model": "Qwen/Qwen3.5-9B", "kind": Full()} | fields))
