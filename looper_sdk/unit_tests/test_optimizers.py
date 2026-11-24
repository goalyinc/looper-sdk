"""The optimizers a run names at create_run."""

from __future__ import annotations

import math

import pytest

from looper_sdk import AdamW, InvalidArgument, Muon
from looper_sdk.optimizers import check_learning_rate


def test_adamw_keeps_its_defaults_as_floats() -> None:
    adamw = AdamW(learning_rate=1, betas=[0, 0.5], max_grad_norm=2)  # pyright: ignore[reportArgumentType]
    assert adamw == AdamW(learning_rate=1.0, betas=(0.0, 0.5), max_grad_norm=2.0)
    assert AdamW(learning_rate=1e-5) == AdamW(
        learning_rate=1e-5, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0
    )


@pytest.mark.parametrize(
    "settings",
    [
        {"learning_rate": 0.0},
        {"learning_rate": math.inf},
        {"learning_rate": "1e-5"},
        {"betas": (0.9, 1.0)},
        {"betas": (0.9,)},
        {"betas": 0.9},
        {"eps": 0.0},
        {"weight_decay": -0.1},
        {"max_grad_norm": 0.0},
        {"max_grad_norm": math.nan},
    ],
    ids=str,
)
def test_adamw_refuses_what_it_cannot_be(settings: dict) -> None:
    with pytest.raises(InvalidArgument, match="AdamW"):
        AdamW(**{"learning_rate": 1e-5, **settings})


def test_a_step_s_learning_rate_is_a_float_of_at_least_0() -> None:
    assert check_learning_rate(0) == 0.0 and isinstance(check_learning_rate(1), float)
    for refused in (-1e-5, math.inf, math.nan, "1e-5", True):
        with pytest.raises(InvalidArgument, match="learning_rate"):
            check_learning_rate(refused)


def test_muon_keeps_its_defaults_as_floats() -> None:
    muon = Muon(learning_rate=1, momentum=0, betas=[0, 0.5])  # pyright: ignore[reportArgumentType]
    assert muon == Muon(learning_rate=1.0, momentum=0.0, betas=(0.0, 0.5))
    assert Muon(learning_rate=1e-4) == Muon(
        learning_rate=1e-4,
        weight_decay=0.1,
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
        betas=(0.9, 0.95),
        eps=1e-8,
    )


@pytest.mark.parametrize(
    "settings",
    [
        {"learning_rate": 0.0},
        {"weight_decay": -0.1},
        {"momentum": 1.0},
        {"momentum": "0.9"},
        {"nesterov": 1},
        {"ns_steps": 0},
        {"ns_steps": 2.5},
        {"betas": (0.9, 1.0)},
        {"eps": 0.0},
        {"max_grad_norm": 0.0},
    ],
    ids=str,
)
def test_muon_refuses_what_it_cannot_be(settings: dict) -> None:
    with pytest.raises(InvalidArgument, match="Muon"):
        Muon(**{"learning_rate": 1e-4, **settings})
