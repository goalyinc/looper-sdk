"""The losses and what a call's data must carry for each."""

from __future__ import annotations

import math
import re

import pytest

from looper_sdk import CrossEntropy, Dapo, Datum, Grpo, InvalidArgument
from looper_sdk.losses import check_batch

LOGPROBS = {"logprobs": [0.0, -0.2]}


def datum(columns=None, datum_id=None, group=None) -> Datum:
    """A two-token datum; ``group`` is (group_id, group_size)."""
    grouped = {} if group is None else {"reward": 1.0, "group_id": group[0], "group_size": group[1]}
    return Datum(
        tokens=[1, 2], weights=[0, 1], columns=columns or {}, datum_id=datum_id, **grouped
    )


def sample(group_id: str, size: int) -> Datum:
    return datum(LOGPROBS, group=(group_id, size))


def test_grpo_and_dapo_defaults() -> None:
    assert Grpo() == Grpo(clip_low=0.2, clip_high=0.2, kl_weight=0.0, scale_by_std=True)
    assert Dapo() == Dapo(clip_low=0.2, clip_high=0.28, kl_weight=0.0, scale_by_std=True)
    assert Grpo(clip_low=1e-3, clip_high=5).clip_high == 5.0
    assert Dapo(kl_weight=1).kl_weight == 1.0


@pytest.mark.parametrize(
    "low, high",
    [(0, 0.2), (1, 0.2), (-0.1, 0.2), (0.2, 0), (0.2, -1), (math.nan, 0.2), (0.2, math.inf)],
)
@pytest.mark.parametrize("loss", [Grpo, Dapo])
def test_clip_bounds_outside_their_ranges_are_refused(loss, low, high) -> None:
    with pytest.raises(InvalidArgument, match=loss.__name__):
        loss(clip_low=low, clip_high=high)


@pytest.mark.parametrize("weight", [-0.1, math.nan, math.inf, "0.1"])
@pytest.mark.parametrize("loss", [Grpo, Dapo])
def test_a_kl_weight_below_zero_or_not_finite_is_refused(loss, weight) -> None:
    with pytest.raises(InvalidArgument, match=f"{loss.__name__}.kl_weight"):
        loss(kl_weight=weight)


@pytest.mark.parametrize("flag", [0, 1, "false", None])
@pytest.mark.parametrize("loss", [Grpo, Dapo])
def test_scale_by_std_is_true_or_false(loss, flag) -> None:
    with pytest.raises(InvalidArgument, match=f"{loss.__name__}.scale_by_std must be True"):
        loss(scale_by_std=flag)


@pytest.mark.parametrize(
    "loss, data",
    [
        (CrossEntropy(), [datum()]),
        (Grpo(), [sample("a", 2), sample("b", 3), sample("a", 2), sample("b", 3), sample("b", 3)]),
        (Dapo(), [sample("a", 2), sample("a", 2)]),
    ],
)
def test_a_batch_carries_exactly_what_its_loss_reads(loss, data) -> None:
    check_batch(data, loss)


@pytest.mark.parametrize(
    "data, loss, says",
    [
        ([], CrossEntropy(), "at least one datum"),
        (
            [datum(LOGPROBS)],
            CrossEntropy(),
            "data[0] carries columns ['logprobs'], but CrossEntropy reads []",
        ),
        (
            [datum(group=("a", 2)), datum(group=("a", 2))],
            Grpo(),
            "data[0] carries columns [], but Grpo reads ['logprobs']",
        ),
        (
            [datum(group=("a", 1))],
            CrossEntropy(),
            "data[0] carries a reward and a group, but CrossEntropy compares no groups",
        ),
        (
            [sample("a", 2), datum(LOGPROBS)],
            Dapo(),
            "data[1] carries no reward and no group, but Dapo compares groups",
        ),
        (
            [sample("a", 2), sample("a", 3)],
            Grpo(),
            "data[1] declares group 'a' of 3 datums, but an earlier member of it declares 2",
        ),
        ([sample("a", 2), sample("b", 2), sample("b", 2)], Grpo(), "Groups ['a'] have [1] datums"),
        ([sample("a", 2)] * 3, Grpo(), "Groups ['a'] have [3] datums"),
        ([sample("a", 1)], Grpo(), "Groups ['a'] have [1] datums, but Grpo compares 2 or more"),
        ([datum(datum_id="a"), datum(datum_id="a")], CrossEntropy(), "repeats datum_id 'a'"),
        (["datum"], CrossEntropy(), "data[0] is not a Datum"),
        ([datum()], "cross_entropy", "Not a loss"),
    ],
)
def test_a_batch_is_refused(data, loss, says) -> None:
    with pytest.raises(InvalidArgument, match=re.escape(says)):
        check_batch(data, loss)


def test_datums_without_ids_never_collide() -> None:
    check_batch([datum(), datum()], CrossEntropy())
