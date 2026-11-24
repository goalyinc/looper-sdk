"""The losses, chosen per call.

Each is a frozen value: a loss's settings, defaults and refusals, and none
of its math, which runs on Looper's trainer. The trainer runs the whole
algorithm, forming the advantages of a loss that compares groups from the
datums' rewards. Each is a sum over targets, the tokens of weight 1. With
r = exp(log π_θ(y) − q), q the ``logprobs`` column, and A a datum's reward
less its group's mean, over the group's sample standard deviation unless
``scale_by_std`` is False:

| Loss | Objective, summed over targets |
|---|---|
| ``CrossEntropy`` | −Σ log π_θ(y) |
| ``Grpo`` | −Σ min(r·A, clip(r, 1 − clip_low, 1 + clip_high)·A) + kl_weight·Σ(r − 1 − log r) |
| ``Dapo`` | as ``Grpo``, with DAPO's clip-higher |

A call's data must fit its loss as the trainer requires (``check_batch``).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import ClassVar

from looper_sdk._checks import _finite, _flag
from looper_sdk.errors import InvalidArgument
from looper_sdk.values import Datum


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class CrossEntropy:
    """SFT: −Σ log π_θ(y), the targets imitated. Reads no column and compares
    no groups.

    Attributes:
        reads: The columns its datums carry.
        min_group_size: None: it compares no groups, so its datums carry no
            reward and no group.
    """

    reads: ClassVar[frozenset[str]] = frozenset()
    min_group_size: ClassVar[int | None] = None


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class Grpo:
    """GRPO: each datum's reward normalized within its group, trained with
    PPO's clipped surrogate, and with ``kl_weight`` above 0 a penalty on the
    policy drifting from the sampler. ``clip_high > clip_low`` is DAPO's
    clip-higher.

    Attributes:
        clip_low: The ratio's floor, as an offset below 1: in (0, 1).
        clip_high: The ratio's ceiling, as an offset above 1: above 0.
        kl_weight: The penalty's weight, 0 or above: the k3 estimate of
            KL(π_sampler ‖ π_θ), Σ(r − 1 − log r), added to the surrogate.
            0, the default, adds none.
        scale_by_std: Whether a datum's advantage, its reward less its
            group's mean, is divided by the group's standard deviation.
            False leaves it undivided, as Dr. GRPO does.
        reads: The columns its datums carry: the sampler's ``logprobs``. The
            trainer forms the advantages, so a datum never carries them.
        min_group_size: The fewest datums a group may have: a mean needs
            something to compare against.
    """

    reads: ClassVar[frozenset[str]] = frozenset({"logprobs"})
    min_group_size: ClassVar[int | None] = 2

    clip_low: float = 0.2
    clip_high: float = 0.2
    kl_weight: float = 0.0
    scale_by_std: bool = True

    def __post_init__(self) -> None:
        _flag(f"{type(self).__name__}.scale_by_std", self.scale_by_std)
        low = _finite(f"{type(self).__name__}.clip_low", self.clip_low)
        high = _finite(f"{type(self).__name__}.clip_high", self.clip_high)
        weight = _finite(f"{type(self).__name__}.kl_weight", self.kl_weight)
        if not 0 < low < 1:
            raise InvalidArgument(f"{type(self).__name__}.clip_low must be in (0, 1), not {low}")
        if high <= 0:
            raise InvalidArgument(f"{type(self).__name__}.clip_high must be above 0, not {high}")
        if weight < 0:
            raise InvalidArgument(
                f"{type(self).__name__}.kl_weight must be 0 or above, not {weight}"
            )
        object.__setattr__(self, "clip_low", low)
        object.__setattr__(self, "clip_high", high)
        object.__setattr__(self, "kl_weight", weight)


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class Dapo(Grpo):
    """DAPO: GRPO's advantages and loss with the clip's upper bound at 0.28."""

    clip_high: float = 0.28


Loss = CrossEntropy | Grpo | Dapo


def check_batch(data: Sequence[Datum], loss: Loss) -> None:
    """Refuses a call's data that its loss cannot train, as the trainer's
    rank 0 would: no data; columns other than those the loss
    reads; a reward and a group on a datum of a loss that compares none, or
    none on one of a loss that does; a group whose members declare different
    sizes, that is not whole within the call, or that is smaller than the
    loss compares; or a ``datum_id`` given twice."""
    if not isinstance(loss, CrossEntropy | Grpo):
        raise InvalidArgument(f"Not a loss: {loss!r}")
    if not data:
        raise InvalidArgument("A call needs at least one datum")
    name = type(loss).__name__
    seen: set[str] = set()
    counts: dict[str, int] = {}
    declared: dict[str, int] = {}
    for i, datum in enumerate(data):
        if not isinstance(datum, Datum):
            raise InvalidArgument(f"data[{i}] is not a Datum: {datum!r}")
        if (columns := frozenset(datum.columns)) != loss.reads:
            raise InvalidArgument(
                f"data[{i}] carries columns {sorted(columns)}, but {name} reads "
                f"{sorted(loss.reads)}"
            )
        if (datum.group_id is None) != (loss.min_group_size is None):
            carried = "no reward and no group" if datum.group_id is None else "a reward and a group"
            compares = "compares no groups" if loss.min_group_size is None else "compares groups"
            raise InvalidArgument(f"data[{i}] carries {carried}, but {name} {compares}")
        if datum.group_id is not None:
            assert datum.group_size is not None, "a group comes with its size"
            size = declared.setdefault(datum.group_id, datum.group_size)
            if datum.group_size != size:
                raise InvalidArgument(
                    f"data[{i}] declares group {datum.group_id!r} of {datum.group_size} datums, "
                    f"but an earlier member of it declares {size}"
                )
            counts[datum.group_id] = counts.get(datum.group_id, 0) + 1
        if datum.datum_id is not None:
            if datum.datum_id in seen:
                raise InvalidArgument(f"data[{i}] repeats datum_id {datum.datum_id!r}")
            seen.add(datum.datum_id)
    # The trainer compares a group only within one call, so a part of one
    # would be compared against the wrong datums.
    if partial := sorted(group for group, n in counts.items() if n != declared[group]):
        raise InvalidArgument(
            f"Groups {partial} have {[counts[g] for g in partial]} datums in this call where "
            f"their members declare {[declared[g] for g in partial]}: send each group whole, "
            "within one forward_backward"
        )
    minimum = loss.min_group_size
    if minimum is not None and (small := sorted(g for g, n in declared.items() if n < minimum)):
        raise InvalidArgument(
            f"Groups {small} have {[declared[g] for g in small]} datums, but {name} compares "
            f"{minimum} or more"
        )
