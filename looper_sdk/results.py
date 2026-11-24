"""What an operation computed, once it has run:
frozen values, as the service reported them."""

from __future__ import annotations

import dataclasses
import types
from collections.abc import Mapping, Sequence

from looper_sdk.values import Version


def _frozen(metrics: Mapping[str, float]) -> Mapping[str, float]:
    return types.MappingProxyType({name: float(value) for name, value in metrics.items()})


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class DatumResult:
    """One datum's results, in the order of the call's data.

    Attributes:
        datum_id: The datum's own, echoed.
    """

    datum_id: str | None


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class ForwardBackwardResult:
    """What a forward_backward counted from its request. The loss and its
    metrics come with the step.

    Attributes:
        weighted_targets: The targets, the tokens of weight 1.
        data: One entry per datum.
        num_tokens: Every token of every datum, the work billed as trained;
            None in a result recorded before Looper counted it.
    """

    weighted_targets: int
    data: Sequence[DatumResult]
    num_tokens: int | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "data", tuple(self.data))


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class StepResult:
    """Both halves of an optim_step's reduction.

    Attributes:
        step: Completed optimizer steps, this one included.
        version: The weights it made.
        loss_sum: The losses accumulated since the last step, summed.
        denominator: What the gradients were divided by.
        weighted_targets: The targets with nonzero weight accumulated.
        weight_sum: The targets' weights accumulated, summed; with weights
            of 0 or 1, the same count as ``weighted_targets``.
        metrics: The loss's own diagnostics, by name, as the trainer
            reported them.
        grad_norm: The global gradient norm before clipping.
        clipped: Whether clipping scaled the gradients.
        lr: The learning rate the step applied.
        num_tokens: The ``num_tokens`` of the forward_backwards whose
            gradients the step applied, summed; None when one of them has
            none.
    """

    step: int
    version: Version
    loss_sum: float
    denominator: float
    weighted_targets: int
    weight_sum: float
    metrics: Mapping[str, float]
    grad_norm: float
    clipped: bool
    lr: float
    num_tokens: int | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "metrics", _frozen(self.metrics))


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class PublishResult:
    """What a publish_weights moved.

    Attributes:
        version: The version the run's sampler now serves.
        sampler_id: The sampler it reached.
    """

    version: Version
    sampler_id: str
