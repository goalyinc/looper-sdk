"""The optimizer a run trains with, chosen once in its ``TrainingConfig``.

Each is Looper's own frozen value: a kind, its settings and its clipping,
never a schedule nor the parameters it updates: a Muon's are its profile's.
Looper builds the run's trainer with it, so the trainer owns the optimizer,
and every ``optim_step`` steps it at its ``learning_rate``, unless the step
names its own (``check_learning_rate``).
"""

from __future__ import annotations

import dataclasses

from looper_sdk._checks import _count, _finite, _flag
from looper_sdk.errors import InvalidArgument


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class AdamW:
    """AdamW over every trainable parameter: fp32 moments, bias correction
    counted over the run's own steps.

    Attributes:
        learning_rate: The rate every step trains at, above 0.
        betas: The first and second moments' decay rates, each in [0, 1).
        eps: Added to the second moment's root, above 0.
        weight_decay: Decoupled weight decay, at least 0.
        max_grad_norm: The global gradient norm a step clips to before the
            update, above 0. None measures the norm without clipping.
    """

    learning_rate: float
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.0
    max_grad_norm: float | None = None

    def __post_init__(self) -> None:
        _check_shared(self, "AdamW")


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class Muon:
    """Muon over the model's matrices, and AdamW over every other parameter,
    at one learning rate and one weight decay: each matrix's update is
    orthogonalized and scaled to AdamW's update size. Its profile names the
    matrices; one that names none refuses Muon when the run is created.

    Attributes:
        learning_rate: Both halves' rate, above 0, every step trains at.
        weight_decay: Both halves' decoupled weight decay, at least 0.
        momentum: The decay rate of Muon's momentum, in [0, 1).
        nesterov: Whether Muon steps along its momentum's Nesterov
            look-ahead rather than the momentum itself.
        ns_steps: Newton–Schulz iterations orthogonalizing each update, at
            least 1.
        betas: The AdamW half's moment decay rates, each in [0, 1).
        eps: Added to the AdamW half's second-moment root, above 0.
        max_grad_norm: The global gradient norm a step clips to before the
            update, above 0. None measures the norm without clipping.
    """

    learning_rate: float
    weight_decay: float = 0.1
    momentum: float = 0.95
    nesterov: bool = True
    ns_steps: int = 5
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    max_grad_norm: float | None = None

    def __post_init__(self) -> None:
        _check_shared(self, "Muon")
        momentum = _finite("Muon.momentum", self.momentum)
        if not 0 <= momentum < 1:
            raise InvalidArgument(f"Muon.momentum must be in [0, 1), not {momentum}")
        object.__setattr__(self, "momentum", momentum)
        object.__setattr__(self, "nesterov", _flag("Muon.nesterov", self.nesterov))
        object.__setattr__(self, "ns_steps", _count("Muon.ns_steps", self.ns_steps, minimum=1))


# Every optimizer a run may name.
Optimizer = AdamW | Muon


def _check_shared(optimizer: Optimizer, kind: str) -> None:
    """Checks, and stores as floats, the settings every kind has."""
    rate = _finite(f"{kind}.learning_rate", optimizer.learning_rate)
    if rate <= 0:
        raise InvalidArgument(f"{kind}.learning_rate must be above 0, not {rate}")
    given = optimizer.betas
    try:
        betas = tuple(_finite(f"{kind}.betas", beta) for beta in given)
    except TypeError:
        raise InvalidArgument(f"{kind}.betas must be two numbers, not {given!r}") from None
    if len(betas) != 2 or not all(0 <= beta < 1 for beta in betas):
        raise InvalidArgument(f"{kind}.betas must be two values in [0, 1), not {given!r}")
    eps = _finite(f"{kind}.eps", optimizer.eps)
    if eps <= 0:
        raise InvalidArgument(f"{kind}.eps must be above 0, not {eps}")
    decay = _finite(f"{kind}.weight_decay", optimizer.weight_decay)
    if decay < 0:
        raise InvalidArgument(f"{kind}.weight_decay must be at least 0, not {decay}")
    clip = optimizer.max_grad_norm
    if clip is not None:
        clip = _finite(f"{kind}.max_grad_norm", clip)
        if clip <= 0:
            raise InvalidArgument(f"{kind}.max_grad_norm must be above 0, not {clip}")
    object.__setattr__(optimizer, "learning_rate", rate)
    object.__setattr__(optimizer, "betas", betas)
    object.__setattr__(optimizer, "eps", eps)
    object.__setattr__(optimizer, "weight_decay", decay)
    object.__setattr__(optimizer, "max_grad_norm", clip)


def check_learning_rate(learning_rate: object) -> float:
    """The learning rate one ``optim_step`` names, in place of its
    optimizer's: finite and at least 0."""
    rate = _finite("learning_rate", learning_rate)
    if rate < 0:
        raise InvalidArgument(f"learning_rate must be at least 0, not {rate}")
    return rate
