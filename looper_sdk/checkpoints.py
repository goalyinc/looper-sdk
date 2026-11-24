"""Named checkpoint values returned by the registry."""

from __future__ import annotations

import dataclasses
import enum
from datetime import datetime
from typing import ClassVar

from looper_sdk.errors import InvalidArgument
from looper_sdk.values import ModelSpec, Version


class CheckpointState(enum.StrEnum):
    """Where a checkpoint is. A save succeeds once its step is
    on the run's disk, and its checkpoint is ``UPLOADING``; only a
    ``READY_TO_USE`` one is uploaded, named and loadable."""

    UPLOADING = "uploading"
    READY_TO_USE = "ready_to_use"
    FAILED = "failed"


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class Checkpoint:
    """A checkpoint observed at save or registry lookup time: a
    ``StateCheckpoint`` or a ``WeightsCheckpoint``, which ``kind`` names.

    Attributes:
        name: The name it binds once ``READY_TO_USE``; until then, the name its
            save or export asked for, which still resolves to its previous
            target.
        expires_at: Set once ``READY_TO_USE``, when its TTL starts. An export
            takes no TTL of its own but its source's ``expires_at``, copied at
            its first request; a retry after it failed keeps that copy.
        id: The service's name for it, which ``get_checkpoint`` and
            ``wait_checkpoint`` take.
        operation_id: The save operation that made it; None for a checkpoint
            no save made, such as an exported one.
        project: The project it belongs to.
        source_run_id: The run it was saved from.
        model_spec: Source run model name and exact trainable parameters.
        principal: Creating session's user or service identity; an export's
            is its source's.
        version: Exact source run, epoch and committed optimizer step.
        size_bytes: Physical directory size, known once ``READY_TO_USE``, and 0
            before; aliases do not duplicate bytes.
        created_at: Time of registry publication once ``READY_TO_USE``; before,
            the time its save succeeded or, for an export, the time it was
            requested, or last requested again after it failed. With timezone.
        state: Where it is; see ``CheckpointState``.
        failure: Why a ``FAILED`` checkpoint failed, for people; None otherwise.
        kind: ``state`` or ``weights``: which save made it, or ``weights`` for
            an export of a state checkpoint.
    """

    name: str
    expires_at: datetime | None
    id: str
    operation_id: str | None
    project: str
    principal: str
    source_run_id: str
    version: Version
    model_spec: ModelSpec
    size_bytes: int
    created_at: datetime
    state: CheckpointState
    failure: str | None = None
    kind: ClassVar[str]

    def __post_init__(self) -> None:
        if type(self) is Checkpoint:
            raise TypeError("Construct a StateCheckpoint or a WeightsCheckpoint")
        if not isinstance(self.model_spec, ModelSpec):
            raise InvalidArgument("Checkpoint.model_spec must be a ModelSpec")
        if not isinstance(self.state, CheckpointState):
            raise InvalidArgument("Checkpoint.state must be a CheckpointState")
        if (self.failure is not None) != (self.state is CheckpointState.FAILED):
            raise InvalidArgument("Checkpoint.failure is set exactly when its state is FAILED")


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class StateCheckpoint(Checkpoint):
    """Training state, from ``save_state``: the weights with the optimizer's
    state, which a sampler converts at launch."""

    kind: ClassVar[str] = "state"


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class WeightsCheckpoint(Checkpoint):
    """The weights alone, from ``save_weights`` or exported from a
    full-parameter ``StateCheckpoint``: Hugging Face safetensors, or PEFT's
    layout for LoRA, which a sampler loads as they are. It holds no
    optimizer state."""

    kind: ClassVar[str] = "weights"


CHECKPOINT_KINDS: dict[str, type[Checkpoint]] = {
    cls.kind: cls for cls in (StateCheckpoint, WeightsCheckpoint)
}
