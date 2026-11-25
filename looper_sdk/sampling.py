"""A sampler's values: what it
serves and how much, and what the service last observed of it.

A sampler serves one source, fixed at creation: a project checkpoint by its
name, a catalog base model as ``base:<model>``, or a run of its own session
as ``run:<run_id>``, whose weights it serves as the run publishes them. Its
context length and virtual concurrency are fixed for its life too.
"""

from __future__ import annotations

import dataclasses
import enum
import math
from collections.abc import Sequence
from datetime import datetime
from typing import Literal

from looper_sdk._checks import _count, _finite
from looper_sdk.errors import InvalidArgument, LooperError
from looper_sdk.values import Version, _floats, _token_ids

BASE_PREFIX = "base:"
RUN_PREFIX = "run:"


class SamplerState(enum.StrEnum):
    """Where a sampler is. ``failed`` and ``stopped`` are
    final: a failed sampler is never relaunched; create another."""

    STARTING = "starting"
    READY = "ready"
    FAILED = "failed"
    STOPPING = "stopping"
    STOPPED = "stopped"


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class SamplerConfig:
    """What a sampler serves, and how much.

    Attributes:
        checkpoint_name: A published checkpoint of the session's project, by
            its name, or ``base:<model>`` for a catalog model's base weights.
            Resolved once, when the sampler is created: a later publication
            under the same name does not change what it serves. Or
            ``run:<run_id>`` for a run of the session, as
            ``TrainingClient.create_sampling_client`` writes it: the sampler
            serves each version the run publishes to it.
        max_seq_len: The longest prompt plus completion one sample takes, in
            tokens.
        max_concurrent_sequences: The sequences the sampler generates at
            once; a request for ``n`` completions takes ``n`` of them.
    """

    checkpoint_name: str
    max_seq_len: int
    max_concurrent_sequences: int

    def __post_init__(self) -> None:
        name = self.checkpoint_name
        if (
            not isinstance(name, str)
            or not 1 <= len(name) <= 256
            or "\x00" in name
            or name in (BASE_PREFIX, RUN_PREFIX)
        ):
            raise InvalidArgument(
                "SamplerConfig.checkpoint_name is a checkpoint's name, 1 to 256 characters "
                f"with no NUL, or base:<model>, or run:<run_id>, not {name!r}"
            )
        longest = _count("SamplerConfig.max_seq_len", self.max_seq_len, minimum=1)
        concurrent = _count(
            "SamplerConfig.max_concurrent_sequences", self.max_concurrent_sequences, minimum=1
        )
        object.__setattr__(self, "max_seq_len", longest)
        object.__setattr__(self, "max_concurrent_sequences", concurrent)

    @property
    def base_model(self) -> str | None:
        """The catalog model a ``base:`` reference names; None for a checkpoint."""
        if self.checkpoint_name.startswith(BASE_PREFIX):
            return self.checkpoint_name.removeprefix(BASE_PREFIX)
        return None

    @property
    def run_id(self) -> str | None:
        """The run a ``run:`` reference names; None for any other source."""
        if self.checkpoint_name.startswith(RUN_PREFIX):
            return self.checkpoint_name.removeprefix(RUN_PREFIX)
        return None


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class SamplingUsage:
    """The tokens of a request's engine work, or of a sampler's requests
    summed.

    Attributes:
        num_prompt_tokens: The prompt's tokens, counted once per request
            whatever its ``n``: the engine computes a prompt once and its
            ``n`` sequences share it.
        num_generated_tokens: Every token generated, over all ``n``
            sequences.
        num_cached_tokens: The prompt tokens the engine's prefix cache
            served, the most any of the request's sequences reported; at
            most ``num_prompt_tokens``. For ``n`` above 1 the engine first
            computes the prompt once, then reports it cached for each copy,
            so this can count prompt tokens that request computed.
    """

    num_prompt_tokens: int
    num_generated_tokens: int
    num_cached_tokens: int

    def __post_init__(self) -> None:
        for name in ("num_prompt_tokens", "num_generated_tokens", "num_cached_tokens"):
            value = _count(f"SamplingUsage.{name}", getattr(self, name), minimum=0)
            object.__setattr__(self, name, value)
        if self.num_cached_tokens > self.num_prompt_tokens:
            raise InvalidArgument(
                f"SamplingUsage.num_cached_tokens ({self.num_cached_tokens}) is more than "
                f"its num_prompt_tokens ({self.num_prompt_tokens})"
            )


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class SamplerStatus:
    """What the service last observed of a sampler: not proof that it is
    alive now.

    Attributes:
        id: The service's name for it.
        config: What it was created with.
        state: Where it is.
        model: The catalog model it serves, a checkpoint's base included.
        version: The trained weights it serves, as its checkpoint recorded
            them when the sampler was created; None for a base reference.
            A run's sampler: the version its run last published to it, and
            None before the first.
        created_at: When the service accepted it.
        error: Why it failed, once it has; None otherwise.
        usage: Its succeeded requests' usage, summed when the status was
            read; None when one of them was recorded before Looper counted
            tokens.
        waiting_reason: Why a starting sampler has not started: "org_gpu_limit"
            while its organization's GPUs in use leave no room for it; None
            otherwise, and always None for a run's sampler, which waits for
            its run.
    """

    id: str
    config: SamplerConfig
    state: SamplerState
    model: str
    version: Version | None
    created_at: datetime
    error: LooperError | None = dataclasses.field(default=None, compare=False)
    usage: SamplingUsage | None = None
    waiting_reason: Literal["org_gpu_limit"] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.state, SamplerState):
            raise InvalidArgument(f"SamplerStatus.state must be a SamplerState, not {self.state!r}")
        if self.config.run_id is not None:
            return  # as its run publishes: none yet, or the last
        if (self.version is None) != (self.config.base_model is not None):
            raise InvalidArgument(
                "SamplerStatus.version names a checkpoint's trained weights, and a base "
                "reference has none"
            )


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class SamplingParams:
    """How each completion is drawn.

    Attributes:
        max_tokens: The most tokens each completion generates.
        temperature: 0 is greedy.
        top_p: Nucleus sampling's mass, in (0, 1].
        stop_token_ids: Token ids that end a completion.
    """

    max_tokens: int
    temperature: float = 1.0
    top_p: float = 1.0
    stop_token_ids: Sequence[int] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "max_tokens", _count("SamplingParams.max_tokens", self.max_tokens, minimum=1)
        )
        temperature = _finite("SamplingParams.temperature", self.temperature)
        if temperature < 0:
            raise InvalidArgument(f"SamplingParams.temperature must be >= 0, not {temperature}")
        top_p = _finite("SamplingParams.top_p", self.top_p)
        if not 0 < top_p <= 1:
            raise InvalidArgument(f"SamplingParams.top_p must be in (0, 1], not {top_p}")
        object.__setattr__(self, "temperature", temperature)
        object.__setattr__(self, "top_p", top_p)
        object.__setattr__(
            self, "stop_token_ids", _token_ids("SamplingParams.stop_token_ids", self.stop_token_ids)
        )


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class SampleRequest:
    """``n`` completions of one prompt: what ``SamplingClient.sample`` sends.

    Attributes:
        prompt: Token ids, never text; at least one.
        params: How each completion is drawn.
        n: Completions wanted; each takes one of the sampler's
            ``max_concurrent_sequences`` while it generates.
    """

    prompt: Sequence[int]
    params: SamplingParams
    n: int = 1

    def __post_init__(self) -> None:
        prompt = _token_ids("SampleRequest.prompt", self.prompt)
        if not prompt:
            raise InvalidArgument("SampleRequest.prompt is empty")
        if not isinstance(self.params, SamplingParams):
            raise InvalidArgument("SampleRequest.params must be a SamplingParams")
        object.__setattr__(self, "prompt", prompt)
        object.__setattr__(self, "n", _count("SampleRequest.n", self.n, minimum=1))

    def check(self, config: SamplerConfig) -> None:
        """InvalidArgument unless the sampler ``config`` describes can serve
        it: an ``n`` it could never fit would wait forever."""
        if self.n > config.max_concurrent_sequences:
            raise InvalidArgument(
                f"n={self.n} exceeds the sampler's max_concurrent_sequences "
                f"({config.max_concurrent_sequences})"
            )
        longest = len(self.prompt) + self.params.max_tokens
        if longest > config.max_seq_len:
            raise InvalidArgument(
                f"the prompt's {len(self.prompt)} tokens and max_tokens "
                f"{self.params.max_tokens} exceed the sampler's max_seq_len ({config.max_seq_len})"
            )


class FinishReason(enum.StrEnum):
    """Why a completion ended: ``STOP`` at a stop token or the model's end
    of text, ``LENGTH`` at ``max_tokens``."""

    STOP = "stop"
    LENGTH = "length"


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class SampleSequence:
    """One completion.

    Attributes:
        token_ids: The tokens generated.
        logprobs: The engine's log probability of each chosen token, one per
            token; not a normalized distribution over candidates.
        finish_reason: Why it ended.
        cached_tokens: Prompt tokens the engine's prefix cache served.
        version: The trained weights that generated it. From a run's
            sampler, the oldest version that generated one of its tokens: a
            publish during the generation lands under it.
            None for a base reference.
    """

    token_ids: Sequence[int]
    logprobs: Sequence[float]
    finish_reason: FinishReason
    cached_tokens: int = 0
    version: Version | None = None

    def __post_init__(self) -> None:
        tokens = _token_ids("SampleSequence.token_ids", self.token_ids)
        logprobs = _floats("SampleSequence.logprobs", self.logprobs)
        if len(logprobs) != len(tokens):
            raise InvalidArgument(
                f"a sequence has one logprob per token: {len(tokens)} tokens, "
                f"{len(logprobs)} logprobs"
            )
        if not all(math.isfinite(value) for value in logprobs):
            raise InvalidArgument("a sequence's logprobs must be finite")
        if not isinstance(self.finish_reason, FinishReason):
            raise InvalidArgument(f"not a finish reason: {self.finish_reason!r}")
        object.__setattr__(self, "token_ids", tokens)
        object.__setattr__(self, "logprobs", logprobs)
        object.__setattr__(
            self,
            "cached_tokens",
            _count("SampleSequence.cached_tokens", self.cached_tokens, minimum=0),
        )


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class SampleResult:
    """A request's completions, ``n`` of them, and what generated them.

    Attributes:
        sequences: The completions, one per ``n``.
        model: The catalog model the sampler serves.
        checkpoint_name: As the sampler was created with it.
        version: The trained weights it serves; None for a base reference,
            and for a run's sampler, whose sequences each name their own.
    """

    sequences: Sequence[SampleSequence]
    model: str
    checkpoint_name: str
    version: Version | None = None

    def __post_init__(self) -> None:
        sequences = tuple(self.sequences)
        if not sequences or not all(isinstance(s, SampleSequence) for s in sequences):
            raise InvalidArgument("a result holds at least one SampleSequence")
        object.__setattr__(self, "sequences", sequences)
