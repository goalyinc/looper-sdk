"""The standalone sampler's frozen values and its new error codes."""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime

import pytest

from looper_sdk import wire
from looper_sdk.errors import ERRORS, CheckpointNotReady, ErrorCode, InvalidArgument
from looper_sdk.sampling import SamplerConfig, SamplerState, SamplerStatus, SamplingUsage
from looper_sdk.values import Version

CONFIG = SamplerConfig(checkpoint_name="after-sft", max_seq_len=4096, max_concurrent_sequences=4)


def test_a_config_is_frozen_and_compares_by_value() -> None:
    assert CONFIG == SamplerConfig(
        checkpoint_name="after-sft", max_seq_len=4096, max_concurrent_sequences=4
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        CONFIG.max_seq_len = 8  # type: ignore[misc]


@pytest.mark.parametrize(
    "field,value",
    [
        ("checkpoint_name", ""),
        ("checkpoint_name", "a\x00b"),
        ("checkpoint_name", "x" * 257),
        ("checkpoint_name", "base:"),
        ("checkpoint_name", 7),
        ("max_seq_len", 0),
        ("max_seq_len", True),
        ("max_seq_len", 1.5),
        ("max_concurrent_sequences", 0),
        ("max_concurrent_sequences", False),
    ],
)
def test_a_config_refuses_what_no_sampler_can_serve(field: str, value: object) -> None:
    with pytest.raises(InvalidArgument):
        dataclasses.replace(CONFIG, **{field: value})


def test_a_base_reference_names_a_catalog_model() -> None:
    config = dataclasses.replace(CONFIG, checkpoint_name="base:Qwen/Qwen3.5-9B")
    assert config.base_model == "Qwen/Qwen3.5-9B"
    assert CONFIG.base_model is None


def test_a_run_reference_names_a_run_whose_sampler_has_a_version_once_published() -> None:
    config = dataclasses.replace(CONFIG, checkpoint_name="run:6b3c3a52")
    assert (config.run_id, config.base_model, CONFIG.run_id) == ("6b3c3a52", None, None)
    with pytest.raises(InvalidArgument, match="run:<run_id>"):
        dataclasses.replace(CONFIG, checkpoint_name="run:")
    # None before the run's first publish, then the version last published.
    for version in (None, Version(run_id="6b3c3a52", epoch=0, step=3)):
        status = SamplerStatus(
            id="s-1",
            config=config,
            state=SamplerState.READY,
            model="Qwen/Qwen3.5-9B",
            version=version,
            created_at=datetime(2026, 11, 10, tzinfo=UTC),
        )
        assert wire.sampler_status_from_proto(wire.sampler_status_to_proto(status)) == status


def test_a_status_names_its_source_and_only_a_trained_one_has_a_version() -> None:
    status = SamplerStatus(
        id="s-1",
        config=CONFIG,
        state=SamplerState.READY,
        model="Qwen/Qwen3.5-9B",
        version=Version(run_id="r", epoch=0, step=40),
        created_at=datetime(2026, 11, 5, tzinfo=UTC),
    )
    assert status.error is None
    base = dataclasses.replace(CONFIG, checkpoint_name="base:Qwen/Qwen3.5-9B")
    with pytest.raises(InvalidArgument, match="version"):
        dataclasses.replace(status, config=base)
    with pytest.raises(InvalidArgument, match="version"):
        dataclasses.replace(status, version=None)
    assert dataclasses.replace(status, config=base, version=None).version is None


def test_checkpoint_not_ready_is_registered_and_never_retried_automatically() -> None:
    error = CheckpointNotReady("after-sft is still uploading")
    assert ERRORS[ErrorCode.CHECKPOINT_NOT_READY] is CheckpointNotReady
    assert error.retryable is False
    decoded = wire.error_from_proto(wire.error_to_proto(error))
    assert isinstance(decoded, CheckpointNotReady)
    assert (decoded.message, decoded.retryable) == (error.message, False)


# Sampling requests and results.

from looper_sdk.errors import QueueTimeout, SamplerFailed  # noqa: E402
from looper_sdk.sampling import (  # noqa: E402
    FinishReason,
    SampleRequest,
    SampleResult,
    SampleSequence,
    SamplingParams,
)

PARAMS = SamplingParams(max_tokens=16, temperature=0.7, top_p=0.9, stop_token_ids=(2,))


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_tokens", 0),
        ("max_tokens", True),
        ("temperature", -0.1),
        ("temperature", float("nan")),
        ("top_p", 0.0),
        ("top_p", 1.5),
        ("stop_token_ids", (-1,)),
    ],
)
def test_params_refuse_what_no_engine_takes(field: str, value: object) -> None:
    with pytest.raises(InvalidArgument):
        dataclasses.replace(PARAMS, **{field: value})


def test_a_request_must_fit_its_sampler() -> None:
    request = SampleRequest(prompt=(1, 2, 3), params=PARAMS, n=4)
    request.check(CONFIG)  # 3 + 16 <= 4096 and n = 4 <= 4
    with pytest.raises(InvalidArgument, match="max_concurrent_sequences"):
        dataclasses.replace(request, n=5).check(CONFIG)
    with pytest.raises(InvalidArgument, match="max_seq_len"):
        dataclasses.replace(request, params=dataclasses.replace(PARAMS, max_tokens=4094)).check(
            CONFIG
        )
    for bad in ({"prompt": ()}, {"prompt": (1, "x")}, {"n": 0}, {"n": False}):
        with pytest.raises(InvalidArgument):
            dataclasses.replace(request, **bad)


def test_a_sequence_has_one_finite_logprob_per_token() -> None:
    SampleSequence(token_ids=(5, 6), logprobs=(-0.1, -0.2), finish_reason=FinishReason.LENGTH)
    with pytest.raises(InvalidArgument, match="one logprob per token"):
        SampleSequence(token_ids=(5, 6), logprobs=(-0.1,), finish_reason=FinishReason.STOP)
    with pytest.raises(InvalidArgument, match="finite"):
        SampleSequence(token_ids=(5,), logprobs=(float("-inf"),), finish_reason=FinishReason.STOP)


def test_requests_and_results_round_trip_through_the_wire() -> None:
    request = SampleRequest(prompt=(1, 2, 3), params=PARAMS, n=2)
    assert wire.sample_request_from_proto(wire.sample_request_to_proto(request)) == request
    sequence = SampleSequence(
        token_ids=(7, 8), logprobs=(-0.5, -0.25), finish_reason=FinishReason.STOP, cached_tokens=1
    )
    for version in (Version(run_id="r", epoch=0, step=3), None):
        result = SampleResult(
            sequences=(sequence, sequence),
            model="Qwen/Qwen3.5-9B",
            checkpoint_name="after-sft",
            version=version,
        )
        assert wire.sample_result_from_proto(wire.sample_result_to_proto(result)) == result
    # A run's sampler: each sequence names the version that generated it.
    older, newer = (Version(run_id="r", epoch=1, step=step) for step in (3, 4))
    result = SampleResult(
        sequences=(
            dataclasses.replace(sequence, version=older),
            dataclasses.replace(sequence, version=newer),
        ),
        model="Qwen/Qwen3.5-9B",
        checkpoint_name="run:r",
    )
    assert wire.sample_result_from_proto(wire.sample_result_to_proto(result)) == result


USAGE = SamplingUsage(num_prompt_tokens=10, num_generated_tokens=64, num_cached_tokens=4)


@pytest.mark.parametrize(
    "counts",
    [
        {"num_prompt_tokens": -1},
        {"num_generated_tokens": True},
        {"num_cached_tokens": 1.5},
        {"num_cached_tokens": 11},  # more than the prompt holds
    ],
)
def test_usage_refuses_counts_no_request_has(counts: dict) -> None:
    with pytest.raises(InvalidArgument):
        dataclasses.replace(USAGE, **counts)


def test_usage_round_trips_alone_and_on_a_status() -> None:
    assert wire.sampling_usage_from_proto(wire.sampling_usage_to_proto(USAGE)) == USAGE
    status = SamplerStatus(
        id="s-1",
        config=CONFIG,
        state=SamplerState.STOPPED,
        model="Qwen/Qwen3.5-9B",
        version=Version(run_id="r", epoch=0, step=40),
        created_at=datetime(2026, 11, 5, tzinfo=UTC),
        usage=USAGE,
    )
    assert wire.sampler_status_from_proto(wire.sampler_status_to_proto(status)) == status
    # A server that predates the count sends none: unknown, not zero.
    unknown = dataclasses.replace(status, usage=None)
    assert wire.sampler_status_from_proto(wire.sampler_status_to_proto(unknown)).usage is None


def test_a_waiting_reason_round_trips_and_one_this_sdk_does_not_know_reads_as_none() -> None:
    status = SamplerStatus(
        id="s-1",
        config=CONFIG,
        state=SamplerState.STARTING,
        model="Qwen/Qwen3.5-9B",
        version=Version(run_id="r", epoch=0, step=40),
        created_at=datetime(2026, 11, 5, tzinfo=UTC),
        waiting_reason="org_gpu_limit",
    )
    assert wire.sampler_status_from_proto(wire.sampler_status_to_proto(status)) == status
    # A reason a newer service names reads as none, not as an error.
    assert wire.waiting_reason_from_proto(2) is None


@pytest.mark.parametrize("error", [QueueTimeout, SamplerFailed])
def test_terminal_sampling_errors_are_registered_and_not_retried(error: type) -> None:
    raised = error("it ended")
    assert ERRORS[raised.code] is error and raised.retryable is False
    decoded = wire.error_from_proto(wire.error_to_proto(raised))
    assert type(decoded) is error and decoded.retryable is False
