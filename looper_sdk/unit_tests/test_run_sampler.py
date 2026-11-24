"""A run's sampler and ``publish_weights``, against a
scripted API: the sampler is created as a sampler of the session whose
source is the run, and a publish is an ordered operation of the run whose
result names the version its sampler then serves."""

from __future__ import annotations

import asyncio

import httpx
import pytest

from looper_sdk import Full, PublishResult, ServiceClient, TrainingConfig, wire
from looper_sdk.proto.v1 import operations_pb2, runs_pb2, sampling_pb2, sessions_pb2
from looper_sdk.transport import PROTOBUF
from looper_sdk.values import Version

CONFIG = TrainingConfig(model="fake/tiny", kind=Full(), tokens_per_step=64, max_seq_len=16)
PUBLISHED = PublishResult(version=Version(run_id="r", epoch=0, step=1), sampler_id="m")


@pytest.fixture(autouse=True)
def immediate_shutdown(monkeypatch):
    monkeypatch.setattr("looper_sdk.transport.CLOSE_GRACE_S", 0, raising=False)
    monkeypatch.setattr("looper_sdk.sampling_client.READINESS_POLL_S", 0.01)


def answer(message):
    return httpx.Response(
        200, content=message.SerializeToString(), headers={"content-type": PROTOBUF}
    )


class Api:
    """Scripted ``/v1`` answers: the active run ``r``, a sampler ``m`` that
    serves what it was created with, and publishes that succeed at their
    first await."""

    def __init__(self) -> None:
        self.calls: list[httpx.Request] = []
        self.created: sampling_pb2.CreateSamplerRequest | None = None
        self.publishes: list[runs_pb2.PublishWeightsRequest] = []

    def sampler(self, state) -> sampling_pb2.Sampler:
        assert self.created is not None
        return sampling_pb2.Sampler(
            id="m", config=self.created.config, state=state, model="fake/tiny"
        )

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        self.calls.append(request)
        path = request.url.path
        if path == "/v1/sessions":
            return answer(
                sessions_pb2.OpenSessionResponse(
                    session=sessions_pb2.Session(id="s", project="p", heartbeat_interval_s=3600)
                )
            )
        if path.endswith(":close"):
            return answer(sessions_pb2.CloseSessionResponse())
        run = runs_pb2.Run(id="r", writer_session_id="s", state=runs_pb2.Run.ACTIVE)
        if path == "/v1/runs":
            return answer(runs_pb2.CreateRunResponse(run=run))
        if path == "/v1/runs/r":
            return answer(runs_pb2.GetRunResponse(run=run))
        if path == "/v1/samplers":
            self.created = sampling_pb2.CreateSamplerRequest.FromString(request.content)
            starting = self.sampler(sampling_pb2.Sampler.STARTING)
            return answer(sampling_pb2.CreateSamplerResponse(sampler=starting))
        if path == "/v1/samplers/m":
            ready = self.sampler(sampling_pb2.Sampler.READY)
            return answer(sampling_pb2.GetSamplerResponse(sampler=ready))
        if path == "/v1/runs/r:publishWeights":
            body = runs_pb2.PublishWeightsRequest.FromString(request.content)
            self.publishes.append(body)
            operation = operations_pb2.Operation(
                id=f"op-{len(self.publishes)}",
                run_id="r",
                seq=len(self.publishes),
                key=body.submission.key,
                epoch=body.submission.epoch,
                kind=operations_pb2.Operation.PUBLISH_WEIGHTS,
                state=operations_pb2.Operation.QUEUED,
            )
            return answer(runs_pb2.PublishWeightsResponse(operation=operation))
        if path.endswith(":await"):
            operation = operations_pb2.Operation(
                id=path.removeprefix("/v1/operations/").removesuffix(":await"),
                kind=operations_pb2.Operation.PUBLISH_WEIGHTS,
                state=operations_pb2.Operation.SUCCEEDED,
                publish_weights=wire.publish_result_to_proto(PUBLISHED),
            )
            return answer(operations_pb2.AwaitOperationResponse(operation=operation))
        raise AssertionError(f"unscripted {request.method} {path}")


async def connect(api: Api) -> ServiceClient:
    return await ServiceClient.Config(base_url="http://looper", project="p").build(
        ServiceClient.Runtime(transport=httpx.MockTransport(api))
    )


def test_a_run_s_sampler_is_a_sampler_of_the_session_whose_source_is_the_run() -> None:
    async def run() -> None:
        api = Api()
        async with await connect(api) as service:
            training_client = await service.create_training_client(CONFIG)
            sampling_client = await training_client.create_sampling_client(
                max_seq_len=64, max_concurrent_sequences=4, key="c1"
            )
            assert (sampling_client.sampler_id, sampling_client.checkpoint_name) == ("m", "run:r")
        assert api.created is not None
        assert (api.created.session_id, api.created.key) == ("s", "c1")
        asked = wire.sampler_config_from_proto(api.created.config)
        assert (asked.run_id, asked.max_seq_len, asked.max_concurrent_sequences) == ("r", 64, 4)

    asyncio.run(run())


def test_a_publish_is_an_ordered_operation_whose_result_names_the_version_served() -> None:
    async def run() -> None:
        api = Api()
        async with await connect(api) as service:
            training_client = await service.create_training_client(CONFIG)
            first = await training_client.publish_weights(key="k1")
            second = await training_client.publish_weights(key="k2", after=[first])
            assert (second.kind, second.key, second.seq) == ("publish_weights", "k2", 2)
            assert await second.result() == PUBLISHED
        sent = [(body.run_id, body.submission) for body in api.publishes]
        assert [(run_id, s.session_id, s.key, list(s.after)) for run_id, s in sent] == [
            ("r", "s", "k1", []),
            ("r", "s", "k2", ["op-1"]),
        ]

    asyncio.run(run())
