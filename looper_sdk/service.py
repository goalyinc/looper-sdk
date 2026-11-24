"""``ServiceClient``: a program's connection to Looper. It opens the
program's session, heartbeats it, and makes a ``TrainingClient`` for each
run the program creates.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import math
import os
import re
from types import TracebackType
from typing import Self
from urllib.parse import urlencode
from uuid import uuid4

import httpx

from looper_sdk import wire
from looper_sdk.checkpoints import Checkpoint, CheckpointState
from looper_sdk.errors import CheckpointFailed, CreationError, CreationTimeout, LooperError
from looper_sdk.operations import LONG_POLL_S
from looper_sdk.proto.v1 import checkpoints_pb2, runs_pb2, sampling_pb2
from looper_sdk.sampling import SamplerConfig
from looper_sdk.sampling_client import SamplingClient
from looper_sdk.training import TrainingClient
from looper_sdk.transport import (
    CLOSE_TIMEOUT_S,
    Session,
    Transport,
    open_session,
    session_operation,
)
from looper_sdk.values import TrainingConfig

DEFAULT_BASE_URL = "https://api.looper.goaly.ai"

# Where a client finds its API key when its config names none.
API_KEY_VARIABLE = "LOOPER_API_KEY"
# Seconds between a waited checkpoint's reads.
CHECKPOINT_POLL_S = 1.0

_DISPLAY_NAME = re.compile(r"[A-Za-z0-9_-]{1,255}")


def _check_key(key: str, source: str) -> str:
    # One token, so it cannot end the Authorization header or start another.
    if not key or any(c.isspace() for c in key):
        raise ValueError(f"{source} is an API key, one token without spaces")
    return key


class ServiceClient:
    """A program's session with Looper, built from ``ServiceClient.Config``.

    It creates the runs a program trains, each with a ``TrainingClient``,
    and the samplers it samples, each with a ``SamplingClient``, and reads
    the project's checkpoints. Everything it creates belongs to its session:
    closing it, with ``close()`` or by leaving ``async with``, retires the
    session's runs and samplers, while their checkpoints stay.
    """

    @dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
    class Config:
        """Where the service is, who calls it, and how long a call may take.

        Attributes:
            base_url: The API root without ``/v1``: production by default,
                ``https://dev.api.looper.goaly.ai`` for dev.
            project: The project the session opens in; its runs belong to it.
            display_name: A label for the session on the console: 1 to 255
                ASCII letters, digits, ``_`` or ``-``. Not unique, and fixed
                once the session opens. None for no label.
            api_key: The API key every request sends as
                ``Authorization: Bearer``. None reads ``$LOOPER_API_KEY`` when
                the client is built; with neither, no credential is sent and
                ``build()`` raises ``Unauthenticated``. Kept out of the
                config's repr, so a logged config never shows it.
            request_timeout_s: One HTTP request's deadline. It must exceed
                the 30 s an await waits at the API.
            retry_timeout_s: How long a call keeps resending a retryable
                failure, such as the API's restart, before raising it.
        """

        base_url: str = DEFAULT_BASE_URL
        project: str
        display_name: str | None = None
        api_key: str | None = dataclasses.field(default=None, repr=False)
        request_timeout_s: float = 60.0
        retry_timeout_s: float = 300.0

        def __post_init__(self) -> None:
            if not self.base_url.startswith(("http://", "https://")):
                raise ValueError(f"base_url is an http(s) URL, not {self.base_url!r}")
            if self.display_name is not None and not _DISPLAY_NAME.fullmatch(self.display_name):
                raise ValueError(
                    "display_name is 1 to 255 ASCII letters, digits, _ or -, "
                    f"not {self.display_name!r}"
                )
            if self.api_key is not None:
                _check_key(self.api_key, "api_key")
            if not math.isfinite(self.request_timeout_s) or self.request_timeout_s <= LONG_POLL_S:
                raise ValueError(
                    f"request_timeout_s must exceed the {LONG_POLL_S} s an await waits at the "
                    f"API, or every long await times out: not {self.request_timeout_s}"
                )
            if not math.isfinite(self.retry_timeout_s) or self.retry_timeout_s < 0:
                raise ValueError(f"retry_timeout_s must be at least 0, not {self.retry_timeout_s}")

        async def build(self, runtime: ServiceClient.Runtime | None = None) -> ServiceClient:
            """Opens the program's session."""
            runtime = ServiceClient.Runtime() if runtime is None else runtime
            key = self.api_key
            if key is None and (found := os.environ.get(API_KEY_VARIABLE)) is not None:
                key = _check_key(found, f"${API_KEY_VARIABLE}")
            headers = {} if key is None else {"authorization": f"Bearer {key}"}

            def transport() -> Transport:
                client = httpx.AsyncClient(
                    base_url=self.base_url,
                    transport=runtime.transport,
                    timeout=self.request_timeout_s,
                    headers=headers,
                )
                return Transport(client, retry_timeout_s=self.retry_timeout_s)

            # Sampling's long polls get connections of their own.
            main, sampling = transport(), transport()
            try:
                session = await open_session(main, self.project, self.display_name)
                return ServiceClient(session, sampling=sampling)
            except BaseException:
                await main.close()
                await sampling.close()
                raise

    @dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
    class Runtime:
        """What a client is built over, besides its config.

        Attributes:
            transport: The HTTP transport, such as one in front of an API
                in the caller's process; None for httpx's own.
        """

        transport: httpx.AsyncBaseTransport | None = None

    def __init__(self, session: Session, *, sampling: Transport):
        self._session = session
        self._sampling = sampling

    @property
    def session(self) -> Session:
        """The service's shared session; its identity cannot be replaced."""
        return self._session

    async def __aenter__(self) -> Self:
        """Returns this already-built client."""
        self.session.check_open()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Closes the session without suppressing the body's exception."""
        try:
            await self.close()
        except Exception as error:
            if exc_value is None:
                raise
            exc_value.add_note(f"Session cleanup also raised {type(error).__name__}: {error}")

    @session_operation
    async def create_training_client(
        self, config: TrainingConfig, *, key: str | None = None, timeout_s: float | None = None
    ) -> TrainingClient:
        """Create a run and wait until its training node is ready.

        The optional deadline covers submission, retries and readiness. A timeout
        leaves accepted backend work intact; its error identifies the creation.
        """
        if timeout_s is not None and (not math.isfinite(timeout_s) or timeout_s < 0):
            raise ValueError("timeout_s must be finite and nonnegative, or None")
        key = str(uuid4()) if key is None else key
        session_id = self.session.id
        run_id = None
        request = runs_pb2.CreateRunRequest(
            session_id=session_id, key=key, config=wire.training_config_to_proto(config)
        )
        try:
            async with asyncio.timeout(timeout_s):
                answer = await self.session.transport.call(
                    "POST", "/v1/runs", request, runs_pb2.CreateRunResponse
                )
                run_id = answer.run.id
                client = TrainingClient(
                    answer.run, session=self.session, create_sampler=self.create_sampling_client
                )
                await client.wait_until_ready()
                return client
        except TimeoutError:
            raise CreationTimeout(key=key, session_id=session_id, run_id=run_id) from None
        except LooperError as exc:
            raise CreationError(exc, key=key, session_id=session_id, run_id=run_id) from exc

    @session_operation
    async def create_sampling_client(
        self, config: SamplerConfig, *, key: str | None = None, timeout_s: float | None = None
    ) -> SamplingClient:
        """Create a sampler and wait until it serves: its
        source resolved and pinned, its GPUs started, a checkpoint converted
        and its model loaded. As ``create_training_client``: the deadline
        covers submission, retries and readiness, and a timeout leaves the
        accepted sampler starting, its error naming it; resending under the
        same key answers the same sampler. A refusal, such as
        ``CheckpointNotReady`` for a name still uploading or still being
        exported, raises at once."""
        if timeout_s is not None and (not math.isfinite(timeout_s) or timeout_s < 0):
            raise ValueError("timeout_s must be finite and nonnegative, or None")
        key = str(uuid4()) if key is None else key
        session_id = self.session.id
        sampler_id = None
        request = sampling_pb2.CreateSamplerRequest(
            session_id=session_id, key=key, config=wire.sampler_config_to_proto(config)
        )
        try:
            async with asyncio.timeout(timeout_s):
                answer = await self._sampling.call(
                    "POST", "/v1/samplers", request, sampling_pb2.CreateSamplerResponse
                )
                sampler_id = answer.sampler.id
                client = SamplingClient(
                    answer.sampler, session=self.session, transport=self._sampling
                )
                await client.wait_until_ready()
                return client
        except TimeoutError:
            raise CreationTimeout(key=key, session_id=session_id, run_id=sampler_id) from None
        except LooperError as exc:
            raise CreationError(exc, key=key, session_id=session_id, run_id=sampler_id) from exc

    def _checkpoint_path(self, suffix: str = "", **query: object) -> str:
        return (
            "/v1/checkpoints" + suffix + "?" + urlencode({"session_id": self.session.id, **query})
        )

    @session_operation
    async def get_checkpoint(self, checkpoint_id: str) -> Checkpoint:
        """The project's checkpoint of that id, in whatever stage it is.
        ``NotFound`` if the project has none."""
        answer = await self.session.transport.call(
            "GET",
            self._checkpoint_path("/" + checkpoint_id),
            None,
            checkpoints_pb2.GetCheckpointResponse,
        )
        return wire.checkpoint_from_proto(answer.checkpoint)

    async def wait_checkpoint(
        self, checkpoint_id: str, *, timeout_s: float | None = None
    ) -> Checkpoint:
        """The checkpoint once it is ``READY_TO_USE``: uploaded, named and
        loadable. ``CheckpointFailed`` once it failed, and
        TimeoutError once ``timeout_s`` has passed; the upload goes on."""
        try:
            async with asyncio.timeout(timeout_s):
                while True:
                    checkpoint = await self.get_checkpoint(checkpoint_id)
                    if checkpoint.state is CheckpointState.READY_TO_USE:
                        return checkpoint
                    if checkpoint.state is CheckpointState.FAILED:
                        assert checkpoint.failure is not None
                        raise CheckpointFailed(
                            checkpoint_id=checkpoint_id, failure=checkpoint.failure
                        )
                    # The coordinator moves a checkpoint on in a pass, once a second.
                    await asyncio.sleep(CHECKPOINT_POLL_S)
        except TimeoutError:
            raise TimeoutError(
                f"Checkpoint {checkpoint_id} was not ready in {timeout_s} s"
            ) from None

    @session_operation
    async def get_checkpoint_by_name(self, name: str) -> Checkpoint:
        """The checkpoint the project's ``name`` binds: the last one that
        reached ``READY_TO_USE`` under it. ``NotFound`` if the
        name binds none, as before the first one is ready."""
        answer = await self.session.transport.call(
            "GET",
            self._checkpoint_path(name=name),
            None,
            checkpoints_pb2.GetCheckpointByNameResponse,
        )
        return wire.checkpoint_from_proto(answer.checkpoint)

    @session_operation
    async def list_checkpoints(
        self, *, limit: int = 100, after_name: str | None = None
    ) -> tuple[Checkpoint, ...]:
        """The project's named checkpoints, in name order: up to ``limit``,
        1 to 1000, of those named after ``after_name``, or from the first.
        Pass the last name returned as ``after_name`` to read the next page."""
        query = {"limit": str(limit)}
        if after_name is not None:
            query["after_name"] = after_name
        answer = await self.session.transport.call(
            "GET",
            self._checkpoint_path(**query),
            None,
            checkpoints_pb2.ListCheckpointsResponse,
        )
        return tuple(wire.checkpoint_from_proto(value) for value in answer.checkpoints)

    @session_operation
    async def delete_checkpoint(self, checkpoint_id: str) -> None:
        """Deletes the checkpoint's name; its files are reclaimed later. An id
        the project does not name deletes nothing: a ``FAILED`` checkpoint
        holds no name, so ``get_checkpoint`` still reads it. A checkpoint still
        being exported raises ``CheckpointNotReady``: delete it once it is
        ``READY_TO_USE``."""
        await self.session.transport.call(
            "DELETE",
            self._checkpoint_path("/" + checkpoint_id),
            None,
            checkpoints_pb2.DeleteCheckpointResponse,
        )

    async def close(self, *, timeout_s: float | None = CLOSE_TIMEOUT_S) -> None:
        """Closes the session. Its backend runs and samplers are retired
        asynchronously."""
        try:
            await self.session.close(timeout_s=timeout_s)
        except BaseException:
            # The session's own outcome, a cancellation included, is what the caller sees.
            with contextlib.suppress(Exception):
                await self._sampling.close()
            raise
        await self._sampling.close()
