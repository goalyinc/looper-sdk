"""HTTP to Looper's API, and the program's session: binary protobuf bodies,
typed errors, the retries every call takes, and the session's heartbeats.

- **Retries.** Every call of ``looper.v1`` is safe to send again: an ordered
  one carries its key, and the rest change nothing twice, but for opening a
  session, whose extra sessions expire unused. So a call is sent
  again while its failure is retryable, such as ``UNAVAILABLE`` while the
  coordinator restarts, or while its answer is lost to the network, until
  ``retry_timeout_s`` has passed. An answer that is not ``looper.v1``'s, such
  as a proxy's, counts as ``UNAVAILABLE`` from a 5xx status, and otherwise as
  a failure that no retry fixes.
  Session-capacity refusals warn immediately, once per open attempt, with
  the server's limit and recovery instructions.
- **The session** has one immutable identity and heartbeats at the interval
  its service names. Expiry is terminal; it never renews automatically.
  Explicit close invalidates handles, bounds its caller's wait, and retains
  cleanup ownership when a caller is cancelled or its deadline expires.
"""

from __future__ import annotations

import asyncio
import logging
import math
from collections.abc import Callable, Coroutine
from functools import wraps
from typing import Any, Concatenate

import httpx
from google.protobuf.message import DecodeError, Message

from looper_sdk import wire
from looper_sdk.errors import (
    CapacityExhausted,
    LooperError,
    SessionClosed,
    SessionCloseTimeout,
    SessionExpired,
    Unavailable,
)
from looper_sdk.proto.v1 import errors_pb2, sessions_pb2

log = logging.getLogger(__name__)

PROTOBUF = "application/x-protobuf"
# The first wait between attempts, which doubles up to the last.
FIRST_WAIT_S, LAST_WAIT_S = 0.1, 2.0
# Bound the server-close RPC separately from draining admitted calls.
CLOSE_WAIT_S = 5.0
CLOSE_GRACE_S = 60.0
# Leave time for the close RPC and local cleanup after the grace period.
CLOSE_TIMEOUT_S = CLOSE_GRACE_S + 2 * CLOSE_WAIT_S


def _error(answer: httpx.Response) -> LooperError:
    """The typed error an answer carries."""
    if answer.headers.get("content-type", "").startswith(PROTOBUF):
        try:
            return wire.error_from_proto(errors_pb2.Error.FromString(answer.content))
        except DecodeError:
            pass
    message = f"HTTP {answer.status_code} from {answer.request.url}, not looper.v1's answer"
    if answer.status_code >= 500:
        return Unavailable(message)
    return LooperError(message)


def session_operation[**P, R](
    method: Callable[Concatenate[Any, P], Coroutine[Any, Any, R]],
) -> Callable[Concatenate[Any, P], Coroutine[Any, Any, R]]:
    """Keep a whole SDK call, including its sleeps, within its session's life."""

    @wraps(method)
    async def bound(self: Any, /, *args: P.args, **kwargs: P.kwargs) -> R:
        return await self._session.run(lambda: method(self, *args, **kwargs))

    return bound


class Transport:
    """Calls the API through one HTTP client, which it closes."""

    def __init__(self, client: httpx.AsyncClient, *, retry_timeout_s: float):
        self.client, self.retry_timeout_s = client, retry_timeout_s

    async def call[M: Message](
        self, method: str, path: str, request: Message | None, response: type[M]
    ) -> M:
        """Sends the request until it is answered, or its failure is not
        retryable, or ``retry_timeout_s`` has passed; raises its typed error."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.retry_timeout_s
        content = None if request is None else request.SerializeToString()
        wait = FIRST_WAIT_S
        capacity_warned = False
        while True:
            try:
                answer = await self.client.request(
                    method, path, content=content, headers={"content-type": PROTOBUF}
                )
            except httpx.TransportError as exc:
                error: LooperError = Unavailable(f"{method} {path} got no answer: {exc!r}")
            else:
                if answer.status_code == 200:
                    return response.FromString(answer.content)
                error = _error(answer)
            if (
                method == "POST"
                and path == "/v1/sessions"
                and isinstance(error, CapacityExhausted)
                and not capacity_warned
            ):
                log.warning("Service session concurrency limit reached: %s", error.message)
                capacity_warned = True
            pause = wait if error.retry_after_s is None else error.retry_after_s
            if not error.retryable or loop.time() + pause > deadline:
                raise error
            log.info("%s %s: %s; sending it again in %.1f s", method, path, error.message, pause)
            await asyncio.sleep(pause)
            wait = min(2 * wait, LAST_WAIT_S)

    async def close(self) -> None:
        await self.client.aclose()


async def open_session(
    transport: Transport, project: str, display_name: str | None = None
) -> Session:
    """Opens a session in the project, and starts its heartbeats."""
    answer = await transport.call(
        "POST",
        "/v1/sessions",
        sessions_pb2.OpenSessionRequest(project=project, display_name=display_name or ""),
        sessions_pb2.OpenSessionResponse,
    )
    return Session(transport, answer.session)


class Session:
    """The program's session, which its clients share.

    Attributes:
        id: The service's permanent session identity.
        transport: How the clients reach the API.
    """

    def __init__(self, transport: Transport, message: sessions_pb2.Session):
        self.transport, self._id, self.project = transport, message.id, message.project
        self.heartbeat_interval_s = message.heartbeat_interval_s
        self._tasks: set[asyncio.Task[Any]] = set()
        self._closing: asyncio.Task[None] | None = None
        self._stopping = False
        self._heartbeats = asyncio.create_task(self._beat_forever())

    @property
    def id(self) -> str:
        return self._id

    def check_open(self) -> None:
        if self._closing is not None and (
            self._stopping or asyncio.current_task() not in self._tasks
        ):
            raise SessionClosed("The service session is closed; build a new ServiceClient")

    def _finished(self, task: asyncio.Task[Any]) -> None:
        self._tasks.discard(task)
        if not task.cancelled():
            task.exception()

    async def run[T](self, call: Callable[[], Coroutine[Any, Any, T]]) -> T:
        self.check_open()
        if asyncio.current_task() in self._tasks:
            return await call()
        task = asyncio.create_task(call())
        self._tasks.add(task)
        task.add_done_callback(self._finished)
        try:
            result = await asyncio.shield(task)
            if self._stopping and task.cancelling():
                raise SessionClosed("The service session closed while this call was pending")
            return result
        except asyncio.CancelledError:
            caller = asyncio.current_task()
            if caller is not None and caller.cancelling():
                task.cancel()
                raise
            if self._closing is not None:
                raise SessionClosed(
                    "The service session closed while this call was pending"
                ) from None
            raise

    async def _heartbeat(self) -> None:
        await self.transport.call(
            "POST",
            f"/v1/sessions/{self.id}:heartbeat",
            sessions_pb2.HeartbeatSessionRequest(session_id=self.id),
            sessions_pb2.HeartbeatSessionResponse,
        )

    async def _beat_forever(self) -> None:
        while True:
            await asyncio.sleep(self.heartbeat_interval_s)
            try:
                await self._heartbeat()
            except SessionExpired:
                log.warning("Session %s expired; create a new service and run", self.id)
                return
            except Exception as exc:
                log.warning(
                    "Session %s missed a heartbeat: %s. Each miss brings its expiry closer; "
                    "expiry requires a new service",
                    self.id,
                    exc,
                )

    def _cleanup_finished(self, task: asyncio.Task[None]) -> None:
        if not task.cancelled() and (error := task.exception()) is not None:
            log.error(
                "Session %s cleanup failed: %s",
                self.id,
                error,
                exc_info=(type(error), error, error.__traceback__),
            )

    async def close(self, *, timeout_s: float | None = CLOSE_TIMEOUT_S) -> None:
        """Wait for shared cleanup, bounded independently of cleanup's lifetime.

        Admitted calls have 60 seconds to finish while heartbeats continue.
        Cancellation or SessionCloseTimeout leaves cleanup owned by the session;
        another close can wait again. Server closure ends its runs; workers are
        retired asynchronously.
        """
        if timeout_s is not None and (not math.isfinite(timeout_s) or timeout_s < 0):
            raise ValueError("timeout_s must be finite and nonnegative, or None")
        if self._closing is None:
            self._closing = asyncio.create_task(self._close())
            self._closing.add_done_callback(self._cleanup_finished)
        if not self._closing.done():
            done, _ = await asyncio.wait({self._closing}, timeout=timeout_s)
            if not done:
                raise SessionCloseTimeout(
                    f"Session {self.id} cleanup did not finish in {timeout_s} s; "
                    "it continues in the background; close again to wait"
                )
        self._closing.result()

    async def _close(self) -> None:
        tasks = tuple(self._tasks)
        if tasks:
            await asyncio.wait(tasks, timeout=CLOSE_GRACE_S)
        self._stopping = True
        self._heartbeats.cancel()
        for task in tasks:
            task.cancel()
        try:
            async with asyncio.timeout(CLOSE_WAIT_S):
                await self.transport.call(
                    "POST",
                    f"/v1/sessions/{self.id}:close",
                    sessions_pb2.CloseSessionRequest(session_id=self.id),
                    sessions_pb2.CloseSessionResponse,
                )
        except (LooperError, TimeoutError) as exc:
            log.warning("Session %s was not closed (%s); it expires on its own", self.id, exc)
        finally:
            try:
                await self.transport.close()
            finally:
                await asyncio.gather(self._heartbeats, *tasks, return_exceptions=True)
