"""The download of a weights checkpoint's files into a directory.

Each file comes from a signed URL that the API gives for 15 minutes. The
files on disk are the only record of progress, so a stopped download
continues when it runs again:

- A file downloads into ``<path>.part`` and gets its final name only once
  its CRC32C matches storage's. A file with its final name and the right
  size is therefore correct, and a later run trusts it unread.
- A partial file continues from its size, with ``Range``.
- The directory's marker file names the checkpoint, so two checkpoints'
  files never mix, and an OS lock on it keeps a second download out.
- A network or storage failure is retried with backoff, with no time limit;
  an expired URL is signed again, only for the files not yet finished. A
  file that fails its checksum three times stops the download.
- A URL is never logged: it reads its file for anyone who holds it.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import errno
import json
import logging
import os
import shutil
import sys
import time
from collections.abc import Awaitable, Callable, Iterator, Sequence
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import IO, ClassVar, Protocol

import google_crc32c
import httpx

from looper_sdk.errors import LooperError

log = logging.getLogger(__name__)

MARKER = ".looper-download.json"
PARTIAL = ".part"
# A file whose checksum fails this many downloads stops the download.
CHECKSUM_ATTEMPTS = 3
# Backoff between tries of storage or the API, doubling up to the last.
FIRST_WAIT_S, LAST_WAIT_S = 1.0, 60.0
# URLs are signed again this long before they expire.
REFRESH_MARGIN_S = 60.0
# A notice when no byte has arrived for this long, while files remain.
STALL_NOTICE_S = 300.0
# Fresh URLs that storage refuses in a row before the download stops: a
# refusal of a fresh URL is not expiry, and support must look at it.
REFUSALS = 3
# Bytes held in memory per file before a write to disk.
WRITE_BYTES = 8 << 20


class _RedactSignedUrls(logging.Filter):
    """httpx logs each request's URL at INFO. A signed URL's query is its
    credential, so its log line keeps the path alone."""

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.args, tuple):
            record.args = tuple(
                value.copy_with(query=None)
                if isinstance(value, httpx.URL) and b"signature" in value.query.lower()
                else value
                for value in record.args
            )
        return True


_REDACT = _RedactSignedUrls()
logging.getLogger("httpx").addFilter(_REDACT)


class DownloadError(RuntimeError):
    """A download that cannot go on without the user: the directory holds
    something else, or a file is wrong again and again. Its message says
    what to do. The files on disk stay, and a later run continues them."""

    retryable: ClassVar[bool] = False


class InsufficientSpace(DownloadError):
    """The disk lacks the room for the bytes still to come. Make space, then
    run the download again; it continues from the bytes on disk."""

    retryable: ClassVar[bool] = False


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class SignedFile:
    """One file of a checkpoint, as the API answers it.

    Attributes:
        path: Relative to the checkpoint, ``/``-separated.
        crc32c: The file's CRC32C (Castagnoli).
        url: Reads this file alone, until its answer's ``expires_at``.
    """

    path: str
    size_bytes: int
    crc32c: int
    url: str = dataclasses.field(repr=False)


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class SignedFiles:
    """The API's answer: the files, and when their URLs expire."""

    files: tuple[SignedFile, ...]
    expires_at: datetime


# Signs the named files' URLs, or every file's with None.
Sign = Callable[[Sequence[str] | None], Awaitable[SignedFiles]]


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class DownloadPlan:
    """What a download found on disk before its first byte.

    Attributes:
        files: The checkpoint's files.
        size_bytes: The checkpoint's bytes.
        on_disk_bytes: Bytes already on disk, finished or partial, which the
            download keeps.
    """

    files: int
    size_bytes: int
    on_disk_bytes: int


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class DownloadResult:
    """A finished download: every file complete and correct.

    Attributes:
        directory: Where the files are, as a model directory.
        files: How many files the checkpoint has.
        size_bytes: The checkpoint's bytes, all on disk.
    """

    directory: Path
    files: int
    size_bytes: int


class DownloadObserver(Protocol):
    """What a download tells its caller as it goes, such as a progress bar.
    None of its methods ever receives a URL."""

    def started(self, plan: DownloadPlan) -> None:
        """Once, before the first byte: what is to come and what is on disk."""
        ...

    def advanced(self, size_bytes: int) -> None:
        """Bytes written; negative when a file starts again from byte 0."""
        ...

    def finished(self, path: str) -> None:
        """A file is complete and correct, with its final name."""
        ...

    def notice(self, message: str) -> None:
        """A message for people: a lost connection, a file downloaded again."""
        ...


class LogObserver:
    """Logs each notice at WARNING and each finished file at DEBUG."""

    def started(self, plan: DownloadPlan) -> None:
        log.info(
            "%d files, %s, of which %s is on disk",
            plan.files,
            size(plan.size_bytes),
            size(plan.on_disk_bytes),
        )

    def advanced(self, size_bytes: int) -> None:
        pass

    def finished(self, path: str) -> None:
        log.debug("%s is complete", path)

    def notice(self, message: str) -> None:
        log.warning("%s", message)


def size(value: float) -> str:
    """Bytes for people, in decimal units: ``712 GB``, ``1.84 TB``."""
    for unit in ("B", "KB", "MB", "GB"):
        if abs(value) < 1000:
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.3g} {unit}"
        value /= 1000
    return f"{value:.3g} TB"


def check_path(path: str) -> PurePosixPath:
    """A file's path from the server, refused unless it stays inside the
    directory and names no file of the download's own."""
    parts = PurePosixPath(path)
    if (
        not path
        or "\\" in path
        or "\0" in path
        or parts.is_absolute()
        or any(part in ("", ".", "..") or ":" in part for part in path.split("/"))
        or path == MARKER
        or path.endswith(PARTIAL)
    ):
        raise DownloadError(f"The server sent a file path that is not safe: {path!r}")
    return parts


@contextlib.contextmanager
def _locked(directory: Path, checkpoint_id: str, *, verify: bool) -> Iterator[None]:
    """Holds the directory for this checkpoint while the download runs:
    its marker file names the checkpoint, and an OS lock on the marker
    keeps a second download out until this process ends."""
    marker = directory / MARKER
    shown = str(directory)
    if not marker.exists() and any(directory.iterdir()) and not verify:
        raise DownloadError(
            f"{shown} is not empty and is not a Looper download. Use an empty directory, "
            "or add --verify to keep the files in it that are correct."
        )
    descriptor = os.open(marker, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        if not _lock(descriptor):
            holder = _read_marker(marker).get("pid")
            process = f" (process {holder})" if holder else ""
            raise DownloadError(f"A download into {shown} is already in progress{process}.")
        found = _read_marker(marker).get("checkpoint_id")
        if found and found != checkpoint_id:
            raise DownloadError(
                f"{shown} holds files of checkpoint {found}. Use a different directory."
            )
        content = json.dumps({"checkpoint_id": checkpoint_id, "pid": os.getpid()}).encode()
        os.ftruncate(descriptor, 0)
        os.lseek(descriptor, 0, os.SEEK_SET)
        os.write(descriptor, content)
        os.fsync(descriptor)
        yield
    finally:
        # Closing the descriptor releases the lock.
        os.close(descriptor)


def _lock(descriptor: int) -> bool:
    if sys.platform == "win32":
        import msvcrt

        try:
            msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
        except OSError:
            return False
        return True
    import fcntl

    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return False
    return True


def _read_marker(marker: Path) -> dict[str, object]:
    try:
        value = json.loads(marker.read_bytes() or b"{}")
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _crc32c(path: Path, *, limit: int | None = None) -> int:
    """The CRC32C of a file's first ``limit`` bytes, or of all of it."""
    crc = 0
    left = path.stat().st_size if limit is None else limit
    with path.open("rb") as file:
        while left > 0:
            block = file.read(min(WRITE_BYTES, left))
            if not block:
                break
            crc = google_crc32c.extend(crc, block)
            left -= len(block)
    return crc


class _Urls:
    """The files' current URLs, signed again before they expire or after
    storage refuses one, for the files not yet finished alone."""

    def __init__(self, sign: Sign, observer: DownloadObserver, signed: SignedFiles):
        self.sign, self.observer = sign, observer
        self.lock = asyncio.Lock()
        self.unfinished: set[str] = set()
        self._take(signed)

    def _take(self, signed: SignedFiles) -> None:
        self.urls = {file.path: file.url for file in signed.files}
        self.expires_at = signed.expires_at.timestamp()
        self.generation = getattr(self, "generation", 0) + 1

    async def get(self, path: str) -> tuple[str, int]:
        """The file's URL, and its generation, for ``expired``."""
        if time.time() >= self.expires_at - REFRESH_MARGIN_S:
            await self.expired(self.generation)
        if path not in self.urls:
            raise DownloadError(f"The checkpoint no longer lists {path}. Send this to support.")
        return self.urls[path], self.generation

    async def expired(self, generation: int) -> None:
        """Signs again, unless another file did since ``generation``."""
        async with self.lock:
            if generation != self.generation or not self.unfinished:
                return
            wait = FIRST_WAIT_S
            while True:
                try:
                    self._take(await self.sign(sorted(self.unfinished)))
                    return
                except LooperError as error:
                    if not error.retryable:
                        raise
                    self.observer.notice(
                        f"The Looper API did not answer ({error.message}). "
                        f"The next try is in {wait:.0f} s."
                    )
                await asyncio.sleep(wait)
                wait = min(2 * wait, LAST_WAIT_S)


class _File:
    """One file's download into ``<path>.part``, then its rename.

    ``offset`` and ``crc`` follow the partial file's bytes as they are
    written, so a lost connection continues without reading them again."""

    def __init__(self, download: _Download, file: SignedFile):
        self.download, self.file = download, file
        self.final = download.directory / file.path
        self.partial = self.final.with_name(self.final.name + PARTIAL)
        self.offset = self.crc = 0

    async def run(self) -> None:
        observer, path = self.download.observer, self.file.path
        bad = 0
        while True:
            await self._fetch()
            if self.crc == self.file.crc32c:
                await asyncio.to_thread(os.replace, self.partial, self.final)
                self.download.urls.unfinished.discard(path)
                observer.finished(path)
                return
            bad += 1
            self._restart()
            if bad >= CHECKSUM_ATTEMPTS:
                raise DownloadError(
                    f"{path} is not correct after {CHECKSUM_ATTEMPTS} downloads. Send this "
                    f"message to support: {self.download.checkpoint_id}, {path}."
                )
            observer.notice(f"{path} is not correct. The command downloads it again.")

    def _restart(self) -> None:
        """Deletes the partial file, and takes its bytes off the progress."""
        written = self.partial.stat().st_size if self.partial.exists() else 0
        self.partial.unlink(missing_ok=True)
        self.download.observer.advanced(-written)
        self.offset = self.crc = 0

    async def _fetch(self) -> None:
        """Brings ``<path>.part`` to the file's size, with its CRC32C in ``crc``."""
        self.partial.parent.mkdir(parents=True, exist_ok=True)
        # An empty file gets no byte, but still needs its partial file to rename.
        self.partial.touch()
        self.offset = self.partial.stat().st_size
        if self.offset > self.file.size_bytes:
            self._restart()
        # The checksum covers the bytes already on disk, read once here.
        self.crc = (
            await asyncio.to_thread(_crc32c, self.partial, limit=self.offset) if self.offset else 0
        )
        wait, refusals = FIRST_WAIT_S, 0
        while self.offset < self.file.size_bytes:
            url, generation = await self.download.urls.get(self.file.path)
            before = self.offset
            try:
                await self._stream(url)
                wait, refusals = FIRST_WAIT_S, 0
                continue
            except _Refused as refused:
                refusals += 1
                if refusals >= REFUSALS:
                    raise DownloadError(
                        f"Storage refused {self.file.path} {REFUSALS} times in a row with HTTP "
                        f"{refused.status}. Make sure that this computer can connect to "
                        "storage.googleapis.com, then send this message to support: "
                        f"{self.download.checkpoint_id}, {self.file.path}, {refused.status}."
                    ) from None
                # Most often the URL expired; a fresh one settles it.
                await self.download.urls.expired(generation)
                continue
            except (httpx.TransportError, _Unavailable) as error:
                # A try that wrote bytes was working, so earlier failures no longer count.
                if self.offset > before:
                    wait, refusals = FIRST_WAIT_S, 0
                # httpx's timeouts carry no text, so the error's type names them.
                self.download.observer.notice(
                    f"Connection lost for {self.file.path} ({str(error) or type(error).__name__}). "
                    f"The next try is in {wait:.0f} s."
                )
            await asyncio.sleep(wait)
            wait = min(2 * wait, LAST_WAIT_S)

    async def _stream(self, url: str) -> None:
        """One GET from ``offset``, appended to the partial file."""
        headers = {"range": f"bytes={self.offset}-"} if self.offset else {}
        async with self.download.client.stream("GET", url, headers=headers) as answer:
            status = answer.status_code
            if status == 416:
                # Nothing past the offset, which is short of the size: start again.
                self._restart()
                return
            if status >= 500 or status == 429:
                raise _Unavailable(status)
            if status not in (200, 206):
                raise _Refused(status)
            if status == 200 and self.offset:
                # Storage ignored the range and sends the file from byte 0.
                self._restart()
            buffer: list[bytes] = []
            held = 0
            with self.partial.open("ab") as out:
                try:
                    async for chunk in answer.aiter_raw():
                        buffer.append(chunk)
                        held += len(chunk)
                        if held >= WRITE_BYTES:
                            await self._write(out, buffer)
                            buffer, held = [], 0
                finally:
                    # What arrived before a failure is kept, and the next try continues it.
                    if buffer:
                        await self._write(out, buffer)

    async def _write(self, out: IO[bytes], chunks: list[bytes]) -> None:
        data = b"".join(chunks)
        try:
            await asyncio.to_thread(out.write, data)
            await asyncio.to_thread(out.flush)
        except OSError as error:
            if error.errno == errno.ENOSPC:
                raise InsufficientSpace(
                    f"The disk is full. {size(self.download.on_disk())} is on disk."
                ) from error
            raise
        self.offset += len(data)
        self.crc = google_crc32c.extend(self.crc, data)
        self.download.observer.advanced(len(data))
        self.download.last_byte_at = time.monotonic()


class _Unavailable(Exception):
    def __init__(self, status: int):
        super().__init__(f"HTTP {status}")


class _Refused(Exception):
    def __init__(self, status: int):
        super().__init__(f"HTTP {status}")
        self.status = status


class _Download:
    def __init__(
        self,
        *,
        checkpoint_id: str,
        directory: Path,
        client: httpx.AsyncClient,
        observer: DownloadObserver,
    ):
        self.checkpoint_id, self.directory = checkpoint_id, directory
        self.client, self.observer = client, observer
        self.last_byte_at = time.monotonic()
        self.files: tuple[SignedFile, ...] = ()
        self.urls: _Urls

    def on_disk(self) -> int:
        total = 0
        for file in self.files:
            for candidate in (
                self.directory / file.path,
                self.directory / (file.path + PARTIAL),
            ):
                with contextlib.suppress(OSError):
                    total += candidate.stat().st_size
        return total

    async def watch(self) -> None:
        """A notice each ``STALL_NOTICE_S`` with no byte, while files remain."""
        while True:
            await asyncio.sleep(STALL_NOTICE_S / 10)
            if time.monotonic() - self.last_byte_at >= STALL_NOTICE_S:
                self.observer.notice(
                    "No data for 5 minutes. Make sure that this computer can connect to "
                    "storage.googleapis.com. The download continues when it can."
                )
                self.last_byte_at = time.monotonic()


def _plan(directory: Path, files: Sequence[SignedFile], *, verify: bool) -> tuple[int, list[str]]:
    """The bytes on disk that the download keeps, and the files still to
    download, after it removes what it cannot keep."""
    on_disk, todo = 0, []
    for file in files:
        final = directory / file.path
        partial = final.with_name(final.name + PARTIAL)
        if final.is_file() and final.stat().st_size == file.size_bytes:
            if not verify or _crc32c(final) == file.crc32c:
                partial.unlink(missing_ok=True)
                on_disk += file.size_bytes
                continue
        final.unlink(missing_ok=True)
        if partial.is_file():
            offset = partial.stat().st_size
            if offset <= file.size_bytes:
                on_disk += offset
            else:
                partial.unlink()
        todo.append(file.path)
    return on_disk, todo


async def download(
    *,
    checkpoint_id: str,
    directory: Path,
    sign: Sign,
    client: httpx.AsyncClient,
    verify: bool = False,
    parallel: int = 8,
    observer: DownloadObserver | None = None,
) -> DownloadResult:
    """Downloads the checkpoint's files into ``directory``, keeping what a
    stopped download left there.

    ``client`` reaches storage and must send no credential of Looper's: a
    signed URL is its own. ``verify`` checks each finished file's checksum
    again, and also accepts a directory that holds other files and no
    marker. ``parallel`` files download at once. A network failure never
    ends it; ``DownloadError`` does, and the files on disk stay.
    """
    if not 1 <= parallel <= 64:
        raise ValueError(f"parallel must be from 1 to 64, not {parallel}")
    observer = observer or LogObserver()
    directory.mkdir(parents=True, exist_ok=True)
    with _locked(directory, checkpoint_id, verify=verify):
        signed = await sign(None)
        for file in signed.files:
            check_path(file.path)
        state = _Download(
            checkpoint_id=checkpoint_id, directory=directory, client=client, observer=observer
        )
        state.files = signed.files
        total = sum(file.size_bytes for file in signed.files)
        on_disk, todo = await asyncio.to_thread(_plan, directory, signed.files, verify=verify)
        needed = total - on_disk
        free = shutil.disk_usage(directory).free
        if needed > free:
            raise InsufficientSpace(
                f"The disk at {directory} has {size(free)} free. The download needs "
                f"{size(needed)} more."
            )
        observer.started(
            DownloadPlan(files=len(signed.files), size_bytes=total, on_disk_bytes=on_disk)
        )
        state.urls = _Urls(sign, observer, signed)
        state.urls.unfinished = set(todo)
        by_path = {file.path: file for file in signed.files}
        limit = asyncio.Semaphore(parallel)

        async def one(path: str) -> None:
            async with limit:
                await _File(state, by_path[path]).run()

        watcher = asyncio.create_task(state.watch())
        try:
            async with asyncio.TaskGroup() as group:
                for path in todo:
                    group.create_task(one(path))
        except* Exception as errors:
            # The first failure, as itself: the others are its consequences.
            raise errors.exceptions[0] from None
        finally:
            watcher.cancel()
    return DownloadResult(directory=directory, files=len(signed.files), size_bytes=total)
