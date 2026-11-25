"""The downloader over a fake storage and a fake API: each event a user
sees, and each guarantee. Storage answers ``Range``, cuts connections,
serves wrong bytes, and refuses URLs of an expired signature, as Google
Cloud Storage does."""

from __future__ import annotations

import asyncio
import errno
import functools
import json
import logging
import os
from collections.abc import AsyncIterator, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import IO, cast

import google_crc32c
import httpx
import pytest

from looper_sdk import download as downloads
from looper_sdk.download import (
    MARKER,
    DownloadError,
    DownloadPlan,
    InsufficientSpace,
    SignedFile,
    SignedFiles,
)
from looper_sdk.errors import NotFound, Unavailable

FILES = {
    "config.json": b'{"model_type": "qwen3"}',
    "model-00001-of-00002.safetensors": bytes(range(256)) * 400,
    "model-00002-of-00002.safetensors": bytes(reversed(range(256))) * 300,
    "nested/tokenizer.json": b"{}" * 1000,
    "empty.txt": b"",
}
# The files storage serves: an empty one needs no request.
FETCHED = sorted(name for name, data in FILES.items() if data)
CHECKPOINT = "8d7e4a8e-6c41-4b67-9e55-0a3f8c7b0e11"
SECRET = "X-Goog-Signature"


def synchronous(test):
    """The async test run in a loop of its own: the SDK's tests also run
    with pytest alone, outside the repository (looper-isolated)."""

    @functools.wraps(test)
    def run(*args, **kwargs):
        return asyncio.run(test(*args, **kwargs))

    return run


@pytest.fixture(autouse=True)
def fast(monkeypatch):
    monkeypatch.setattr(downloads, "FIRST_WAIT_S", 0.001)
    monkeypatch.setattr(downloads, "LAST_WAIT_S", 0.002)
    monkeypatch.setattr(downloads, "WRITE_BYTES", 4096)


class _Body(httpx.AsyncByteStream):
    """A body in 1 KiB chunks, as a network delivers it; with ``keep``, one
    that stops after that many bytes, as a dropped connection does."""

    def __init__(self, data: bytes, keep: int | None = None):
        self.data, self.keep = data, keep

    async def __aiter__(self) -> AsyncIterator[bytes]:
        end = len(self.data) if self.keep is None else self.keep
        for start in range(0, end, 1024):
            yield self.data[start : min(start + 1024, end)]
        if self.keep is not None:
            raise httpx.ReadError("connection reset")


class Storage:
    """Objects behind signed URLs. Only the current signature reads; each
    request is recorded with its range."""

    def __init__(self, files: dict[str, bytes]):
        self.files = dict(files)
        self.generation = 1
        self.requests: list[tuple[str, str | None]] = []
        self.cut: dict[str, list[int]] = {}
        self.corrupt: dict[str, int] = {}
        self.unavailable = 0
        self.ignore_range = False
        self.refuse_all = False

    def handle(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path.removeprefix("/")
        span = request.headers.get("range")
        self.requests.append((path, span))
        assert "authorization" not in request.headers
        if self.unavailable:
            self.unavailable -= 1
            return httpx.Response(503)
        if self.refuse_all or int(request.url.params["g"]) != self.generation:
            return httpx.Response(403, text="ExpiredToken")
        data = self.files[path]
        if self.corrupt.get(path):
            self.corrupt[path] -= 1
            data = bytes([data[0] ^ 1]) + data[1:]
        start = 0
        if span and not self.ignore_range:
            start = int(span.removeprefix("bytes=").removesuffix("-"))
            if start >= len(data):
                return httpx.Response(416)
        body = data[start:]
        status = 206 if start else 200
        keep = self.cut[path].pop(0) if self.cut.get(path) else None
        return httpx.Response(status, stream=_Body(body, keep))


class Api:
    """The download route: each file's size and checksum, and URLs of the
    storage's current signature."""

    def __init__(self, storage: Storage, *, ttl_s: float = 900):
        self.storage, self.ttl_s = storage, ttl_s
        self.calls: list[list[str] | None] = []
        self.deleted = False
        self.unavailable = 0

    async def __call__(self, paths: Sequence[str] | None) -> SignedFiles:
        self.calls.append(None if paths is None else list(paths))
        if self.deleted:
            raise NotFound("No current checkpoint visible in this project")
        if self.unavailable:
            self.unavailable -= 1
            raise Unavailable("The API is restarting")
        names = sorted(FILES) if paths is None else paths
        return SignedFiles(
            files=tuple(
                SignedFile(
                    path=name,
                    size_bytes=len(FILES[name]),
                    crc32c=google_crc32c.value(FILES[name]),
                    url=f"https://storage.test/{name}?g={self.storage.generation}&{SECRET}=s",
                )
                for name in names
            ),
            expires_at=datetime.now(UTC) + timedelta(seconds=self.ttl_s),
        )


class Seen:
    """What the observer was told."""

    def __init__(self):
        self.plans: list[DownloadPlan] = []
        self.notices: list[str] = []
        self.done: list[str] = []
        self.bytes = 0

    def started(self, plan: DownloadPlan) -> None:
        self.plans.append(plan)
        self.bytes = plan.on_disk_bytes

    def advanced(self, size_bytes: int) -> None:
        self.bytes += size_bytes

    def finished(self, path: str) -> None:
        self.done.append(path)

    def notice(self, message: str) -> None:
        self.notices.append(message)


@pytest.fixture
def storage() -> Storage:
    return Storage(FILES)


@pytest.fixture
def api(storage) -> Api:
    return Api(storage)


async def run(tmp_path: Path, storage: Storage, api: Api, *, seen=None, **options):
    client = httpx.AsyncClient(transport=httpx.MockTransport(storage.handle))
    async with client:
        return await downloads.download(
            checkpoint_id=CHECKPOINT,
            directory=tmp_path / "model",
            sign=api,
            client=client,
            observer=seen or Seen(),
            **options,
        )


def assert_complete(directory: Path) -> None:
    for name, data in FILES.items():
        assert (directory / name).read_bytes() == data
    assert not list(directory.rglob("*.part"))
    assert json.loads((directory / MARKER).read_text())["checkpoint_id"] == CHECKPOINT


@synchronous
async def test_a_download_writes_every_file_and_the_marker(tmp_path, storage, api):
    seen = Seen()
    result = await run(tmp_path, storage, api, seen=seen)
    assert (result.files, result.size_bytes) == (len(FILES), sum(map(len, FILES.values())))
    assert_complete(tmp_path / "model")
    assert sorted(seen.done) == sorted(FILES)
    assert seen.bytes == result.size_bytes
    assert seen.plans == [
        DownloadPlan(files=len(FILES), size_bytes=result.size_bytes, on_disk_bytes=0)
    ]


@synchronous
async def test_a_second_run_trusts_finished_files_and_reads_nothing(tmp_path, storage, api):
    await run(tmp_path, storage, api)
    storage.requests.clear()
    seen = Seen()
    await run(tmp_path, storage, api, seen=seen)
    assert storage.requests == []
    assert seen.plans[0].on_disk_bytes == seen.plans[0].size_bytes


@synchronous
async def test_a_partial_file_continues_from_its_last_byte(tmp_path, storage, api):
    name = "model-00001-of-00002.safetensors"
    directory = tmp_path / "model"
    directory.mkdir()
    (directory / MARKER).write_text(json.dumps({"checkpoint_id": CHECKPOINT}))
    (directory / (name + ".part")).write_bytes(FILES[name][:5000])
    await run(tmp_path, storage, api)
    assert (name, "bytes=5000-") in storage.requests
    assert_complete(directory)


@synchronous
async def test_a_lost_connection_continues_from_the_last_byte_written(tmp_path, storage, api):
    name = "model-00001-of-00002.safetensors"
    storage.cut[name] = [10_000, 7_000]
    seen = Seen()
    await run(tmp_path, storage, api, seen=seen)
    assert [span for path, span in storage.requests if path == name] == [
        None,
        "bytes=10000-",
        "bytes=17000-",
    ]
    assert_complete(tmp_path / "model")
    assert any(n.startswith(f"Connection lost for {name}") for n in seen.notices)


@synchronous
async def test_a_notice_names_an_error_that_has_no_message(tmp_path, storage, api):
    # httpx's timeouts carry no text: a silent network would read "()".
    name = "model-00001-of-00002.safetensors"
    original = storage.handle
    timeouts = [httpx.ReadTimeout("")]

    def silent_once(request):
        if timeouts and request.url.path.endswith(name):
            raise timeouts.pop()
        return original(request)

    storage.handle = silent_once
    seen = Seen()
    await run(tmp_path, storage, api, seen=seen)
    assert_complete(tmp_path / "model")
    assert f"Connection lost for {name} (ReadTimeout). The next try is in 0 s." in seen.notices


@synchronous
async def test_storage_errors_are_tried_again(tmp_path, storage, api):
    storage.unavailable = 5
    await run(tmp_path, storage, api)
    assert_complete(tmp_path / "model")


@synchronous
async def test_an_expired_url_is_signed_again_for_unfinished_files_alone(tmp_path, storage, api):
    # One file finishes on the first signature; then every URL expires.
    original = storage.handle

    def expire_after_first(request):
        answer = original(request)
        if answer.status_code == 200 and storage.generation == 1:
            storage.generation = 2
        return answer

    storage.handle = expire_after_first
    await run(tmp_path, storage, api, parallel=1)
    assert_complete(tmp_path / "model")
    assert api.calls[0] is None
    assert api.calls[1] == FETCHED[1:]


@synchronous
async def test_urls_that_expire_between_working_tries_never_stop_a_file(tmp_path, storage, api):
    # A computer whose clock is behind uses each URL past its expiry: every
    # lost connection meets a refusal, which fresh URLs settle.
    name = "model-00001-of-00002.safetensors"
    storage.cut[name] = [10_000] * (downloads.REFUSALS + 1)
    original = storage.handle

    def expire_at_each_cut(request):
        cutting = bool(storage.cut.get(name)) and request.url.path.endswith(name)
        answer = original(request)
        if cutting and answer.status_code in (200, 206):
            storage.generation += 1
        return answer

    storage.handle = expire_at_each_cut
    await run(tmp_path, storage, api, parallel=1)
    assert_complete(tmp_path / "model")
    assert len(api.calls) == 1 + downloads.REFUSALS + 1


class _Clock:
    """``asyncio`` as the downloader sees it, with each wait recorded."""

    def __init__(self):
        self.waits: list[float] = []

    def __getattr__(self, name):
        return getattr(asyncio, name)

    async def sleep(self, seconds: float) -> None:
        self.waits.append(seconds)
        await asyncio.sleep(seconds)


@synchronous
async def test_a_try_that_wrote_bytes_starts_the_wait_over(tmp_path, storage, api, monkeypatch):
    monkeypatch.setattr(downloads, "LAST_WAIT_S", 1.0)
    clock = _Clock()
    monkeypatch.setattr(downloads, "asyncio", clock)
    name = "model-00001-of-00002.safetensors"
    storage.cut[name] = [10_000] * 4
    await run(tmp_path, storage, api, parallel=1)
    assert_complete(tmp_path / "model")
    # The stall watch's waits are longer; each retry's is the first wait.
    retries = [wait for wait in clock.waits if wait < 1]
    assert retries == [downloads.FIRST_WAIT_S] * 4


@synchronous
async def test_urls_near_their_expiry_are_signed_again_before_use(tmp_path, storage, api):
    api.ttl_s = 30  # inside the refresh margin
    await run(tmp_path, storage, api, parallel=1)
    assert_complete(tmp_path / "model")
    # Each file asked for fresh URLs before its GET, so storage refused none.
    assert len(api.calls) == 1 + len(FETCHED)
    assert all(paths for paths in api.calls[1:])


@synchronous
async def test_the_api_down_during_a_refresh_is_waited_out(tmp_path, storage, api):
    api.ttl_s = 30
    seen = Seen()
    original = api.__call__

    calls = 0

    async def flaky(paths):
        nonlocal calls
        calls += 1
        if calls == 2:
            api.unavailable = 2
        return await original(paths)

    await downloads_run(tmp_path, storage, flaky, seen)
    assert_complete(tmp_path / "model")
    assert any(n.startswith("The Looper API did not answer") for n in seen.notices)


async def downloads_run(tmp_path, storage, sign, seen):
    client = httpx.AsyncClient(transport=httpx.MockTransport(storage.handle))
    async with client:
        return await downloads.download(
            checkpoint_id=CHECKPOINT,
            directory=tmp_path / "model",
            sign=sign,
            client=client,
            observer=seen,
            parallel=1,
        )


@synchronous
async def test_a_wrong_file_downloads_again(tmp_path, storage, api):
    name = "config.json"
    storage.corrupt[name] = 1
    seen = Seen()
    await run(tmp_path, storage, api, seen=seen)
    assert_complete(tmp_path / "model")
    assert f"{name} is not correct. The command downloads it again." in seen.notices
    assert seen.bytes == sum(map(len, FILES.values()))


@synchronous
async def test_three_wrong_downloads_stop_with_a_message_for_support(tmp_path, storage, api):
    name = "config.json"
    storage.corrupt[name] = 3
    with pytest.raises(DownloadError) as raised:
        await run(tmp_path, storage, api)
    assert str(raised.value) == (
        f"{name} is not correct after 3 downloads. Send this message to support: "
        f"{CHECKPOINT}, {name}."
    )
    assert not (tmp_path / "model" / name).exists()


@synchronous
async def test_storage_that_always_refuses_stops_the_download(tmp_path, storage, api):
    storage.refuse_all = True
    with pytest.raises(DownloadError, match="storage.googleapis.com"):
        await run(tmp_path, storage, api, parallel=1)


@synchronous
async def test_storage_that_ignores_the_range_restarts_the_file(tmp_path, storage, api):
    name = "model-00002-of-00002.safetensors"
    directory = tmp_path / "model"
    directory.mkdir()
    (directory / MARKER).write_text(json.dumps({"checkpoint_id": CHECKPOINT}))
    (directory / (name + ".part")).write_bytes(FILES[name][:100])
    storage.ignore_range = True
    seen = Seen()
    await run(tmp_path, storage, api, seen=seen)
    assert_complete(directory)
    assert seen.bytes == sum(map(len, FILES.values()))


@synchronous
async def test_too_little_disk_stops_before_the_first_byte(tmp_path, storage, api, monkeypatch):
    usage = type("Usage", (), {"free": 10})
    monkeypatch.setattr(downloads.shutil, "disk_usage", lambda _: usage)
    with pytest.raises(InsufficientSpace) as raised:
        await run(tmp_path, storage, api)
    assert str(raised.value).startswith(f"The disk at {tmp_path / 'model'} has 10 B free.")
    assert storage.requests == []


@synchronous
async def test_a_full_disk_stops_and_keeps_what_is_on_disk(tmp_path, storage, api, monkeypatch):
    real = downloads._File._write
    writes = 0

    class Full:
        def write(self, _):
            raise OSError(errno.ENOSPC, "No space left on device")

    async def fill(self, out, chunks):
        nonlocal writes
        writes += 1
        return await real(self, cast(IO[bytes], Full()) if writes == 3 else out, chunks)

    monkeypatch.setattr(downloads._File, "_write", fill)
    with pytest.raises(InsufficientSpace, match="^The disk is full"):
        await run(tmp_path, storage, api, parallel=1)


@synchronous
async def test_a_deleted_checkpoint_stops_the_download(tmp_path, storage, api):
    api.ttl_s = 30
    original = api.__call__

    async def deleted_after_the_first(paths):
        result = await original(paths)
        api.deleted = True
        return result

    with pytest.raises(NotFound):
        await downloads_run(tmp_path, storage, deleted_after_the_first, Seen())


@synchronous
async def test_a_second_download_into_the_directory_is_refused(tmp_path, storage, api):
    directory = tmp_path / "model"
    directory.mkdir()
    with downloads._locked(directory, CHECKPOINT, verify=False):
        with pytest.raises(DownloadError) as raised:
            await run(tmp_path, storage, api)
    assert str(raised.value) == (
        f"A download into {directory} is already in progress (process {os.getpid()})."
    )


@synchronous
async def test_a_directory_of_another_checkpoint_is_refused(tmp_path, storage, api):
    directory = tmp_path / "model"
    directory.mkdir()
    (directory / MARKER).write_text(json.dumps({"checkpoint_id": "other"}))
    with pytest.raises(DownloadError) as raised:
        await run(tmp_path, storage, api)
    assert str(raised.value) == (
        f"{directory} holds files of checkpoint other. Use a different directory."
    )


@synchronous
async def test_a_directory_with_other_files_and_no_marker_needs_verify(tmp_path, storage, api):
    directory = tmp_path / "model"
    directory.mkdir()
    (directory / "config.json").write_bytes(FILES["config.json"])
    wrong = bytearray(FILES["model-00001-of-00002.safetensors"])
    wrong[7] ^= 1
    (directory / "model-00001-of-00002.safetensors").write_bytes(wrong)
    with pytest.raises(DownloadError, match="is not empty and is not a Looper download"):
        await run(tmp_path, storage, api)
    await run(tmp_path, storage, api, verify=True)
    assert_complete(directory)
    fetched = {path for path, _ in storage.requests}
    assert "config.json" not in fetched
    assert "model-00001-of-00002.safetensors" in fetched


@pytest.mark.parametrize("deleted", ["finished", "partial", "directory"])
@synchronous
async def test_deleted_files_download_again(tmp_path, storage, api, deleted):
    name = "model-00001-of-00002.safetensors"
    directory = tmp_path / "model"
    await run(tmp_path, storage, api)
    storage.requests.clear()
    if deleted == "finished":
        (directory / name).unlink()
    elif deleted == "partial":
        (directory / name).rename(directory / (name + ".part"))
        (directory / (name + ".part")).unlink()
    else:
        for path in sorted(directory.rglob("*"), reverse=True):
            path.rmdir() if path.is_dir() else path.unlink()
        directory.rmdir()
    await run(tmp_path, storage, api)
    assert_complete(directory)
    fetched = sorted({path for path, _ in storage.requests})
    assert fetched == (FETCHED if deleted == "directory" else [name])


@synchronous
async def test_a_deleted_marker_needs_verify_which_keeps_the_correct_files(tmp_path, storage, api):
    directory = tmp_path / "model"
    await run(tmp_path, storage, api)
    (directory / MARKER).unlink()
    storage.requests.clear()
    with pytest.raises(DownloadError, match="is not empty"):
        await run(tmp_path, storage, api)
    await run(tmp_path, storage, api, verify=True)
    assert storage.requests == []
    assert_complete(directory)


@synchronous
async def test_verify_finds_a_changed_file_of_the_right_size(tmp_path, storage, api):
    name = "model-00002-of-00002.safetensors"
    directory = tmp_path / "model"
    await run(tmp_path, storage, api)
    changed = bytearray(FILES[name])
    changed[-1] ^= 1
    (directory / name).write_bytes(changed)
    await run(tmp_path, storage, api)
    assert (directory / name).read_bytes() == changed  # trusted by its size
    await run(tmp_path, storage, api, verify=True)
    assert_complete(directory)


@pytest.mark.parametrize(
    "path",
    ["../escape", "/etc/passwd", "a/../../b", "a\\b", "C:/x", "", MARKER, "x.part", "a//b", "./a"],
)
def test_a_path_that_leaves_the_directory_is_refused(path):
    with pytest.raises(DownloadError, match="not safe"):
        downloads.check_path(path)


@synchronous
async def test_no_url_reaches_a_notice_or_a_log(tmp_path, storage, api, caplog):
    caplog.set_level(logging.DEBUG)
    storage.cut["config.json"] = [3]
    storage.unavailable = 2
    storage.corrupt["nested/tokenizer.json"] = 1
    seen = Seen()
    await run(tmp_path, storage, api, seen=seen)
    assert seen.notices
    assert all(SECRET not in notice for notice in seen.notices)
    assert SECRET not in caplog.text
    assert SECRET not in repr(api.calls)


def test_sizes_read_as_people_write_them():
    assert [downloads.size(n) for n in (512, 42_000_000, 712e9, 1.84e12)] == [
        "512 B",
        "42 MB",
        "712 GB",
        "1.84 TB",
    ]
