"""``looper``, the SDK's command. It has one
subcommand, ``looper download``, a thin layer on
``ServiceClient.download_checkpoint``: it finds the checkpoint by name or
ID, waits for its upload, and shows the progress.

Exit codes: 0 once every file is complete and correct, 1 when an
error stops the download, and 130 after Ctrl-C. A network problem never
stops it, so a script that needs a time limit uses ``timeout``.
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import os
import shlex
import sys
import time
import uuid
from typing import TextIO

from looper_sdk.checkpoints import Checkpoint, CheckpointState
from looper_sdk.download import DownloadError, DownloadPlan, InsufficientSpace, size
from looper_sdk.errors import LooperError, NotFound, SessionExpired, Unauthenticated
from looper_sdk.service import API_KEY_VARIABLE, DEFAULT_BASE_URL, ServiceClient

# Where the command finds the API, when it is not production's.
BASE_URL_VARIABLE = "LOOPER_BASE_URL"
# A log gets a summary line this often.
SUMMARY_S = 60.0
# A terminal's bar is drawn again at most this often.
REDRAW_S = 0.2
# The speed is the mean over this window.
SPEED_WINDOW_S = 20.0
BAR_WIDTH = 20


def _duration(seconds: float) -> str:
    seconds = int(seconds)
    if seconds >= 3600:
        return f"{seconds // 3600} h {seconds % 3600 // 60} min"
    if seconds >= 60:
        return f"{seconds // 60} min"
    return f"{seconds} s"


class _Progress:
    """``DownloadObserver`` for a person: a bar on a terminal, and lines a
    log can keep everywhere else."""

    def __init__(self, out: TextIO, checkpoint: Checkpoint, resume: str):
        self.out, self.checkpoint, self.resume = out, checkpoint, resume
        self.terminal = out.isatty()
        self.total = self.done = self.files = self.finished_files = 0
        self.samples: collections.deque[tuple[float, int]] = collections.deque()
        self.drawn_at = self.summary_at = 0.0
        self.bar_shown = False

    def _line(self, text: str) -> None:
        if self.bar_shown:
            self.out.write("\r\033[K")
            self.bar_shown = False
        self.out.write(text + "\n")
        self.out.flush()

    def started(self, plan: DownloadPlan) -> None:
        self.total, self.done, self.files = plan.size_bytes, plan.on_disk_bytes, plan.files
        self.summary_at = time.monotonic()
        self.samples.append((time.monotonic(), self.done))
        version = self.checkpoint.version
        self._line(
            f"Checkpoint {self.checkpoint.id} ({self.checkpoint.name}, step {version.step}): "
            f"{plan.files} files, {size(plan.size_bytes)}"
        )
        self._line(f"To continue after a stop, run: {self.resume}")

    def advanced(self, size_bytes: int) -> None:
        self.done += size_bytes
        now = time.monotonic()
        self.samples.append((now, self.done))
        while len(self.samples) > 2 and now - self.samples[0][0] > SPEED_WINDOW_S:
            self.samples.popleft()
        if self.terminal and now - self.drawn_at >= REDRAW_S:
            self.drawn_at = now
            self.out.write("\r\033[K" + self.status(bar=True))
            self.out.flush()
            self.bar_shown = True
        elif not self.terminal and now - self.summary_at >= SUMMARY_S:
            self.summary_at = now
            self._line(self.status(bar=False))

    def finished(self, path: str) -> None:
        self.finished_files += 1
        if not self.terminal:
            self._line(f"{path} is complete ({self.finished_files} files this run).")

    def notice(self, message: str) -> None:
        self._line(message)

    def status(self, *, bar: bool) -> str:
        fraction = self.done / self.total if self.total else 1.0
        parts = [f"{fraction:.0%}", f"{size(self.done)} of {size(self.total)}"]
        (first_at, first), (last_at, last) = self.samples[0], self.samples[-1]
        if last_at > first_at and last > first:
            speed = (last - first) / (last_at - first_at)
            parts += [f"{size(speed)}/s", f"{_duration((self.total - self.done) / speed)} left"]
        if bar:
            filled = int(BAR_WIDTH * fraction)
            parts.insert(0, "[" + "█" * filled + "░" * (BAR_WIDTH - filled) + "]")
        return "  ".join(parts)

    def stopped(self) -> str:
        return f"{size(self.done)} of {size(self.total)} is on disk"


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="looper", description="Looper's command")
    commands = root.add_subparsers(dest="command", required=True)
    download = commands.add_parser(
        "download",
        help="download a weights checkpoint into a directory",
        description="Download a weights checkpoint. If the download stops, run the same "
        "command again: it continues from the bytes on disk. The API key comes from "
        f"${API_KEY_VARIABLE}.",
    )
    download.add_argument("--project", required=True, help="the checkpoint's project")
    download.add_argument(
        "--verify",
        action="store_true",
        help="examine the checksum of each file on disk, and accept a directory with no "
        "marker file",
    )
    download.add_argument(
        "--parallel", type=int, default=8, help="files to download at the same time (default 8)"
    )
    download.add_argument("checkpoint", help="the name or the ID of a weights checkpoint")
    download.add_argument("directory", help="the download directory; made if it does not exist")
    return root


async def _find(service: ServiceClient, checkpoint: str) -> Checkpoint:
    try:
        uuid.UUID(checkpoint)
    except ValueError:
        return await service.get_checkpoint_by_name(checkpoint)
    return await service.get_checkpoint(checkpoint)


async def _download(args: argparse.Namespace, out: TextIO, state: dict[str, object]) -> int:
    base_url = os.environ.get(BASE_URL_VARIABLE) or DEFAULT_BASE_URL
    service = await ServiceClient.Config(base_url=base_url, project=args.project).build()
    try:
        try:
            checkpoint = await _find(service, args.checkpoint)
        except NotFound:
            print(
                f"Checkpoint {args.checkpoint} does not exist in project {args.project}.",
                file=sys.stderr,
            )
            return 1
        if checkpoint.kind != "weights":
            print(
                f"{checkpoint.id} is a state checkpoint. Only weights checkpoints can be "
                "downloaded. Use save_weights to make one.",
                file=sys.stderr,
            )
            return 1
        if checkpoint.state is CheckpointState.UPLOADING:
            print(
                "The upload of this checkpoint is not complete. The download starts when it "
                "is complete.",
                file=out,
            )
            checkpoint = await service.wait_checkpoint(checkpoint.id)
        elif checkpoint.state is CheckpointState.FAILED:
            print(f"Checkpoint {checkpoint.id} failed: {checkpoint.failure}", file=sys.stderr)
            return 1
        resume = shlex.join(
            ["looper", "download", "--project", args.project, checkpoint.id, args.directory]
        )
        progress = _Progress(out, checkpoint, resume)
        state.update(progress=progress, resume=resume, checkpoint=checkpoint.id)
        result = await service.download_checkpoint(
            checkpoint.id, args.directory, verify=args.verify, parallel=args.parallel,
            observer=progress,
        )  # fmt: skip
        progress.notice(
            f"Done. {result.files} files, {size(result.size_bytes)} in {args.directory}"
        )
        return 0
    finally:
        await service.close()


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if not 1 <= args.parallel <= 64:
        print("--parallel must be from 1 to 64.", file=sys.stderr)
        return 1
    state: dict[str, object] = {}
    try:
        return asyncio.run(_download(args, sys.stdout, state))
    except KeyboardInterrupt:
        progress = state.get("progress")
        on_disk = f" {progress.stopped()}." if isinstance(progress, _Progress) else ""
        resume = state.get("resume")
        again = f" To continue, run: {resume}" if resume else ""
        print(f"\nStopped.{on_disk}{again}", file=sys.stderr)
        return 130
    except Unauthenticated:
        print(f"No valid API key. Set {API_KEY_VARIABLE}.", file=sys.stderr)
    except InsufficientSpace as error:
        again = f" To continue, make space and run: {state['resume']}" if "resume" in state else ""
        print(f"{error}{again}", file=sys.stderr)
    except DownloadError as error:
        print(str(error), file=sys.stderr)
    except SessionExpired:
        # Its heartbeats stopped for a minute, as while the computer slept.
        again = f" To continue, run: {state['resume']}" if "resume" in state else ""
        print(f"The session with Looper expired.{again}", file=sys.stderr)
    except NotFound:
        checkpoint = state.get("checkpoint", args.checkpoint)
        print(
            f"Checkpoint {checkpoint} does not exist. It was deleted or its TTL expired. "
            "The files on disk are not complete.",
            file=sys.stderr,
        )
    except LooperError as error:
        print(f"{error.message}", file=sys.stderr)
    return 1


def entry() -> None:
    sys.exit(main())


if __name__ == "__main__":
    entry()
