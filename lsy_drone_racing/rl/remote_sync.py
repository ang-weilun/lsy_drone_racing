"""Thin rclone wrapper for off-instance checkpoint storage.

rclone is invoked via subprocess so the same code works against R2, S3,
GCS, B2, etc. — the user picks the backend by writing the rclone config
file. We never parse rclone output for correctness; we trust the exit
code and let stdout/stderr surface for debugging.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class RcloneSync:
    """Background uploader + foreground downloader."""

    remote: str
    """rclone remote spec, e.g. ``r2:lsy-drone-racing/ckpts/run-xyz``."""

    rclone_bin: str = "rclone"

    def __post_init__(self) -> None:
        if shutil.which(self.rclone_bin) is None:
            log.warning(
                "rclone not found on PATH (looked for %r). Remote sync will fail; "
                "training will still run locally.",
                self.rclone_bin,
            )

    def upload_async(self, local_path: Path) -> threading.Thread:
        """Fire-and-forget copy of ``local_path`` to the remote.

        Returns the thread so the caller can ``.join()`` during shutdown
        if it wants to guarantee the last ckpt made it off-instance.
        """
        t = threading.Thread(
            target=self._copy, args=(local_path,), name=f"rclone-up-{local_path.name}", daemon=True
        )
        t.start()
        return t

    def upload_sync(self, local_path: Path) -> None:
        """Block until upload completes. Used on emergency-save."""
        self._copy(local_path)

    def download_latest(self, dest_dir: Path, glob: str = "ckpt-*.pt") -> Path | None:
        """Pull the newest matching ckpt from the remote into ``dest_dir``.

        Returns the local path, or ``None`` if the remote has nothing.
        """
        dest_dir.mkdir(parents=True, exist_ok=True)
        latest = self._latest_remote_name(glob)
        if latest is None:
            log.info("No remote ckpt found under %s matching %s", self.remote, glob)
            return None
        dest = dest_dir / latest
        cmd = [self.rclone_bin, "copyto", f"{self.remote}/{latest}", str(dest)]
        log.info("Downloading %s -> %s", cmd[-2], dest)
        subprocess.run(cmd, check=True)
        return dest

    def prune_remote(self, keep: int, glob: str = "ckpt-*.pt") -> None:
        """Delete all but the ``keep`` newest ckpts on the remote."""
        names = self._list_remote(glob)
        for old in names[:-keep]:
            cmd = [self.rclone_bin, "deletefile", f"{self.remote}/{old}"]
            log.info("Pruning remote ckpt %s", old)
            # deletefile returns non-zero if missing; tolerate that.
            subprocess.run(cmd, check=False)

    def _copy(self, local_path: Path) -> None:
        cmd = [self.rclone_bin, "copyto", str(local_path), f"{self.remote}/{local_path.name}"]
        log.info("Uploading %s -> %s", local_path, cmd[-1])
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            log.error("rclone upload failed (exit %d): %s", e.returncode, e)

    def _list_remote(self, glob: str) -> list[str]:
        cmd = [self.rclone_bin, "lsf", "--include", glob, self.remote]
        try:
            out = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout
        except subprocess.CalledProcessError as e:
            log.warning("rclone lsf failed: %s", e)
            return []
        names = sorted(n.strip().rstrip("/") for n in out.splitlines() if n.strip())
        return names

    def _latest_remote_name(self, glob: str) -> str | None:
        names = self._list_remote(glob)
        return names[-1] if names else None
