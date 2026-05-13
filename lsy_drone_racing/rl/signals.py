"""SIGTERM/SIGINT trap for last-gasp checkpoint on spot preemption.

vast.ai (and most spot providers) deliver SIGTERM before yanking the
instance. We catch it, flip a flag, and let the training loop notice at
the next safe boundary. If the user double-taps Ctrl+C we honour it and
exit immediately rather than wedging.
"""

from __future__ import annotations

import logging
import signal
import threading
from types import FrameType

log = logging.getLogger(__name__)


class SignalGuard:
    """Cooperative shutdown flag plus a final hard-exit on double-signal."""

    def __init__(self) -> None:
        self._stop = threading.Event()
        self._hits = 0
        self._installed = False

    @property
    def should_stop(self) -> bool:
        return self._stop.is_set()

    def install(self) -> None:
        if self._installed:
            return
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, self._handle)
        self._installed = True
        log.info("Signal guard installed (SIGTERM, SIGINT).")

    def _handle(self, signum: int, frame: FrameType | None) -> None:
        self._hits += 1
        sig_name = signal.Signals(signum).name
        if self._hits == 1:
            log.warning("Received %s — requesting graceful checkpoint and exit.", sig_name)
            self._stop.set()
        else:
            log.warning(
                "Received %s again (hit #%d) — exiting immediately without further save.",
                sig_name,
                self._hits,
            )
            raise SystemExit(130)
