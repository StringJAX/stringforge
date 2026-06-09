# Copyright 2026 Andreas Schachner
#
# This file is part of StringForge.
#
# StringForge is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

r"""
Rolling-window commit-rate budget for the Vulcan sync tier.

HuggingFace caps direct commits at **100 / hour per repo**.  Vulcan's
sync tier maintains a rolling-window counter (default ceiling: 90
commits/hour, leaving a 10-commit safety margin) and refuses to
``create_commit`` while the window is full.  The budget is persisted
to disk so multiple sync invocations (cron jobs, daemon restarts)
share the same accounting.

This module also exposes :func:`sleep_with_backoff`, an exponential-
backoff helper that honours ``Retry-After`` headers on HTTP 429 / 503
responses (whichever the sync tier classifies as rate-limit-like).
"""
from __future__ import annotations

import json
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Deque, Optional

#: Hard ceiling enforced by HuggingFace.  Our default budget sits a
#: safety margin below this.
HF_COMMITS_PER_HOUR_CAP: int = 100

#: Default budget consumed by a single sync run.  10-commit margin
#: below the HF cap leaves head-room for ad-hoc operations the user
#: may perform manually (e.g. README edits).
DEFAULT_BUDGET: int = 90

#: Default window over which the budget applies (seconds).
DEFAULT_WINDOW_S: float = 3600.0

#: Initial backoff for the exponential-backoff helper, in seconds.
INITIAL_BACKOFF_S: float = 1.0

#: Cap on a single backoff sleep.  Five minutes is enough to ride out
#: most transient HF outages; anything longer should error.
MAX_BACKOFF_S: float = 300.0


@dataclass
class CommitWindow:
    r"""
    Rolling-window record of recent commit timestamps for a single
    repo.  Persisted as a JSON file under the staging directory.

    Attributes:
        budget: Maximum commits permitted within ``window_s``.
        window_s: Window length in seconds.
        timestamps: Monotonic-clock timestamps of commits recorded so
            far.  Older entries are pruned lazily on every
            :meth:`reserve` call.
    """
    budget: int = DEFAULT_BUDGET
    window_s: float = DEFAULT_WINDOW_S
    timestamps: Deque[float] = field(default_factory=deque)

    def _prune(self, now: float) -> None:
        cutoff = now - self.window_s
        while self.timestamps and self.timestamps[0] < cutoff:
            self.timestamps.popleft()

    def n_recent(self, now: Optional[float] = None) -> int:
        r"""
        **Description:**
        Count commits inside the current rolling window.

        Args:
            now: Wall-clock override for tests.  Defaults to
                :func:`time.time`.

        Returns:
            int: Number of timestamps within the last ``window_s``
            seconds.
        """
        if now is None:
            now = time.time()
        self._prune(now)
        return len(self.timestamps)

    def remaining(self, now: Optional[float] = None) -> int:
        r"""
        **Description:**
        Compute the number of commits still available within the
        current rolling window.

        Args:
            now: Wall-clock override for tests.

        Returns:
            int: Non-negative slot count.
        """
        return max(0, self.budget - self.n_recent(now=now))

    def seconds_until_slot(self, now: Optional[float] = None) -> float:
        r"""
        **Description:**
        Compute the wall-clock delay before a new slot opens.

        Args:
            now: Wall-clock override for tests.

        Returns:
            float: ``0.0`` if a slot is currently free; otherwise the
            seconds until the oldest recorded commit ages out of the
            window.
        """
        if now is None:
            now = time.time()
        self._prune(now)
        if len(self.timestamps) < self.budget:
            return 0.0
        oldest = self.timestamps[0]
        return max(0.0, (oldest + self.window_s) - now)

    def reserve(self, now: Optional[float] = None) -> bool:
        r"""
        **Description:**
        Attempt to record a new commit at ``now``.

        Args:
            now: Wall-clock override for tests.

        Returns:
            bool: ``True`` if the slot was recorded; ``False`` if the
            budget is exhausted.
        """
        if now is None:
            now = time.time()
        self._prune(now)
        if len(self.timestamps) >= self.budget:
            return False
        self.timestamps.append(now)
        return True

    # ── persistence ────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "budget": int(self.budget),
            "window_s": float(self.window_s),
            "timestamps": [float(t) for t in self.timestamps],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CommitWindow":
        return cls(
            budget=int(data.get("budget", DEFAULT_BUDGET)),
            window_s=float(data.get("window_s", DEFAULT_WINDOW_S)),
            timestamps=deque(float(t) for t in data.get("timestamps", [])),
        )


class CommitBudget:
    r"""
    Process-safe wrapper around a :class:`CommitWindow` persisted to
    disk.  Multiple processes calling ``reserve()`` against the same
    state file see a consistent view modulo flush latency.  Within a
    single process the wrapper is thread-safe.

    Note: this is *advisory* rate limiting at the process boundary.
    The state file (``{state_path}``) is written atomically per-process,
    but two processes hitting the same file concurrently can race; the
    budget can therefore overshoot by up to (number of concurrent
    writers - 1) commits per window.  The cluster best-practice (see
    section "Best practices for cluster runs" in the docs) is to run a
    SINGLE head-node sync process, which makes this advisory-only
    guarantee tight in practice.
    """

    def __init__(
        self,
        state_path: Path,
        *,
        budget: int = DEFAULT_BUDGET,
        window_s: float = DEFAULT_WINDOW_S,
    ) -> None:
        self.state_path = Path(state_path)
        self._lock = threading.Lock()
        self._default_budget = int(budget)
        self._default_window_s = float(window_s)

    def _load(self) -> CommitWindow:
        if self.state_path.is_file():
            try:
                data = json.loads(self.state_path.read_text())
                win = CommitWindow.from_dict(data)
                # Allow programmatic override of budget without losing
                # the recorded history.
                win.budget = self._default_budget
                win.window_s = self._default_window_s
                return win
            except (json.JSONDecodeError, OSError):
                pass
        return CommitWindow(
            budget=self._default_budget, window_s=self._default_window_s,
        )

    def _save(self, win: CommitWindow) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        tmp.write_text(json.dumps(win.to_dict(), sort_keys=True))
        os.replace(tmp, self.state_path)

    def remaining(self, now: Optional[float] = None) -> int:
        with self._lock:
            win = self._load()
            return win.remaining(now=now)

    def seconds_until_slot(self, now: Optional[float] = None) -> float:
        with self._lock:
            win = self._load()
            return win.seconds_until_slot(now=now)

    def reserve(self, now: Optional[float] = None) -> bool:
        with self._lock:
            win = self._load()
            ok = win.reserve(now=now)
            if ok:
                self._save(win)
            return ok


def sleep_with_backoff(
    attempt: int,
    *,
    retry_after_s: Optional[float] = None,
    initial: float = INITIAL_BACKOFF_S,
    cap: float = MAX_BACKOFF_S,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> float:
    r"""
    Sleep for an exponentially-backed-off interval, honouring
    ``Retry-After`` when provided.

    Args:
        attempt: 0 for the first retry, 1 for the next, ...
        retry_after_s: Server-reported retry-after (HF returns this on
            429).  When supplied, the sleep duration is the maximum of
            this and the exponential value -- never less than what the
            server asked for.
        initial: Initial backoff (seconds).
        cap: Maximum sleep duration (seconds).
        sleep_fn: Override for tests.

    Returns:
        Actual sleep duration in seconds.
    """
    if attempt < 0:
        raise ValueError("attempt must be >= 0")
    base = min(cap, initial * (2 ** attempt))
    if retry_after_s is not None:
        base = max(base, float(retry_after_s))
    sleep_fn(base)
    return base


def parse_retry_after(value: Optional[str]) -> Optional[float]:
    r"""
    Parse an HTTP ``Retry-After`` header value into seconds.

    The header may be either an integer (seconds) or an HTTP-date.
    Only the integer form is honoured here -- HF returns seconds.
    Unparseable values return ``None``.
    """
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
