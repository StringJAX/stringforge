# Copyright 2026 Andreas Schachner
#
# This file is part of StringForge.
#
# StringForge is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
r"""
Tests for :mod:`stringforge.vulcan.ratelimit`.

* Rolling-window counter: pruning, remaining-slot accounting,
  ``seconds_until_slot``.
* Persistent budget across process restarts (simulated by
  re-instantiating :class:`CommitBudget` against the same state file).
* ``sleep_with_backoff`` exponential growth + ``Retry-After`` floor.
* ``parse_retry_after`` accepts integer seconds, rejects HTTP-date.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent))

from stringforge.vulcan.ratelimit import (
    CommitBudget,
    CommitWindow,
    DEFAULT_BUDGET,
    INITIAL_BACKOFF_S,
    MAX_BACKOFF_S,
    parse_retry_after,
    sleep_with_backoff,
)


# ── CommitWindow ──────────────────────────────────────────────────────────

def test_window_starts_empty():
    win = CommitWindow(budget=3, window_s=10.0)
    assert win.n_recent(now=1000.0) == 0
    assert win.remaining(now=1000.0) == 3


def test_window_records_reservations():
    win = CommitWindow(budget=3, window_s=10.0)
    assert win.reserve(now=1000.0)
    assert win.reserve(now=1001.0)
    assert win.reserve(now=1002.0)
    assert not win.reserve(now=1003.0)  # budget exhausted
    assert win.n_recent(now=1003.0) == 3
    assert win.remaining(now=1003.0) == 0


def test_window_prunes_old_entries():
    win = CommitWindow(budget=2, window_s=10.0)
    assert win.reserve(now=1000.0)
    assert win.reserve(now=1001.0)
    assert not win.reserve(now=1002.0)
    # 11s later the oldest entry has aged out.
    assert win.reserve(now=1011.0)


def test_seconds_until_slot_zero_when_room():
    win = CommitWindow(budget=2, window_s=10.0)
    assert win.seconds_until_slot(now=1000.0) == 0.0


def test_seconds_until_slot_reports_remaining_window():
    win = CommitWindow(budget=1, window_s=10.0)
    win.reserve(now=1000.0)
    assert pytest.approx(win.seconds_until_slot(now=1003.0), abs=1e-6) == 7.0


def test_window_roundtrips_through_dict():
    win = CommitWindow(budget=5, window_s=60.0)
    for t in (10.0, 20.0, 30.0):
        win.reserve(now=t)
    blob = win.to_dict()
    restored = CommitWindow.from_dict(blob)
    assert restored.budget == 5
    assert restored.window_s == 60.0
    assert list(restored.timestamps) == [10.0, 20.0, 30.0]


# ── CommitBudget persistence ──────────────────────────────────────────────

def test_budget_persists_across_instances(tmp_path):
    state = tmp_path / "budget.json"
    b1 = CommitBudget(state, budget=2, window_s=1000.0)
    assert b1.reserve(now=1000.0)
    assert b1.reserve(now=1001.0)
    assert not b1.reserve(now=1002.0)
    # A fresh CommitBudget pointing at the same state file sees the
    # same exhausted budget.
    b2 = CommitBudget(state, budget=2, window_s=1000.0)
    assert b2.remaining(now=1002.0) == 0


def test_budget_recovers_when_window_advances(tmp_path):
    state = tmp_path / "budget.json"
    b = CommitBudget(state, budget=1, window_s=10.0)
    assert b.reserve(now=1000.0)
    assert b.remaining(now=1005.0) == 0
    # 10s later the slot is back.
    assert b.remaining(now=1010.5) == 1
    assert b.reserve(now=1010.5)


def test_budget_handles_corrupt_state_file(tmp_path):
    state = tmp_path / "budget.json"
    state.write_text("not json")
    b = CommitBudget(state, budget=3, window_s=10.0)
    # Falls back to a fresh window.
    assert b.remaining(now=1000.0) == 3


def test_default_budget_matches_module_default():
    assert DEFAULT_BUDGET == 90


# ── sleep_with_backoff ────────────────────────────────────────────────────

def test_backoff_grows_exponentially():
    calls: list[float] = []
    for attempt in range(4):
        dt = sleep_with_backoff(attempt, sleep_fn=calls.append)
        assert dt == calls[-1]
    assert calls == [
        INITIAL_BACKOFF_S,
        INITIAL_BACKOFF_S * 2,
        INITIAL_BACKOFF_S * 4,
        INITIAL_BACKOFF_S * 8,
    ]


def test_backoff_respects_retry_after_floor():
    calls: list[float] = []
    dt = sleep_with_backoff(
        attempt=0, retry_after_s=5.0, sleep_fn=calls.append,
    )
    assert dt == 5.0
    assert calls == [5.0]


def test_backoff_capped_by_max():
    calls: list[float] = []
    dt = sleep_with_backoff(
        attempt=100, sleep_fn=calls.append,  # would overflow if uncapped
    )
    assert dt == MAX_BACKOFF_S
    assert calls == [MAX_BACKOFF_S]


def test_backoff_rejects_negative_attempt():
    with pytest.raises(ValueError):
        sleep_with_backoff(-1, sleep_fn=lambda _: None)


# ── parse_retry_after ─────────────────────────────────────────────────────

def test_parse_retry_after_seconds():
    assert parse_retry_after("5") == 5.0
    assert parse_retry_after("0") == 0.0


def test_parse_retry_after_none():
    assert parse_retry_after(None) is None


def test_parse_retry_after_unparseable():
    # HF returns integer seconds; an HTTP-date is rare but must not
    # crash.
    assert parse_retry_after("Wed, 01 Jan 2026 00:00:00 GMT") is None
    assert parse_retry_after("") is None
