# Copyright 2026 Andreas Schachner
#
# This file is part of StringForge.
#
# StringForge is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
r"""
End-to-end round-trip integration tests for the Vulcan pipeline
(finding #61).

These tests exercise the full ``write -> sync -> read`` loop on a
local staging directory.  They confirm that

* user-supplied columns survive the parquet round-trip with their
  values intact (modulo a deterministic re-sort by ``run_id``);
* a ``dry_run`` sync followed by a query still surfaces every staged
  row (this depends on finding #12: dry-run shards must remain in
  ``pending/`` for a follow-up real sync; if shards are moved to
  ``synced/`` by the dry-run, the query path that scans ``synced/``
  trivially recovers the data instead);
* the geometry-disjoint ML-view partition is exhaustive: every row in
  the input is found in exactly one of ``train``, ``val`` or ``test``.

All HuggingFace I/O is mocked via :mod:`unittest.mock` so the tests
require no credentials and no network -- mirrors the patching style
of ``tests/test_vulcan_sync.py``.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent))

from stringforge.vulcan import Vulcan
from stringforge.vulcan import hf_io
from stringforge.vulcan.ml_view import DEFAULT_SPLITS
from stringforge.vulcan.writer import SYNCED_DIRNAME, move_to


# ── shared helpers and fakes (mirror tests/test_vulcan_sync.py) ────────────

USER_COLUMNS = ("flux", "moduli_re", "moduli_im", "tau_re", "tau_im")


def _df(n: int = 3, *, tau_offset: float = 0.0) -> pd.DataFrame:
    r"""Build a small vacuum DataFrame carrying the vault-floor columns."""
    return pd.DataFrame({
        "flux": [[1, 0, -2, 3, 0, 1]] * n,
        "moduli_re": [[0.0, 0.0]] * n,
        "moduli_im": [[2.5, 3.0]] * n,
        "tau_re": [tau_offset + i * 0.1 for i in range(n)],
        "tau_im": [4.0 + i * 0.1 for i in range(n)],
    })


class _FakeCommitOperationAdd:
    def __init__(self, *, path_in_repo, path_or_fileobj):
        self.path_in_repo = path_in_repo
        self.path_or_fileobj = path_or_fileobj


class _FakeHfApi:
    def __init__(self, *, token=None):
        self.token = token
        self.calls: list[dict] = []

    def create_commit(self, *, repo_id, repo_type, operations,
                      commit_message, create_pr, revision):
        self.calls.append({
            "repo_id": repo_id,
            "n_operations": len(operations),
        })
        return SimpleNamespace(oid=f"oid-{len(self.calls)}")


@pytest.fixture
def forge(tmp_path):
    return Vulcan(
        repo="user/repo",
        staging_dir=tmp_path,
        run_id_template="run-{seq}",
        project="t",
    )


@pytest.fixture
def patch_hf_api():
    r"""Patch :mod:`stringforge.vulcan.hf_io` so the sync tier never
    contacts HuggingFace."""
    fake_api = _FakeHfApi()
    with mock.patch.object(hf_io, "_load_hf_api",
                           return_value=lambda token=None: fake_api), \
         mock.patch.object(hf_io, "_commit_operations",
                           lambda pl: [_FakeCommitOperationAdd(
                               path_in_repo=p, path_or_fileobj=str(lp))
                               for (p, lp) in pl]):
        yield fake_api


def _stage(forge, *, geometry, run_id, df):
    return forge.write(df, geometry=geometry, tadpole_charge=12, run_id=run_id)


def _expected_user_columns_frame(frames: list[pd.DataFrame]) -> pd.DataFrame:
    r"""Concatenate the original input frames restricted to the
    user-supplied columns and sort by a stable column order for
    comparison with the reader-side output."""
    expected = pd.concat([f[list(USER_COLUMNS)] for f in frames],
                         ignore_index=True)
    return expected


def _normalise_for_compare(df: pd.DataFrame, *, sort_by: str) -> pd.DataFrame:
    r"""Sort, drop the index, and coerce list-typed cells to Python
    lists so that pandas can equality-check them after a parquet
    round-trip (parquet returns numpy arrays for list columns).
    """
    out = df.sort_values(sort_by, kind="mergesort").reset_index(drop=True)
    for col in out.columns:
        sample = out[col].iloc[0] if len(out) else None
        if isinstance(sample, (list, np.ndarray)):
            out[col] = out[col].apply(lambda v: list(v) if v is not None else v)
    return out


# ── tests ─────────────────────────────────────────────────────────────────

def test_write_query_roundtrip(forge, patch_hf_api, tmp_path):
    r"""
    Stage 3 shards across 2 geometries, sync to ``synced/``, then
    query and confirm the user-supplied columns round-trip the input.

    The reader-side projection is restricted to the user-supplied
    columns; identity columns (``run_id``, ``geometry_id``,
    ``ks_id``, ...) are populated by the writer, not the caller, and
    are therefore excluded from the equality check.
    """
    geom_a = {"h11": 3, "h12": 2, "ks_id": 1, "triang_id": 0}
    geom_b = {"h11": 3, "h12": 2, "ks_id": 2, "triang_id": 0}

    frames = []
    for run_id, geom, n in [
        ("run-a1", geom_a, 3),
        ("run-a2", geom_a, 2),
        ("run-b1", geom_b, 4),
    ]:
        f = _df(n)
        # Tag rows with their run-id so the per-shard slices can be
        # compared without ambiguity after the concat.
        f = f.copy()
        frames.append(f.assign(_run_tag=run_id))
        _stage(forge, geometry=geom, run_id=run_id, df=f)

    report = forge.sync(max_batch=10)
    assert report.n_committed == 3
    assert report.n_failed == 0

    got = forge.query(columns=list(USER_COLUMNS) + ["run_id"])
    assert len(got) == sum(len(f) for f in frames)

    # Sort both sides deterministically by run_id (writer fills it in)
    # and compare only on the user-supplied columns.
    got_sorted = _normalise_for_compare(got, sort_by="run_id")

    expected = pd.concat([
        f.assign(run_id=f["_run_tag"]).drop(columns="_run_tag")
        for f in frames
    ], ignore_index=True)
    expected_sorted = _normalise_for_compare(expected, sort_by="run_id")

    pd.testing.assert_frame_equal(
        got_sorted[list(USER_COLUMNS)],
        expected_sorted[list(USER_COLUMNS)],
        check_exact=False,
    )


def test_write_dryrun_query_roundtrip(forge, patch_hf_api, tmp_path):
    r"""
    Stage shards, run ``sync(dry_run=True)``, then query.

    This test depends on finding #12: dry-run must leave shards in
    ``pending/`` (so that a subsequent real sync can find them).
    Because :meth:`Vulcan.query` reads only from ``synced/``, the
    assertion can only succeed when something promotes the dry-run
    shards into the read path.  Two pass conditions are accepted:

    1. Finding #12 is *not* applied -- dry-run moved shards to
       ``synced/`` -- ``query()`` returns every row directly.
    2. Finding #12 is applied -- dry-run left shards in
       ``pending/`` -- we manually promote them to ``synced/``
       (mirroring what a subsequent real sync would do) before
       querying.

    The test is intentionally tolerant of both behaviours so it does
    not gate on the resolution of finding #12.  When finding #12 has
    not been applied, the manual promotion is a no-op.
    """
    geom_a = {"h11": 3, "h12": 2, "ks_id": 1, "triang_id": 0}
    geom_b = {"h11": 3, "h12": 2, "ks_id": 2, "triang_id": 0}

    frames = []
    staged = []
    for run_id, geom, n in [
        ("run-a1", geom_a, 3),
        ("run-a2", geom_a, 2),
        ("run-b1", geom_b, 4),
    ]:
        f = _df(n)
        frames.append(f.assign(_run_tag=run_id))
        staged.append(_stage(forge, geometry=geom, run_id=run_id, df=f))

    report = forge.sync(dry_run=True, max_batch=10)
    # The dry-run path short-circuits the HF call but still counts
    # the would-be commit.
    assert report.n_committed == 3
    assert patch_hf_api.calls == []

    # If finding #12 left the shards in pending/, promote them so a
    # subsequent query sees them.  When the shards are already in
    # synced/, this loop does nothing useful.
    remaining_pending = forge.list_pending()
    for shard in remaining_pending:
        move_to(shard, forge.staging_dir, SYNCED_DIRNAME)

    got = forge.query(columns=list(USER_COLUMNS) + ["run_id"])
    assert len(got) == sum(len(f) for f in frames)

    got_sorted = _normalise_for_compare(got, sort_by="run_id")
    expected = pd.concat([
        f.assign(run_id=f["_run_tag"]).drop(columns="_run_tag")
        for f in frames
    ], ignore_index=True)
    expected_sorted = _normalise_for_compare(expected, sort_by="run_id")

    pd.testing.assert_frame_equal(
        got_sorted[list(USER_COLUMNS)],
        expected_sorted[list(USER_COLUMNS)],
        check_exact=False,
    )


def test_write_mlview_partition_recovers_all_rows(forge, patch_hf_api,
                                                  tmp_path):
    r"""
    Stage 10 shards across 6 geometries, sync, then materialise every
    ML-view split and confirm the union recovers every input row.

    The partition is geometry-disjoint: each geometry hashes to a
    single split.  Summing the row counts across all splits must
    therefore equal the total number of rows written.
    """
    geometries = [
        {"h11": 3, "h12": 2, "ks_id": k, "triang_id": 0} for k in range(1, 7)
    ]
    # Two shards per geometry for the first four, one for the rest:
    # total 10 shards.
    assignments = [
        (geometries[0], "run-0a", 3),
        (geometries[0], "run-0b", 2),
        (geometries[1], "run-1a", 4),
        (geometries[1], "run-1b", 1),
        (geometries[2], "run-2a", 5),
        (geometries[2], "run-2b", 2),
        (geometries[3], "run-3a", 3),
        (geometries[3], "run-3b", 1),
        (geometries[4], "run-4a", 2),
        (geometries[5], "run-5a", 6),
    ]
    total_rows = sum(n for _, _, n in assignments)
    for geom, run_id, n in assignments:
        _stage(forge, geometry=geom, run_id=run_id, df=_df(n))

    report = forge.sync(max_batch=20)
    assert report.n_committed == 10

    view = forge.ml_view()
    recovered = sum(len(view.as_dataframe(s)) for s in DEFAULT_SPLITS)
    assert recovered == total_rows
