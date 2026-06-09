# Copyright 2026 Andreas Schachner
#
# This file is part of StringForge.
#
# StringForge is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
r"""
Tests for the Vulcan sync tier (:mod:`stringforge.vulcan.sync`,
:mod:`stringforge.vulcan.hf_io`).

These tests never contact HuggingFace.  ``huggingface_hub.HfApi`` and
``CommitOperationAdd`` are patched so the sync logic, batching, rate
budget, and 429 backoff can be exercised in isolation.
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
from stringforge.vulcan import sync as sync_module
from stringforge.vulcan.ratelimit import CommitBudget
from stringforge.vulcan.sync import (
    BUDGET_STATE_FILENAME,
    SyncReport,
    sync_pending,
)
from stringforge.vulcan.writer import (
    FAILED_DIRNAME,
    PENDING_DIRNAME,
    SYNCED_DIRNAME,
    list_pending,
)


# ── shared helpers ────────────────────────────────────────────────────────

def _df(n: int = 3):
    return pd.DataFrame({
        "flux": [[1, 0, -2, 3, 0, 1]] * n,
        "moduli_re": [[0.0, 0.0]] * n,
        "moduli_im": [[2.5, 3.0]] * n,
        "tau_re": [0.0] * n,
        "tau_im": [4.0] * n,
    })


@pytest.fixture
def forge(tmp_path):
    return Vulcan(
        repo="user/repo",
        staging_dir=tmp_path,
        run_id_template="run-{seq}",
        project="t",
    )


def _stage_n(forge, n):
    geom = {"h11": 3, "h12": 2, "ks_id": 1, "triang_id": 0}
    shards = []
    for i in range(n):
        s = forge.write(
            _df(), geometry=geom, tadpole_charge=12,
            run_id=f"run-{i}",
        )
        shards.append(s)
    return shards


# ── fake HfApi infrastructure ─────────────────────────────────────────────

class _FakeCommitOperationAdd:
    def __init__(self, *, path_in_repo, path_or_fileobj):
        self.path_in_repo = path_in_repo
        self.path_or_fileobj = path_or_fileobj


class _FakeResponse:
    def __init__(self, status_code, retry_after=None):
        self.status_code = status_code
        self.headers = {"Retry-After": retry_after} if retry_after else {}


class _Fake429(Exception):
    def __init__(self, retry_after="2"):
        super().__init__("rate limited")
        self.response = _FakeResponse(429, retry_after=retry_after)


class _Fake503(Exception):
    def __init__(self, retry_after=None):
        super().__init__("service unavailable")
        self.response = _FakeResponse(503, retry_after=retry_after)


class _FakeHfApi:
    def __init__(self, *, raise_429=False, raise_429_then_succeed=False,
                 raise_503=False, raise_503_then_succeed=False, token=None):
        self.token = token
        self.raise_429 = raise_429
        self.raise_429_then_succeed = raise_429_then_succeed
        self.raise_503 = raise_503
        self.raise_503_then_succeed = raise_503_then_succeed
        self.calls: list[dict] = []
        self._429_count = 0
        self._503_count = 0

    def create_commit(self, *, repo_id, repo_type, operations,
                      commit_message, create_pr, revision):
        call = {
            "repo_id": repo_id,
            "repo_type": repo_type,
            "n_operations": len(operations),
            "commit_message": commit_message,
            "create_pr": create_pr,
            "revision": revision,
        }
        self.calls.append(call)
        if self.raise_429:
            raise _Fake429()
        if self.raise_429_then_succeed:
            self._429_count += 1
            if self._429_count <= 1:
                raise _Fake429()
        if self.raise_503:
            raise _Fake503()
        if self.raise_503_then_succeed:
            self._503_count += 1
            if self._503_count <= 1:
                raise _Fake503()
        return SimpleNamespace(oid=f"oid-{len(self.calls)}")


@pytest.fixture
def patch_hf_api():
    r"""
    Patch the lazy loaders in hf_io so create_batched_commit uses
    fakes instead of the real huggingface_hub SDK.  Yields the
    instantiated _FakeHfApi so tests can inspect ``.calls``.
    """
    fake_api = _FakeHfApi()
    with mock.patch.object(hf_io, "_load_hf_api", return_value=lambda token=None: fake_api), \
         mock.patch.object(hf_io, "_commit_operations",
                           lambda pl: [_FakeCommitOperationAdd(path_in_repo=p, path_or_fileobj=str(lp))
                                       for (p, lp) in pl]):
        yield fake_api


@pytest.fixture
def patch_hf_api_429():
    fake_api = _FakeHfApi(raise_429=True)
    with mock.patch.object(hf_io, "_load_hf_api", return_value=lambda token=None: fake_api), \
         mock.patch.object(hf_io, "_commit_operations",
                           lambda pl: [_FakeCommitOperationAdd(path_in_repo=p, path_or_fileobj=str(lp))
                                       for (p, lp) in pl]):
        yield fake_api


@pytest.fixture
def patch_hf_api_429_then_ok():
    fake_api = _FakeHfApi(raise_429_then_succeed=True)
    with mock.patch.object(hf_io, "_load_hf_api", return_value=lambda token=None: fake_api), \
         mock.patch.object(hf_io, "_commit_operations",
                           lambda pl: [_FakeCommitOperationAdd(path_in_repo=p, path_or_fileobj=str(lp))
                                       for (p, lp) in pl]):
        yield fake_api


@pytest.fixture
def patch_hf_api_503():
    fake_api = _FakeHfApi(raise_503=True)
    with mock.patch.object(hf_io, "_load_hf_api", return_value=lambda token=None: fake_api), \
         mock.patch.object(hf_io, "_commit_operations",
                           lambda pl: [_FakeCommitOperationAdd(path_in_repo=p, path_or_fileobj=str(lp))
                                       for (p, lp) in pl]):
        yield fake_api


@pytest.fixture
def patch_hf_api_503_then_ok():
    fake_api = _FakeHfApi(raise_503_then_succeed=True)
    with mock.patch.object(hf_io, "_load_hf_api", return_value=lambda token=None: fake_api), \
         mock.patch.object(hf_io, "_commit_operations",
                           lambda pl: [_FakeCommitOperationAdd(path_in_repo=p, path_or_fileobj=str(lp))
                                       for (p, lp) in pl]):
        yield fake_api


@pytest.fixture(autouse=True)
def patch_sleep():
    r"""Speed up exponential-backoff tests by stubbing time.sleep."""
    with mock.patch("stringforge.vulcan.ratelimit.time.sleep", lambda _t: None):
        yield


# ── tests: empty / dry-run / basic batching ──────────────────────────────

def test_sync_empty_staging_returns_zero_report(forge, patch_hf_api):
    report = forge.sync()
    assert report.n_committed == 0
    assert report.n_failed == 0
    assert report.n_commits == 0
    assert patch_hf_api.calls == []


def test_sync_batches_pending_into_one_commit(forge, patch_hf_api):
    _stage_n(forge, 5)
    report = forge.sync(max_batch=10)
    assert report.n_commits == 1
    # One operation per parquet file (sidecars stay local).
    assert patch_hf_api.calls[0]["n_operations"] == 5
    assert report.n_committed == 5
    assert report.n_failed == 0
    assert report.n_remaining == 0


def test_sync_splits_into_max_batch_chunks(forge, patch_hf_api):
    _stage_n(forge, 7)
    report = forge.sync(max_batch=3)
    # 7 items -> ceil(7/3) = 3 commits, sizes 3, 3, 1.
    assert report.n_commits == 3
    assert [c["n_operations"] for c in patch_hf_api.calls] == [3, 3, 1]
    assert report.n_committed == 7


def test_sync_does_not_upload_sidecars(forge, patch_hf_api):
    r"""Sidecars are local-only; never committed to HuggingFace."""
    _stage_n(forge, 3)
    forge.sync(max_batch=10)
    # Inspect the operations that were "uploaded" (i.e. recorded by
    # the fake API): no path may end with ``.meta.json``.
    # We can't see operation paths through ``n_operations`` alone, so
    # this is the structural check: total ops == parquet count.
    assert patch_hf_api.calls[0]["n_operations"] == 3


def test_synced_shards_move_to_synced_dir(forge, patch_hf_api, tmp_path):
    _stage_n(forge, 2)
    forge.sync(max_batch=10)
    assert list_pending(forge.staging_dir) == []
    synced = list((tmp_path / SYNCED_DIRNAME).glob("*.parquet"))
    assert len(synced) == 2


def test_dry_run_does_not_invoke_hf(forge, patch_hf_api):
    _stage_n(forge, 3)
    report = forge.sync(dry_run=True)
    assert report.n_committed == 3
    # The dry-run path short-circuits before HfApi is called.
    assert patch_hf_api.calls == []


def test_dry_run_leaves_shards_in_pending(forge, patch_hf_api, tmp_path):
    r"""
    A dry-run reserves a budget slot but must not move shards out of
    ``pending/`` -- a subsequent real sync needs to find them there.
    """
    _stage_n(forge, 3)
    forge.sync(dry_run=True)
    pending = list_pending(forge.staging_dir)
    assert len(pending) == 3
    synced = list((tmp_path / SYNCED_DIRNAME).glob("*.parquet"))
    assert synced == []


# ── tests: rate budget ────────────────────────────────────────────────────

def test_sync_stops_at_budget_boundary(forge, patch_hf_api, tmp_path):
    # Stage many shards but allow only 2 commits.
    _stage_n(forge, 6)
    report = forge.sync(max_batch=2, budget=2)
    assert report.n_commits == 2
    assert report.n_committed == 4  # 2 commits * 2 shards each
    assert report.n_remaining == 2  # the third batch was budget-blocked


def test_remaining_pending_after_budget_block_still_queryable(forge, patch_hf_api):
    _stage_n(forge, 6)
    forge.sync(max_batch=2, budget=2)
    pending = list_pending(forge.staging_dir)
    assert len(pending) == 2


# ── tests: 429 backoff ────────────────────────────────────────────────────

def test_persistent_429_lands_in_failed_dir(forge, patch_hf_api_429, tmp_path):
    _stage_n(forge, 2)
    report = forge.sync(max_batch=10, max_retries=2)
    assert report.n_committed == 0
    assert report.n_failed == 2
    assert report.errors  # populated
    failed = list((tmp_path / FAILED_DIRNAME).glob("*.parquet"))
    assert len(failed) == 2


def test_429_then_success_lands_in_synced_dir(
    forge, patch_hf_api_429_then_ok, tmp_path,
):
    _stage_n(forge, 2)
    report = forge.sync(max_batch=10, max_retries=2)
    # First attempt raised 429, second succeeded.
    assert report.n_committed == 2
    assert report.n_failed == 0
    # The fake recorded TWO calls (one raised, one succeeded).
    assert len(patch_hf_api_429_then_ok.calls) == 2


# ── tests: 503 backoff ────────────────────────────────────────────────────

def test_persistent_503_lands_in_failed_dir(forge, patch_hf_api_503, tmp_path):
    r"""
    A 503 with no ``Retry-After`` header still counts as a rate-limit
    response: the batch must be retried up to ``max_retries`` times and
    then relegated to ``failed/``.
    """
    _stage_n(forge, 2)
    report = forge.sync(max_batch=10, max_retries=2)
    assert report.n_committed == 0
    assert report.n_failed == 2
    assert report.errors
    # One initial attempt + ``max_retries`` retries == 3 calls.
    assert len(patch_hf_api_503.calls) == 3
    failed = list((tmp_path / FAILED_DIRNAME).glob("*.parquet"))
    assert len(failed) == 2


def test_503_then_success_lands_in_synced_dir(
    forge, patch_hf_api_503_then_ok, tmp_path,
):
    _stage_n(forge, 2)
    report = forge.sync(max_batch=10, max_retries=2)
    # First attempt raised 503, second succeeded.
    assert report.n_committed == 2
    assert report.n_failed == 0
    assert len(patch_hf_api_503_then_ok.calls) == 2
    synced = list((tmp_path / SYNCED_DIRNAME).glob("*.parquet"))
    assert len(synced) == 2


# ── tests: integration with the Vulcan instance ──────────────────────────

def test_forge_remaining_budget_reflects_persisted_state(forge, patch_hf_api):
    _stage_n(forge, 1)
    before = forge.remaining_budget()
    forge.sync(max_batch=10)
    after = forge.remaining_budget()
    assert after == before - 1


def test_budget_state_file_persists_under_staging_dir(forge, patch_hf_api, tmp_path):
    _stage_n(forge, 1)
    forge.sync(max_batch=10)
    state = tmp_path / BUDGET_STATE_FILENAME
    assert state.is_file()
