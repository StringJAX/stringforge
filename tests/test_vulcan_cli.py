# Copyright 2026 Andreas Schachner
#
# This file is part of StringForge.
#
# StringForge is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
r"""
CLI tests for :mod:`stringforge.vulcan.__main__` (finding #58).

These tests exercise the entry-point used by cron / head-node
workflows: ``python -m stringforge.vulcan sync ...``.  They confirm
that

* a ``--dry-run`` sync against a populated staging directory exits
  cleanly and prints the standard ``committed=...`` status line;
* the ``--repo`` argument falls back to the
  ``STRINGFORGE_VULCAN_REPO`` environment variable;
* both sources missing yields exit code 2 (argument error).

All HuggingFace traffic is suppressed via ``--dry-run`` and via
fixtures that patch :mod:`stringforge.vulcan.hf_io` so the tests
require no credentials and no network -- mirrors the patching style
of ``tests/test_vulcan_sync.py``.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent))

from stringforge.vulcan import Vulcan
from stringforge.vulcan import __main__ as cli
from stringforge.vulcan import hf_io


# ── fakes shared with tests/test_vulcan_sync.py ──────────────────────────


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
        self.calls.append({"repo_id": repo_id, "n_operations": len(operations)})
        return SimpleNamespace(oid=f"oid-{len(self.calls)}")


def _vacuum_df(n: int = 2) -> pd.DataFrame:
    return pd.DataFrame({
        "flux": [[1, 0, -2, 3, 0, 1]] * n,
        "moduli_re": [[0.0, 0.0]] * n,
        "moduli_im": [[2.5, 3.0]] * n,
        "tau_re": [0.0] * n,
        "tau_im": [4.0] * n,
    })


def _stage_one_shard(staging_dir: Path) -> None:
    r"""Stage a single shard into ``staging_dir/pending/``."""
    forge = Vulcan(
        repo="user/repo",
        staging_dir=staging_dir,
        run_id_template="run-{seq}",
        project="t",
    )
    forge.write(
        _vacuum_df(),
        geometry={"h11": 3, "h12": 2, "ks_id": 1, "triang_id": 0},
        tadpole_charge=12,
        run_id="run-cli-1",
    )


@pytest.fixture
def patch_hf_api():
    r"""
    Patch the lazy loaders in :mod:`hf_io` so the dry-run path never
    triggers any real HF SDK import.  Even with ``--dry-run`` the
    sync layer touches the operation-construction helper before
    short-circuiting; the fixture keeps it safe.
    """
    fake_api = _FakeHfApi()
    with mock.patch.object(hf_io, "_load_hf_api",
                           return_value=lambda token=None: fake_api), \
         mock.patch.object(hf_io, "_commit_operations",
                           lambda pl: [_FakeCommitOperationAdd(
                               path_in_repo=p, path_or_fileobj=str(lp))
                               for (p, lp) in pl]):
        yield fake_api


@pytest.fixture
def isolated_env(monkeypatch):
    r"""Ensure no STRINGFORGE_VULCAN_* state leaks in from the host."""
    for var in (
        "STRINGFORGE_VULCAN_REPO",
        "STRINGFORGE_VULCAN_STAGING_DIR",
        "STRINGFORGE_VULCAN_TOKEN",
        "STRINGFORGE_VULCAN_PROJECT",
        "STRINGFORGE_VULCAN_BUDGET",
        "HF_TOKEN",
    ):
        monkeypatch.delenv(var, raising=False)
    yield monkeypatch


# ── tests ─────────────────────────────────────────────────────────────────

def test_cli_sync_dry_run(tmp_path, patch_hf_api, isolated_env, capsys):
    r"""
    ``vulcan sync --repo u/r --staging-dir TMP --dry-run`` must return
    0 and print a ``committed=N`` status line on stdout.
    """
    _stage_one_shard(tmp_path)

    rc = cli.main([
        "sync",
        "--repo", "u/r",
        "--staging-dir", str(tmp_path),
        "--dry-run",
    ])
    assert rc == 0

    out = capsys.readouterr().out
    assert "committed=" in out, f"expected status line; got {out!r}"


def test_cli_sync_env_fallback(tmp_path, patch_hf_api, isolated_env, capsys):
    r"""
    With ``STRINGFORGE_VULCAN_REPO`` set, the sync subcommand may be
    invoked with no ``--repo`` flag.
    """
    _stage_one_shard(tmp_path)
    isolated_env.setenv("STRINGFORGE_VULCAN_REPO", "u/r")

    rc = cli.main([
        "sync",
        "--staging-dir", str(tmp_path),
        "--dry-run",
    ])
    assert rc == 0


def test_cli_sync_missing_repo_errors(tmp_path, patch_hf_api,
                                      isolated_env, capsys):
    r"""
    With no ``--repo`` and no ``STRINGFORGE_VULCAN_REPO`` env var,
    the sync subcommand must return the conventional argument-error
    exit code (2).
    """
    rc = cli.main([
        "sync",
        "--staging-dir", str(tmp_path),
        "--dry-run",
    ])
    assert rc == 2
