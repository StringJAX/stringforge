# Copyright 2026 Andreas Schachner
#
# This file is part of StringForge.
#
# StringForge is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
r"""
Tests for Phase 2: frozen citeable snapshots
(:mod:`stringforge.vulcan.snapshot`) and ``VulcanReader`` revision
pinning.

These tests are local-only (no HuggingFace); the HF tag/commit path is
exercised via ``dry_run`` + a mocked ``HfApi``.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent))

from stringforge.vulcan import (
    SnapshotManifest,
    Vulcan,
    VulcanReader,
    build_manifest,
    freeze_snapshot,
)
from stringforge.vulcan import snapshot as snapshot_mod
from stringforge.vulcan.snapshot import MANIFEST_FILENAME, UNVERIFIED_SENTINEL
from stringforge.vulcan.writer import SYNCED_DIRNAME, move_to

FIXED_TS = "2026-06-04-000000"


@pytest.fixture
def tmp_staging():
    with tempfile.TemporaryDirectory() as td:
        yield Path(td)


def _df(n, *, verifier_id=None, cert_status=None, solver="newton", metric_pd_ok=True):
    cols = {
        "flux": [[1, 0, -2, 3, 0, 1]] * n,
        "moduli_re": [[0.0, 0.0]] * n,
        "moduli_im": [[2.5, 3.0]] * n,
        "tau_re": [0.0] * n,
        "tau_im": [4.0] * n,
        "solver_name": [solver] * n,
    }
    if verifier_id is not None:
        cols["verifier_id"] = [verifier_id] * n
    if cert_status is not None:
        cols["cert_status"] = [cert_status] * n
    # A realistic certified/provisional shard carries the metric-PD
    # evidence a real certifier emits; without it, the positive-assertion
    # gate (correctly) treats the row as an omitted-check / 2026-06-02
    # signature.
    if verifier_id is not None or cert_status is not None:
        import json as _json
        cols["cert_checks"] = [_json.dumps({"kahler_metric_pd": bool(metric_pd_ok)})] * n
        cols["kahler_metric_min_eig"] = [0.2 if metric_pd_ok else -0.57] * n
    return pd.DataFrame(cols)


@pytest.fixture
def certified_forge(tmp_staging):
    r"""Two geometries, all rows certified under one verifier."""
    forge = Vulcan(repo="user/vacua_forge", staging_dir=tmp_staging, project="t")
    ga = {"h11": 3, "h12": 2, "ks_id": 1, "triang_id": 0}
    gb = {"h11": 5, "h12": 2, "ks_id": 2, "triang_id": 0}
    s1 = forge.write(_df(3, verifier_id="v1-aaaa", cert_status="certified"),
                     geometry=ga, tadpole_charge=12, run_id="run-a")
    s2 = forge.write(_df(2, verifier_id="v1-aaaa", cert_status="certified"),
                     geometry=gb, tadpole_charge=14, run_id="run-b")
    for s in (s1, s2):
        move_to(s, tmp_staging, SYNCED_DIRNAME)
    return forge


# ── manifest correctness ──────────────────────────────────────────────────

def test_manifest_counts(certified_forge):
    m = build_manifest(certified_forge.reader(), snapshot_tag="v1", created_at=FIXED_TS)
    assert m.n_shards == 2
    assert m.n_records == 5
    assert m.records_by_hodge == {"h11=3,h12=2": 3, "h11=5,h12=2": 2}


def test_manifest_verifier_histogram(certified_forge):
    m = build_manifest(certified_forge.reader(), snapshot_tag="v1", created_at=FIXED_TS)
    assert m.verifier_id_histogram == {"v1-aaaa": 5}
    assert m.cert_status_histogram == {"certified": 5}


def test_manifest_fully_certified_true(certified_forge):
    m = build_manifest(certified_forge.reader(), snapshot_tag="v1", created_at=FIXED_TS)
    assert m.fully_certified() is True


def test_manifest_flags_unverified_records(tmp_staging):
    # A shard with no verifier_id / cert_status columns must tally under
    # the UNVERIFIED sentinel and fail fully_certified().
    forge = Vulcan(repo="user/r", staging_dir=tmp_staging, project="t")
    g = {"h11": 3, "h12": 2, "ks_id": 1, "triang_id": 0}
    s = forge.write(_df(4), geometry=g, tadpole_charge=12, run_id="run-x")
    move_to(s, tmp_staging, SYNCED_DIRNAME)
    m = build_manifest(forge.reader(), snapshot_tag="v1", created_at=FIXED_TS)
    assert m.verifier_id_histogram == {UNVERIFIED_SENTINEL: 4}
    assert m.fully_certified() is False


def test_manifest_provisional_not_fully_certified(tmp_staging):
    forge = Vulcan(repo="user/r", staging_dir=tmp_staging, project="t")
    g = {"h11": 3, "h12": 2, "ks_id": 1, "triang_id": 0}
    s = forge.write(_df(2, verifier_id="v1-aaaa", cert_status="provisional"),
                    geometry=g, tadpole_charge=12, run_id="run-p")
    move_to(s, tmp_staging, SYNCED_DIRNAME)
    m = build_manifest(forge.reader(), snapshot_tag="v1", created_at=FIXED_TS)
    assert m.fully_certified() is False
    assert m.cert_status_histogram == {"provisional": 2}


def test_manifest_content_sha_stable_and_sensitive(certified_forge, tmp_staging):
    m1 = build_manifest(certified_forge.reader(), snapshot_tag="v1", created_at=FIXED_TS)
    m2 = build_manifest(certified_forge.reader(), snapshot_tag="v1", created_at=FIXED_TS)
    assert m1.content_sha256 == m2.content_sha256
    # Add a shard -> content hash must change.
    g = {"h11": 7, "h12": 2, "ks_id": 9, "triang_id": 0}
    s = certified_forge.write(_df(1, verifier_id="v1-aaaa", cert_status="certified"),
                              geometry=g, tadpole_charge=16, run_id="run-c")
    move_to(s, certified_forge.staging_dir, SYNCED_DIRNAME)
    m3 = build_manifest(certified_forge.reader(), snapshot_tag="v1", created_at=FIXED_TS)
    assert m3.content_sha256 != m1.content_sha256


def test_manifest_to_dict_roundtrips(certified_forge):
    m = build_manifest(certified_forge.reader(), snapshot_tag="v1", created_at=FIXED_TS)
    d = m.to_dict()
    assert d["snapshot_tag"] == "v1"
    assert d["n_records"] == 5
    assert d["verifier_id_histogram"] == {"v1-aaaa": 5}


# ── freeze_snapshot ───────────────────────────────────────────────────────

def test_freeze_dry_run_builds_manifest_no_network(certified_forge):
    with mock.patch("stringforge.vulcan.snapshot._push_and_tag") as pushed:
        m = freeze_snapshot(
            certified_forge.reader(), tag="v2026.06", created_at=FIXED_TS,
            dry_run=True,
        )
        pushed.assert_not_called()
    assert m.snapshot_tag == "v2026.06"


def test_freeze_writes_local_manifest(certified_forge, tmp_path):
    out = tmp_path / "manifest_out"
    freeze_snapshot(
        certified_forge.reader(), tag="v1", created_at=FIXED_TS,
        dry_run=True, local_manifest_dir=out,
    )
    written = out / MANIFEST_FILENAME
    assert written.is_file()
    import json
    blob = json.loads(written.read_text())
    assert blob["n_records"] == 5


def test_freeze_refuses_uncertified_by_default(tmp_staging):
    forge = Vulcan(repo="user/r", staging_dir=tmp_staging, project="t")
    g = {"h11": 3, "h12": 2, "ks_id": 1, "triang_id": 0}
    s = forge.write(_df(3), geometry=g, tadpole_charge=12, run_id="run-x")  # no verifier
    move_to(s, tmp_staging, SYNCED_DIRNAME)
    with pytest.raises(ValueError, match="not fully\\s+certified"):
        freeze_snapshot(forge.reader(), tag="v1", created_at=FIXED_TS, dry_run=True)


def test_freeze_override_allows_uncertified(tmp_staging):
    forge = Vulcan(repo="user/r", staging_dir=tmp_staging, project="t")
    g = {"h11": 3, "h12": 2, "ks_id": 1, "triang_id": 0}
    s = forge.write(_df(3), geometry=g, tadpole_charge=12, run_id="run-x")
    move_to(s, tmp_staging, SYNCED_DIRNAME)
    m = freeze_snapshot(
        forge.reader(), tag="v1", created_at=FIXED_TS, dry_run=True,
        require_fully_certified=False,
    )
    assert not m.fully_certified()


def test_freeze_real_requires_repo(certified_forge):
    with pytest.raises(ValueError, match="requires repo"):
        freeze_snapshot(
            certified_forge.reader(), tag="v1", created_at=FIXED_TS,
            repo=None, dry_run=False,
        )


def test_freeze_real_calls_push_and_tag(certified_forge):
    with mock.patch("stringforge.vulcan.snapshot._push_and_tag") as pushed:
        freeze_snapshot(
            certified_forge.reader(), tag="v2026.06", created_at=FIXED_TS,
            repo="user/vacua_forge", dry_run=False,
        )
        pushed.assert_called_once()
        _, kwargs = pushed.call_args
        assert kwargs["tag"] == "v2026.06"
        assert kwargs["repo"] == "user/vacua_forge"


def test_vulcan_freeze_snapshot_convenience(certified_forge):
    with mock.patch("stringforge.vulcan.snapshot._push_and_tag") as pushed:
        m = certified_forge.freeze_snapshot(tag="v9", created_at=FIXED_TS)
        pushed.assert_called_once()
    assert m.snapshot_tag == "v9"


# ── reader revision plumbing ──────────────────────────────────────────────

def test_reader_revision_default_none(certified_forge):
    assert certified_forge.reader().revision is None


def test_from_hf_records_revision():
    reader = VulcanReader.from_hf("user/repo", revision="v2026.06")
    assert reader.revision == "v2026.06"


def test_from_hf_revision_threaded_to_download():
    # The revision must reach hf_hub_download / list_repo_files.
    reader = VulcanReader.from_hf("user/repo", revision="v2026.06")
    fake_api = mock.MagicMock()
    fake_api.list_repo_files.return_value = ["h12_2/h11_3/bucket_0/run_x.parquet"]
    with mock.patch.dict("sys.modules", {"huggingface_hub": mock.MagicMock(
        HfApi=mock.MagicMock(return_value=fake_api),
        hf_hub_download=mock.MagicMock(return_value="/tmp/nonexistent.parquet"),
    )}):
        import huggingface_hub  # the mock
        list(reader._iter_shards())
        fake_api.list_repo_files.assert_called_once()
        _, kwargs = fake_api.list_repo_files.call_args
        assert kwargs.get("revision") == "v2026.06"
        _, dl_kwargs = huggingface_hub.hf_hub_download.call_args
        assert dl_kwargs.get("revision") == "v2026.06"


# ── hardening: empty corpus / metric-PD / corrupt shard / registry ───────

def test_empty_corpus_not_fully_certified(tmp_staging):
    forge = Vulcan(repo="user/r", staging_dir=tmp_staging, project="t")
    m = build_manifest(forge.reader(), snapshot_tag="v1", created_at=FIXED_TS)
    assert m.n_records == 0
    assert m.fully_certified() is False


def test_freeze_refuses_empty_corpus(tmp_staging):
    forge = Vulcan(repo="user/r", staging_dir=tmp_staging, project="t")
    with pytest.raises(ValueError, match="empty"):
        freeze_snapshot(forge.reader(), tag="v1", created_at=FIXED_TS, dry_run=True)


def test_freeze_empty_refused_even_with_override(tmp_staging):
    # The empty guard fires even when require_fully_certified=False.
    forge = Vulcan(repo="user/r", staging_dir=tmp_staging, project="t")
    with pytest.raises(ValueError, match="empty"):
        freeze_snapshot(forge.reader(), tag="v1", created_at=FIXED_TS, dry_run=True,
                        require_fully_certified=False)


def test_manifest_detects_metric_pd_failure(tmp_staging):
    # A row stamped 'certified' but whose cert_checks says metric-PD
    # failed (negative eig) is the 2026-06-02 signature -- it must be
    # counted and must fail fully_certified().
    forge = Vulcan(repo="user/r", staging_dir=tmp_staging, project="t")
    g = {"h11": 3, "h12": 2, "ks_id": 1, "triang_id": 0}
    import json as _json
    df = pd.DataFrame({
        "flux": [[1, 0, -2, 3, 0, 1]] * 2,
        "moduli_re": [[0.0, 0.0]] * 2, "moduli_im": [[2.5, 3.0]] * 2,
        "tau_re": [0.0] * 2, "tau_im": [4.0] * 2,
        "verifier_id": ["v1-aaaa"] * 2,
        "cert_status": ["certified"] * 2,
        "cert_checks": [_json.dumps({"kahler_metric_pd": False})] * 2,
        "kahler_metric_min_eig": [-0.57, -0.57],
    })
    s = forge.write(df, geometry=g, tadpole_charge=12, run_id="run-bad")
    move_to(s, tmp_staging, SYNCED_DIRNAME)
    m = build_manifest(forge.reader(), snapshot_tag="v1", created_at=FIXED_TS)
    assert m.metric_pd_failures == 2
    assert m.fully_certified() is False


def test_manifest_catches_OMITTED_metric_pd_check(tmp_staging):
    # The literal 2026-06-02 signature: a row stamped 'certified' whose
    # verifier OMITTED the metric-PD check entirely -- cert_checks has no
    # kahler_metric_pd key and kahler_metric_min_eig is NaN.  A negative
    # ("explicitly failed?") test would miss this; the positive-assertion
    # gate must count it as a failure and refuse to freeze.
    forge = Vulcan(repo="user/r", staging_dir=tmp_staging, project="t")
    g = {"h11": 3, "h12": 2, "ks_id": 1, "triang_id": 0}
    import json as _json
    df = pd.DataFrame({
        "flux": [[1, 0, -2, 3, 0, 1]] * 3,
        "moduli_re": [[0.0, 0.0]] * 3, "moduli_im": [[2.5, 3.0]] * 3,
        "tau_re": [0.0] * 3, "tau_im": [4.0] * 3,
        "verifier_id": ["v1-omitted"] * 3,
        "cert_status": ["certified"] * 3,
        "cert_checks": [_json.dumps({"is_physical": True})] * 3,  # NO kahler_metric_pd key
        "kahler_metric_min_eig": [float("nan")] * 3,              # never computed
    })
    s = forge.write(df, geometry=g, tadpole_charge=12, run_id="run-omitted")
    move_to(s, tmp_staging, SYNCED_DIRNAME)
    m = build_manifest(forge.reader(), snapshot_tag="v1", created_at=FIXED_TS)
    assert m.metric_pd_failures == 3
    assert m.fully_certified() is False
    with pytest.raises(ValueError, match="not fully\\s+certified"):
        freeze_snapshot(forge.reader(), tag="v1", created_at=FIXED_TS, dry_run=True)


def test_manifest_catches_certified_shard_missing_cert_checks_column(tmp_staging):
    # A shard with cert_status='certified' but no cert_checks column at
    # all: every certified row fails the positive assertion.
    forge = Vulcan(repo="user/r", staging_dir=tmp_staging, project="t")
    g = {"h11": 3, "h12": 2, "ks_id": 1, "triang_id": 0}
    df = pd.DataFrame({
        "flux": [[1, 0, -2, 3, 0, 1]] * 2,
        "moduli_re": [[0.0, 0.0]] * 2, "moduli_im": [[2.5, 3.0]] * 2,
        "tau_re": [0.0] * 2, "tau_im": [4.0] * 2,
        "verifier_id": ["v1-x"] * 2,
        "cert_status": ["certified"] * 2,  # but no cert_checks / min_eig columns
    })
    s = forge.write(df, geometry=g, tadpole_charge=12, run_id="run-nochecks")
    move_to(s, tmp_staging, SYNCED_DIRNAME)
    m = build_manifest(forge.reader(), snapshot_tag="v1", created_at=FIXED_TS)
    assert m.metric_pd_failures == 2
    assert m.fully_certified() is False


def test_fully_certified_histogram_coverage_guard():
    # A hand-built manifest whose histograms don't sum to n_records must
    # not pass (defensive guard against a future/hand-built manifest).
    m = SnapshotManifest(
        snapshot_tag="v1", created_at=FIXED_TS, n_shards=1, n_records=5,
        records_by_hodge={"h11=3,h12=2": 5},
        verifier_id_histogram={"v1-x": 3},   # only 3, not 5
        cert_status_histogram={"certified": 3},
        schema_version=1, content_sha256="deadbeef", metric_pd_failures=0,
    )
    assert m.fully_certified() is False


def test_build_manifest_raises_on_corrupt_shard(certified_forge, tmp_staging):
    # Drop a non-parquet file with a .parquet extension into synced/;
    # the reader catalogue should skip it (n_skipped), and build_manifest
    # must refuse rather than silently omit it.
    synced = tmp_staging / SYNCED_DIRNAME
    (synced / "h12_2__h11_9__bucket_0__run_corrupt.parquet").write_text("not a parquet")
    (synced / "h12_2__h11_9__bucket_0__run_corrupt.parquet.meta.json").write_text(
        '{"path_in_repo": "h12_2/h11_9/bucket_0/run_corrupt.parquet", '
        '"run_id": "corrupt", "n_rows": 1}'
    )
    reader = certified_forge.reader()
    with pytest.warns(RuntimeWarning):
        with pytest.raises(ValueError, match="skipped|unreadable"):
            build_manifest(reader, snapshot_tag="v1", created_at=FIXED_TS)


def test_fully_certified_registry_cross_check(certified_forge):
    from stringforge.vulcan import VerifierRegistry, VerifierSpec
    from stringforge.vulcan.verifier import FULL_CASCADE_CHECKS
    m = build_manifest(certified_forge.reader(), snapshot_tag="v1", created_at=FIXED_TS)
    # The corpus uses the placeholder verifier_id "v1-aaaa", which does
    # not resolve in an empty/real registry -> registry gate fails.
    reg = VerifierRegistry([VerifierSpec("realsha", FULL_CASCADE_CHECKS)])
    assert m.fully_certified() is True            # no registry: passes
    assert m.fully_certified(registry=reg) is False  # registry: unresolved id


def test_mlview_split_reproducible_across_readers(certified_forge):
    # Two independent readers over the same synced tree must produce
    # byte-identical splits (the snapshot-reproducibility guarantee).
    v1 = certified_forge.ml_view().split_assignments()
    v2 = certified_forge.ml_view().split_assignments()
    pd.testing.assert_frame_equal(
        v1.sort_values("geometry_id").reset_index(drop=True),
        v2.sort_values("geometry_id").reset_index(drop=True),
    )
