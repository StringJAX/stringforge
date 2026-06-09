# Copyright 2026 Andreas Schachner
#
# This file is part of StringForge.
#
# StringForge is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
r"""
Tests for Phase 1: certification + record stamping
(:mod:`stringforge.vulcan.certify`).

Solver-free: a stub :class:`Certifier` stands in for the real physics
cascade.  Includes an explicit encoding of the 2026-06-02 saddle case
(a point with a negative Kähler-metric eigenvalue must be rejected and
its eigenvalue recorded), and an adapter-wiring test that injects stub
``is_physical_fn`` / ``stability_fn`` so the lazy jaxvacua adapter is
covered without JAX.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent))

from stringforge.vulcan import Vulcan
from stringforge.vulcan.certify import (
    CertRow,
    CertifyResult,
    StabilityRecord,
    build_jaxvacua_certifier,
    check_endpoint_stability,
    certify_shard,
)
from stringforge.vulcan.schema import CERT_COLUMNS, record_id
from stringforge.vulcan.verifier import FULL_CASCADE_CHECKS, VerifierSpec
from stringforge.vulcan.writer import SYNCED_DIRNAME, move_to


GEOM = {"h11": 3, "h12": 2, "ks_id": 1, "triang_id": 0}


def _spec(sha="abc123"):
    return VerifierSpec(jaxvacua_git_sha=sha, checks=FULL_CASCADE_CHECKS,
                        tolerances={"residual": 1e-6}, stability_suite=True)


def _candidate_df(n=4):
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "flux": [list(rng.integers(-3, 4, size=6)) for _ in range(n)],
        "moduli_re": [[0.0, 0.0]] * n,
        "moduli_im": [[2.5, 3.0]] * n,
        "tau_re": [0.0] * n,
        "tau_im": [4.0] * n,
    })


def _pass_certifier(row):
    return CertRow(passed=True,
                   checks={c: True for c in FULL_CASCADE_CHECKS},
                   kahler_metric_min_eig=0.12, hessian_min_eig=0.05, mass_min_eig=2.3)


def _fail_metric_certifier(row):
    # The 2026-06-02 signature: passes the cone test but the Kähler
    # metric is indefinite (median min-eig -0.57 in the real incident).
    checks = {c: True for c in FULL_CASCADE_CHECKS}
    checks["kahler_metric_pd"] = False
    return CertRow(passed=False, checks=checks,
                   kahler_metric_min_eig=-0.57, hessian_min_eig=-0.4, mass_min_eig=-1.0)


# ── basic stamping ────────────────────────────────────────────────────────

def test_certify_all_pass_stamps_every_row():
    df = _candidate_df(4)
    res = certify_shard(df, certifier=_pass_certifier, verifier_spec=_spec(), geometry=GEOM)
    assert res.n_input == 4 and res.n_certified == 4 and res.n_rejected == 0
    out = res.certified
    for col in CERT_COLUMNS:
        assert col in out.columns
    assert (out["verifier_id"] == _spec().verifier_id).all()
    assert (out["cert_status"] == "certified").all()
    assert (out["kahler_metric_min_eig"] == 0.12).all()


def test_certify_cert_checks_roundtrip_json():
    df = _candidate_df(1)
    res = certify_shard(df, certifier=_pass_certifier, verifier_spec=_spec(), geometry=GEOM)
    checks = json.loads(res.certified["cert_checks"].iloc[0])
    assert checks["kahler_metric_pd"] is True


def test_certify_record_id_matches_helper():
    df = _candidate_df(1)
    res = certify_shard(df, certifier=_pass_certifier, verifier_spec=_spec(), geometry=GEOM)
    flux = res.certified["flux"].iloc[0]
    expected = record_id(GEOM, flux, _spec().verifier_id)
    assert res.certified["record_id"].iloc[0] == expected


# ── the 2026-06-02 saddle case ────────────────────────────────────────────

def test_certify_rejects_negative_metric_eig_saddle():
    df = _candidate_df(3)
    res = certify_shard(df, certifier=_fail_metric_certifier, verifier_spec=_spec(),
                        geometry=GEOM)
    # All rejected; none enter the certified output.
    assert res.n_certified == 0
    assert res.n_rejected == 3
    assert len(res.certified) == 0
    # The certified frame still carries the cert columns (empty).
    for col in CERT_COLUMNS:
        assert col in res.certified.columns


def test_certify_keep_rejected_records_eigenvalue_for_diagnosis():
    df = _candidate_df(2)
    res = certify_shard(df, certifier=_fail_metric_certifier, verifier_spec=_spec(),
                        geometry=GEOM, keep_rejected=True)
    assert len(res.rejected) == 2
    # The min-eig that the 2026-06-02 incident lacked is recorded...
    assert (res.rejected["kahler_metric_min_eig"] == -0.57).all()
    # ...and rejected rows are NOT given a 'certified' status.
    assert (res.rejected["cert_status"].isna()).all()
    checks = json.loads(res.rejected["cert_checks"].iloc[0])
    assert checks["kahler_metric_pd"] is False


def test_certify_mixed_pass_fail():
    df = _candidate_df(4)
    flags = [True, False, True, False]
    def certifier(row, _it=iter(flags)):
        ok = next(_it)
        return (_pass_certifier(row) if ok else _fail_metric_certifier(row))
    res = certify_shard(df, certifier=certifier, verifier_spec=_spec(), geometry=GEOM)
    assert res.n_certified == 2 and res.n_rejected == 2
    assert (res.certified["cert_status"] == "certified").all()


# ── contract guards ───────────────────────────────────────────────────────

def test_certify_requires_flux_column():
    df = _candidate_df(2).drop(columns=["flux"])
    with pytest.raises(ValueError, match="flux"):
        certify_shard(df, certifier=_pass_certifier, verifier_spec=_spec(), geometry=GEOM)


def test_certify_requires_verifier_spec():
    df = _candidate_df(1)
    with pytest.raises(ValueError, match="VerifierSpec"):
        certify_shard(df, certifier=_pass_certifier, verifier_spec="v1-abc", geometry=GEOM)


def test_certify_empty_input():
    df = _candidate_df(0)
    res = certify_shard(df, certifier=_pass_certifier, verifier_spec=_spec(), geometry=GEOM)
    assert res.n_input == 0 and res.n_certified == 0
    for col in CERT_COLUMNS:
        assert col in res.certified.columns


# ── certify -> write -> read integration ──────────────────────────────────

def test_certified_shard_writes_and_reads_back():
    with tempfile.TemporaryDirectory() as td:
        staging = Path(td)
        forge = Vulcan(repo="user/r", staging_dir=staging, project="t")
        df = _candidate_df(3)
        res = certify_shard(df, certifier=_pass_certifier, verifier_spec=_spec(),
                            geometry=GEOM)
        shard = forge.write(res.certified, geometry=GEOM, tadpole_charge=12,
                            run_id="run-cert")
        move_to(shard, staging, SYNCED_DIRNAME)
        rows = forge.query(run_id="run-cert")
        assert len(rows) == 3
        assert (rows["verifier_id"] == _spec().verifier_id).all()
        assert (rows["cert_status"] == "certified").all()
        assert "record_id" in rows.columns


# ── jaxvacua adapter wiring (stubbed; no JAX) ─────────────────────────────

def test_adapter_packs_moduli_interleaved_for_jaxvacua():
    # The HIGH bug: jaxvacua reads moduli interleaved
    # (x[0:-2:2] + 1j*x[1:-2:2], tau at x[-2], x[-1]).  A block layout
    # would scramble the evaluation point for h12>=2.  Capture x and
    # assert the round-trip with DISTINCT real parts (equal parts could
    # mask a partial mismatch).
    seen = {}

    def fake_is_physical(model, sampler, x, moduli_max=None):
        seen["x"] = np.asarray(x, dtype=float)
        return True

    def fake_stability(eft, *, z, tau, flux, source_index, W0_abs, residual,
                       metric_tol, hessian_tol, mass_tol, min_im_tau):
        return SimpleNamespace(
            kahler_metric_positive=True, hessian_minimum=True, mass_minimum=True,
            stable_minimum=True,
            kahler_metric_min_eig=0.3, real_hessian_min_eig=0.1, mass_matrix_min_eig=1.5,
        )

    certifier = build_jaxvacua_certifier(
        model=object(), sampler=object(),
        is_physical_fn=fake_is_physical, stability_fn=fake_stability,
    )
    row = {"moduli_re": [1.0, 2.0], "moduli_im": [2.5, 3.0],
           "tau_re": 0.5, "tau_im": 4.0, "flux": [1, 0, -2, 3, 0, 1]}
    certifier(row)
    x = seen["x"]
    assert np.allclose(x[0:-2:2] + 1j * x[1:-2:2],
                       np.array(row["moduli_re"]) + 1j * np.array(row["moduli_im"]))
    assert np.allclose([x[-2], x[-1]], [row["tau_re"], row["tau_im"]])


def test_build_jaxvacua_certifier_with_injected_stubs():
    # Inject is_physical_fn / stability_fn so the lazy import is bypassed
    # and the adapter's row->CertRow mapping is exercised without JAX.
    def fake_is_physical(model, sampler, x, moduli_max=None):
        return True

    def fake_stability(eft, *, z, tau, flux, source_index, W0_abs, residual,
                       metric_tol, hessian_tol, mass_tol, min_im_tau):
        # Use the public StabilityRecord field names so this test guards
        # against the adapter reading the wrong attribute (which getattr
        # would silently turn into nan): real_hessian_min_eig /
        # mass_matrix_min_eig, NOT hessian_min_eig / mass_min_eig.
        return SimpleNamespace(
            kahler_metric_positive=True, hessian_minimum=True, mass_minimum=True,
            stable_minimum=True,
            kahler_metric_min_eig=0.3, real_hessian_min_eig=0.1, mass_matrix_min_eig=1.5,
        )

    certifier = build_jaxvacua_certifier(
        model=object(), sampler=object(),
        is_physical_fn=fake_is_physical, stability_fn=fake_stability,
    )
    row = {"moduli_re": [0.0, 0.0], "moduli_im": [2.5, 3.0],
           "tau_re": 0.0, "tau_im": 4.0, "flux": [1, 0, -2, 3, 0, 1]}
    cr = certifier(row)
    assert cr.passed is True
    assert cr.checks["kahler_metric_pd"] is True
    assert cr.kahler_metric_min_eig == 0.3
    # The diagnostic eigenvalues must be read from the real fields, not nan.
    assert cr.hessian_min_eig == 0.1
    assert cr.mass_min_eig == 1.5


def test_build_jaxvacua_certifier_rejects_unstable():
    def fake_is_physical(model, sampler, x, moduli_max=None):
        return True  # is_physical says ok...

    def fake_stability(eft, *, z, tau, flux, source_index, W0_abs, residual,
                       metric_tol, hessian_tol, mass_tol, min_im_tau):
        # ...but the stability suite finds an indefinite metric.
        return SimpleNamespace(
            kahler_metric_positive=False, hessian_minimum=False, mass_minimum=False,
            stable_minimum=False,
            kahler_metric_min_eig=-0.57, real_hessian_min_eig=-0.4, mass_matrix_min_eig=-1.0,
        )

    certifier = build_jaxvacua_certifier(
        model=object(), sampler=object(),
        is_physical_fn=fake_is_physical, stability_fn=fake_stability,
    )
    row = {"moduli_re": [0.0, 0.0], "moduli_im": [2.5, 3.0],
           "tau_re": 0.0, "tau_im": 4.0, "flux": [1, 0, -2, 3, 0, 1]}
    cr = certifier(row)
    assert cr.passed is False
    assert cr.checks["kahler_metric_pd"] is False
    assert cr.kahler_metric_min_eig == -0.57


def test_check_endpoint_stability_local_matrix_diagnostics():
    class FakeEFT:
        def kahler_metric(self, *, z, tau, flux):
            return np.diag([0.2, 0.4])

        def hessian(self, *, z, tau, flux):
            return np.diag([0.1, 0.3])

        def mass_matrix(self, *, z, tau, flux):
            return np.diag([1.5, 2.0])

    rec = check_endpoint_stability(
        FakeEFT(),
        z=np.array([1.0 + 2.0j]),
        tau=0.0 + 3.0j,
        flux=np.array([1, 0, -1]),
    )

    assert isinstance(rec, StabilityRecord)
    assert rec.stable_minimum is True
    assert rec.kahler_metric_min_eig == 0.2
    assert rec.real_hessian_min_eig == 0.1
    assert rec.mass_matrix_min_eig == 1.5
