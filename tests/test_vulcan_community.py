# Copyright 2026 Andreas Schachner
#
# This file is part of StringForge.
#
# StringForge is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
r"""
Tests for Phase 3: the community submission gate
(:mod:`stringforge.vulcan.community`).

Solver-free (stub certifier).  Covers the three guarantees the plan
calls out: a valid submission is admitted; a *forged* submission
(client claims certified, server re-cert fails) is rejected and the
forged columns never survive; and a verifier lacking the Kähler-metric
PD check is refused outright (the 2026-06-02 guard).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent))

from stringforge.vulcan.certify import CertRow
from stringforge.vulcan.community import dedup_key, review_submission
from stringforge.vulcan.schema import CERT_COLUMNS
from stringforge.vulcan.verifier import FULL_CASCADE_CHECKS, VerifierSpec

GEOM = {"h11": 3, "h12": 2, "ks_id": 1, "triang_id": 0}


def _full_spec():
    return VerifierSpec("abc123", FULL_CASCADE_CHECKS, tolerances={"residual": 1e-6})


def _spec_without_metric_pd():
    return VerifierSpec(
        "abc123",
        tuple(c for c in FULL_CASCADE_CHECKS if c != "kahler_metric_pd"),
        tolerances={"residual": 1e-6},
    )


def _submission(n=3, *, forged=False):
    rng = np.random.default_rng(1)
    cols = {
        "flux": [list(rng.integers(-3, 4, size=6)) for _ in range(n)],
        "moduli_re": [[0.0, 0.0]] * n,
        "moduli_im": [[2.5, 3.0]] * n,
        "tau_re": [0.0] * n,
        "tau_im": [4.0] * n,
    }
    if forged:
        # A malicious submitter pre-stamps "certified" + a fake verifier.
        cols["verifier_id"] = ["v1-FORGED"] * n
        cols["cert_status"] = ["certified"] * n
        cols["is_susy"] = [True] * n
    return pd.DataFrame(cols)


def _pass_certifier(row):
    return CertRow(passed=True, checks={c: True for c in FULL_CASCADE_CHECKS},
                   kahler_metric_min_eig=0.2, hessian_min_eig=0.1, mass_min_eig=1.0)


def _saddle_certifier(row):
    checks = {c: True for c in FULL_CASCADE_CHECKS}
    checks["kahler_metric_pd"] = False
    return CertRow(passed=False, checks=checks,
                   kahler_metric_min_eig=-0.57, hessian_min_eig=-0.4, mass_min_eig=-1.0)


# ── valid submission ──────────────────────────────────────────────────────

def test_valid_submission_admitted():
    res = review_submission(_submission(3), certifier=_pass_certifier,
                            verifier_spec=_full_spec(), geometry=GEOM)
    assert res.accepted is True
    assert res.n_admitted == 3
    assert (res.certified["cert_status"] == "certified").all()
    assert (res.certified["verifier_id"] == _full_spec().verifier_id).all()


# ── forged submission ─────────────────────────────────────────────────────

def test_forged_submission_rejected_when_recert_fails():
    # Submitter claims certified; the server re-cert finds saddles.
    res = review_submission(_submission(3, forged=True), certifier=_saddle_certifier,
                            verifier_spec=_full_spec(), geometry=GEOM)
    assert res.accepted is False
    assert res.n_admitted == 0
    assert any("rejected" in r for r in res.reasons)


def test_forged_cert_columns_are_stripped_before_recert():
    # Even when the server re-cert PASSES, the forged verifier_id must
    # not survive -- the server stamps its own.
    res = review_submission(_submission(2, forged=True), certifier=_pass_certifier,
                            verifier_spec=_full_spec(), geometry=GEOM)
    assert res.accepted is True
    assert (res.certified["verifier_id"] == _full_spec().verifier_id).all()
    assert "v1-FORGED" not in set(res.certified["verifier_id"])
    assert any("stripped client-supplied" in r for r in res.reasons)


# ── metric-PD gate ────────────────────────────────────────────────────────

def test_gate_refuses_verifier_without_metric_pd():
    with pytest.raises(ValueError, match="kahler_metric_pd|metric"):
        review_submission(_submission(2), certifier=_pass_certifier,
                          verifier_spec=_spec_without_metric_pd(), geometry=GEOM)


def test_gate_allows_override_for_degraded_test():
    # The override exists but must be explicit.
    res = review_submission(_submission(2), certifier=_pass_certifier,
                            verifier_spec=_spec_without_metric_pd(), geometry=GEOM,
                            require_metric_pd=False)
    assert res.accepted is True


# ── dedup ─────────────────────────────────────────────────────────────────

def test_dedup_drops_already_seen_vacua():
    sub = _submission(3)
    # Pretend the first row's vacuum is already in the DB.
    seen = {dedup_key(GEOM, sub["flux"].iloc[0])}
    res = review_submission(sub, certifier=_pass_certifier, verifier_spec=_full_spec(),
                            geometry=GEOM, seen_keys=seen)
    assert res.n_certified == 3
    assert res.n_duplicates == 1
    assert res.n_admitted == 2
    assert any("duplicate" in r for r in res.reasons)


def test_dedup_key_is_verifier_independent():
    # Same vacuum -> same dedup key regardless of float vs int flux repr.
    k1 = dedup_key(GEOM, [1.0, 0.0, -2.0])
    k2 = dedup_key(GEOM, [1, 0, -2])
    assert k1 == k2


def test_partial_admission_mixed_pass_fail():
    sub = _submission(4)
    flags = iter([True, False, True, True])
    def certifier(row):
        return _pass_certifier(row) if next(flags) else _saddle_certifier(row)
    res = review_submission(sub, certifier=certifier, verifier_spec=_full_spec(),
                            geometry=GEOM)
    assert res.n_certified == 3
    assert res.n_admitted == 3
    assert res.accepted is True


# ── intra-batch dedup + record_id uniqueness ──────────────────────────────

def test_intra_batch_dedup_collapses_identical_flux():
    # Three identical-flux rows, empty DB: must admit ONE, and record_id
    # must be unique among admitted rows (the schema invariant).
    sub = pd.DataFrame({
        "flux": [[1, 0, -2, 3, 0, 1]] * 3,
        "moduli_re": [[0.0, 0.0]] * 3, "moduli_im": [[2.5, 3.0]] * 3,
        "tau_re": [0.0] * 3, "tau_im": [4.0] * 3,
    })
    res = review_submission(sub, certifier=_pass_certifier, verifier_spec=_full_spec(),
                            geometry=GEOM)
    assert res.n_certified == 3
    assert res.n_admitted == 1
    assert res.n_duplicates == 2
    assert res.certified["record_id"].is_unique


# ── per-row metric-PD consistency filter ──────────────────────────────────

def test_inconsistent_certifier_row_is_dropped():
    # A certifier bug: reports passed=True but its own evidence shows
    # kahler_metric_pd False / negative min-eig.  The gate must drop it.
    def inconsistent(row):
        checks = {c: True for c in FULL_CASCADE_CHECKS}
        checks["kahler_metric_pd"] = False
        return CertRow(passed=True, checks=checks,
                       kahler_metric_min_eig=-0.57, hessian_min_eig=-0.4, mass_min_eig=-1.0)
    res = review_submission(_submission(2), certifier=inconsistent,
                            verifier_spec=_full_spec(), geometry=GEOM)
    assert res.n_certified == 2          # certify_shard trusted passed=True
    assert res.n_admitted == 0           # ...but the gate caught the inconsistency
    assert any("inconsistency" in r for r in res.reasons)


# ── identity + physics-flag stripping ─────────────────────────────────────

def test_forged_geometry_id_is_overwritten():
    sub = _submission(2)
    sub["geometry_id"] = "ffffffffffffffff"   # forged
    sub["ks_id"] = 999999                       # forged geometry key
    res = review_submission(sub, certifier=_pass_certifier, verifier_spec=_full_spec(),
                            geometry=GEOM)
    from stringforge.vulcan.schema import geometry_id
    assert (res.certified["geometry_id"] == geometry_id(GEOM)).all()
    assert (res.certified["ks_id"] == GEOM["ks_id"]).all()


def test_forged_physics_flags_stripped():
    sub = _submission(2)
    sub["is_susy"] = [True, True]
    sub["residual"] = [1e-30, 1e-30]   # implausibly good, unverified
    res = review_submission(sub, certifier=_pass_certifier, verifier_spec=_full_spec(),
                            geometry=GEOM)
    # The unverified client physics flags must not survive into the
    # admitted rows (the certifier produced none, so they are absent).
    assert "is_susy" not in res.certified.columns or res.certified["is_susy"].isna().all()
    assert any("stripped" in r for r in res.reasons)


def test_forged_tadpole_charge_stripped():
    # tadpole_charge is a physics quantity the certifier does NOT
    # recompute; a forged value must not survive into the admitted rows
    # (the server supplies the authoritative value at write time).
    sub = _submission(2)
    sub["tadpole_charge"] = [-99999, -99999]   # fraudulent orientifold charge
    res = review_submission(sub, certifier=_pass_certifier, verifier_spec=_full_spec(),
                            geometry=GEOM)
    assert res.accepted is True
    assert "tadpole_charge" not in res.certified.columns
    assert any("tadpole_charge" in r for r in res.reasons)
