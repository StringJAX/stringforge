# Copyright 2026 Andreas Schachner
#
# This file is part of StringForge.
#
# StringForge is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
r"""
Tests for the certified-vacuum record contract (Phase 0):

* the additive certification columns are present in the schema and in
  ``pyarrow_schema`` without breaking the vault-floor superset invariant;
* ``record_id`` is verifier-dependent (new verifier -> new row) while
  ``canonical_flux`` is verifier-independent (the dedup key);
* ``cert_status`` validation;
* :mod:`stringforge.vulcan.verifier` content-hashing, ordering
  enforcement, the metric-PD-check guard (the 2026-06-02 fix), and the
  registry round-trip with corruption detection.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pyarrow as pa
import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent))

from stringforge.vacuavault.schema import REQUIRED_COLUMNS as VAULT_FLOOR
from stringforge.vulcan.schema import (
    CERT_COLUMNS,
    CERT_STATUSES,
    OPTIONAL_COLUMNS,
    REQUIRED_COLUMNS,
    canonical_flux,
    pyarrow_schema,
    record_id,
    validate_cert_status,
    vault_floor_compatible,
)
from stringforge.vulcan.verifier import (
    CANONICAL_CHECK_NAMES,
    FULL_CASCADE_CHECKS,
    VerifierRegistry,
    VerifierSpec,
    verifier_id,
)


# ── schema additions are additive + non-breaking ──────────────────────────

def test_cert_columns_are_in_optional_not_required():
    for col in CERT_COLUMNS:
        assert col in OPTIONAL_COLUMNS
        assert col not in REQUIRED_COLUMNS


def test_cert_columns_do_not_disturb_vault_floor():
    # The vault floor must remain a strict subset of REQUIRED_COLUMNS
    # after the additive change (the promotion-compatibility invariant).
    for col in VAULT_FLOOR:
        assert col in REQUIRED_COLUMNS


def test_pyarrow_schema_includes_cert_columns_with_expected_types():
    schema = pyarrow_schema()
    by_name = {f.name: f.type for f in schema}
    assert by_name["verifier_id"] == pa.string()
    assert by_name["cert_status"] == pa.string()
    assert by_name["cert_checks"] == pa.string()
    assert by_name["kahler_metric_min_eig"] == pa.float64()
    assert by_name["hessian_min_eig"] == pa.float64()
    assert by_name["mass_min_eig"] == pa.float64()
    assert by_name["record_id"] == pa.string()


def test_extra_columns_still_rejected_against_cert_fields():
    # The shadowing guard must now also protect the cert columns.
    with pytest.raises(ValueError):
        pyarrow_schema(extra_columns={"verifier_id": pa.int64()})


# ── canonical_flux + record_id ────────────────────────────────────────────

def test_canonical_flux_rounds_to_int_tuple():
    assert canonical_flux([1.0, -2.0, 3.0]) == (1, -2, 3)
    assert canonical_flux((1, 0, -2, 3, 0, 1)) == (1, 0, -2, 3, 0, 1)


def test_record_id_is_verifier_dependent():
    geom = {"h11": 3, "h12": 2, "ks_id": 1, "triang_id": 0}
    flux = [1, 0, -2, 3, 0, 1]
    rid_a = record_id(geom, flux, "v1-aaaaaaaaaaaaaaaa")
    rid_b = record_id(geom, flux, "v1-bbbbbbbbbbbbbbbb")
    # Same physical vacuum, different verifier -> different row id
    # (preserves history; must NOT be used for physical dedup).
    assert rid_a != rid_b


def test_record_id_stable_for_same_inputs():
    geom = {"h11": 3, "h12": 2, "ks_id": 1, "triang_id": 0}
    flux = [1, 0, -2, 3, 0, 1]
    assert record_id(geom, flux, "v1-x") == record_id(geom, flux, "v1-x")


def test_record_id_flux_float_int_equivalent():
    geom = {"h11": 3, "h12": 2, "ks_id": 1, "triang_id": 0}
    # The physical identity is the integer flux, independent of float repr.
    assert record_id(geom, [1.0, 0.0, -2.0], "v1-x") == record_id(geom, [1, 0, -2], "v1-x")


def test_physical_dedup_key_is_verifier_independent():
    geom = {"h11": 3, "h12": 2, "ks_id": 1, "triang_id": 0}
    flux = [1.0, 0.0, -2.0]
    # The dedup key the plan specifies is (geometry_id, canonical_flux),
    # which carries no verifier component -- so the same vacuum certified
    # by two verifiers deduplicates to one physical entry.
    from stringforge.vulcan.schema import geometry_id
    key = (geometry_id(geom), canonical_flux(flux))
    assert key == (geometry_id(geom), canonical_flux([1, 0, -2]))


# ── cert_status validation ────────────────────────────────────────────────

def test_validate_cert_status_accepts_known():
    for status in CERT_STATUSES:
        validate_cert_status(status)


def test_validate_cert_status_rejects_unknown():
    with pytest.raises(ValueError, match="invalid cert_status"):
        validate_cert_status("approved")


def test_cert_statuses_are_the_expected_three():
    assert set(CERT_STATUSES) == {"certified", "provisional", "invalidated"}


# ── VerifierSpec content hashing ──────────────────────────────────────────

def _full_spec(sha="abc123"):
    return VerifierSpec(
        jaxvacua_git_sha=sha,
        checks=FULL_CASCADE_CHECKS,
        tolerances={"residual": 1e-6, "metric_tol": 1e-10},
        stability_suite=True,
    )


def test_verifier_id_is_deterministic_and_prefixed():
    a = _full_spec().verifier_id
    b = _full_spec().verifier_id
    assert a == b
    assert a.startswith("v1-")
    assert verifier_id(_full_spec()) == a


def test_verifier_id_changes_with_jaxvacua_sha():
    assert _full_spec("sha-A").verifier_id != _full_spec("sha-B").verifier_id


def test_verifier_id_changes_when_metric_pd_check_dropped():
    # This is the 2026-06-02 case made queryable: a verifier that drops
    # the Kähler-metric PD check must hash to a DIFFERENT id.
    full = _full_spec()
    without_metric = VerifierSpec(
        jaxvacua_git_sha="abc123",
        checks=tuple(c for c in FULL_CASCADE_CHECKS if c != "kahler_metric_pd"),
        tolerances={"residual": 1e-6, "metric_tol": 1e-10},
        stability_suite=True,
    )
    assert full.verifier_id != without_metric.verifier_id
    assert full.includes_metric_pd_check() is True
    assert without_metric.includes_metric_pd_check() is False


def test_verifier_id_tolerance_float_format_stable():
    # Float formatting must not perturb the id.
    a = VerifierSpec("s", FULL_CASCADE_CHECKS, tolerances={"residual": 1e-6})
    b = VerifierSpec("s", FULL_CASCADE_CHECKS, tolerances={"residual": 0.000001})
    assert a.verifier_id == b.verifier_id


def test_verifier_notes_not_part_of_hash():
    a = VerifierSpec("s", FULL_CASCADE_CHECKS, notes="first")
    b = VerifierSpec("s", FULL_CASCADE_CHECKS, notes="second")
    assert a.verifier_id == b.verifier_id


def test_verifier_rejects_unknown_check():
    with pytest.raises(ValueError, match="unknown check names"):
        VerifierSpec("s", ("kahler_metric_pd", "not_a_real_check"))


def test_verifier_rejects_non_canonical_order():
    # Same checks, wrong order -> rejected (so one procedure has one id).
    reversed_checks = tuple(reversed(FULL_CASCADE_CHECKS))
    with pytest.raises(ValueError, match="canonical order"):
        VerifierSpec("s", reversed_checks)


# ── VerifierRegistry ──────────────────────────────────────────────────────

def test_registry_register_and_get():
    spec = _full_spec()
    reg = VerifierRegistry()
    vid = reg.register(spec)
    assert vid in reg
    assert reg.get(vid).hashable_payload() == spec.hashable_payload()
    assert len(reg) == 1


def test_registry_register_idempotent():
    reg = VerifierRegistry()
    vid1 = reg.register(_full_spec())
    vid2 = reg.register(_full_spec())
    assert vid1 == vid2
    assert len(reg) == 1


def test_registry_roundtrips_through_dict():
    reg = VerifierRegistry([_full_spec("A"), _full_spec("B")])
    restored = VerifierRegistry.from_dict(reg.to_dict())
    assert reg.to_dict() == restored.to_dict()
    assert len(restored) == 2


def test_registry_from_dict_detects_corruption():
    reg = VerifierRegistry([_full_spec()])
    blob = reg.to_dict()
    # Hand-corrupt: rename the key so it no longer matches its spec hash.
    (vid, entry), = blob.items()
    corrupt = {"v1-deadbeefdeadbeef": entry}
    with pytest.raises(ValueError, match="does not match recomputed"):
        VerifierRegistry.from_dict(corrupt)


def test_canonical_check_names_includes_metric_pd():
    # Guard: the canonical cascade must always offer the metric-PD check.
    assert "kahler_metric_pd" in CANONICAL_CHECK_NAMES
    assert "kahler_metric_pd" in FULL_CASCADE_CHECKS
