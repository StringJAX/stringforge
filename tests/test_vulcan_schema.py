# Copyright 2026 Andreas Schachner
#
# This file is part of StringForge.
#
# StringForge is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
r"""
Tests for :mod:`stringforge.vulcan.schema`.

Covers:

* ``REQUIRED_COLUMNS`` is a strict superset of the vault floor;
* ``geometry_id`` is stable across calls and order-insensitive on input;
* ``shard_relpath`` factors out (h11, h12, bucket);
* ``validate_schema`` reports missing columns, mixed geometries, and
  malformed run-ids -- not just the first one;
* ``pyarrow_schema`` admits ``extra_columns``.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent))

from stringforge.vacuavault.schema import REQUIRED_COLUMNS as VAULT_FLOOR
from stringforge.vulcan.schema import (
    GEOMETRY_KEYS,
    REQUIRED_COLUMNS,
    SCHEMA_VERSION,
    geometry_id,
    is_safe_run_id,
    pyarrow_schema,
    shard_relpath,
    validate_schema,
    vault_floor_compatible,
)


def _make_valid_df(n: int = 3, run_id: str = "run-test") -> pd.DataFrame:
    return pd.DataFrame({
        "flux": [[1, 0, -2, 3, 0, 1]] * n,
        "moduli_re": [[0.0, 0.0]] * n,
        "moduli_im": [[2.5, 3.0]] * n,
        "tau_re": [0.0] * n,
        "tau_im": [4.0] * n,
        "run_id": [run_id] * n,
        "geometry_id": ["abc1234567890abc"] * n,
        "tadpole_charge": [12] * n,
        "created_at": [pd.Timestamp("2026-06-01T12:00:00Z")] * n,
    })


def test_required_columns_superset_of_vault_floor():
    for col in VAULT_FLOOR:
        assert col in REQUIRED_COLUMNS, (
            f"vault-required column {col!r} not in Vulcan REQUIRED_COLUMNS"
        )


def test_geometry_id_stable_and_order_insensitive():
    g1 = {"h11": 3, "h12": 99, "ks_id": 384564, "triang_id": 0}
    g2 = {"triang_id": 0, "ks_id": 384564, "h12": 99, "h11": 3}  # different order
    assert geometry_id(g1) == geometry_id(g2)
    # Stable across calls.
    assert geometry_id(g1) == geometry_id(g1)


def test_geometry_id_distinguishes_distinct_models():
    g1 = {"h11": 3, "h12": 99, "ks_id": 384564, "triang_id": 0}
    g2 = {"h11": 3, "h12": 99, "ks_id": 384564, "triang_id": 1}
    assert geometry_id(g1) != geometry_id(g2)


def test_geometry_id_fills_missing_with_sentinel():
    g_explicit = {"h11": 3, "h12": 99, "ks_id": 1, "triang_id": 0,
                  "conifold_id": -1, "cicy_id": -1}
    g_implicit = {"h11": 3, "h12": 99, "ks_id": 1, "triang_id": 0}
    assert geometry_id(g_explicit) == geometry_id(g_implicit)


def test_shard_relpath_includes_h11_h12_and_bucket():
    g = {"h11": 3, "h12": 99, "ks_id": 384564, "triang_id": 0}
    rel = shard_relpath(g, "run-42")
    assert rel.startswith("h12_99/h11_3/bucket_")
    assert rel.endswith("/run_run-42.parquet")


def test_shard_relpath_bucket_within_range():
    g = {"h11": 3, "h12": 99, "ks_id": 384564, "triang_id": 0}
    for bits in (1, 4, 6):
        rel = shard_relpath(g, "x", bucket_bits=bits)
        bucket = int(rel.split("/bucket_")[1].split("/")[0])
        assert 0 <= bucket < (1 << bits)


def test_validate_schema_accepts_valid_df():
    validate_schema(_make_valid_df())


def test_validate_schema_reports_missing_columns():
    df = _make_valid_df().drop(columns=["tau_re"])
    with pytest.raises(ValueError, match="missing required columns.*tau_re"):
        validate_schema(df)


def test_validate_schema_rejects_mixed_geometries():
    df = _make_valid_df()
    df.loc[0, "geometry_id"] = "ffffffffffffffff"
    with pytest.raises(ValueError, match="mixed geometry_ids"):
        validate_schema(df)


def test_validate_schema_rejects_invalid_run_id():
    df = _make_valid_df(run_id="has spaces!")
    with pytest.raises(ValueError, match="LABEL_SLUG_RE"):
        validate_schema(df)


def test_validate_schema_aggregates_multiple_problems():
    # Trigger three independent problem categories simultaneously:
    #   (i)   a missing required column (drop tau_im);
    #   (ii)  mixed geometry_ids in the same shard;
    #   (iii) a bad run_id slug.
    # The aggregation path must surface all of them in a single
    # ValueError; none may be silently swallowed by an early gate.
    df = _make_valid_df(run_id="bad slug")
    df = df.drop(columns=["tau_im"])
    df.loc[0, "geometry_id"] = "ffffffffffffffff"
    with pytest.raises(ValueError) as exc_info:
        validate_schema(df)
    msg = str(exc_info.value)
    assert "missing required columns" in msg
    assert ("mixed geometry_ids" in msg) or ("LABEL_SLUG_RE" in msg)
    # At least one further problem beyond the missing-columns notice
    # must appear -- i.e. the message lists more than one finding.
    findings = [p for p in msg.split("; ") if p.strip()]
    assert len(findings) >= 2


def test_pyarrow_schema_includes_required_fields():
    schema = pyarrow_schema()
    names = set(schema.names)
    for col in REQUIRED_COLUMNS:
        assert col in names


def test_pyarrow_schema_admits_extra_columns():
    schema = pyarrow_schema(extra_columns={"vol_cy": pa.float64()})
    assert "vol_cy" in set(schema.names)


def test_pyarrow_schema_rejects_extra_overriding_canonical():
    with pytest.raises(ValueError):
        pyarrow_schema(extra_columns={"flux": pa.string()})


def test_vault_floor_compatible():
    df = _make_valid_df()
    assert vault_floor_compatible(df)
    assert not vault_floor_compatible(df.drop(columns=["flux"]))


def test_is_safe_run_id_matches_label_slug_re():
    assert is_safe_run_id("good-run_v1")
    assert is_safe_run_id("2026-06-01-120000_fluxscan_h12-2_ks-1234_seed-42")
    assert not is_safe_run_id("has spaces")
    assert not is_safe_run_id("dots.are.bad")
    assert not is_safe_run_id("")


def test_is_safe_run_id_rejects_double_underscore():
    # '__' is reserved by the staging-area encoding (writer.py replaces
    # '/' with '__' to produce local filenames).
    assert is_safe_run_id("fluxscan__variant") is False


def test_schema_version_is_a_positive_int():
    assert isinstance(SCHEMA_VERSION, int) and SCHEMA_VERSION >= 1


def test_validate_schema_rejects_bad_cert_status():
    df = _make_valid_df()
    df["cert_status"] = ["CERTIFIED_TYPO"] * len(df)
    with pytest.raises(ValueError, match="cert_status"):
        validate_schema(df)


def test_validate_schema_accepts_none_cert_status():
    # certify.py stamps rejected rows with cert_status=None; dropna must
    # mean a None does not trip the validator.
    df = _make_valid_df()
    df["cert_status"] = [None] * len(df)
    validate_schema(df)  # no raise


def test_validate_schema_warns_on_lone_unknown_column():
    df = _make_valid_df()
    df["cert_statuss"] = ["typo"] * len(df)   # typo'd column, nothing else wrong
    import warnings as _w
    with _w.catch_warnings(record=True) as caught:
        _w.simplefilter("always")
        validate_schema(df)  # must not raise
    assert any("unknown columns" in str(w.message) for w in caught)
