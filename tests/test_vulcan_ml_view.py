# Copyright 2026 Andreas Schachner
#
# This file is part of StringForge.
#
# StringForge is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
r"""
Tests for :mod:`stringforge.vulcan.ml_view`.

Coverage:

* deterministic split assignment: same input -> same split;
* uniform distribution over many synthetic geometry IDs;
* no geometry leakage across train/val/test;
* feature projection drops geometry / catalogue columns;
* list-column padding to a fixed length;
* salt perturbs assignments;
* ``Vulcan.ml_view`` wires everything together.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent))

from stringforge.vulcan import Vulcan, VulcanMLView, VulcanReader
from stringforge.vulcan.ml_view import (
    DEFAULT_FRACTIONS,
    DEFAULT_SPLITS,
    FeatureSpec,
    assign_split,
    shape_features,
)
from stringforge.vulcan.writer import SYNCED_DIRNAME, move_to


# ── assign_split ──────────────────────────────────────────────────────────

def test_assign_split_is_deterministic():
    gid = "abc1234567890abc"
    a = assign_split(gid)
    b = assign_split(gid)
    assert a == b


def test_assign_split_distribution_close_to_fractions():
    # Sample many synthetic geometry IDs and confirm the empirical
    # distribution lands close to the requested fractions.
    rng = np.random.default_rng(0)
    n = 4000
    ids = [hex(int(x))[2:].zfill(16) for x in rng.integers(0, 1 << 60, size=n)]
    counts = {s: 0 for s in DEFAULT_SPLITS}
    for gid in ids:
        counts[assign_split(gid)] += 1
    for split, frac in zip(DEFAULT_SPLITS, DEFAULT_FRACTIONS):
        empirical = counts[split] / n
        # 4σ band on a binomial sample of size 4000.
        tol = 4 * np.sqrt(frac * (1 - frac) / n)
        assert abs(empirical - frac) < tol, (
            f"{split}: empirical={empirical:.3f}, expected={frac}, tol={tol:.3f}"
        )


def test_assign_split_salt_perturbs_assignment():
    # The same geometry ID with two different salts should land in
    # different splits at least *some* of the time.
    rng = np.random.default_rng(1)
    n = 200
    ids = [hex(int(x))[2:].zfill(16) for x in rng.integers(0, 1 << 60, size=n)]
    diffs = sum(
        1 for gid in ids if assign_split(gid, salt="A") != assign_split(gid, salt="B")
    )
    assert diffs > 0


def test_assign_split_rejects_bad_fractions():
    with pytest.raises(ValueError):
        assign_split("x", fractions=(0.5, 0.6), splits=("a", "b"))
    with pytest.raises(ValueError):
        assign_split("x", fractions=(-0.1, 1.1), splits=("a", "b"))
    with pytest.raises(ValueError):
        assign_split("x", fractions=(0.5,), splits=("a", "b"))


# ── shape_features ────────────────────────────────────────────────────────

def _vacuum_df(n: int = 3):
    return pd.DataFrame({
        "run_id": ["run-x"] * n,
        "geometry_id": ["abc1234567890abc"] * n,
        "h11": [3] * n,
        "h12": [2] * n,
        "ks_id": [1] * n,
        "triang_id": [0] * n,
        "conifold_id": [-1] * n,
        "cicy_id": [-1] * n,
        "flux": [[1, 0, -2, 3, 0, 1]] * n,
        "moduli_re": [[0.0, 0.0]] * n,
        "moduli_im": [[2.5, 3.0]] * n,
        "tau_re": [0.0] * n,
        "tau_im": [4.0] * n,
        "created_at": [pd.Timestamp("2026-06-01T12:00:00Z")] * n,
        "is_susy": [True] * n,
    })


def test_shape_features_default_drops_geometry_columns():
    df = _vacuum_df()
    out = shape_features(df, FeatureSpec())
    for dropped in ("h11", "h12", "ks_id", "triang_id", "conifold_id", "cicy_id",
                    "run_id", "created_at"):
        assert dropped not in out.columns
    # geometry_id is retained by default (keep_geometry_id=True).
    assert "geometry_id" in out.columns


def test_shape_features_explicit_projection():
    df = _vacuum_df()
    spec = FeatureSpec(features=["tau_re", "is_susy"], keep_geometry_id=False)
    out = shape_features(df, spec)
    assert list(out.columns) == ["tau_re", "is_susy"]


def test_shape_features_missing_feature_raises():
    df = _vacuum_df()
    with pytest.raises(KeyError):
        shape_features(df, FeatureSpec(features=["nope"]))


def test_shape_features_pads_list_columns():
    df = _vacuum_df()
    spec = FeatureSpec(list_max_len=8, list_fill=-1)
    out = shape_features(df, spec)
    for value in out["flux"]:
        assert isinstance(value, np.ndarray)
        assert value.shape == (8,)
    # Trailing positions filled with -1.
    assert (out["flux"].iloc[0][6:] == -1).all()


def test_shape_features_truncates_when_max_len_short():
    df = _vacuum_df()
    spec = FeatureSpec(list_max_len=2)
    out = shape_features(df, spec)
    for value in out["flux"]:
        assert value.shape == (2,)


# ── populated reader fixture ─────────────────────────────────────────────

@pytest.fixture
def populated_view(tmp_path):
    forge = Vulcan(repo="user/repo", staging_dir=tmp_path, project="t")
    geoms = [
        {"h11": 3, "h12": 2, "ks_id": k, "triang_id": 0}
        for k in range(1, 21)
    ]

    def _df(susy: bool):
        return pd.DataFrame({
            "flux": [[1, 0, -2, 3, 0, 1]] * 2,
            "moduli_re": [[0.0, 0.0]] * 2,
            "moduli_im": [[2.5, 3.0]] * 2,
            "tau_re": [0.0, 0.1],
            "tau_im": [4.0, 4.1],
            "is_susy": [susy, susy],
        })

    for i, g in enumerate(geoms):
        s = forge.write(_df(i % 2 == 0), geometry=g, tadpole_charge=12,
                        run_id=f"run-{i:03d}")
        move_to(s, tmp_path, SYNCED_DIRNAME)
    return forge.ml_view()


def test_ml_view_split_assignments_cover_all_geometries(populated_view):
    table = populated_view.split_assignments()
    assert len(table) == 20
    assert set(table["split"]).issubset(set(DEFAULT_SPLITS))


def test_ml_view_no_geometry_leakage(populated_view):
    seen: dict[str, str] = {}
    for split in DEFAULT_SPLITS:
        df = populated_view.as_dataframe(split)
        if df.empty:
            continue
        for gid in df["geometry_id"].unique():
            if gid in seen:
                assert seen[gid] == split, (
                    f"geometry {gid} leaked across splits "
                    f"{seen[gid]!r} and {split!r}"
                )
            else:
                seen[gid] = split


def test_ml_view_partition_is_exhaustive(populated_view):
    n_total = sum(
        len(populated_view.as_dataframe(s)) for s in DEFAULT_SPLITS
    )
    # 20 geometries × 2 rows each = 40 rows.
    assert n_total == 40


def test_ml_view_rejects_unknown_split(populated_view):
    with pytest.raises(ValueError):
        populated_view.as_dataframe("validation")


def test_ml_view_passes_through_query_filters(populated_view):
    rows = populated_view.as_dataframe(
        "train",
        query_filters={"is_susy": True},
    )
    if not rows.empty:
        # All rows in the SUSY subset must in fact be SUSY.
        assert (rows["is_susy"] == True).all()


def test_ml_view_uses_default_feature_spec(populated_view):
    rows = populated_view.as_dataframe("train")
    if not rows.empty:
        # geometry_id retained by default; geometry-key columns dropped.
        assert "geometry_id" in rows.columns
        assert "ks_id" not in rows.columns


def test_vulcan_ml_view_factory_wires_reader(tmp_path):
    forge = Vulcan(repo="user/repo", staging_dir=tmp_path, project="t")
    view = forge.ml_view()
    assert isinstance(view, VulcanMLView)
    assert isinstance(view.reader, VulcanReader)
