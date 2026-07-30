"""Regression tests for the ``cicy`` sub-dataset's convention handling.

Two defects were shipped until 2026-07-30 and are pinned here:

1. **Hodge swap.** ``LCSDatabase.load`` swapped ``h11``/``h12`` unconditionally, but the
   ``cicy`` catalogue is *already* in the mirror convention ``lcs_tree`` expects. Since
   ``jaxvacua.lcs`` dimensions the intersection tensor by ``h12``, the swap produced a
   ``(101, 101, 101)`` tensor for the quintic — whose geometry has rank 1.
2. **GV layout.** ``cicy``'s ``gv/`` split is flat, not ``gv/h11_{N}/``, and it carries no
   GW columns. The bucketed path raised ``FileNotFoundError``, and once that was fixed the
   fabricated ``{"charges": None, ...}`` GW dict slipped past ``lcs_tree``'s ``is not None``
   guard and raised ``AttributeError``.

These use only pure helpers plus (where available) the local build, so they never touch the
network. The data-backed tests skip when the local build is absent.
"""

import numpy as np
import pytest

from stringforge.cy_io import _parse_gv_row
from stringforge.lcs_database import _FLAT_GV_DATASETS, _needs_h_swap

# ---------------------------------------------------------------------------- #
# pure-helper tests (always run)
# ---------------------------------------------------------------------------- #


def test_cicy_is_exempt_from_the_hodge_swap():
    """``cicy`` must not be swapped; the catalogue-convention datasets must be."""
    assert _needs_h_swap("cicy") is False
    for ds in ("tdf", "kklt", "toric"):
        assert _needs_h_swap(ds) is True, ds


def test_gv_layout_and_hodge_convention_are_independent_flags():
    """They coincide for cicy but must not be the same constant — a future sub-dataset
    could have one property without the other."""
    assert "cicy" in _FLAT_GV_DATASETS
    assert _FLAT_GV_DATASETS is not None
    assert _needs_h_swap("cicy") is False


def test_parse_gv_row_reports_an_absent_family_as_none():
    """A missing GW family must be ``None``, not ``{"charges": None, ...}`` — the latter
    defeats ``lcs_tree._charges_from_gv_gw``'s ``x is not None`` guard."""
    row = {"gv_charges": [[1, 0], [0, 1]], "gv_invariants": [3, 5], "grading_vector": [1, 1]}
    out = _parse_gv_row(row)
    assert out["GWs"] is None, "absent GW family must collapse to None"
    assert out["GVs"] is not None
    assert set(out["GVs"]) == {"charges", "invariants"}


def test_parse_gv_row_keeps_both_families_when_present():
    row = {
        "gv_charges": [[1]], "gv_invariants": [3],
        "gw_charges": [[1]], "gw_invariants": [7],
        "grading_vector": [1],
    }
    out = _parse_gv_row(row)
    assert out["GVs"] is not None and out["GWs"] is not None


def test_truncate_gv_passes_an_absent_family_through():
    from stringforge.lcs_database import LCSDatabase

    gvs = {"charges": np.array([[1], [4]]), "invariants": np.array([3.0, 5.0])}
    out_gv, out_gw = LCSDatabase._truncate_gv(gvs, None, np.array([1]), 2)
    assert out_gw is None
    assert out_gv["charges"].shape == (1, 1), "degree-4 curve should be truncated away"


# ---------------------------------------------------------------------------- #
# data-backed tests (skip without the local build)
# ---------------------------------------------------------------------------- #
_LOCAL = "private/database/cy-database"


def _db(dataset):
    import os

    if not os.path.isdir(os.path.join(_LOCAL, dataset)):
        pytest.skip(f"local {dataset} build not present")
    from stringforge import LCSDatabase

    return LCSDatabase(dataset=dataset, cache_dir=_LOCAL, offline=True)


def test_quintic_has_rank_one_geometry_and_chi_minus_200():
    """``cicy_id=7890`` is the quintic in P^4: h11=1, h21=101, chi=-200, kappa=5, c2.J=50."""
    tree = _db("cicy").load(cicy_id=7890)
    kappa = np.asarray(tree.intnums)
    c2 = np.asarray(tree.c2)
    assert kappa.shape == (1, 1, 1), f"expected rank-1 tensor, got {kappa.shape}"
    assert len(c2) == 1
    assert int(kappa[0, 0, 0]) == 5
    assert int(round(float(c2[0]))) == 50
    assert int(tree.chi) == -200
    # mirror convention: lcs_tree.h11 is the CY's h21
    assert (int(tree.h11), int(tree.h12)) == (101, 1)


def test_cicy_gv_loads_from_the_flat_split_and_has_no_gw():
    tree = _db("cicy").load(cicy_id=7890, include_gv=True, maximum_degree=10)
    assert tree.gv_charges is not None, "flat gv/ split should load"
    assert np.asarray(tree.gv_charges).shape[1] == 1, "charge width must match the rank"
    assert tree.gw_charges is None, "cicy carries no GW family"


def test_cicy_rank_is_self_consistent_across_models():
    db = _db("cicy")
    for cicy_id in (7890, 7447):
        tree = db.load(cicy_id=cicy_id)
        kappa = np.asarray(tree.intnums)
        assert kappa.shape[0] == len(np.asarray(tree.c2)), cicy_id
        assert int(tree.chi) == -2 * (int(tree.h11) - int(tree.h12)), cicy_id


def test_tdf_load_is_unaffected_by_the_cicy_fix():
    """tdf stores the catalogue convention and must still be swapped, with both GV/GW."""
    db = _db("tdf")
    row = db.query().head(1).iloc[0]
    tree = db.load(ks_id=int(row["ks_id"]), triang_id=int(row["triang_id"]),
                   include_gv=True, maximum_degree=3)
    kappa = np.asarray(tree.intnums)
    assert kappa.shape[0] == len(np.asarray(tree.c2))
    assert tree.gv_charges is not None and tree.gw_charges is not None
