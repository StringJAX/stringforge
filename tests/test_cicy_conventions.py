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


# ---------------------------------------------------------------------------- #
# CICYPhase (always run: a stub tree, so neither the build nor jaxvacua is needed)
# ---------------------------------------------------------------------------- #
class _StubTree:
    """The minimal ``lcs_tree`` surface ``CICYPhase.from_row`` reads.

    ``lcs_tree`` is in **mirror** convention, so ``h11`` here is the CY's :math:`h^{2,1}` and
    ``h12`` -- the Kahler-moduli count that dimensions kappa -- is its :math:`h^{1,1}`.
    """

    def __init__(self, *, mirror_h11, mirror_h12, intnums_coo, c2, extra_data=None):
        self.h11 = mirror_h11
        self.h12 = mirror_h12
        self.intnums_coo = intnums_coo
        self.c2 = c2
        self.extra_data = extra_data


def _quintic_stub(**over):
    """The quintic in P^4 as stored: mirror h11=101, h12=1, kappa=5, c2.J=50."""
    kw = dict(mirror_h11=101, mirror_h12=1, intnums_coo=[[0, 0, 0, 5]], c2=[50],
              extra_data={"h11": 1, "h12": 101, "favorable": True,
                          "Kahler favorable": True, "cicy_id": 7890})
    kw.update(over)
    return _StubTree(**kw)


def test_cicy_phase_unswaps_the_hodge_labels_and_chi():
    """The stored labels are mirror-convention; CICYPhase must report the CY's own."""
    from stringforge import CICYPhase, CYPhase

    p = CICYPhase.from_row(_quintic_stub(), cicy_id=7890)
    assert isinstance(p, CYPhase) and p.construction == "cicy"
    assert (p.h11, p.h12) == (1, 101), "labels must be un-swapped relative to the store"
    assert p.euler_characteristic == -200, "chi sign must follow the CY, not the catalogue"
    assert p.basis_rank == 1 and p.basis_is_complete is True
    assert p.to_dense().shape == (1, 1, 1)
    assert int(p.to_dense()[0, 0, 0]) == 5
    assert list(p.second_chern_class()) == [50]


def test_cicy_phase_has_no_wall_hash_and_no_in_basis():
    """No basis identification exists for cicy, so neither may be offered."""
    from stringforge import CICYPhase

    p = CICYPhase.from_row(_quintic_stub(), cicy_id=7890)
    assert not hasattr(p, "wall_hash"), "a cicy wall_hash would be uncomparable"
    for call in (lambda: p.intersection_numbers(in_basis=True),
                 lambda: p.second_chern_class(in_basis=True)):
        with pytest.raises(NotImplementedError):
            call()


def test_cicy_phase_flags_an_incomplete_basis():
    """len(c2) < h11(X): the stored classes span only a subspace of H^{1,1}(X)."""
    from stringforge import CICYPhase

    p = CICYPhase.from_row(_quintic_stub(
        mirror_h11=15, mirror_h12=15, intnums_coo=[[0, 0, 0, 1]], c2=[1] * 7,
        extra_data={"h11": 15, "h12": 15, "Kahler favorable": False, "favorable": False},
    ), cicy_id=1)
    assert (p.h11, p.h12) == (15, 15) and p.euler_characteristic == 0
    assert p.basis_rank == 7 < p.h11
    assert p.basis_is_complete is False


def test_cicy_phase_rejects_the_degenerate_product_rows():
    """The 22 'product' entries store h11 = h12 = 0, which is not a CY threefold."""
    from stringforge import CICYPhase

    with pytest.raises(ValueError, match="not a valid Calabi-Yau"):
        CICYPhase.from_row(_quintic_stub(
            mirror_h11=0, mirror_h12=0, intnums_coo=[[0, 1, 1, 12]], c2=[72, 0],
            extra_data={"h11": 0, "h12": 0, "product": True},
        ), cicy_id=31)


def test_cicy_phase_refuses_a_flag_contradicted_by_the_data():
    """Kahler favorable requires len(c2) == h11(X); a mismatch must not pass silently."""
    from stringforge import CICYPhase

    with pytest.raises(ValueError, match="contradicts the flag"):
        CICYPhase.from_row(_quintic_stub(
            mirror_h11=101, mirror_h12=4, intnums_coo=[[0, 0, 0, 5]], c2=[50],
            extra_data={"h11": 4, "h12": 101, "Kahler favorable": True},
        ), cicy_id=-1)


def test_cicy_phase_refuses_a_contradictory_stored_convention():
    """extra_data is the un-swapped truth; if it disagrees with the swap, do not guess."""
    from stringforge import CICYPhase

    with pytest.raises(ValueError, match="Refusing to guess the convention"):
        CICYPhase.from_row(_quintic_stub(
            extra_data={"h11": 77, "h12": 101, "Kahler favorable": True},
        ), cicy_id=7890)


def test_cicy_phase_rejects_a_catalogue_only_database():
    """CICYDatabase has no load(); the error must name the right replacement."""
    from stringforge import CICYDatabase, CICYPhase

    db = CICYDatabase.__new__(CICYDatabase)      # no I/O; only .dataset is consulted
    db.dataset = "cicy"
    with pytest.raises(TypeError, match="catalogue-only"):
        CICYPhase.from_database(db, cicy_id=7890)


def test_cicy_phase_is_registered_for_base_class_dispatch():
    from stringforge import CYPhase

    assert CYPhase._DB_DISPATCH == {"toric": "ToricCYPhase", "cicy": "CICYPhase"}


def test_zero_kahler_moduli_load_raises_a_clear_error():
    """A degenerate row must not reach jaxvacua, where it raises a bare IndexError."""
    import os

    if not os.path.isdir(os.path.join(_LOCAL, "cicy")):
        pytest.skip("local cicy build not present")
    with pytest.raises(ValueError, match="no Kahler moduli recorded"):
        _db("cicy").load(cicy_id=31)
