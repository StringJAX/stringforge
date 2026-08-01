"""Tests for the promoted ``toric`` consumer and the ``CYPhase`` class family.

The ``test_kklt_database.py`` pattern (inject a catalogue as an in-memory ``DataFrame``) does
not transfer here: the toric sub-dataset has **no monolithic ``catalog.parquet``** — its
catalogue is sharded per ``(mode, h11)``. So these tests synthesise a minimal sharded tree with
``pyarrow`` under ``tmp_path`` (a few hundred bytes, no network, no committed fixtures) and
exercise the consumer against it.

What is pinned here, and why each one bit at some point:

* ``from_local`` on the sub-dataset directory itself. The base-class step-up guard used to key
  only on ``catalog.parquet``, so for a sharded layout it silently resolved ``.../toric`` to
  ``.../toric/toric`` *and created that directory*.
* Every inherited member that assumes the monolithic layout is closed. Left inherited they
  raise a bare ``FileNotFoundError`` or demand a ``cicy_id``.
* ``get_polytope`` is keyword-only, because the inherited signature takes ``(ks_id, h11)`` —
  the opposite order — so a positional call would silently mean the wrong thing.
* ``CYPhase(**toric_kwargs)`` still returns a ``ToricCYPhase`` (back-compat for the pre-split
  17-argument constructor).
* The toric ``schema.json`` version check actually fires; it used to be a bare ``return None``
  that accepted any version.
"""

import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from stringforge import CYPhase, ToricCYDatabase, ToricCYPhase
from stringforge.cy_io import SchemaVersionError
from stringforge.toric_normalize import TORIC_SCHEMA_VERSION

H11 = 2
# two polytopes: ks_id 0 favorable (basis_dim == h11), ks_id 1 NON-favorable (basis_dim < h11)
_POLYS = [
    dict(ks_id=0, h12=29, fav_N=True, basis_dim=2, n_classes=2),
    dict(ks_id=1, h12=30, fav_N=False, basis_dim=1, n_classes=1),
]


def _idx_table(rows):
    return pa.Table.from_pylist(
        rows, schema=pa.schema([("ks_id", pa.int64()), ("part", pa.int32()),
                                ("row0", pa.int32()), ("n", pa.int32())]))


@pytest.fixture
def toric_root(tmp_path):
    """A minimal but structurally faithful sharded ``toric/`` tree."""
    root = tmp_path / "db"
    t = root / "toric"
    for split in ("polytope_catalog", "polytope", "frst/catalog", "frst/geom"):
        (t / split / f"h11_{H11}").mkdir(parents=True)
    (t / "schema.json").write_text(json.dumps(
        {"schema_version": TORIC_SCHEMA_VERSION, "dataset": "toric",
         "modes": ["frst", "vex"], "convention": "prime-toric-0indexed-v1"}))

    pcat, poly, cat, geom, pidx, cidx = [], [], [], [], [], []
    crow = 0
    for p in _POLYS:
        oob = p["basis_dim"] + 4
        pcat.append(dict(h11=H11, ks_id=p["ks_id"], h12=p["h12"],
                         polytope_hash=f"hash{p['ks_id']}", fav_N=p["fav_N"], fav_M=True,
                         trilayer=True, n_rigids=0, n_rigids_dual=0, n_frsts=p["n_classes"],
                         n_ntfe_frsts=None, n_frst_classes=p["n_classes"],
                         oob_dim=oob, basis_dim=p["basis_dim"]))
        poly.append(dict(h11=H11, ks_id=p["ks_id"], polytope_hash=f"hash{p['ks_id']}",
                         vertices=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0],
                                   [0, 0, 0, 1], [-1, -1, -1, -1]],
                         glsm_basis=list(range(p["basis_dim"])),
                         glsm_charge_matrix=[[1] * oob]))
        pidx.append(dict(ks_id=p["ks_id"], part=0, row0=len(pcat) - 1, n=1))
        cidx.append(dict(ks_id=p["ks_id"], part=0, row0=crow, n=p["n_classes"]))
        for t_id in range(p["n_classes"]):
            cat.append(dict(h11=H11, ks_id=p["ks_id"], triang_id=t_id, h12=p["h12"],
                            fav_N=p["fav_N"], fav_M=True, trilayer=True,
                            wall_hash=bytes([t_id]) * 32, geom_shard_id=0, geom_row_index=crow))
            geom.append(dict(h11=H11, ks_id=p["ks_id"], triang_id=t_id,
                             heights=[1.0] * oob,
                             intnums_coo_i=[0], intnums_coo_j=[0], intnums_coo_k=[0],
                             intnums_coo_v=[5 + t_id], c2=[10 + t_id] * oob, c2_origin=-62))
            crow += 1

    for split, rows in (("polytope_catalog", pcat), ("polytope", poly),
                        ("frst/catalog", cat), ("frst/geom", geom)):
        pq.write_table(pa.Table.from_pylist(rows),
                       t / split / f"h11_{H11}" / "data-00000.parquet")
    pq.write_table(_idx_table(pidx),
                   t / "polytope_catalog" / f"h11_{H11}" / "_ksid_index.parquet")
    pq.write_table(_idx_table(cidx), t / "frst" / "catalog" / f"h11_{H11}" / "_ksid_index.parquet")
    return root


# --------------------------------------------------------------------------- #
# from_local path resolution
# --------------------------------------------------------------------------- #
def test_from_local_accepts_parent_and_subdataset_dir(toric_root):
    """Both spellings must land on the same cache_dir, and neither may create a stray dir."""
    a = ToricCYDatabase.from_local(str(toric_root))
    b = ToricCYDatabase.from_local(str(toric_root / "toric"))
    assert a.cache_dir == b.cache_dir == toric_root / "toric"
    assert not (toric_root / "toric" / "toric").exists(), "stepped into toric/toric/"


# --------------------------------------------------------------------------- #
# the inherited surface the sharded layout invalidates
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("call", [
    lambda db: db.query_conifolds(),
    lambda db: db._ensure_catalog(),
    lambda db: db._ensure_conifold_catalog(),
    lambda db: db._lookup(1, 2, None),
    lambda db: db._validate_key(1, 2, None),
    lambda db: db._identifiers_to_list(None),
])
def test_monolithic_only_members_are_closed(toric_root, call):
    db = ToricCYDatabase.from_local(str(toric_root))
    with pytest.raises(NotImplementedError, match="sharded 'toric'"):
        call(db)


def test_info_works_on_the_sharded_layout(toric_root, capsys):
    ToricCYDatabase.from_local(str(toric_root)).info()
    out = capsys.readouterr().out
    assert "ToricCYDatabase" in out and f"{H11:>4}" in out


# --------------------------------------------------------------------------- #
# query / load / get_polytope
# --------------------------------------------------------------------------- #
def test_query_and_load_round_trip(toric_root):
    db = ToricCYDatabase.from_local(str(toric_root))
    assert len(db.query_polytopes(h11=H11)) == 2
    assert len(db.query("frst", H11)) == 3                    # 2 + 1 classes
    assert len(db.query("frst", H11, ks_id=0)) == 2
    g = db.load("frst", H11, 0, 1, in_basis=True)
    assert int(g["intnums_coo"][0][3]) == 6                   # 5 + triang_id
    assert g["polytope_hash"] == "hash0"                      # joined via ks_id
    assert isinstance(g["wall_hash"], (bytes, bytearray))
    assert g["intnums_coo_in_basis"].shape[1] == 4


def test_query_rejects_a_bad_mode(toric_root):
    db = ToricCYDatabase.from_local(str(toric_root))
    with pytest.raises(ValueError, match="mode must be one of"):
        db.query("bogus", H11)


def test_get_polytope_is_keyword_only(toric_root):
    """Positional use must be impossible: the inherited signature has the opposite order."""
    db = ToricCYDatabase.from_local(str(toric_root))
    assert db.get_polytope(h11=H11, ks_id=0)["basis_dim"] == 2
    with pytest.raises(TypeError):
        db.get_polytope(H11, 0)


# --------------------------------------------------------------------------- #
# schema versioning
# --------------------------------------------------------------------------- #
def test_schema_version_mismatch_raises(toric_root):
    p = toric_root / "toric" / "schema.json"
    for bad in (TORIC_SCHEMA_VERSION - 1, TORIC_SCHEMA_VERSION + 997):
        p.write_text(json.dumps({"schema_version": bad, "dataset": "toric"}))
        with pytest.raises(SchemaVersionError):
            ToricCYDatabase.from_local(str(toric_root))._check_schema()


def test_missing_or_unversioned_schema_only_warns(toric_root):
    p = toric_root / "toric" / "schema.json"
    p.write_text(json.dumps({"dataset": "toric"}))            # no schema_version
    ToricCYDatabase.from_local(str(toric_root))._check_schema()
    p.unlink()
    with pytest.warns(UserWarning, match="schema.json"):
        ToricCYDatabase.from_local(str(toric_root))._check_schema()


# --------------------------------------------------------------------------- #
# the CYPhase family
# --------------------------------------------------------------------------- #
def test_from_database_returns_a_toric_subclass(toric_root):
    db = ToricCYDatabase.from_local(str(toric_root))
    cp = CYPhase.from_database(db, "frst", H11, 0, 0)
    assert isinstance(cp, ToricCYPhase) and isinstance(cp, CYPhase)
    assert cp.mode == "frst" and cp.construction == "toric"
    assert cp.hodge_numbers == (H11, 29)
    assert cp.euler_characteristic == 2 * (H11 - 29)
    assert cp.basis_is_complete is True and cp.basis_rank == 2


def test_to_dense_uses_the_out_of_basis_side_length(toric_root):
    """A toric phase stores kappa out-of-basis, so the dense default must be ``oob_dim``.

    The inherited base default is ``basis_rank`` (== ``basis_dim``), which is *smaller* than
    ``oob_dim`` for essentially every toric phase, so a bare ``to_dense()`` raised
    ``IndexError`` on almost all of them.
    """
    db = ToricCYDatabase.from_local(str(toric_root))
    cp = CYPhase.from_database(db, "frst", H11, 0, 0)
    assert cp.oob_dim > cp.basis_dim, "fixture must exercise the out-of-basis case"
    dense = cp.to_dense()
    assert dense.shape == (cp.oob_dim,) * 3
    # ... and it must agree with the explicit out-of-basis dense request
    assert np.array_equal(dense, cp.intersection_numbers(in_basis=False, format="dense"))
    # the in-basis dense form still uses the basis side length
    assert (cp.intersection_numbers(in_basis=True, format="dense").shape
            == (cp.basis_dim,) * 3)
    # a base-class geometry keeps the base default (kappa already in a divisor basis)
    q = CYPhase(construction="cicy", h11=1, h12=101, intnums_coo=[[0, 0, 0, 5]], c2=[50])
    assert q.to_dense().shape == (1, 1, 1)


def test_dispatch_rejects_an_unsupported_database(toric_root):
    """The base factory dispatches on ``db.dataset``; an unreadable one must say so clearly."""
    db = ToricCYDatabase.from_local(str(toric_root))
    db.dataset = "tdf"                          # no CYPhase subclass reads tdf geometry
    with pytest.raises(TypeError, match="no CYPhase subclass reads that sub-dataset"):
        CYPhase.from_database(db, "frst", H11, 0, 0)
    # adding a construction means adding a registry entry, not editing the dispatch logic
    assert set(CYPhase._DB_DISPATCH) == {"toric", "cicy"}


def test_incomplete_key_is_rejected(toric_root):
    db = ToricCYDatabase.from_local(str(toric_root))
    with pytest.raises(ValueError, match=r"missing \['mode'\]"):
        CYPhase.from_database(db, h11=H11, ks_id=0, triang_id=0)
    with pytest.raises(ValueError, match="does not match the stored h12"):
        CYPhase.from_database(db, mode="frst", h11=H11, ks_id=0, triang_id=0, h12=999)


def test_non_favorable_flags_incomplete_basis_and_defers_full_h11(toric_root):
    db = ToricCYDatabase.from_local(str(toric_root))
    cp = CYPhase.from_database(db, "frst", H11, 1, 0)          # ks_id 1 is fav_N=False
    assert cp.basis_is_complete is False and cp.covers_full_h11 is False
    assert cp.basis_rank == 1 < cp.h11
    with pytest.warns(UserWarning, match="non-favorable"):
        cp.intersection_numbers(in_basis=True)                 # toric part only, with a warning
    for name in ("full_intersection_numbers", "full_second_chern_class"):
        with pytest.raises(NotImplementedError):
            getattr(cp, name)()


def test_legacy_constructor_form_still_dispatches(toric_root):
    """The pre-split call — CYPhase(dataset=..., <17 kwargs>) — must still work."""
    db = ToricCYDatabase.from_local(str(toric_root))
    g = db.load("frst", H11, 0, 0)
    poly = db.get_polytope(h11=H11, ks_id=0)
    cp = CYPhase(
        dataset="frst", h11=H11, h12=29, ks_id=0, triang_id=0, heights=g["heights"],
        intnums_coo=g["intnums_coo"], c2=g["c2"], c2_origin=g["c2_origin"],
        vertices=poly["vertices"], glsm_basis=poly["glsm_basis"],
        glsm_charge_matrix=poly["glsm_charge_matrix"], fav_N=True, fav_M=True,
        trilayer=True, wall_hash=g["wall_hash"], polytope_hash=poly["polytope_hash"],
    )
    assert isinstance(cp, ToricCYPhase)          # __new__ dispatched on the toric-only kwargs
    assert cp.mode == "frst"                     # `dataset=` was accepted as the legacy spelling
    with pytest.deprecated_call():
        assert cp.dataset == "frst"              # ... and the alias itself warns


def test_base_class_needs_no_polytope(toric_root):
    """A construction-independent geometry: the quintic's Wall data, no polytope anywhere."""
    q = CYPhase(construction="cicy", h11=1, h12=101, intnums_coo=[[0, 0, 0, 5]], c2=[50])
    assert type(q) is CYPhase
    assert q.euler_characteristic == -200 and q.basis_rank == 1
    assert q.to_dense().shape == (1, 1, 1)
    assert np.asarray(q.intersection_numbers()).tolist() == [[0, 0, 0, 5]]
    with pytest.raises(NotImplementedError, match="toric notion"):
        q.intersection_numbers(in_basis=True)


@pytest.mark.parametrize("kwargs, match", [
    (dict(construction="bogus", h11=1, h12=2, intnums_coo=[[0, 0, 0, 1]], c2=[1]),
     "construction must be one of"),
    (dict(construction="cicy", h11=1, h12=101, intnums_coo=[[0, 0, 0, 5]], c2=[50], chi=7),
     "contradicts the Hodge numbers"),
])
def test_base_class_guards(kwargs, match):
    with pytest.raises(ValueError, match=match):
        CYPhase(**kwargs)
