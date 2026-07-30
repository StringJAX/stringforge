"""Tests for the ``jaxvacua.vacuum`` <-> stringforge writer adapter.

These exercise the *finder-free* path end-to-end (no CYTools / model load
needed): a hand-built ``Vacuum`` / ``PFV`` carrying its geometry identity in
``metadata["identity"]`` is written through ``LCSDatabase.write_vacua`` and read
back exactly via ``read_vacua``.  The authoritative record travels as JSON in
the free ``extra_data`` column (never pickle), so read-back is exact and
model-independent.

Accuracy note: the tests assert *exact* round-trip (``Vacuum.equals``) and the
queryable typed projection (``moduli``/``tau`` from the fixed real/imag
interleaving), not merely that a write "succeeds".
"""
import json

import gzip
import pickle
from pathlib import Path

import numpy as np
import pytest

# jaxvacua is required for these tests (the adapter's whole point); skip cleanly
# where it is unavailable so the rest of the stringforge suite still runs.
jaxvacua_vacuum = pytest.importorskip("jaxvacua.vacuum")
Vacuum = jaxvacua_vacuum.Vacuum
PFV = jaxvacua_vacuum.PFV
PFVData = jaxvacua_vacuum.PFVData

from stringforge._vacuum_adapter import is_vacuum, row_to_vacuum, vacuum_to_row
from stringforge.lcs_database import LCSDatabase


# --------------------------------------------------------------------------- #
# Fixtures / builders
# --------------------------------------------------------------------------- #
_IDENTITY = {
    "h11": 5, "h12": 2, "ks_id": 10, "triang_id": 0,
    "conifold_id": -1, "cicy_id": -1, "model_name": "test_geo",
}


def _make_vacuum(model_name="test_geo", identity=None):
    """A solved h12=2 ``Vacuum`` with a known ``x``/``flux``.

    ``x`` decodes (fixed real/imag interleaving) to moduli ``z=(2.5i, 3i)`` and
    ``tau=4i`` — matching the shapes used elsewhere in the writer tests.
    """
    x = np.array([0.0, 2.5, 0.0, 3.0, 0.0, 4.0], dtype=float)
    flux = np.array([1, 0, -2, 3, 0, 1], dtype=float)
    ident = dict(identity if identity is not None else _IDENTITY)
    if model_name is not None:
        ident.setdefault("model_name", model_name)
    return Vacuum(
        x=x, flux=flux,
        W0=complex(0.05, 0.01),
        DW=np.array([1e-10, 2e-10, 1e-10], dtype=float),
        residual=2e-10,
        gs=0.25,
        metadata={"model_name": model_name, "identity": ident},
    )


def _make_pfv():
    x = np.array([0.0, 2.5, 0.0, 3.0, 0.0, 4.0], dtype=float)
    flux = np.array([1, 0, -2, 3, 0, 1], dtype=float)
    data = PFVData(M=np.array([2.0, 1.0]), K=np.array([1.0, 0.0]))
    return PFV(
        x=x, flux=flux, data=data, tau_input=complex(0.0, 4.0),
        W0=complex(0.05, 0.01),
        DW=np.array([1e-10, 2e-10, 1e-10], dtype=float),
        residual=2e-10,
        metadata={"model_name": "test_geo", "identity": dict(_IDENTITY)},
    )


def _db(tmp_path, monkeypatch):
    monkeypatch.setenv("STRINGFORGE_VAULT", str(tmp_path / "vault"))
    return LCSDatabase(dataset="tdf", cache_dir=str(tmp_path / "cache"), offline=True)


# --------------------------------------------------------------------------- #
# 1. Pure adapter (no DB)
# --------------------------------------------------------------------------- #
def test_is_vacuum_short_circuits_on_containers():
    # dict/tuple/list must be False *without* importing jaxvacua machinery
    assert is_vacuum({"flux": [1]}) is False
    assert is_vacuum((1, 2, 3)) is False
    assert is_vacuum([1, 2, 3]) is False
    assert is_vacuum(_make_vacuum()) is True


def test_adapter_roundtrip_vacuum():
    v = _make_vacuum()
    result, extra = vacuum_to_row(v, finder=None, store_trajectory=True)
    # queryable typed projection
    assert np.allclose(np.asarray(result["moduli"]).imag, [2.5, 3.0])
    assert np.isclose(complex(result["tau"]).imag, 4.0)
    assert result["N_flux"] is None            # finder-free
    assert result["is_susy"] is True           # residual < 1e-8
    # exact record round-trip
    v2 = row_to_vacuum(extra)
    assert v2 is not None
    assert v2.equals(v)
    assert type(v2).__name__ == "Vacuum"


def test_adapter_roundtrip_pfv():
    v = _make_pfv()
    _result, extra = vacuum_to_row(v, finder=None)
    v2 = row_to_vacuum(extra)
    assert type(v2).__name__ == "PFV"
    assert v2.equals(v)
    assert np.allclose(np.asarray(v2.data.M), [2.0, 1.0])
    assert np.allclose(np.asarray(v2.data.K), [1.0, 0.0])


def test_adapter_extra_data_is_json_not_pickle():
    _result, extra = vacuum_to_row(_make_vacuum(), finder=None)
    blob = extra["vacuum"]
    # The stored record is JSON-serialisable (round-trips through json), never a
    # pickle byte-string (public vault -> no unpickle-on-load RCE).  (Direct dict
    # equality is not used: an LCS vacuum carries NaN fields, and NaN != NaN.)
    text = json.dumps(blob)
    assert isinstance(text, str) and text.lstrip().startswith("{")
    reloaded = json.loads(text)
    assert reloaded["_kind"] == "Vacuum"
    assert "x" in reloaded                       # coordinate record present
    assert extra["kind"] == "Vacuum"


def test_row_to_vacuum_none_on_missing_blob():
    assert row_to_vacuum(None) is None
    assert row_to_vacuum({"model_name": "x"}) is None   # no "vacuum" key
    assert row_to_vacuum("not json") is None


# --------------------------------------------------------------------------- #
# 2. DB round-trip (finder-free)
# --------------------------------------------------------------------------- #
def test_write_read_vacua_finder_free(tmp_path, monkeypatch):
    db = _db(tmp_path, monkeypatch)
    v = _make_vacuum()
    report = db.write_vacua([v], store_trajectory=True)

    assert report.n_written == 1
    assert report.geometries == ["test_geo"]
    run_id = report.run_ids_by_geometry["test_geo"]

    # read back as Vacuum objects
    got = db.read_vacua(run_id)
    assert len(got) == 1
    assert got[0].equals(v)

    # typed projection is queryable via the DataFrame path
    df = db.load_vacua(run_id)
    assert np.isclose(float(df["tau_im"].iloc[0]), 4.0)
    assert np.allclose(sorted(np.asarray(df["moduli_im"].iloc[0])), [2.5, 3.0])

    # extra_data is valid JSON carrying the record
    payload = json.loads(df["extra_data"].iloc[0])
    assert "vacuum" in payload and payload["kind"] == "Vacuum"


def test_write_vacua_single_object_ok(tmp_path, monkeypatch):
    db = _db(tmp_path, monkeypatch)
    report = db.write_vacua(_make_vacuum())      # single object, not a list
    assert report.n_written == 1


def test_write_vacua_only_solved_filters(tmp_path, monkeypatch):
    """``only_solved`` filters on the duck-typed ``is_solved()`` contract.

    Promotion provenance (``success`` / ``is_solved``) lives on the ``afvs``
    subclasses, not on a base ``Vacuum`` -- so the writer treats an object
    *without* ``is_solved`` as solved (``getattr(v, "is_solved", lambda: True)``)
    and only filters those that expose it and return ``False``.  A local stand-in
    exercises that contract without depending on the private ``afvs`` package.
    """
    from dataclasses import dataclass

    @dataclass(eq=False)
    class _Unsolved(Vacuum):
        def is_solved(self):
            return False

    good = _make_vacuum()
    # A DIFFERENT FLUX: the writer deduplicates on the flux vector alone
    # (vacua_writer.py:845), so sharing `good`'s flux would make the counts
    # report dedup rather than the filter.
    bad = _Unsolved(x=np.array([0.0, 2.5, 0.0, 3.5, 0.0, 4.0]),
                    flux=np.array([1, 0, -2, 3, 0, 2], dtype=float),
                    metadata=dict(good.metadata))
    assert not hasattr(good, "is_solved")        # base Vacuum: no provenance
    assert bad.is_solved() is False

    # Separate vaults: the writer deduplicates, so a second write into the same
    # one would report duplicates rather than the filter's effect.
    filtered = _db(tmp_path / "filtered", monkeypatch).write_vacua(
        [good, bad], only_solved=True)
    assert filtered.n_written == 1                # the unsolved one is dropped

    unfiltered = _db(tmp_path / "unfiltered", monkeypatch).write_vacua(
        [good, bad], only_solved=False)
    assert unfiltered.n_written == 2              # ... and kept without the flag


def test_write_vacua_mixed_geometry(tmp_path, monkeypatch):
    db = _db(tmp_path, monkeypatch)
    va = _make_vacuum(model_name="geo_a",
                      identity={**_IDENTITY, "h11": 5, "model_name": "geo_a"})
    vb = _make_vacuum(model_name="geo_b",
                      identity={**_IDENTITY, "h11": 6, "model_name": "geo_b"})
    report = db.write_vacua([va, vb])
    assert report.n_written == 2
    assert set(report.geometries) == {"geo_a", "geo_b"}
    assert set(report.run_ids_by_geometry) == {"geo_a", "geo_b"}
    # each geometry read back exactly
    got_a = db.read_vacua(report.run_ids_by_geometry["geo_a"])
    assert len(got_a) == 1 and got_a[0].equals(va)


def test_write_vacua_guard_no_identity_no_models(tmp_path, monkeypatch):
    db = _db(tmp_path, monkeypatch)
    v = _make_vacuum(model_name=None, identity={})   # no identity, no finder
    v.metadata = {}
    with pytest.raises(ValueError, match="no resolvable finder"):
        db.write_vacua([v])


def test_pfv_survives_db_roundtrip(tmp_path, monkeypatch):
    db = _db(tmp_path, monkeypatch)
    v = _make_pfv()
    report = db.write_vacua([v], store_trajectory=True)
    got = db.read_vacua(report.run_ids_by_geometry["test_geo"])
    assert len(got) == 1
    assert type(got[0]).__name__ == "PFV"
    assert got[0].equals(v)


# --------------------------------------------------------------------------- #
# 3. Real vacua (dataset_B: 12196 h12=2 flux vacua, model_ID=1, Q_D3=276)
# --------------------------------------------------------------------------- #
def _dataset_B_path():
    """Locate the shipped ``dataset_B.p`` relative to the installed jaxvacua repo.

    ``dataset_B`` holds 12196 real h12=2 flux vacua (degree-18 hypersurface in
    :math:`\\mathbb{CP}^{1,1,1,6,9}`, ``model_ID=1``) as rows ``[x(6), flux(12)]``.
    Returns ``None`` when the notebook data file is absent (e.g. a wheel install),
    so the test skips cleanly rather than failing.
    """
    import jaxvacua
    repo = Path(jaxvacua.__file__).resolve().parents[1]
    p = (repo / "documentation" / "source" / "notebooks"
         / "02_vacuum_finding" / "dataset_B.p")
    return p if p.exists() else None


def _lcs_vacuum_from_row(finder, row, tol=1e-4):
    """Build a NaN-free LCS ``Vacuum`` from a dataset_B row.

    The conifold diagnostics (``zcf``, ``alignment``, ``residual_conifold``) are
    ``None`` — legitimately *not applicable* to a pure-LCS vacuum — so the object
    is NaN-free by construction.
    """
    x = np.asarray(row[:6], dtype=float)
    flux = np.asarray(row[6:18], dtype=float)
    z, _, tau, _ = finder._convert_real_to_complex(x)
    DW = np.asarray(finder.DW_x(x, flux))
    res = float(np.max(np.abs(DW)))
    W0 = complex(finder.W(z, tau, flux, normalise=True))
    return Vacuum(
        x=x, flux=flux, W0=W0, DW=DW, residual=res,
        residual_bulk=res, residual_conifold=None,
        zcf=None, gs=float(1.0 / tau.imag),
        metadata={"model_name": "CP11169_deg18", "is_susy": bool(res < tol)},
    )


def test_real_dataset_B_roundtrip(tmp_path, monkeypatch):
    path = _dataset_B_path()
    if path is None:
        pytest.skip("dataset_B.p not present (notebook data not installed)")
    try:
        import jaxvacua as jvc
        jvc.set_precision("float64")
        finder = jvc.FluxVacuaFinder(h12=2, model_ID=1, maximum_degree=2)
    except Exception as exc:                       # bundled model unavailable
        pytest.skip(f"cannot build model_ID=1 finder: {exc}")

    with gzip.open(path, "rb") as fh:
        A = pickle.load(fh)
    vac = [_lcs_vacuum_from_row(finder, A[i]) for i in (0, 2, 5, 100, 5000)]

    # every built vacuum is NaN-free (the point of the alignment fix + N/A None)
    from dataclasses import fields

    def _num_fields(v):
        for f in fields(v):
            if f.name in ("metadata", "trajectory", "data", "_model"):
                continue
            val = getattr(v, f.name)
            if val is None or isinstance(val, (str, bool, list, dict)):
                continue
            yield np.asarray(val, dtype=complex)
    assert not any(np.any(np.isnan(a)) for v in vac for a in _num_fields(v))

    db = _db(tmp_path, monkeypatch)
    report = db.write_vacua(vac, models=finder, store_trajectory=True)
    assert report.n_written == 5
    run_id = list(report.run_ids_by_geometry.values())[0]

    got = db.read_vacua(run_id)
    assert len(got) == 5
    assert all(a.equals(b) for a, b in zip(vac, got))     # exact round-trip

    # the finder path populates N_flux = tadpole(flux)
    df = db.load_vacua(run_id)
    assert all(int(n) > 0 for n in df["N_flux"])
