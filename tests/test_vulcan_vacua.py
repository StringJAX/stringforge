"""Vulcan production-tier support for ``jaxvacua.vacuum`` objects.

Mirrors the curated-tier ``test_vacuum_adapter`` round-trip on the Vulcan
staging path: ``Vulcan.write_vacua`` stages a per-geometry parquet shard whose
``extra_data`` column carries the authoritative JSON record, and
``Vulcan.read_vacua`` rebuilds the exact ``Vacuum`` objects from a local shard —
no HuggingFace I/O, no pickle.
"""
import json

import numpy as np
import pytest

jaxvacua_vacuum = pytest.importorskip("jaxvacua.vacuum")
Vacuum = jaxvacua_vacuum.Vacuum

from stringforge.vulcan import Vulcan, vacua_to_vulcan_df


_IDENT = {
    "h11": 5, "h12": 2, "ks_id": 10, "triang_id": 0,
    "conifold_id": -1, "cicy_id": -1, "model_name": "test_geo",
}


def _vacuum():
    """A NaN-free solved h12=2 vacuum (conifold diagnostics are None = N/A)."""
    x = np.array([0.0, 2.5, 0.0, 3.0, 0.0, 4.0], dtype=float)
    flux = np.array([1, 0, -2, 3, 0, 1], dtype=float)
    return Vacuum(
        x=x, flux=flux, W0=complex(0.05, 0.01),
        DW=np.array([1e-10, 2e-10, 1e-10], dtype=float), residual=2e-10,
        residual_bulk=2e-10, residual_conifold=None,
        zcf=None, gs=0.25,
        metadata={"model_name": "test_geo", "identity": dict(_IDENT)},
    )


def _vulcan(tmp_path):
    return Vulcan(repo="test/vulcan", staging_dir=str(tmp_path / "staging"))


def test_vulcan_write_read_roundtrip(tmp_path):
    v = _vacuum()
    vulc = _vulcan(tmp_path)
    # finder-free -> geometry from metadata['identity'], explicit tadpole_charge
    shard = vulc.write_vacua([v], tadpole_charge=5, store_trajectory=True)
    assert shard.n_rows == 1
    got = vulc.read_vacua(shard)
    assert len(got) == 1
    assert got[0].equals(v)


def test_vulcan_df_schema_columns():
    df = vacua_to_vulcan_df([_vacuum()], finder=None)
    for col in ("flux", "moduli_re", "moduli_im", "tau_re", "tau_im", "extra_data"):
        assert col in df.columns
    # extra_data is valid JSON carrying the record (never pickle)
    payload = json.loads(df["extra_data"].iloc[0])
    assert "vacuum" in payload and payload["kind"] == "Vacuum"


def test_vulcan_finder_free_requires_tadpole(tmp_path):
    vulc = _vulcan(tmp_path)
    with pytest.raises(ValueError, match="tadpole"):
        vulc.write_vacua([_vacuum()])          # no finder, no tadpole_charge


def test_vulcan_store_full_false_drops_blob(tmp_path):
    import pandas as pd
    v = _vacuum()
    vulc = _vulcan(tmp_path)
    shard = vulc.write_vacua([v], tadpole_charge=5, store_full=False)
    # no record embedded -> read-back skips the (record-less) row
    assert vulc.read_vacua(shard) == []
    df = pd.read_parquet(shard.parquet_path)
    assert "vacuum" not in json.loads(df["extra_data"].iloc[0])
