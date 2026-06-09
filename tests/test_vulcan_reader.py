# Copyright 2026 Andreas Schachner
#
# This file is part of StringForge.
#
# StringForge is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
r"""
Tests for the Vulcan read path (:mod:`stringforge.vulcan.reader`).

Coverage:

* LRU shard cache: eviction order, capacity, clear.
* Local-source catalogue scan from a populated ``synced/`` tree.
* Query filters (catalogue-level + row-level).
* ``fetch_run`` returns the rows of a known run.
* ``Vulcan.query`` / ``Vulcan.fetch_run`` convenience wrappers.
* Cache is honoured -- second read does not re-touch the disk.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent))

from stringforge.vulcan import Vulcan, VulcanReader
from stringforge.vulcan.reader import (
    CATALOG_COLUMNS,
    DEFAULT_CACHE_SIZE,
    _LRUShardCache,
)
from stringforge.vulcan.writer import SYNCED_DIRNAME, move_to


# ── fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_staging():
    with tempfile.TemporaryDirectory() as td:
        yield Path(td)


@pytest.fixture
def populated_forge(tmp_staging):
    r"""
    Build a Vulcan staging dir with three synced shards (two distinct
    geometries; one solver matches across all rows).
    """
    forge = Vulcan(repo="user/repo", staging_dir=tmp_staging, project="t")
    geom_a = {"h11": 3, "h12": 2, "ks_id": 1, "triang_id": 0}
    geom_b = {"h11": 3, "h12": 2, "ks_id": 2, "triang_id": 0}

    def _df(n, susy: bool, solver: str):
        return pd.DataFrame({
            "flux": [[1, 0, -2, 3, 0, 1]] * n,
            "moduli_re": [[0.0, 0.0]] * n,
            "moduli_im": [[2.5, 3.0]] * n,
            "tau_re": [0.0] * n,
            "tau_im": [4.0] * n,
            "is_susy": [susy] * n,
            "solver_name": [solver] * n,
        })

    s1 = forge.write(_df(3, True,  "newton"), geometry=geom_a, tadpole_charge=12, run_id="run-a-1")
    s2 = forge.write(_df(2, False, "newton"), geometry=geom_a, tadpole_charge=12, run_id="run-a-2")
    s3 = forge.write(_df(4, True,  "agm"),    geometry=geom_b, tadpole_charge=12, run_id="run-b-1")
    # Move them all to synced/ as the reader scans only that subtree.
    for s in (s1, s2, s3):
        move_to(s, tmp_staging, SYNCED_DIRNAME)
    return forge


# ── LRU cache ─────────────────────────────────────────────────────────────

def test_lru_basic_get_put():
    c = _LRUShardCache(capacity=2)
    df = pd.DataFrame({"x": [1, 2]})
    c.put("a", df)
    assert c.get("a") is df
    assert c.get("missing") is None
    assert len(c) == 1


def test_lru_evicts_least_recently_used():
    c = _LRUShardCache(capacity=2)
    c.put("a", pd.DataFrame({"x": [1]}))
    c.put("b", pd.DataFrame({"x": [2]}))
    c.get("a")  # promote "a"
    c.put("c", pd.DataFrame({"x": [3]}))  # should evict "b"
    assert c.get("a") is not None
    assert c.get("b") is None
    assert c.get("c") is not None


def test_lru_rejects_invalid_capacity():
    with pytest.raises(ValueError):
        _LRUShardCache(capacity=0)


def test_lru_clear_drops_everything():
    c = _LRUShardCache(capacity=4)
    c.put("a", pd.DataFrame({"x": [1]}))
    c.put("b", pd.DataFrame({"x": [2]}))
    c.clear()
    assert len(c) == 0
    assert c.get("a") is None


# ── catalogue scan ────────────────────────────────────────────────────────

def test_local_reader_catalogue_columns(populated_forge):
    reader = populated_forge.reader()
    cat = reader.catalog()
    assert list(cat.columns) == list(CATALOG_COLUMNS)


def test_local_reader_catalogue_has_one_row_per_shard(populated_forge):
    cat = populated_forge.reader().catalog()
    assert len(cat) == 3


def test_catalogue_carries_geometry_keys(populated_forge):
    cat = populated_forge.reader().catalog()
    assert set(cat["h12"].unique()) == {2}
    assert set(cat["h11"].unique()) == {3}
    assert set(cat["ks_id"].unique()) == {1, 2}


def test_catalogue_carries_run_ids(populated_forge):
    cat = populated_forge.reader().catalog()
    assert set(cat["run_id"]) == {"run-a-1", "run-a-2", "run-b-1"}


def test_catalogue_refresh_picks_up_new_shards(populated_forge, tmp_staging):
    reader = populated_forge.reader()
    first = reader.catalog()
    # Add another shard.
    geom = {"h11": 3, "h12": 2, "ks_id": 7, "triang_id": 0}
    df = pd.DataFrame({
        "flux": [[1, 0, -2, 3, 0, 1]],
        "moduli_re": [[0.0, 0.0]],
        "moduli_im": [[2.5, 3.0]],
        "tau_re": [0.0],
        "tau_im": [4.0],
    })
    s = populated_forge.write(df, geometry=geom, tadpole_charge=12, run_id="run-new")
    move_to(s, tmp_staging, SYNCED_DIRNAME)
    # Without refresh: stale catalogue.
    cached = reader.catalog()
    assert len(cached) == len(first)
    # With refresh: fresh scan.
    fresh = reader.catalog(refresh=True)
    assert len(fresh) == len(first) + 1


# ── query ─────────────────────────────────────────────────────────────────

def test_query_by_run_id(populated_forge):
    rows = populated_forge.query(run_id="run-a-1")
    assert len(rows) == 3


def test_query_by_ks_id(populated_forge):
    rows = populated_forge.query(ks_id=2)
    assert len(rows) == 4  # only the b-1 shard


def test_query_with_row_level_predicate(populated_forge):
    rows = populated_forge.query(solver_name="agm")
    assert len(rows) == 4
    assert set(rows["solver_name"].unique()) == {"agm"}


def test_query_columns_projection(populated_forge):
    rows = populated_forge.query(
        run_id="run-a-1",
        columns=["run_id", "is_susy", "tau_re"],
    )
    assert list(rows.columns) == ["run_id", "is_susy", "tau_re"]


def test_query_missing_match_returns_empty(populated_forge):
    rows = populated_forge.query(ks_id=99999)
    assert len(rows) == 0


def test_query_chains_filters(populated_forge):
    rows = populated_forge.query(h12=2, ks_id=1, is_susy=False)
    # Only the run-a-2 shard has is_susy=False.
    assert len(rows) == 2
    assert (rows["is_susy"] == False).all()


# ── fetch_run ─────────────────────────────────────────────────────────────

def test_fetch_run_returns_all_run_rows(populated_forge):
    rows = populated_forge.fetch_run("run-b-1")
    assert len(rows) == 4
    assert (rows["solver_name"] == "agm").all()


def test_fetch_run_empty_for_unknown_id(populated_forge):
    rows = populated_forge.fetch_run("does-not-exist")
    assert len(rows) == 0


# ── shard cache integration ──────────────────────────────────────────────

def test_repeated_fetch_uses_cache(populated_forge):
    reader = populated_forge.reader()
    cat = reader.catalog()
    path = cat.iloc[0]["path_in_repo"]
    # First read populates the cache.
    df1 = reader.fetch_shard(path)
    with mock.patch("stringforge.vulcan.reader.pd.read_parquet") as patched:
        df2 = reader.fetch_shard(path)
        patched.assert_not_called()
    # Both views point at the same underlying DataFrame.
    assert df1 is df2


def test_clear_cache_forces_reload(populated_forge):
    reader = populated_forge.reader()
    cat = reader.catalog()
    path = cat.iloc[0]["path_in_repo"]
    reader.fetch_shard(path)
    reader.clear_cache()
    with mock.patch("stringforge.vulcan.reader.pd.read_parquet",
                    wraps=pd.read_parquet) as patched:
        reader.fetch_shard(path)
        patched.assert_called_once()


def test_fetch_unknown_shard_raises(populated_forge):
    reader = populated_forge.reader()
    with pytest.raises(FileNotFoundError):
        reader.fetch_shard("h12_2/h11_3/bucket_0/run_does-not-exist.parquet")


# ── factory dispatch ─────────────────────────────────────────────────────

def test_from_local_factory_returns_reader(tmp_staging):
    reader = VulcanReader.from_local(tmp_staging)
    assert isinstance(reader, VulcanReader)


def test_from_hf_factory_returns_reader():
    reader = VulcanReader.from_hf("user/repo")
    assert isinstance(reader, VulcanReader)


# ── catalogue: parallel scan, unreadable shards, schema-version guard ────

def test_catalogue_handles_many_shards(tmp_staging):
    r"""
    Parallel catalogue scan should return one row per shard for a
    moderately large fanout without raising.
    """
    forge = Vulcan(repo="user/repo", staging_dir=tmp_staging, project="t")
    n_shards = 16

    def _df(n: int) -> pd.DataFrame:
        return pd.DataFrame({
            "flux": [[1, 0, -2, 3, 0, 1]] * n,
            "moduli_re": [[0.0, 0.0]] * n,
            "moduli_im": [[2.5, 3.0]] * n,
            "tau_re": [0.0] * n,
            "tau_im": [4.0] * n,
        })

    for i in range(n_shards):
        geom = {"h11": 3, "h12": 2, "ks_id": i, "triang_id": 0}
        s = forge.write(_df(1), geometry=geom, tadpole_charge=12, run_id=f"run-{i}")
        move_to(s, tmp_staging, SYNCED_DIRNAME)

    cat = forge.reader().catalog()
    assert len(cat) == n_shards
    assert cat.attrs["n_skipped"] == 0


def test_catalogue_skips_unreadable_shard_with_warning(populated_forge, tmp_staging):
    r"""
    A non-parquet file with a ``.parquet`` extension dropped into the
    synced tree must be skipped with a RuntimeWarning and counted in
    ``n_skipped``.
    """
    synced_root = tmp_staging / SYNCED_DIRNAME
    bogus = synced_root / "bogus_shard.parquet"
    bogus.write_text("this is not a parquet file")
    reader = populated_forge.reader()
    with pytest.warns(RuntimeWarning, match="skipping unreadable shard"):
        cat = reader.catalog()
    assert cat.attrs["n_skipped"] == 1


def test_catalogue_warns_on_schema_version_mismatch(populated_forge, tmp_staging):
    r"""
    Mutating a shard's kv-metadata to a different ``schema_version``
    triggers a RuntimeWarning; ``strict=True`` promotes it to ValueError.
    """
    from stringforge.vulcan.writer import (
        PROVENANCE_METADATA_KEY,
        SCHEMA_VERSION_METADATA_KEY,
    )

    synced_root = tmp_staging / SYNCED_DIRNAME
    target = next(synced_root.rglob("*.parquet"))
    table = pq.read_table(target)
    raw_meta = dict(table.schema.metadata or {})
    raw_meta[SCHEMA_VERSION_METADATA_KEY] = b"999"
    new_schema = table.schema.with_metadata(raw_meta)
    pq.write_table(table.replace_schema_metadata(new_schema.metadata), target)

    reader = populated_forge.reader()
    with pytest.warns(RuntimeWarning, match="schema_version=999"):
        reader.catalog()

    # strict=True promotes to ValueError on a fresh scan.
    reader2 = populated_forge.reader()
    with pytest.raises(ValueError, match="schema_version mismatch"):
        reader2.catalog(strict=True)
