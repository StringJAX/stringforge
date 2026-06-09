# Copyright 2026 Andreas Schachner
#
# This file is part of StringForge.
#
# StringForge is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
r"""
Tests for the worker-side Vulcan write path
(:class:`stringforge.vulcan.Vulcan`, :mod:`stringforge.vulcan.writer`).

These tests must run with no HuggingFace credentials and no network.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent))

from stringforge.vulcan import Vulcan, resolve_staging_dir
from stringforge.vulcan.runid import DEFAULT_TEMPLATE
from stringforge.vulcan.schema import (
    GEOMETRY_KEYS,
    REQUIRED_COLUMNS,
    geometry_id,
)
from stringforge.vulcan.writer import (
    FAILED_DIRNAME,
    PENDING_DIRNAME,
    SYNCED_DIRNAME,
    list_pending,
    move_to,
    write_shard,
)


# ── fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_staging():
    with tempfile.TemporaryDirectory() as td:
        yield Path(td)


@pytest.fixture
def small_vacua_df():
    n = 4
    return pd.DataFrame({
        "flux": [[1, 0, -2, 3, 0, 1]] * n,
        "moduli_re": [[0.0, 0.0]] * n,
        "moduli_im": [list(np.linspace(2.0, 3.0, 2))] * n,
        "tau_re": [0.0] * n,
        "tau_im": [4.0] * n,
        "residual": [1e-10] * n,
        "is_susy": [True] * n,
        "solver_name": ["newton"] * n,
    })


@pytest.fixture
def geometry():
    return {"h11": 3, "h12": 2, "ks_id": 384564, "triang_id": 0}


@pytest.fixture
def forge(tmp_staging):
    return Vulcan(
        repo="aschachner/vacua_forge_test",
        staging_dir=tmp_staging,
        run_id_template=DEFAULT_TEMPLATE,
        project="testpkg",
    )


# ── tests ─────────────────────────────────────────────────────────────────

def test_constructor_requires_repo(tmp_staging):
    with pytest.raises(ValueError, match="HuggingFace repo"):
        Vulcan(repo="", staging_dir=tmp_staging)


def test_constructor_records_repo_without_mkdir(tmp_staging):
    # The constructor must not eagerly create the staging layout;
    # ``write_shard`` / ``move_to`` / ``list_pending`` materialise the
    # subdirs on demand.
    forge = Vulcan(repo="user/repo", staging_dir=tmp_staging)
    assert forge.repo == "user/repo"
    for sub in (PENDING_DIRNAME, SYNCED_DIRNAME, FAILED_DIRNAME):
        assert not (tmp_staging / sub).exists()


def test_write_lands_parquet_and_sidecar(forge, small_vacua_df, geometry):
    shard = forge.write(
        small_vacua_df,
        geometry=geometry,
        tadpole_charge=12,
        solver={"name": "newton", "config_hash": "abc"},
        provenance={"git_sha": "deadbeef", "seed": 42, "wall_clock_s": 1.5},
    )
    assert shard.parquet_path.is_file()
    assert shard.sidecar_path.is_file()
    assert shard.n_rows == 4
    assert shard.path_in_repo.startswith("h12_2/h11_3/")


def test_written_parquet_includes_all_required_columns(forge, small_vacua_df, geometry):
    shard = forge.write(small_vacua_df, geometry=geometry, tadpole_charge=12)
    table = pq.read_table(shard.parquet_path)
    columns = set(table.column_names)
    for col in REQUIRED_COLUMNS:
        assert col in columns


def test_geometry_id_matches_independent_computation(forge, small_vacua_df, geometry):
    shard = forge.write(small_vacua_df, geometry=geometry, tadpole_charge=12)
    df = pq.read_table(shard.parquet_path).to_pandas()
    expected = geometry_id(geometry)
    assert (df["geometry_id"] == expected).all()


def test_run_id_template_applied(forge, small_vacua_df, geometry):
    shard = forge.write(
        small_vacua_df,
        geometry=geometry,
        tadpole_charge=12,
        provenance={"seed": 7},
    )
    # Default template includes the geometry hodge numbers, ks/triang
    # IDs and the seed.
    assert "h11-3" in shard.run_id
    assert "h12-2" in shard.run_id
    assert "ks-384564" in shard.run_id
    assert "triang-0" in shard.run_id
    assert "seed-7" in shard.run_id
    assert shard.run_id.endswith("seed-7")


def test_run_id_override(forge, small_vacua_df, geometry):
    shard = forge.write(
        small_vacua_df, geometry=geometry, tadpole_charge=12,
        run_id="custom-tag_v2",
    )
    assert shard.run_id == "custom-tag_v2"


def test_extra_kwargs_supply_template_keys(small_vacua_df, geometry, tmp_staging):
    forge = Vulcan(
        repo="user/repo",
        staging_dir=tmp_staging,
        run_id_template="{date}_{project}_{variant}",
        project="x",
    )
    shard = forge.write(
        small_vacua_df, geometry=geometry, tadpole_charge=12,
        extra_kwargs={"variant": "isd-only"},
    )
    assert "isd-only" in shard.run_id


def test_write_rejects_empty_df(forge, geometry):
    with pytest.raises(ValueError, match="empty DataFrame"):
        forge.write(pd.DataFrame(), geometry=geometry, tadpole_charge=12)


def test_write_rejects_unsafe_run_id(forge, small_vacua_df, geometry):
    with pytest.raises(ValueError, match="slug"):
        forge.write(
            small_vacua_df, geometry=geometry, tadpole_charge=12,
            run_id="has spaces!",
        )


def test_sidecar_carries_provenance(forge, small_vacua_df, geometry):
    shard = forge.write(
        small_vacua_df, geometry=geometry, tadpole_charge=12,
        solver={"name": "newton"},
        provenance={"git_sha": "deadbeef"},
    )
    sidecar = json.loads(shard.sidecar_path.read_text())
    assert sidecar["run_id"] == shard.run_id
    assert sidecar["path_in_repo"] == shard.path_in_repo
    assert sidecar["provenance"]["solver"]["name"] == "newton"
    assert sidecar["provenance"]["provenance"]["git_sha"] == "deadbeef"


def test_list_pending_returns_written_shards(forge, small_vacua_df, geometry):
    shard1 = forge.write(small_vacua_df, geometry=geometry, tadpole_charge=12,
                         provenance={"seed": 1})
    shard2 = forge.write(small_vacua_df, geometry=geometry, tadpole_charge=12,
                         provenance={"seed": 2})
    pending = list_pending(forge.staging_dir)
    assert len(pending) == 2
    run_ids = {s.run_id for s in pending}
    assert {shard1.run_id, shard2.run_id} == run_ids


def test_move_to_relocates_artifacts(forge, small_vacua_df, geometry):
    shard = forge.write(small_vacua_df, geometry=geometry, tadpole_charge=12)
    moved = move_to(shard, forge.staging_dir, SYNCED_DIRNAME)
    assert moved.parquet_path.parent.name == SYNCED_DIRNAME
    assert moved.sidecar_path.parent.name == SYNCED_DIRNAME
    assert moved.parquet_path.is_file()
    # The original locations are now gone.
    assert not shard.parquet_path.exists()
    assert not shard.sidecar_path.exists()


def test_two_writes_same_geometry_same_run_id_overwrite_atomically(forge, small_vacua_df, geometry):
    rid = "stable-run_v1"
    forge.write(small_vacua_df, geometry=geometry, tadpole_charge=12, run_id=rid)
    # A second write with the same run+geometry should overwrite in place.
    df2 = small_vacua_df.head(2)
    shard = forge.write(df2, geometry=geometry, tadpole_charge=12, run_id=rid)
    assert shard.n_rows == 2
    pending = list_pending(forge.staging_dir)
    # Only one pending entry; the second write replaced the first.
    by_run = [s for s in pending if s.run_id == rid]
    assert len(by_run) == 1


def test_resolve_staging_dir_explicit_wins(tmp_path, monkeypatch):
    monkeypatch.setenv("STRINGFORGE_VULCAN_STAGING_DIR", str(tmp_path / "env"))
    explicit = tmp_path / "explicit"
    chosen = resolve_staging_dir(explicit)
    assert chosen == explicit.resolve()


def test_resolve_staging_dir_env_var(tmp_path, monkeypatch):
    target = tmp_path / "from-env"
    monkeypatch.setenv("STRINGFORGE_VULCAN_STAGING_DIR", str(target))
    chosen = resolve_staging_dir(None)
    assert chosen == target.resolve()


def test_from_env_constructor(monkeypatch, tmp_path):
    monkeypatch.setenv("STRINGFORGE_VULCAN_REPO", "user/from-env")
    monkeypatch.setenv("STRINGFORGE_VULCAN_PROJECT", "envproj")
    monkeypatch.setenv("STRINGFORGE_VULCAN_STAGING_DIR", str(tmp_path))
    forge = Vulcan.from_env()
    assert forge.repo == "user/from-env"
    assert forge.project == "envproj"
    assert forge.staging_dir == tmp_path.resolve()


def test_from_env_requires_repo(monkeypatch):
    monkeypatch.delenv("STRINGFORGE_VULCAN_REPO", raising=False)
    with pytest.raises(ValueError, match="no repo specified"):
        Vulcan.from_env()


def test_from_env_tolerates_empty_budget(monkeypatch, tmp_path):
    monkeypatch.setenv("STRINGFORGE_VULCAN_REPO", "user/x")
    monkeypatch.setenv("STRINGFORGE_VULCAN_STAGING_DIR", str(tmp_path))
    monkeypatch.setenv("STRINGFORGE_VULCAN_BUDGET", "")
    forge = Vulcan.from_env()
    # Falls back to the module default rather than crashing.
    assert forge.budget > 0


def test_from_env_rejects_non_integer_budget(monkeypatch, tmp_path):
    monkeypatch.setenv("STRINGFORGE_VULCAN_REPO", "user/x")
    monkeypatch.setenv("STRINGFORGE_VULCAN_STAGING_DIR", str(tmp_path))
    monkeypatch.setenv("STRINGFORGE_VULCAN_BUDGET", "lots")
    with pytest.raises(ValueError, match="must be an integer"):
        Vulcan.from_env()


def test_from_env_explicit_falsy_budget_is_respected(monkeypatch, tmp_path):
    # A caller-supplied ``budget=0`` must not be silently overridden
    # by the env-var / default fallback.
    monkeypatch.delenv("STRINGFORGE_VULCAN_BUDGET", raising=False)
    forge = Vulcan.from_env(repo="u/r", budget=0, staging_dir=tmp_path)
    assert forge.budget == 0


def test_writer_handles_object_dtype_extension_column(forge, geometry):
    # An extension column whose values are Python lists land in
    # pandas as object dtype.  The writer must convert these via
    # value-level inference rather than the (broken) numpy-dtype
    # path.
    df = pd.DataFrame({
        "flux": [[1, 0, -2, 3, 0, 1]] * 2,
        "moduli_re": [[0.0, 0.0]] * 2,
        "moduli_im": [[2.5, 3.0]] * 2,
        "tau_re": [0.0, 0.1],
        "tau_im": [4.0, 4.1],
        # Extension columns -- not in the canonical schema.
        "cone_volume": [[1.0, 2.0, 3.0], [1.5, 2.5, 3.5]],  # object dtype
        "variant": ["alpha", "beta"],                       # object dtype
    })
    shard = forge.write(df, geometry=geometry, tadpole_charge=12)
    table = pq.read_table(shard.parquet_path)
    assert "cone_volume" in table.column_names
    assert "variant" in table.column_names


def test_writer_reports_canonical_column_type_mismatch(forge, geometry):
    # A required column whose values cannot be coerced to the
    # canonical Arrow type should fail with a message naming the
    # column.
    df = pd.DataFrame({
        "flux": ["not-an-int-list"] * 2,
        "moduli_re": [[0.0, 0.0]] * 2,
        "moduli_im": [[2.5, 3.0]] * 2,
        "tau_re": [0.0, 0.1],
        "tau_im": [4.0, 4.1],
    })
    with pytest.raises(ValueError, match="flux"):
        forge.write(df, geometry=geometry, tadpole_charge=12)


def test_solver_name_denormalised_to_column(forge, geometry):
    # Finding #9: solver['name'] must land in the on-disk parquet as
    # the row-level column ``solver_name`` when the caller has not
    # already supplied that column.
    df = pd.DataFrame({
        "flux": [[1, 0, -2, 3, 0, 1]] * 2,
        "moduli_re": [[0.0, 0.0]] * 2,
        "moduli_im": [[2.5, 3.0]] * 2,
        "tau_re": [0.0, 0.1],
        "tau_im": [4.0, 4.1],
    })
    shard = forge.write(
        df, geometry=geometry, tadpole_charge=12,
        solver={"name": "newton"},
    )
    reloaded = pq.read_table(shard.parquet_path).to_pandas()
    assert "solver_name" in reloaded.columns
    assert (reloaded["solver_name"] == "newton").all()


def test_geometry_key_columns_always_overwritten(forge, geometry):
    # Finding #10: caller-supplied geometry-key columns must be
    # silently overwritten by the writer-authoritative values from
    # ``geometry``, mirroring run_id/geometry_id.
    df = pd.DataFrame({
        "flux": [[1, 0, -2, 3, 0, 1]] * 3,
        "moduli_re": [[0.0, 0.0]] * 3,
        "moduli_im": [[2.5, 3.0]] * 3,
        "tau_re": [0.0, 0.1, 0.2],
        "tau_im": [4.0, 4.1, 4.2],
        "ks_id": [99, 99, 99],
    })
    geom = {**geometry, "ks_id": 1}
    shard = forge.write(df, geometry=geom, tadpole_charge=12)
    reloaded = pq.read_table(shard.parquet_path).to_pandas()
    assert (reloaded["ks_id"] == 1).all()


def test_write_rejects_empty_geometry(forge, small_vacua_df):
    # Finding #20: a geometry mapping carrying no non-sentinel keys
    # must be rejected upfront.
    with pytest.raises(ValueError, match="non-sentinel"):
        forge.write(small_vacua_df, geometry={}, tadpole_charge=12)


def test_write_accepts_single_non_sentinel_geometry_key(forge, small_vacua_df):
    # Finding #20: a single non-sentinel geometry key (e.g. cicy_id) is
    # sufficient for the write to succeed.
    shard = forge.write(
        small_vacua_df, geometry={"cicy_id": 7}, tadpole_charge=12,
    )
    assert shard.parquet_path.is_file()


def test_cli_status_smoke(tmp_path):
    r"""``python -m stringforge.vulcan status`` runs end-to-end."""
    import subprocess
    result = subprocess.run(
        [sys.executable, "-m", "stringforge.vulcan",
         "status", "--staging-dir", str(tmp_path)],
        capture_output=True, text=True, check=True,
    )
    assert "pending shards: 0" in result.stdout
