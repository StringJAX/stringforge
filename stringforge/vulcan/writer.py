# Copyright 2026 Andreas Schachner
#
# This file is part of StringForge.
#
# StringForge is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

r"""
Worker-side shard writer for Vulcan.

Worker code never touches HuggingFace.  :func:`write_shard` lands a
validated parquet file inside ``{staging_dir}/pending/`` together with
a per-file ``.meta.json`` sidecar carrying the geometry, solver, and
provenance dicts.  The sync tier (:mod:`stringforge.vulcan.sync`)
later batches these into HuggingFace commits.

Atomicity: each of the parquet body and the JSON sidecar is staged as a .tmp file and os.replace-d into place, so neither file is ever observed half-written.  The two os.replace calls are sequential, not atomic as a pair: a crash between them can leave a parquet without a sidecar.  :func:`list_pending` silently skips such orphan parquets, so the sync tier never uploads a half-pair, and a re-run of the same write rebuilds the missing sidecar.
"""
from __future__ import annotations

import json
import os
import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .schema import (
    GEOMETRY_KEYS,
    GEOMETRY_KEY_SENTINEL,
    REQUIRED_COLUMNS,
    SCHEMA_VERSION,
    geometry_id as _geometry_id,
    is_safe_run_id,
    pyarrow_schema,
    shard_relpath,
    validate_schema,
)

#: Parquet kv-metadata key under which we stash the full provenance
#: blob.  Mirrors the pattern at vacua_writer.py:1714-1727.
PROVENANCE_METADATA_KEY: bytes = b"stringforge.vulcan.provenance"
#: Parquet kv-metadata key carrying the schema version.
SCHEMA_VERSION_METADATA_KEY: bytes = b"schema_version"

PENDING_DIRNAME: str = "pending"
SYNCED_DIRNAME: str = "synced"
FAILED_DIRNAME: str = "failed"


@dataclass
class StagedShard:
    r"""
    Reference to a freshly staged shard on local disk.

    Attributes:
        run_id: The run identifier carried on every row.
        parquet_path: Absolute path to the staged parquet file.
        sidecar_path: Absolute path to the per-file ``.meta.json``.
        path_in_repo: Canonical HF-repo-relative target path.
        n_rows: Number of vacuum rows written.
    """
    run_id: str
    parquet_path: Path
    sidecar_path: Path
    path_in_repo: str
    n_rows: int


def ensure_staging_layout(staging_dir: Path) -> Path:
    r"""
    **Description:**
    Ensure the Vulcan staging directory carries the canonical
    ``pending/``, ``synced/``, and ``failed/`` subdirectories.

    Idempotent: safe to call before every :func:`write_shard` or
    :func:`~stringforge.vulcan.sync.sync_pending` invocation.

    Args:
        staging_dir: Root staging directory.

    Returns:
        Path: The resolved staging root (with the three subdirectories
        guaranteed to exist).
    """
    root = Path(staging_dir).resolve()
    for sub in (PENDING_DIRNAME, SYNCED_DIRNAME, FAILED_DIRNAME):
        (root / sub).mkdir(parents=True, exist_ok=True)
    return root


def _coerce_dataframe(
    df: pd.DataFrame,
    *,
    run_id: str,
    geometry: Mapping[str, Any],
    tadpole_charge: Optional[int],
    solver: Optional[Mapping[str, Any]] = None,
    provenance: Optional[Mapping[str, Any]] = None,
) -> pd.DataFrame:
    r"""
    Fill in the production-tier required columns that callers shouldn't
    have to set by hand.

    ``run_id``, ``geometry_id``, and the geometry-key columns overwrite
    any existing column with the same name -- the writer is the
    authority on identity, not the caller.  ``created_at`` is filled in
    only if absent.  Selected solver/provenance fields are denormalised
    into row-level columns when the caller has not already supplied them.
    """
    out = df.copy()
    out["run_id"] = run_id
    out["geometry_id"] = _geometry_id(geometry)

    if "created_at" not in out.columns:
        out["created_at"] = pd.Timestamp(datetime.now(tz=timezone.utc))
    else:
        out["created_at"] = pd.to_datetime(out["created_at"], utc=True)

    # Denormalise geometry keys -- writer is authoritative, mirroring run_id/geometry_id.
    for key in GEOMETRY_KEYS:
        out[key] = int(geometry.get(key, GEOMETRY_KEY_SENTINEL))

    if tadpole_charge is not None and "tadpole_charge" not in out.columns:
        out["tadpole_charge"] = int(tadpole_charge)

    # Denormalise solver / provenance entries into row-level columns
    # (only when the caller did not already supply the column and the
    # source value is non-None).
    if solver is not None:
        if "solver_name" not in out.columns and solver.get("name") is not None:
            out["solver_name"] = solver["name"]
        if "solver_config_hash" not in out.columns and solver.get("config_hash") is not None:
            out["solver_config_hash"] = solver["config_hash"]
    if provenance is not None:
        if "git_sha" not in out.columns and provenance.get("git_sha") is not None:
            out["git_sha"] = provenance["git_sha"]
        if "seed" not in out.columns and provenance.get("seed") is not None:
            out["seed"] = provenance["seed"]
        if "wall_clock_s" not in out.columns and provenance.get("wall_clock_s") is not None:
            out["wall_clock_s"] = provenance["wall_clock_s"]

    return out


def _arrow_table(df: pd.DataFrame, provenance: Mapping[str, Any]) -> pa.Table:
    r"""
    **Description:**
    Convert ``df`` to a pyarrow Table that conforms to the canonical
    Vulcan schema, attaching schema-version and provenance metadata.

    The schema is *intersected* with ``df.columns`` so callers may
    omit optional columns -- only :data:`REQUIRED_COLUMNS` must be
    present (and :func:`validate_schema` has already checked that).
    Extension columns -- anything ``df`` carries that the canonical
    schema does not name -- are inferred by Arrow from the underlying
    numpy/pandas values rather than from the pandas dtype, so
    ``object``-dtype columns (lists, strings, ``Decimal``, ...) round-trip
    correctly.

    Args:
        df: Coerced DataFrame whose required columns are all populated.
        provenance: JSON-serialisable provenance dict to embed as
            parquet kv-metadata under
            :data:`PROVENANCE_METADATA_KEY`.

    Returns:
        pa.Table: An Arrow table ready for :func:`pyarrow.parquet.write_table`.
    """
    canonical = pyarrow_schema()
    known = {f.name for f in canonical}

    # Build the canonical-column slice first so we get the correct
    # logical types (`list<int32>` for `flux`, timestamp[us, tz=UTC]
    # for `created_at`, etc.) rather than whatever Arrow would infer.
    column_arrays: dict[str, pa.Array] = {}
    selected_fields: list[pa.Field] = []
    for field in canonical:
        if field.name not in df.columns:
            continue
        try:
            column_arrays[field.name] = pa.array(df[field.name].to_list(), type=field.type)
        except (pa.ArrowInvalid, pa.ArrowTypeError):
            # Surface this with the field name; the caller's df has a
            # value that doesn't fit the canonical type.
            raise ValueError(
                f"column {field.name!r} cannot be cast to canonical type "
                f"{field.type}; check the underlying values"
            )
        selected_fields.append(field)

    # Extension columns: let Arrow infer the type from the values.  We
    # use ``pa.array`` on the column's Python list (not the numpy dtype)
    # because pandas object dtype loses the per-element type info that
    # numpy would otherwise paper over with ``object``.
    extra_fields: list[pa.Field] = []
    for name in df.columns:
        if name in known:
            continue
        arr = pa.array(df[name].to_list())
        extra_fields.append(pa.field(name, arr.type))
        column_arrays[name] = arr

    schema = pa.schema(selected_fields + extra_fields).with_metadata({
        SCHEMA_VERSION_METADATA_KEY: str(SCHEMA_VERSION).encode("ascii"),
        PROVENANCE_METADATA_KEY: json.dumps(provenance, sort_keys=True, default=str).encode("utf-8"),
    })
    ordered = [column_arrays[f.name] for f in schema]
    return pa.Table.from_arrays(ordered, schema=schema)


def write_shard(
    df: pd.DataFrame,
    *,
    staging_dir: Path,
    run_id: str,
    geometry: Mapping[str, Any],
    tadpole_charge: Optional[int] = None,
    solver: Optional[Mapping[str, Any]] = None,
    provenance: Optional[Mapping[str, Any]] = None,
    bucket_bits: int = 4,
) -> StagedShard:
    r"""
    **Description:**
    Write a single vacuum-batch parquet shard to ``staging_dir/pending/``.

    The writer is the authority on identity: ``run_id``,
    ``geometry_id``, and the geometry-key columns are unconditionally
    overwritten here, regardless of any per-row values the caller may
    have supplied.  ``tadpole_charge`` and ``created_at``, by contrast,
    are filled in only when absent from ``df``.  The default ``run_id``
    substitution dict used by the calling Vulcan handle is the union of
    ``geometry``, ``solver['name']`` (when supplied), ``provenance['seed']``
    (when supplied), and ``extra_kwargs``.

    Schema validation runs after this coercion --
    :func:`~stringforge.vulcan.schema.validate_schema` must succeed
    before any I/O happens, so a bad batch never leaves a half-written
    shard behind.

    The on-disk write is atomic per-file -- not per-pair: each of the
    parquet body and the JSON sidecar is staged as a ``.tmp`` file and
    ``os.replace``-d into place, but the two replacements are sequential.
    A crash between them can leave a parquet without a sidecar;
    :func:`list_pending` silently skips such orphans, so the sync tier
    never uploads a half-pair, and a re-run of the same write rebuilds
    the missing sidecar.  The ``.tmp`` suffix carries a per-process,
    per-call nonce so concurrent writers on the same
    ``(geometry, run_id)`` do not corrupt each other's staged files.

    Args:
        df: Per-vacuum DataFrame.  Must carry the vault-floor columns
            (see :data:`~stringforge.vulcan.schema.REQUIRED_COLUMNS`).
        staging_dir: Root staging directory.  Subdirs are created on
            demand via :func:`ensure_staging_layout`.
        run_id: Identifier carried on every row.  Must satisfy
            :func:`~stringforge.vulcan.schema.is_safe_run_id`.
        geometry: ``{h11, h12, ks_id, triang_id, conifold_id, cicy_id}``
            mapping.  h11, h12 are interpreted in mirror (jaxvacua /
            lcs_tree) convention.  Missing keys are filled with
            :data:`~stringforge.vulcan.schema.GEOMETRY_KEY_SENTINEL`,
            but at least one key must be set to a non-sentinel value.
        tadpole_charge: ``N_flux`` for the batch.  Required unless
            already present on ``df`` as a column.
        solver: Optional ``{name, config_hash, ...}`` mapping; embedded
            in the parquet kv-metadata blob.
        provenance: Optional ``{git_sha, seed, wall_clock_s, ...}``
            mapping; embedded alongside ``solver`` in the same blob.
        bucket_bits: Width of the bucket prefix in the HF path layout.

    Returns:
        StagedShard: Reference to the on-disk parquet + sidecar pair
        and the canonical HF-repo-relative path.

    Raises:
        ValueError: ``run_id`` is not slug-safe, ``df`` is empty,
            ``geometry`` carries no non-sentinel keys, or
            :func:`~stringforge.vulcan.schema.validate_schema` rejects
            the coerced DataFrame.
    """
    if not is_safe_run_id(run_id):
        raise ValueError(
            f"run_id is not slug-safe: {run_id!r}.  "
            f"Must match LABEL_SLUG_RE."
        )
    if len(df) == 0:
        raise ValueError("Vulcan.write: cannot stage an empty DataFrame")

    provided = {k: geometry.get(k) for k in GEOMETRY_KEYS}
    if all(v is None or int(v) == GEOMETRY_KEY_SENTINEL for v in provided.values()):
        raise ValueError(
            "Vulcan.write: geometry must carry at least one of "
            f"{GEOMETRY_KEYS} set to a non-sentinel value; got {dict(geometry)!r}"
        )

    root = ensure_staging_layout(Path(staging_dir))
    pending = root / PENDING_DIRNAME

    coerced = _coerce_dataframe(
        df, run_id=run_id, geometry=geometry, tadpole_charge=tadpole_charge,
        solver=solver, provenance=provenance,
    )
    # Extension columns are a designed writer feature, so don't warn on
    # them here (validate_schema still aggregates real schema problems).
    validate_schema(coerced, warn_unknown=False)

    provenance_blob: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "geometry": {k: int(geometry.get(k, GEOMETRY_KEY_SENTINEL)) for k in GEOMETRY_KEYS},
        "solver": dict(solver) if solver else {},
        "provenance": dict(provenance) if provenance else {},
        "n_rows": int(len(coerced)),
    }
    table = _arrow_table(coerced, provenance=provenance_blob)

    # Path in the HF repo.  We also use this (slashes -> "__") as the
    # local filename so two shards with identical (run_id, geometry)
    # collide at the same local path -- deliberate: a retry of the
    # same call overwrites cleanly.
    path_in_repo = shard_relpath(geometry, run_id, bucket_bits=bucket_bits)
    local_name = path_in_repo.replace("/", "__")
    parquet_path = pending / local_name
    sidecar_path = pending / (local_name + ".meta.json")
    nonce = f".{os.getpid()}.{uuid.uuid4().hex}.tmp"
    tmp_parquet = parquet_path.with_suffix(parquet_path.suffix + nonce)
    tmp_sidecar = sidecar_path.with_suffix(sidecar_path.suffix + nonce)

    pq.write_table(table, tmp_parquet)
    tmp_sidecar.write_text(json.dumps({
        "path_in_repo": path_in_repo,
        "run_id": run_id,
        "n_rows": int(len(coerced)),
        "provenance": provenance_blob,
    }, indent=2, sort_keys=True, default=str))

    os.replace(tmp_parquet, parquet_path)
    os.replace(tmp_sidecar, sidecar_path)

    return StagedShard(
        run_id=run_id,
        parquet_path=parquet_path,
        sidecar_path=sidecar_path,
        path_in_repo=path_in_repo,
        n_rows=int(len(coerced)),
    )


def list_pending(staging_dir: Path) -> list[StagedShard]:
    r"""
    **Description:**
    Enumerate currently-staged shards awaiting sync.

    Each ``.parquet`` in ``pending/`` is paired with its
    ``.parquet.meta.json`` sidecar.  Shards without a sidecar (e.g.
    after a crashed write where the parquet flushed but the sidecar
    did not) are skipped silently -- the next worker call recreates
    them.

    Args:
        staging_dir: Root staging directory.

    Returns:
        list[StagedShard]: All complete shard records found in
        ``pending/``, sorted by filename for deterministic ordering.
    """
    root = Path(staging_dir).resolve()
    pending = root / PENDING_DIRNAME
    if not pending.is_dir():
        return []
    out: list[StagedShard] = []
    for parquet_path in sorted(pending.glob("*.parquet")):
        sidecar = parquet_path.with_suffix(parquet_path.suffix + ".meta.json")
        if not sidecar.is_file():
            continue
        try:
            meta = json.loads(sidecar.read_text())
        except json.JSONDecodeError:
            continue
        out.append(StagedShard(
            run_id=meta["run_id"],
            parquet_path=parquet_path,
            sidecar_path=sidecar,
            path_in_repo=meta["path_in_repo"],
            n_rows=int(meta.get("n_rows", 0)),
        ))
    return out


def move_to(shard: StagedShard, staging_dir: Path, target: str) -> StagedShard:
    r"""
    **Description:**
    Move a staged shard's ``(parquet, sidecar)`` pair into a sibling
    subdirectory.

    The two files are moved sequentially, not as an atomic pair: a
    crash between the two ``shutil.move`` calls can leave the parquet
    in the destination directory while the sidecar remains in the
    source.  Orphan recovery is the responsibility of
    :func:`list_pending` (which skips parquets without a sidecar) and
    of a future ``prune_orphans`` helper.

    Args:
        shard: The shard reference returned by :func:`write_shard`.
        staging_dir: Root staging directory.
        target: Target subdirectory name -- typically
            :data:`SYNCED_DIRNAME` or :data:`FAILED_DIRNAME`.

    Returns:
        StagedShard: A new shard record whose paths point at the
        post-move locations.
    """
    root = ensure_staging_layout(Path(staging_dir))
    dest_dir = root / target
    dest_parquet = dest_dir / shard.parquet_path.name
    dest_sidecar = dest_dir / shard.sidecar_path.name
    # shutil.move falls back to copy+remove across filesystems.
    shutil.move(str(shard.parquet_path), str(dest_parquet))
    shutil.move(str(shard.sidecar_path), str(dest_sidecar))
    return StagedShard(
        run_id=shard.run_id,
        parquet_path=dest_parquet,
        sidecar_path=dest_sidecar,
        path_in_repo=shard.path_in_repo,
        n_rows=shard.n_rows,
    )


def vacua_to_vulcan_df(
    vacua: Any,
    finder: Any = None,
    *,
    store_full: bool = True,
    store_trajectory: bool = False,
) -> "pd.DataFrame":
    r"""
    **Description:**
    Build a Vulcan-schema :class:`pandas.DataFrame` from
    ``jaxvacua.vacuum.Vacuum`` objects (or ``PFV`` / ``AFV``).

    Each vacuum maps to one row via
    :func:`stringforge._vacuum_adapter.vacuum_to_row`: the vault-floor columns
    (``flux``, ``moduli_re``, ``moduli_im``, ``tau_re``, ``tau_im``) plus the
    optional physics columns (``W_re``/``W_im``, ``F_terms_re``/``F_terms_im``,
    ``g_s``, ``residual``, ``is_susy``, ``tadpole_charge``).  The authoritative
    JSON-encoded ``Vacuum.to_dict()`` record travels in ``extra_data`` for exact,
    finder-free read-back (never pickle).

    Args:
        vacua: A ``Vacuum`` or an iterable of them.
        finder: Optional finder — used only for ``tadpole_charge`` =
            ``finder.tadpole(flux)`` (the geometry-independent D3 charge).
        store_full (bool): If True (default), embed the full record in
            ``extra_data``.  If False, only the readable ``kind`` / ``model_name``
            keys are kept (a shard-size escape hatch; read-back then needs a
            finder to rebuild from the typed columns).
        store_trajectory (bool): Persist the verbose optimisation trajectory in
            the stored record.  Defaults to ``False``.

    Returns:
        pandas.DataFrame: One row per vacuum, ready for :meth:`Vulcan.write`.
    """
    import json

    import numpy as np

    from .._vacuum_adapter import is_vacuum, vacuum_to_row

    items = [vacua] if is_vacuum(vacua) else list(vacua)
    rows: list[dict] = []
    for v in items:
        result, extra = vacuum_to_row(v, finder, store_trajectory=store_trajectory)
        z = np.asarray(result["moduli"])
        tau = complex(result["tau"])
        fl = np.real(np.asarray(result["flux"]))
        row: dict[str, Any] = {
            "flux": [int(round(float(a))) for a in fl.ravel()],
            "moduli_re": [float(a) for a in np.real(z).ravel()],
            "moduli_im": [float(a) for a in np.imag(z).ravel()],
            "tau_re": float(tau.real),
            "tau_im": float(tau.imag),
            "is_susy": bool(result.get("is_susy")),
        }
        res = result.get("residual")
        row["residual"] = None if res is None else float(res)
        nflux = result.get("N_flux")
        if nflux is not None:
            row["tadpole_charge"] = int(nflux)
        W = result.get("W")
        if W is not None:
            Wc = complex(W)
            row["W_re"], row["W_im"] = float(Wc.real), float(Wc.imag)
        F = result.get("F_terms")
        if F is not None:
            Fa = np.asarray(F)
            row["F_terms_re"] = [float(a) for a in np.real(Fa).ravel()]
            row["F_terms_im"] = [float(a) for a in np.imag(Fa).ravel()]
        gs = getattr(v, "gs", None)
        if gs is not None:
            try:
                gsf = float(gs)
                if np.isfinite(gsf):
                    row["g_s"] = gsf
            except (TypeError, ValueError):
                pass
        payload = dict(extra)
        if not store_full:
            payload.pop("vacuum", None)
        row["extra_data"] = json.dumps(payload)
        rows.append(row)
    return pd.DataFrame(rows)
