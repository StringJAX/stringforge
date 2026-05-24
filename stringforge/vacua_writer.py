# ==============================================================================
# stringforge / vacua_writer
#
# Vacua persistence layer — writes designated vacua to disk, retracts/purges
# them, uploads and fetches from the community HuggingFace vault, and runs
# physics validation.  The `VacuaWriter` class wraps a `CYDatabase` instance
# and forwards any attribute not defined locally to the wrapped database via
# `__getattr__`, so method bodies that read `self.cache_dir`, `self._fetch_*`,
# `self._catalog`, etc. keep working unchanged.
#
# Method names are kept identical to the originals on `CYDatabase` (e.g.
# `designate_vacua`, `push_vacua_to_hub`).  The `LCSDatabase` layer in
# `lcs_database.py` forwards its own methods of the same names to these.
#
# No imports from `lcs_tree` / `flux_eft` / `flux_vacua_finder` at module
# load — any physics dependency is imported lazily inside the method that
# needs it.
# ==============================================================================

from __future__ import annotations

import datetime
import hashlib
import json
import os
import threading
import uuid
import warnings
from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np

# The vacua layer depends on the pure-I/O layer for constants, exceptions and
# helpers (all of which live in stringforge.cy_io after the 2026-04-30 split).
# We re-import the ones the method bodies reference.
from .cy_io import (
    SCHEMA_VERSION,
    SchemaVersionError,
    ValidationError,
    _decode_array,
    _require_pandas,
    _require_pyarrow,
    _require_hf_hub,
    _require_hf_api,
    _safe_concat,
    _resolve_vault_repo,
    _resolve_vault_dir,
    VAULT_DIRNAME,
    DEFAULT_VAULT_REPO,
    HF_REPO_ID,
)


@contextmanager
def _file_lock(path: Path):
    r"""
    **Description:**
    Advisory exclusive lock on a sibling ``.lock`` file.  Used to
    serialise multi-process appends to the designated-vacua catalog so
    that concurrent ``designate_vacua`` calls do not allocate
    overlapping ``designated_id`` ranges.

    Falls back to a no-op on platforms without :mod:`fcntl` (Windows).
    """
    try:
        import fcntl
    except ImportError:
        yield
        return
    lock_path = path.with_suffix(path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(lock_path, "a+")
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        finally:
            fh.close()


def _compute_model_hash(tree: Any) -> str:
    r"""
    **Description:**
    Compute a deterministic SHA-256 hash of the geometric fingerprint of an
    ``lcs_tree``.  The fingerprint consists of ``(h11, h12, chi, intnums_coo,
    c2)`` — the minimal set of invariants that uniquely identify the geometry.

    Args:
        tree: An ``lcs_tree`` instance (or any object exposing ``h11``,
            ``h12``, ``chi``, ``intnums_coo``, ``c2`` attributes).

    Returns:
        str: Hex-encoded SHA-256 hash string.
    """
    h = hashlib.sha256()
    h.update(f"h11={tree.h11}".encode())
    h.update(f"h12={tree.h12}".encode())
    h.update(f"chi={tree.chi}".encode())
    if tree.intnums_coo is not None:
        h.update(np.asarray(tree.intnums_coo).tobytes())
    if tree.c2 is not None:
        h.update(np.asarray(tree.c2).tobytes())
    return h.hexdigest()


def _extract_model_identity(
    model: Any = None,
    h11: Optional[int] = None,
    h12: Optional[int] = None,
    ks_id: Optional[int] = None,
    triang_id: Optional[int] = None,
    conifold_id: int = -1,
    cicy_id: Optional[int] = None,
    model_name: Optional[str] = None,
) -> Dict[str, Any]:
    r"""
    **Description:**
    Extract or assemble model identity fields.  If *model* is provided
    (a ``FluxVacuaFinder``, ``FluxEFT``, or ``lcs_tree``), identity is read
    from the tree; explicit keyword arguments override.

    Returns:
        Dict with keys ``h11``, ``h12``, ``ks_id``, ``triang_id``,
        ``conifold_id``, ``cicy_id``, ``model_hash``, ``model_name``.
    """
    tree = None
    if model is not None:
        # Accept JAXVacua model objects (with .periods.lcs_tree) or lcs_tree.
        tree = getattr(getattr(model, "periods", None), "lcs_tree", model)

    _h11 = h11 if h11 is not None else (getattr(tree, "h11", None) if tree else None)
    _h12 = h12 if h12 is not None else (getattr(tree, "h12", None) if tree else None)

    # Try to get ks_id/triang_id from tree.extra_data
    _extra = getattr(tree, "extra_data", None) or {}
    _ks  = ks_id if ks_id is not None else _extra.get("ks_id", -1)
    _tr  = triang_id if triang_id is not None else _extra.get("triang_id", -1)
    _cid = cicy_id if cicy_id is not None else _extra.get("cicy_id", -1)

    # conifold_id: check tree.limit and conifold metadata
    _cf = conifold_id
    if _cf == -1 and tree is not None:
        limit = getattr(tree, "limit", None)
        if limit is not None and "coni" in str(limit).lower():
            _cf = _extra.get("conifold_id", 0)

    _hash = _compute_model_hash(tree) if tree is not None else ""
    _name = model_name

    # For local models loaded via model_ID, auto-generate a model_name
    # so they are human-readable in queries.
    if _name is None and _ks == -1 and tree is not None:
        _mid = getattr(tree, "model_ID", None)
        if _mid is not None:
            _name = f"local_h12_{_h12}_ID_{_mid}"

    if _h11 is None or _h12 is None:
        raise ValueError(
            "Cannot determine h11/h12. Pass them explicitly or provide a model."
        )

    return {
        "h11": int(_h11),
        "h12": int(_h12),
        "ks_id": int(_ks),
        "triang_id": int(_tr),
        "conifold_id": int(_cf),
        "cicy_id": int(_cid),
        "model_hash": _hash,
        "model_name": _name,
    }


# Catalog columns that are *run-level* and should not be smeared
# onto every loaded row.  ``n_vacua`` is a count of the run, not a
# per-row property; ``file_path`` is internal provenance.
_RUN_FIELDS_SKIP_ON_LOAD: frozenset[str] = frozenset({"n_vacua", "file_path"})


def _propagate_run_fields(df: Any, run_row: Any) -> Any:
    r"""
    **Description:**
    Copy the run-level fields from a vacua-catalog row onto every row
    of a freshly-loaded vacua DataFrame.  Lets callers do
    ``df.groupby("method")``, ``df.query("created > '2026-04-01'")``,
    or ``df["model_hash"]`` without re-joining against the catalog.

    Skips columns in :data:`_RUN_FIELDS_SKIP_ON_LOAD` (``n_vacua`` and
    ``file_path``) and leaves existing columns on ``df`` untouched.

    Args:
        df:        A pandas DataFrame as just loaded from a run's parquet.
        run_row:   The matching catalog row (a pandas Series).

    Returns:
        The same DataFrame (mutated in place); also returned for
        chaining.
    """
    for col in run_row.index:
        if col in _RUN_FIELDS_SKIP_ON_LOAD:
            continue
        if col not in df.columns:
            df[col] = run_row[col]
    return df


def _synthesize_complex_columns(df: Any) -> Any:
    r"""
    **Description:**
    Add convenience complex-valued ``moduli`` and ``tau`` columns to a
    freshly-loaded vacua DataFrame, derived from the stored
    ``moduli_re`` / ``moduli_im`` and ``tau_re`` / ``tau_im`` float
    columns.

    The on-disk parquet keeps real and imaginary parts as separate
    columns (parquet has no native complex dtype, and ``moduli`` is a
    variable-length vector per row).  Synthesising the complex form
    on read lets callers write the natural ``df["moduli"]`` /
    ``df["tau"]`` patterns.  Existing complex columns (e.g. on a
    re-load) are left untouched.

    Args:
        df: A pandas DataFrame as just loaded from a run's parquet.

    Returns:
        The same DataFrame (mutated in place); also returned for
        chaining.
    """
    if ("moduli" not in df.columns
            and "moduli_re" in df.columns and "moduli_im" in df.columns):
        df["moduli"] = [
            np.asarray(re, dtype=np.float64) + 1j * np.asarray(im, dtype=np.float64)
            for re, im in zip(df["moduli_re"], df["moduli_im"])
        ]
    if ("tau" not in df.columns
            and "tau_re" in df.columns and "tau_im" in df.columns):
        df["tau"] = df["tau_re"].astype(np.float64) + 1j * df["tau_im"].astype(np.float64)
    return df


def _vacua_row_from_dict(
    result: dict,
    vacuum_id: int,
    tags: Optional[Any] = None,
    extra_data: Optional[dict] = None,
) -> dict:
    r"""
    Normalise a single result dict (as returned by ``enumerate_fluxes`` or
    ``sample_bounded_fluxes``) into a flat row matching the vacua parquet
    schema.

    Args:
        result: Dict with keys ``"flux"``, ``"moduli"``, ``"tau"``, and
            optionally ``"residual"``, ``"is_susy"``, ``"N_flux"``, ``"W"``,
            ``"F_terms"``.
        vacuum_id: 0-based index within the run.
        tags: String or list of strings.
        extra_data: Optional dict of arbitrary metadata.

    Returns:
        dict: Flat row dict ready for DataFrame construction.
    """
    flux = np.asarray(result["flux"], dtype=np.int32)
    moduli = np.asarray(result["moduli"], dtype=np.complex128)
    tau = complex(result["tau"])

    # Normalise tags
    if tags is None:
        tags_json = "[]"
    elif isinstance(tags, str):
        tags_json = json.dumps([tags])
    else:
        tags_json = json.dumps(list(tags))

    row = {
        "vacuum_id": vacuum_id,
        "flux":      flux.tolist(),
        "moduli_re": moduli.real.tolist(),
        "moduli_im": moduli.imag.tolist(),
        "tau_re":    tau.real,
        "tau_im":    tau.imag,
        "W_re":      None,
        "W_im":      None,
        "F_terms_re": None,
        "F_terms_im": None,
        "N_flux":    result.get("N_flux"),
        "residual":  result.get("residual"),
        "is_susy":   result.get("is_susy", False),
        "tags":      tags_json,
        "extra_data": json.dumps(extra_data) if extra_data else None,
        "timestamp": datetime.datetime.now(datetime.timezone.utc),
    }

    # Optional pre-computed derived quantities
    W = result.get("W")
    if W is not None:
        W = complex(W)
        row["W_re"] = W.real
        row["W_im"] = W.imag

    F = result.get("F_terms")
    if F is not None:
        F = np.asarray(F, dtype=np.complex128)
        row["F_terms_re"] = F.real.tolist()
        row["F_terms_im"] = F.imag.tolist()

    return row


def _vacua_rows_from_batch(
    moduli: Any,
    tau: Any,
    fluxes: Any,
    tags: Optional[Any] = None,
    id_offset: int = 0,
    **extra_fields: Any,
) -> List[dict]:
    r"""
    **Description:**
    Convert batched arrays (as returned by ``initial_guesses_ISD``) into a
    list of row dicts matching the vacua parquet schema.

    Args:
        moduli: Array shape ``(N, h12)``, complex.
        tau: Array shape ``(N,)``, complex.
        fluxes: Array shape ``(N, 2*n_fluxes)``, int.
        tags: String or list of strings applied to all rows.
        id_offset: Starting vacuum_id.
        **extra_fields: Additional per-vacuum arrays of length N (e.g.
            ``residual``, ``N_flux``).

    Returns:
        List[dict]: One row dict per vacuum.
    """
    moduli_np = np.asarray(moduli, dtype=np.complex128)
    tau_np    = np.asarray(tau, dtype=np.complex128)
    fluxes_np = np.asarray(fluxes, dtype=np.int32)
    N = len(fluxes_np)

    # Normalise tags once
    if tags is None:
        tags_json = "[]"
    elif isinstance(tags, str):
        tags_json = json.dumps([tags])
    else:
        tags_json = json.dumps(list(tags))

    rows = []
    for i in range(N):
        row = {
            "vacuum_id": id_offset + i,
            "flux":      fluxes_np[i].tolist(),
            "moduli_re": moduli_np[i].real.tolist(),
            "moduli_im": moduli_np[i].imag.tolist(),
            "tau_re":    float(tau_np[i].real),
            "tau_im":    float(tau_np[i].imag),
            "W_re":      None,
            "W_im":      None,
            "F_terms_re": None,
            "F_terms_im": None,
            "N_flux":    float(extra_fields["N_flux"][i]) if "N_flux" in extra_fields else None,
            "residual":  float(extra_fields["residual"][i]) if "residual" in extra_fields else None,
            "is_susy":   bool(extra_fields["is_susy"][i]) if "is_susy" in extra_fields else False,
            "tags":      tags_json,
            "extra_data": None,
            "timestamp": datetime.datetime.now(datetime.timezone.utc),
        }
        rows.append(row)
    return rows


class _VacuaStreamWriter:
    r"""
    Context manager for buffered writing of vacuum solutions to local parquet
    storage.  Intended to be created via :meth:`CYDatabase.vacua_writer`.

    Usage::

        with db.vacua_writer(model=my_model) as writer:
            for result in results:
                writer.append(result, tags=["PFV"])
        # flushes to disk on __exit__

    Args:
        vacua_dir (Path): Root directory for vacua storage.
        identity (dict): Model identity fields (h11, h12, ks_id, ...).
        run_id (str): UUID of this run.
        method (str): Sampling method label.
        metadata (dict | None): Arbitrary run-level metadata (JSON-serialisable).
        model: Optional JAXVacua model for derived-quantity computation.
        compute_derived (bool): If True and *model* is set, compute W, DW,
            N_flux at flush time.
        filter_fn: Optional callable ``result_dict → bool``.
        deduplicate (bool): If True, skip flux vectors already stored for
            this model.
        map_to_fd (bool): If True and *model* has ``map_to_FD``, map each
            vacuum to the fundamental domain before storing and deduplicating.
    """

    def __init__(
        self,
        vacua_dir: Path,
        identity: Dict[str, Any],
        run_id: str,
        method: str = "manual",
        metadata: Optional[dict] = None,
        model: Any = None,
        compute_derived: bool = False,
        filter_fn: Any = None,
        deduplicate: bool = True,
        map_to_fd: bool = False,
    ) -> None:
        """Initialise the vacua writer for a given model identity and run."""
        self._vacua_dir = vacua_dir
        self._identity  = identity
        self._run_id    = run_id
        self._method    = method
        self._metadata  = metadata
        self._model     = model
        self._compute_derived = compute_derived
        self._filter_fn = filter_fn
        self._deduplicate = deduplicate
        self._map_to_fd = map_to_fd and model is not None and hasattr(model, 'map_to_FD')

        self._buffer: List[dict] = []
        self._flushed_count: int = 0
        self._n_duplicates_skipped: int = 0
        self._seen_fluxes: set = set()

        # Determine output path
        h11 = identity["h11"]
        h12 = identity["h12"]
        ks = identity["ks_id"]
        tr = identity["triang_id"]
        if ks >= 0 and tr >= 0:
            self._run_dir = (
                vacua_dir / f"h11_{h11}_h12_{h12}_ksid_{ks}" / f"triang_{tr}"
            )
        else:
            mhash = identity["model_hash"]
            self._run_dir = vacua_dir / "custom" / mhash
        self._run_path = self._run_dir / f"run_{self._run_id}.parquet"

    def __enter__(self) -> "_VacuaStreamWriter":
        """Enter context manager, loading existing fluxes if deduplicating."""
        if self._deduplicate:
            self._load_existing_fluxes()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context manager, flushing buffered rows and updating the catalog."""
        if self._buffer:
            self.flush()
        # Update catalog regardless (even if 0 new vacua, to record the run)
        if self._flushed_count > 0:
            self._update_catalog()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def append(
        self,
        result: Any,
        tags: Optional[Any] = None,
        extra_data: Optional[dict] = None,
    ) -> None:
        r"""
        **Description:**
    Append a single vacuum solution.

        Args:
            result: Either a dict ``{"flux", "moduli", "tau", ...}`` (as
                returned by ``enumerate_fluxes``) or a tuple
                ``(moduli, tau, flux)``.
            tags: String or list of tag strings.
            extra_data: Optional dict of arbitrary per-vacuum metadata.
        """
        if isinstance(result, (tuple, list)) and not isinstance(result, dict):
            moduli, tau, flux = result
            result = {"moduli": moduli, "tau": tau, "flux": flux}

        # Map to fundamental domain (monodromy + SL(2,Z))
        if self._map_to_fd:
            import jax.numpy as jnp
            moduli_fd, tau_fd, fluxes_fd = self._model.map_to_FD(
                jnp.asarray(result["moduli"]),
                complex(result["tau"]),
                jnp.asarray(result["flux"]),
            )
            result = {**result, "moduli": moduli_fd, "tau": tau_fd, "flux": fluxes_fd}

        # Filter
        if self._filter_fn is not None and not self._filter_fn(result):
            return

        # Dedup
        if self._deduplicate:
            flux_key = tuple(int(x) for x in np.asarray(result["flux"]))
            if flux_key in self._seen_fluxes:
                self._n_duplicates_skipped += 1
                return
            self._seen_fluxes.add(flux_key)

        vid = self._flushed_count + len(self._buffer)
        self._buffer.append(_vacua_row_from_dict(result, vid, tags, extra_data))

    def append_batch(
        self,
        moduli: Any,
        tau: Any,
        fluxes: Any,
        tags: Optional[Any] = None,
        **extra_fields: Any,
    ) -> None:
        r"""
        **Description:**
    Append a batch of vacuum solutions from array data.

        Args:
            moduli: Array shape ``(N, h12)``, complex.
            tau: Array shape ``(N,)``, complex.
            fluxes: Array shape ``(N, 2*n_fluxes)``, int.
            tags: String or list applied to all rows.
            **extra_fields: Per-vacuum arrays of length N (``residual``,
                ``N_flux``, ``is_susy``).
        """
        # Strip any zero-magnitude imaginary part on the input fluxes
        # (JAX → numpy roundoff on the upstream solver) before casting
        # to int32; assert it's negligible so we never silently round
        # away a meaningful complex value.
        fluxes_arr = np.asarray(fluxes)
        if np.iscomplexobj(fluxes_arr):
            max_im = float(np.max(np.abs(fluxes_arr.imag)))
            assert max_im < 1e-9, (
                f"fluxes have non-negligible imaginary part "
                f"(max |imag|={max_im:.3e}) — should be a real integer "
                f"array."
            )
            fluxes_arr = fluxes_arr.real
        fluxes_np = np.asarray(fluxes_arr, dtype=np.int32)
        moduli_np = np.asarray(moduli, dtype=np.complex128)
        tau_np = np.asarray(tau, dtype=np.complex128)
        N = len(fluxes_np)

        # Map to fundamental domain (monodromy + SL(2,Z))
        if self._map_to_fd:
            import jax.numpy as jnp
            for i in range(N):
                m_fd, t_fd, f_fd = self._model.map_to_FD(
                    jnp.asarray(moduli_np[i]),
                    complex(tau_np[i]),
                    jnp.asarray(fluxes_np[i]),
                )
                moduli_np[i] = np.asarray(m_fd)
                tau_np[i] = complex(t_fd)
                fluxes_np[i] = np.asarray(f_fd, dtype=np.int32)

        # Build keep mask for dedup + filter
        keep = np.ones(N, dtype=bool)
        if self._deduplicate:
            for i in range(N):
                flux_key = tuple(int(x) for x in fluxes_np[i])
                if flux_key in self._seen_fluxes:
                    keep[i] = False
                    self._n_duplicates_skipped += 1
                else:
                    self._seen_fluxes.add(flux_key)

        if self._filter_fn is not None:
            for i in range(N):
                if keep[i]:
                    r = {"flux": fluxes_np[i], "moduli": moduli_np[i], "tau": tau_np[i]}
                    if not self._filter_fn(r):
                        keep[i] = False

        # Subset arrays
        idx = np.where(keep)[0]
        if len(idx) == 0:
            return

        sub_moduli = moduli_np[idx]
        sub_tau    = tau_np[idx]
        sub_fluxes = fluxes_np[idx]
        sub_extra  = {
            k: np.asarray(v)[idx]
            for k, v in extra_fields.items()
        }

        vid_offset = self._flushed_count + len(self._buffer)
        rows = _vacua_rows_from_batch(
            sub_moduli, sub_tau, sub_fluxes,
            tags=tags, id_offset=vid_offset, **sub_extra,
        )
        self._buffer.extend(rows)

    def flush(self) -> None:
        r"""Write buffered rows to the run parquet file."""
        if not self._buffer:
            return

        pd = _require_pandas()
        df = pd.DataFrame(self._buffer)

        # Optional derived-quantity computation
        if self._compute_derived and self._model is not None:
            df = self._compute_derived_quantities(df)

        # Write (append to existing file if present from prior flush)
        self._run_dir.mkdir(parents=True, exist_ok=True)
        if self._run_path.exists():
            existing = pd.read_parquet(self._run_path)
            df = _safe_concat([existing, df], ignore_index=True)
        df.to_parquet(self._run_path, index=False)

        self._flushed_count += len(self._buffer)
        self._buffer.clear()

    @property
    def count(self) -> int:
        r"""Total number of vacua accepted (buffered + flushed)."""
        return self._flushed_count + len(self._buffer)

    @property
    def n_duplicates_skipped(self) -> int:
        r"""Number of duplicate flux vectors skipped."""
        return self._n_duplicates_skipped

    @property
    def run_id(self) -> str:
        r"""UUID of this run (matches the ``run_id`` column on every
        appended row).  Use this to query back the written rows via
        :meth:`VacuaWriter.query_vacua(run_id=writer.run_id)`."""
        return self._run_id

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_existing_fluxes(self) -> None:
        r"""Load all flux vectors from prior runs for this model into the
        dedup set."""
        pd = _require_pandas()
        if not self._run_dir.exists():
            return
        for f in self._run_dir.glob("run_*.parquet"):
            try:
                df = pd.read_parquet(f, columns=["flux"])
                for flux_list in df["flux"]:
                    self._seen_fluxes.add(tuple(int(x) for x in flux_list))
            except Exception:
                pass  # skip corrupt files

    def _update_catalog(self) -> None:
        r"""Append this run to the vacua catalog."""
        pd = _require_pandas()
        catalog_path = self._vacua_dir / "vacua_catalog.parquet"

        new_row = {
            "run_id":       self._run_id,
            "h11":          self._identity["h11"],
            "h12":          self._identity["h12"],
            "ks_id":        self._identity["ks_id"],
            "triang_id":    self._identity["triang_id"],
            "conifold_id":  self._identity["conifold_id"],
            "cicy_id":      self._identity["cicy_id"],
            "model_hash":   self._identity["model_hash"],
            "model_name":   self._identity["model_name"],
            "n_vacua":      self._flushed_count,
            "method":       self._method,
            "created":      datetime.datetime.now(datetime.timezone.utc),
            "metadata":     json.dumps(self._metadata) if self._metadata else None,
            "file_path":    str(self._run_path.relative_to(self._vacua_dir)),
        }

        self._vacua_dir.mkdir(parents=True, exist_ok=True)
        if catalog_path.exists():
            existing = pd.read_parquet(catalog_path)
            catalog = _safe_concat([existing, pd.DataFrame([new_row])], ignore_index=True)
        else:
            catalog = pd.DataFrame([new_row])
        catalog.to_parquet(catalog_path, index=False)

    def _compute_derived_quantities(self, df: Any) -> Any:
        r"""Compute W, DW, N_flux for all rows in df using self._model."""
        model = self._model
        rows_updated = []
        for _, row in df.iterrows():
            row = dict(row)
            flux = np.array(row["flux"], dtype=np.int32)
            moduli = np.array(row["moduli_re"]) + 1j * np.array(row["moduli_im"])
            tau = row["tau_re"] + 1j * row["tau_im"]

            try:
                # Tadpole
                if row.get("N_flux") is None:
                    row["N_flux"] = float(model.tadpole(flux))

                # Superpotential
                W = complex(model.superpotential(moduli, tau, flux))
                row["W_re"] = W.real
                row["W_im"] = W.imag

                # F-terms
                moduli_c = np.conj(moduli)
                tau_c = np.conj(tau)
                DW = np.asarray(model.DW(moduli, moduli_c, tau, tau_c, flux),
                                dtype=np.complex128)
                row["F_terms_re"] = DW.real.tolist()
                row["F_terms_im"] = DW.imag.tolist()

                # is_susy: check if all F-terms are small
                if not row.get("is_susy"):
                    row["is_susy"] = bool(np.max(np.abs(DW)) < 1e-6)
            except Exception:
                pass  # leave derived fields as None if computation fails

            rows_updated.append(row)

        pd = _require_pandas()
        return pd.DataFrame(rows_updated)



class VacuaWriter:
    """Vacua persistence / HuggingFace-upload handler.

    Wraps a :class:`stringforge.cy_io.CYDatabase` (or any subclass) and delegates
    all data-access calls — catalog reads, shard fetches, path resolution — to
    the wrapped instance via :meth:`__getattr__`.  This keeps the extracted
    method bodies untouched from their original `CYDatabase` form.

    Construct with ``VacuaWriter(db)`` where ``db`` is an instance of
    :class:`stringforge.cy_io.CYDatabase` (or any class implementing the same
    I/O surface — e.g. :class:`jaxvacua.lcs_database.LCSDatabase`).
    """

    # Attribute names that live on the VacuaWriter itself rather than being
    # forwarded to the wrapped database.
    _VW_OWN_ATTRS = frozenset({"db", "_vault_dir_override"})

    def __init__(self, db, *, vault_dir: Optional[Path] = None):
        object.__setattr__(self, "db", db)
        object.__setattr__(self, "_vault_dir_override", Path(vault_dir) if vault_dir is not None else None)

    # ----- Attribute proxy -------------------------------------------------
    # Method bodies were copied verbatim from CYDatabase, so they use `self.X`
    # for attributes that actually belong to the wrapped database (the shard
    # cache, schema state, per-model vault subdir, etc.).  Both __getattr__
    # *and* __setattr__ forward to self.db so reads and writes land on the
    # same object — otherwise state set on the VacuaWriter would shadow but
    # never update the db, and methods like _ensure_designated_catalog(self)
    # (which dispatches on db) would never see the update.
    def __getattr__(self, name: str):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return getattr(self.db, name)

    def __setattr__(self, name: str, value) -> None:
        if name in self._VW_OWN_ATTRS:
            object.__setattr__(self, name, value)
        else:
            setattr(self.db, name, value)

    # ------------------------------------------------------------------
    # Vault-path resolution.  Moved here from CYDatabase because it depends
    # on `_extract_model_identity`, which reads `lcs_tree` attributes and
    # therefore cannot live in the pure-I/O layer.
    # ------------------------------------------------------------------
    def _resolve_vacua_dir(self, model: Any = None, **identity_kwargs: Any) -> Path:
        r"""Return the per-model subdirectory inside the vault for designated
        vacua belonging to *model*.

        Path layout:

        - Local KS models (bundled; loaded via ``(h12, model_ID)``) →
          ``<vault>/KS/h12_{h12}_model_{model_ID}/``.
        - HF-downloaded TDF models →
          ``<vault>/tdf/h11_{h11}_h12_{h12}/ks_{ks_id}_tri_{triang_id}/``.
        - HF-downloaded CICY models → ``<vault>/cicy/cicy_{cicy_id}/``.
        - Fallback (custom model) → ``<vault>/custom/{model_hash}/``.
        """
        vault = _resolve_vault_dir()
        ident = _extract_model_identity(model=model, **identity_kwargs)

        ks = ident.get("ks_id", -1)
        tr = ident.get("triang_id", -1)
        cicy_id = ident.get("cicy_id", -1)
        h11    = ident.get("h11")
        h12    = ident.get("h12")
        name   = ident.get("model_name") or ""
        mhash  = ident.get("model_hash") or ""

        if ks >= 0 and tr >= 0:
            return (
                vault
                / "tdf"
                / f"h11_{h11}_h12_{h12}"
                / f"ks_{ks}_tri_{tr}"
            )
        if cicy_id >= 0:
            return vault / "cicy" / f"cicy_{cicy_id}"

        model_ID = None
        if name.startswith("local_h12_") and "_ID_" in name:
            try:
                model_ID = int(name.split("_ID_")[-1])
            except ValueError:
                model_ID = None

        if h12 is not None and model_ID is not None:
            return vault / "KS" / f"h12_{h12}_model_{model_ID}"

        return vault / "custom" / (mhash or "unknown")

    def vacua_writer(
        self,
        model: Any = None,
        h11: Optional[int] = None,
        h12: Optional[int] = None,
        ks_id: Optional[int] = None,
        triang_id: Optional[int] = None,
        conifold_id: int = -1,
        cicy_id: Optional[int] = None,
        model_name: Optional[str] = None,
        run_id: Optional[str] = None,
        method: str = "manual",
        compute_derived: bool = False,
        filter_fn: Any = None,
        deduplicate: bool = True,
        map_to_fd: bool = False,
        metadata: Optional[dict] = None,
        tags: Optional[list] = None,
    ) -> "_VacuaStreamWriter":
        r"""
        **Description:**
        Return a context manager for writing vacuum solutions to local
        parquet storage.

        Identity fields (``h11``, ``h12``, ``ks_id``, ``triang_id``,
        ``conifold_id``) are extracted from *model* when available; explicit
        keyword arguments override.  For custom models (no database entry),
        set ``model_name`` for human-readable identification.

        Args:
            model: A JAXVacua model or ``lcs_tree`` instance.  Used both
                for identity extraction and (if *compute_derived* is True)
                for computing W, DW, N_flux.
            h11 (int | None): Override :math:`h^{1,1}`.
            h12 (int | None): Override :math:`h^{1,2}`.
            ks_id (int | None): Override Kreuzer-Skarke polytope index.
            triang_id (int | None): Override triangulation index.
            conifold_id (int): Conifold index (-1 for LCS).
            cicy_id (int | None): Override CICY index.
            model_name (str | None): Human-readable label for custom models.
            run_id (str | None): UUID for this run.  Auto-generated if None.
            method (str): Sampling method label.  Defaults to ``"manual"``.
            compute_derived (bool): Compute W, DW, N_flux at flush time.
            filter_fn: Optional callable ``result_dict → bool``.
            deduplicate (bool): Skip duplicate flux vectors.  Defaults to
                ``True``.
            map_to_fd (bool): If True and *model* has ``map_to_FD``, map each
                vacuum to the fundamental domain (SL(2,Z) for tau, monodromy
                for z) before storing and deduplicating.  Defaults to ``False``.
            metadata (dict | None): Arbitrary JSON-serialisable run metadata.
            tags (list | None): Searchable tags folded into *metadata* under
                the ``"tags"`` key.  Accepted so callers that pass ``tags=``
                (e.g. :func:`jaxvacua.flux_bounding.bounded_fluxes.merge_cluster_results`)
                don't raise ``TypeError``.

        Returns:
            _VacuaStreamWriter: A context manager.  Use with ``with`` statement.

        Example::

            with db.vacua_writer(model=my_model, method="enumerate") as w:
                for result in results:
                    w.append(result, tags=["PFV"])
            print(f"Stored {w.count} vacua, skipped {w.n_duplicates_skipped}")
        """
        self._check_cache_size()
        identity = _extract_model_identity(
            model=model, h11=h11, h12=h12, ks_id=ks_id,
            triang_id=triang_id, conifold_id=conifold_id,
            cicy_id=cicy_id, model_name=model_name,
        )
        _run_id = run_id or str(uuid.uuid4())

        # Fold `tags` into metadata so we don't need to thread an extra
        # kwarg through _VacuaStreamWriter.
        if tags is not None:
            metadata = dict(metadata) if metadata else {}
            metadata.setdefault("tags", list(tags))

        return _VacuaStreamWriter(
            vacua_dir=self._vacua_dir,
            identity=identity,
            run_id=_run_id,
            method=method,
            metadata=metadata,
            model=model,
            compute_derived=compute_derived,
            filter_fn=filter_fn,
            deduplicate=deduplicate,
            map_to_fd=map_to_fd,
        )
    def query_vacua(
        self,
        h11: Optional[int] = None,
        h12: Optional[int] = None,
        ks_id: Optional[int] = None,
        triang_id: Optional[int] = None,
        conifold_id: Optional[int] = None,
        cicy_id: Optional[int] = None,
        model_hash: Optional[str] = None,
        run_id: Optional[str] = None,
        is_susy: Optional[bool] = None,
        N_flux_max: Optional[float] = None,
        tags: Optional[Any] = None,
        method: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> Any:
        r"""
        **Description:**
        Query stored vacua and return matching rows as a ``pandas.DataFrame``.

        Filtering proceeds in two stages:

        1. **Catalog stage** — filter ``vacua_catalog.parquet`` by model
           identity and run-level fields (fast, in-memory).
        2. **Data stage** — for matching runs, load parquet files and apply
           row-level filters (``is_susy``, ``N_flux_max``, ``tags``).

        Args:
            h11, h12, ks_id, triang_id, conifold_id, cicy_id: Model identity
                filters (exact match).
            model_hash (str | None): Filter by geometric fingerprint hash.
            run_id (str | None): Filter by run UUID.
            is_susy (bool | None): Filter by SUSY flag.
            N_flux_max (float | None): Maximum tadpole value.
            tags: String or list of strings.  Rows must contain **all**
                specified tags.
            method (str | None): Filter by sampling method.
            limit (int | None): Maximum number of rows to return.

        Returns:
            pandas.DataFrame: Matching vacuum rows with ``run_id`` column
            injected.
        """
        pd = _require_pandas()
        self._ensure_vacua_catalog()
        cat = self._vacua_catalog
        if cat is None or len(cat) == 0:
            return pd.DataFrame()

        # Stage 1: catalog-level filters
        mask = np.ones(len(cat), dtype=bool)
        if h11 is not None:
            mask &= (cat["h11"] == h11).values
        if h12 is not None:
            mask &= (cat["h12"] == h12).values
        if ks_id is not None:
            mask &= (cat["ks_id"] == ks_id).values
        if triang_id is not None:
            mask &= (cat["triang_id"] == triang_id).values
        if conifold_id is not None:
            mask &= (cat["conifold_id"] == conifold_id).values
        if cicy_id is not None:
            mask &= (cat["cicy_id"] == cicy_id).values
        if model_hash is not None:
            mask &= (cat["model_hash"] == model_hash).values
        if run_id is not None:
            mask &= (cat["run_id"] == run_id).values
        if method is not None:
            mask &= (cat["method"] == method).values

        matching = cat[mask]
        if len(matching) == 0:
            return pd.DataFrame()

        # Stage 2: load and filter data files
        frames = []
        for _, run_row in matching.iterrows():
            path = self._vacua_dir / run_row["file_path"]
            if not path.exists():
                continue
            df = pd.read_parquet(path)
            # Propagate run-level catalog fields onto each row, then
            # synthesise convenience complex columns.
            df = _propagate_run_fields(df, run_row)
            df = _synthesize_complex_columns(df)

            if is_susy is not None and "is_susy" in df.columns:
                df = df[df["is_susy"] == is_susy]
            if N_flux_max is not None and "N_flux" in df.columns:
                df = df[df["N_flux"].notna() & (df["N_flux"] <= N_flux_max)]
            if tags is not None:
                if isinstance(tags, str):
                    tags = [tags]
                def _has_all_tags(tags_json, required=tags):
                    """Check whether tags_json contains all required tags."""
                    try:
                        stored = json.loads(tags_json) if isinstance(tags_json, str) else []
                    except (json.JSONDecodeError, TypeError):
                        stored = []
                    return all(t in stored for t in required)
                if "tags" in df.columns:
                    df = df[df["tags"].apply(_has_all_tags)]

            frames.append(df)

        if not frames:
            return pd.DataFrame()
        result = _safe_concat(frames, ignore_index=True)
        if limit is not None:
            result = result.head(limit)
        return result
    def load_vacua(self, run_id: str) -> Any:
        r"""
        **Description:**
        Load all vacuum solutions from a specific run.

        Args:
            run_id (str): UUID of the run.

        Returns:
            pandas.DataFrame: All rows from the run's parquet file.

        Raises:
            KeyError: If the run is not found in the catalog.
        """
        pd = _require_pandas()
        self._ensure_vacua_catalog()
        cat = self._vacua_catalog
        if cat is None or len(cat) == 0:
            raise KeyError(f"Run '{run_id}' not found — vacua catalog is empty.")
        matches = cat[cat["run_id"] == run_id]
        if len(matches) == 0:
            raise KeyError(f"Run '{run_id}' not found in vacua catalog.")
        row = matches.iloc[0]
        path = self._vacua_dir / row["file_path"]
        if not path.exists():
            raise FileNotFoundError(f"Run file not found: {path}")
        df = pd.read_parquet(path)
        df = _propagate_run_fields(df, row)
        df = _synthesize_complex_columns(df)
        return df
    def solution_exists(
        self,
        flux: Any,
        h11: Optional[int] = None,
        h12: Optional[int] = None,
        ks_id: Optional[int] = None,
        triang_id: Optional[int] = None,
        model: Any = None,
    ) -> bool:
        r"""
        **Description:**
        Check whether a vacuum with the given flux vector is already stored
        for this model.  Uses exact integer comparison.

        Args:
            flux: Integer array of length ``2 * n_fluxes``.
            h11, h12, ks_id, triang_id: Model identity (used to filter runs).
            model: Alternative — extract identity from a JAXVacua model or
                ``lcs_tree``.

        Returns:
            bool: True if the flux vector is found in any stored run.
        """
        pd = _require_pandas()
        identity = _extract_model_identity(
            model=model, h11=h11, h12=h12, ks_id=ks_id, triang_id=triang_id,
        )
        flux_key = tuple(int(x) for x in np.asarray(flux))

        self._ensure_vacua_catalog()
        cat = self._vacua_catalog
        if cat is None or len(cat) == 0:
            return False

        # Filter catalog to this model
        mask = np.ones(len(cat), dtype=bool)
        if identity["ks_id"] >= 0:
            mask &= (cat["ks_id"] == identity["ks_id"]).values
            mask &= (cat["triang_id"] == identity["triang_id"]).values
            if "h11" in cat.columns:
                mask &= (cat["h11"] == identity["h11"]).values
        elif identity["model_hash"]:
            mask &= (cat["model_hash"] == identity["model_hash"]).values

        for _, run_row in cat[mask].iterrows():
            path = self._vacua_dir / run_row["file_path"]
            if not path.exists():
                continue
            df = pd.read_parquet(path, columns=["flux"])
            for stored_flux in df["flux"]:
                if tuple(int(x) for x in stored_flux) == flux_key:
                    return True
        return False
    def find_similar_vacua(
        self,
        moduli: Any = None,
        tau: Any = None,
        flux: Any = None,
        h11: Optional[int] = None,
        h12: Optional[int] = None,
        ks_id: Optional[int] = None,
        triang_id: Optional[int] = None,
        model: Any = None,
        flux_tol: int = 0,
        moduli_tol: float = 1e-6,
        tau_tol: float = 1e-6,
    ) -> Any:
        r"""
        **Description:**
        Find stored vacua that are similar to the given point within
        specified tolerances.

        Args:
            moduli: Complex array of length h12 (optional).
            tau: Complex scalar (optional).
            flux: Integer array (optional).  Compared with L∞ tolerance.
            h11, h12, ks_id, triang_id: Model identity filters.
            model: Alternative — extract identity from a model object.
            flux_tol (int): L∞ tolerance on flux vector (0 = exact match).
            moduli_tol (float): Relative tolerance on moduli.
            tau_tol (float): Relative tolerance on tau.

        Returns:
            pandas.DataFrame: Matching vacuum rows.
        """
        pd = _require_pandas()
        identity = _extract_model_identity(
            model=model, h11=h11, h12=h12, ks_id=ks_id, triang_id=triang_id,
        )

        self._ensure_vacua_catalog()
        cat = self._vacua_catalog
        if cat is None or len(cat) == 0:
            return pd.DataFrame()

        # Filter catalog to this model
        mask = np.ones(len(cat), dtype=bool)
        if identity["ks_id"] >= 0:
            mask &= (cat["ks_id"] == identity["ks_id"]).values
            mask &= (cat["triang_id"] == identity["triang_id"]).values
            if "h11" in cat.columns:
                mask &= (cat["h11"] == identity["h11"]).values
        elif identity["model_hash"]:
            mask &= (cat["model_hash"] == identity["model_hash"]).values

        matching_runs = cat[mask]
        if len(matching_runs) == 0:
            return pd.DataFrame()

        # Prepare query values
        flux_q = np.asarray(flux, dtype=np.int32) if flux is not None else None
        moduli_q = np.asarray(moduli, dtype=np.complex128) if moduli is not None else None
        tau_q = complex(tau) if tau is not None else None

        frames = []
        for _, run_row in matching_runs.iterrows():
            path = self._vacua_dir / run_row["file_path"]
            if not path.exists():
                continue
            df = pd.read_parquet(path)
            keep = np.ones(len(df), dtype=bool)

            if flux_q is not None and "flux" in df.columns:
                for i, stored in enumerate(df["flux"]):
                    stored_arr = np.array(stored, dtype=np.int32)
                    if np.max(np.abs(stored_arr - flux_q)) > flux_tol:
                        keep[i] = False

            if moduli_q is not None and "moduli_re" in df.columns:
                for i in range(len(df)):
                    if not keep[i]:
                        continue
                    stored_z = (
                        np.array(df.iloc[i]["moduli_re"])
                        + 1j * np.array(df.iloc[i]["moduli_im"])
                    )
                    denom = np.abs(stored_z)
                    denom = np.where(denom > 0, denom, 1.0)
                    if np.max(np.abs(stored_z - moduli_q) / denom) > moduli_tol:
                        keep[i] = False

            if tau_q is not None and "tau_re" in df.columns:
                for i in range(len(df)):
                    if not keep[i]:
                        continue
                    stored_tau = df.iloc[i]["tau_re"] + 1j * df.iloc[i]["tau_im"]
                    denom = abs(stored_tau) if abs(stored_tau) > 0 else 1.0
                    if abs(stored_tau - tau_q) / denom > tau_tol:
                        keep[i] = False

            df = df[keep]
            if len(df) > 0:
                df = _propagate_run_fields(df, run_row)
                df = _synthesize_complex_columns(df)
                frames.append(df)

        if not frames:
            return pd.DataFrame()
        return _safe_concat(frames, ignore_index=True)
    def vacua_info(self) -> None:
        r"""
        **Description:**
        Print a summary of stored vacua.
        """
        self._ensure_vacua_catalog()
        cat = self._vacua_catalog
        if cat is None or len(cat) == 0:
            print("No vacua stored.")
            return

        n_runs   = len(cat)
        n_vacua  = int(cat["n_vacua"].sum())
        n_models = len(cat.groupby(["ks_id", "triang_id", "model_hash"]))
        methods  = cat["method"].value_counts().to_dict()

        print(f"Vacua storage: {n_vacua} vacua across {n_runs} run(s), "
              f"{n_models} distinct model(s)")
        if methods:
            print(f"  Methods: {methods}")

        # Show per-model summary
        groups = cat.groupby(["h11", "ks_id", "triang_id"]).agg(
            n_runs=("run_id", "count"),
            n_vacua=("n_vacua", "sum"),
        ).reset_index()
        if len(groups) <= 20:
            print(f"\n{'h11':>4} {'ks_id':>6} {'trid':>5} {'runs':>5} {'vacua':>7}")
            for _, g in groups.iterrows():
                print(f"{g['h11']:4d} {g['ks_id']:6d} {g['triang_id']:5d} "
                      f"{g['n_runs']:5d} {g['n_vacua']:7d}")
        else:
            print(f"  (showing {len(groups)} models — use query_vacua() for details)")
    def delete_vacua(
        self,
        run_id: Optional[str] = None,
        ks_id: Optional[int] = None,
        triang_id: Optional[int] = None,
    ) -> int:
        r"""
        **Description:**
        Delete stored vacua matching the given filter.

        Args:
            run_id (str | None): Delete a specific run.
            ks_id (int | None): Delete all runs for this model.
            triang_id (int | None): Narrow by triangulation (with ks_id).

        Returns:
            int: Number of runs deleted.
        """
        pd = _require_pandas()
        self._ensure_vacua_catalog()
        cat = self._vacua_catalog
        if cat is None or len(cat) == 0:
            return 0

        mask = np.ones(len(cat), dtype=bool)
        if run_id is not None:
            mask &= (cat["run_id"] == run_id).values
        if ks_id is not None:
            mask &= (cat["ks_id"] == ks_id).values
        if triang_id is not None:
            mask &= (cat["triang_id"] == triang_id).values

        to_delete = cat[mask]
        n_deleted = 0
        for _, row in to_delete.iterrows():
            path = self._vacua_dir / row["file_path"]
            if path.exists():
                path.unlink()
            n_deleted += 1

        # Update catalog
        self._vacua_catalog = cat[~mask].reset_index(drop=True)
        catalog_path = self._vacua_dir / "vacua_catalog.parquet"
        if len(self._vacua_catalog) > 0:
            self._vacua_catalog.to_parquet(catalog_path, index=False)
        elif catalog_path.exists():
            catalog_path.unlink()

        return n_deleted
    def validate_vacua(
        self,
        vacua_df: Any,
        model: Any = None,
        F_term_tol: float = 1e-8,
        check_tadpole: bool = True,
        check_physics: bool = True,
        check_duplicates: bool = True,
        identity: Optional[Dict[str, Any]] = None,
    ) -> List[dict]:
        r"""
        **Description:**
        Validate vacuum solutions before designation.  Each solution is
        checked for schema conformance, deduplication against already
        designated solutions, and (optionally) physics consistency.

        Can be called standalone or is invoked automatically inside
        :meth:`designate_vacua`.

        Args:
            vacua_df: ``pandas.DataFrame`` of vacuum rows (as returned by
                :meth:`query_vacua`).
            model: Optional JAXVacua model for physics re-verification.
                When provided, F-terms and tadpole are recomputed from the
                stored flux and moduli vectors.
            F_term_tol (float): Maximum allowed ``|F_i|`` for a solution to
                pass the physics check.  Defaults to ``1e-8``.
            check_tadpole (bool): Verify ``N_flux ≤ D3_tadpole``.
            check_physics (bool): Recompute F-terms from stored data
                (requires *model*).
            check_duplicates (bool): Check flux vectors against existing
                designated solutions.
            identity (dict | None): Optional model identity used to scope
                duplicate checks.  When provided, a flux vector only counts
                as duplicate if it is already designated for the same model.

        Returns:
            List[dict]: One report entry per row with keys ``"index"``,
            ``"passed"`` (bool), and ``"errors"`` (list of strings).
        """
        pd = _require_pandas()
        report: List[dict] = []

        # Required columns
        required = {"flux", "moduli_re", "moduli_im", "tau_re", "tau_im"}

        # Load existing designated catalog for dedup
        existing_fluxes: set = set()
        if check_duplicates:
            self._ensure_designated_catalog()
            dcat = self._designated_catalog
            if (
                identity is not None
                and dcat is not None
                and len(dcat) > 0
            ):
                mask = np.ones(len(dcat), dtype=bool)
                for key in (
                    "h11", "h12", "ks_id", "triang_id",
                    "conifold_id", "cicy_id", "model_hash", "model_name",
                ):
                    if key not in dcat.columns or key not in identity:
                        continue
                    value = identity.get(key)
                    if value is None or value == "":
                        continue
                    mask &= (dcat[key] == value).values
                dcat = dcat[mask]
            if dcat is not None and len(dcat) > 0 and "flux" in dcat.columns:
                for fl in dcat["flux"]:
                    existing_fluxes.add(tuple(int(x) for x in fl))

        for idx in range(len(vacua_df)):
            row = vacua_df.iloc[idx]
            errors: List[str] = []

            # Schema check
            for col in required:
                if col not in vacua_df.columns:
                    errors.append(f"Missing required column: {col}")
                else:
                    val = row.get(col)
                    if val is None:
                        errors.append(f"Missing required column: {col}")
                    elif not hasattr(val, '__len__') and pd.isna(val):
                        errors.append(f"Missing required column: {col}")

            # Dedup check
            if check_duplicates and "flux" in vacua_df.columns:
                try:
                    flux_key = tuple(int(x) for x in row["flux"])
                    if flux_key in existing_fluxes:
                        errors.append("Flux vector already designated")
                except (TypeError, ValueError):
                    errors.append("Invalid flux vector format")

            # Physics check (requires model)
            if check_physics and model is not None and not errors:
                try:
                    flux = np.array(row["flux"], dtype=np.int32)
                    moduli = (np.array(row["moduli_re"])
                              + 1j * np.array(row["moduli_im"]))
                    tau = row["tau_re"] + 1j * row["tau_im"]
                    tau_c = np.conj(tau)
                    moduli_c = np.conj(moduli)

                    DW = np.asarray(
                        model.DW(moduli, moduli_c, tau, tau_c, flux),
                        dtype=np.complex128,
                    )
                    max_F = float(np.max(np.abs(DW)))
                    if max_F > F_term_tol:
                        errors.append(
                            f"F-term residual too large: max|F_i| = {max_F:.2e} "
                            f"> {F_term_tol:.0e}"
                        )
                except Exception as e:
                    errors.append(f"Physics check failed: {e}")

            # Tadpole check
            if check_tadpole and model is not None and not errors:
                try:
                    flux = np.array(row["flux"], dtype=np.int32)
                    N = float(model.tadpole(flux))
                    D3 = getattr(model, "D3_tadpole", None)
                    if D3 is not None and N > float(D3):
                        errors.append(
                            f"Tadpole violation: N_flux={N:.1f} > "
                            f"D3_tadpole={float(D3):.1f}"
                        )
                except Exception as e:
                    errors.append(f"Tadpole check failed: {e}")

            report.append({
                "index": idx,
                "passed": len(errors) == 0,
                "errors": errors,
            })

        return report
    def _validate_for_upload(
        self,
        vacua_df: Any,
        label: str,
        committed_by: str,
        model: Any = None,
        F_term_tol: float = 1e-6,
        check_remote: bool = True,
        repo_id: Optional[str] = None,
    ) -> dict:
        r"""
        **Description:**
        Upload-specific validation wrapper around :meth:`validate_vacua`.
        Runs the standard physics/dedup checks plus extras required for
        pushing to a HuggingFace dataset repo:

        - ``label`` matches the filesystem-safe slug regex.
        - ``committed_by`` is non-empty.
        - Schema version of this package matches the remote repo's
          ``schema.json`` (when *check_remote* is True and the package
          is online).
        - Optional remote duplicate check against the remote catalog.

        Args:
            vacua_df: Vacuum rows to upload.
            label (str): Dataset label (filesystem-safe).
            committed_by (str): Contributor identifier.
            model: Optional JAXVacua model for F-term re-verification.
            F_term_tol (float): F-term tolerance.  Defaults to ``1e-6``.
            check_remote (bool): Check schema + catalog on the remote
                repo.  Requires network; skipped automatically when
                ``self.offline`` is True.
            repo_id (str | None): Override target repo.  Defaults to the
                package-wide vault repo (see :func:`set_vault_repo`).

        Returns:
            dict: ``{"passed": bool, "errors": list, "warnings": list,
            "per_row_report": list}``.  Suitable for including in a PR
            summary comment.
        """
        import re
        errors: List[str] = []
        warnings_: List[str] = []

        # 1. Upload-specific bookkeeping
        if not label or not label.strip():
            errors.append("label must be non-empty")
        else:
            slug = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
            if not slug.match(label):
                errors.append(
                    f"label {label!r} is not a valid slug "
                    f"(must match {slug.pattern})"
                )
        if not committed_by or not committed_by.strip():
            errors.append("committed_by must be non-empty")

        # 2. Remote schema + catalogue checks
        if check_remote and not self.offline:
            try:
                repo = repo_id or _resolve_vault_repo()
                hf_download = _require_hf_hub()
                schema_path = hf_download(
                    repo_id=repo,
                    filename="schema.json",
                    repo_type="dataset",
                )
                with open(schema_path) as f:
                    remote_schema = json.load(f)
                remote_ver = int(remote_schema.get("schema_version", -1))
                if remote_ver != SCHEMA_VERSION:
                    errors.append(
                        f"Schema version mismatch: remote={remote_ver}, "
                        f"local={SCHEMA_VERSION}.  Upgrade your local "
                        f"`jaxvacua` package or coordinate with the repo "
                        f"maintainer."
                    )
            except Exception as e:
                # Remote schema may not exist yet (fresh repo) — warn only.
                warnings_.append(f"Could not fetch remote schema: {e}")

        # 3. Row-level validation (reuses existing validate_vacua)
        per_row = self.validate_vacua(
            vacua_df, model=model, F_term_tol=F_term_tol,
            check_tadpole=True,
            check_physics=(model is not None),
            check_duplicates=True,
        )
        n_failed = sum(1 for r in per_row if not r["passed"])
        if n_failed > 0:
            errors.append(
                f"{n_failed}/{len(per_row)} rows failed per-row validation"
            )

        return {
            "passed":          len(errors) == 0,
            "errors":          errors,
            "warnings":        warnings_,
            "per_row_report":  per_row,
        }
    def push_vacua_to_hub(
        self,
        vacua_df: Any,
        label: str,
        committed_by: str,
        model: Any,
        tags: Optional[list] = None,
        notes: str = "",
        repo_id: Optional[str] = None,
        create_pr: bool = True,
        partial_failure: str = "strict",
        if_exists: str = "error",
        check_remote: bool = True,
        token: Optional[str] = None,
        classification: Optional[dict] = None,
    ) -> dict:
        r"""
        **Description:**
        Upload a batch of vacuum solutions to the HuggingFace
        ``aschachner/vacua_vault`` dataset repository (or an override
        via *repo_id* / ``STRINGFORGE_VAULT_REPO``).

        The upload lands inside the target model's ``community/``
        directory: ``tdf/h12_{N}/ks_{X}_tri_{Y}/community/{hf_username}_{label}.parquet``
        (or the analogous CICY path).  Filesystem-free-form labels
        (see §5.6 of the vacua-storage plan) mean classification
        (``susy``, ``method``, ``nmax``, ``qualifiers``) is stored as
        parquet-level key-value metadata and extracted by the remote
        catalogue rebuild on merge.

        Args:
            vacua_df: Rows to upload.  Must include at least ``flux``,
                ``moduli_re``, ``moduli_im``, ``tau_re``, ``tau_im``.
            label (str): Filesystem-safe slug for the output file.
            committed_by (str): Contributor identifier (ORCID preferred;
                HF username accepted).
            model: Required JAXVacua model for identity extraction and
                physics validation.
            tags (list | None): Free-form tags attached to the dataset.
            notes (str): Free-text annotation.
            repo_id (str | None): Target dataset repo.  Defaults to
                :func:`_resolve_vault_repo`.
            create_pr (bool): If ``True`` (default), open a pull
                request rather than pushing directly to ``main``.
            partial_failure (str): ``"strict"`` (default, refuse the
                upload if any rows fail validation) or ``"split"``
                (valid rows go to the main file; failing rows go to
                ``_rejected/{filename}.rejected.parquet``).
            if_exists (str): Behaviour on filename collision in the
                target repo.  ``"error"`` (default), ``"append"``, or
                ``"new_version"`` (auto-suffix ``_v2``, ``_v3``, ...).
                ``"append"`` is only supported with
                ``create_pr=False`` because it requires a read-modify-
                write round-trip.
            check_remote (bool): Perform remote schema check during
                validation.  Requires network access.
            token (str | None): Optional HF access token.  Falls back
                to the token cached by ``huggingface-cli login`` and
                then to ``HF_TOKEN``.
            classification (dict | None): Optional parquet-level
                metadata to attach to the uploaded file.  Expected
                keys: ``susy`` (``"SUSY"`` | ``"nonSUSY"`` | ``"mixed"``),
                ``method`` (str), ``nmax`` (int), and ``qualifiers``
                (dict).  When omitted, sensible defaults are inferred.

        Returns:
            dict: ``{"pr_url": str | None, "commit_url": str,
            "file_path": str, "n_uploaded": int, "n_rejected": int,
            "report": ValidationReport}``.

        Raises:
            RuntimeError: If ``self.offline=True``.
            ValidationError: If validation fails and
                ``partial_failure="strict"``.
            FileExistsError: If the target filename exists and
                ``if_exists="error"``.
        """
        if self.offline:
            raise RuntimeError(
                "push_vacua_to_hub requires network access "
                "(self.offline=True)."
            )
        if partial_failure not in ("strict", "split"):
            raise ValueError(
                f"partial_failure must be 'strict' or 'split', "
                f"got {partial_failure!r}"
            )
        if if_exists not in ("error", "append", "new_version"):
            raise ValueError(
                f"if_exists must be 'error', 'append', or 'new_version', "
                f"got {if_exists!r}"
            )
        if if_exists == "append" and create_pr:
            raise ValueError(
                "if_exists='append' requires create_pr=False because it "
                "performs a read-modify-write upload."
            )

        pd = _require_pandas()
        pq = _require_pyarrow()
        try:
            import pyarrow as pa  # for Table-level metadata manipulation
        except ImportError as e:
            raise ImportError(
                "The 'pyarrow' package is required for push_vacua_to_hub."
            ) from e
        hf_api = _require_hf_api()
        hf_download = _require_hf_hub()
        repo = repo_id or _resolve_vault_repo()

        # 1. Client-side validation
        report = self._validate_for_upload(
            vacua_df, label=label, committed_by=committed_by,
            model=model, check_remote=check_remote, repo_id=repo,
        )

        # 2. Apply partial-failure policy
        n_total = len(vacua_df)
        passed_idx = [r["index"] for r in report["per_row_report"] if r["passed"]]
        failed_idx = [r["index"] for r in report["per_row_report"] if not r["passed"]]

        if failed_idx and partial_failure == "strict":
            msg = (
                f"{len(failed_idx)}/{n_total} rows failed validation. "
                f"Use partial_failure='split' to upload only the valid rows."
            )
            raise ValidationError(msg, report=report["per_row_report"])

        valid_df    = vacua_df.iloc[passed_idx].reset_index(drop=True)
        rejected_df = vacua_df.iloc[failed_idx].reset_index(drop=True) if failed_idx else None
        if rejected_df is not None and len(rejected_df) > 0:
            # Attach per-row error strings
            errs = []
            for r in report["per_row_report"]:
                if not r["passed"]:
                    errs.append("; ".join(r["errors"]))
            rejected_df = rejected_df.copy()
            rejected_df["error"] = errs

        # Abort if nothing to upload
        if len(valid_df) == 0:
            raise ValidationError(
                "No valid rows to upload.", report=report["per_row_report"],
            )

        # 3. Identify the authenticated HF user (for filename prefix)
        try:
            who = hf_api.whoami(token=token)
            hf_username = who.get("name") or who.get("username") or "anonymous"
        except Exception as e:
            raise RuntimeError(
                f"Could not identify HF user (check HF_TOKEN / huggingface-cli login): {e}"
            ) from e

        # 4. Resolve remote paths
        identity = _extract_model_identity(model=model)
        model_dir_remote = self._remote_model_dir(identity)
        if model_dir_remote is None:
            raise ValueError(
                "Could not derive a remote model directory from the "
                "given model.  Pass ks_id/triang_id or cicy_id "
                "explicitly."
            )

        base_name = f"{hf_username}_{label}"
        ext = ".parquet"
        target_rel = f"{model_dir_remote}/community/{base_name}{ext}"

        # 5. Handle filename collisions according to if_exists
        final_rel = self._resolve_remote_filename(
            hf_api, hf_download, repo, target_rel, if_exists=if_exists,
            token=token,
        )

        # 6. Write the valid rows (plus pyarrow metadata) to a scratch
        #    parquet file locally, then upload it.
        import tempfile, time
        cls_meta = classification or {}
        arrow_meta = {
            "schema_version": str(SCHEMA_VERSION),
            "susy":           str(cls_meta.get("susy", "unknown")),
            "method":         str(cls_meta.get("method", "unknown")),
            "nmax":           str(cls_meta.get("nmax", "")),
            "qualifiers":     json.dumps(cls_meta.get("qualifiers", {})),
            "label":          label,
            "committed_by":   committed_by,
            "notes":          notes or "",
            "uploaded_at":    datetime.datetime.now(
                                  datetime.timezone.utc).isoformat(),
        }
        if tags:
            arrow_meta["tags"] = json.dumps(list(tags))

        def _write_with_metadata(df, path, meta):
            table = pa.Table.from_pandas(df, preserve_index=False)
            existing = dict(table.schema.metadata or {})
            existing.update({k.encode(): v.encode() for k, v in meta.items()})
            table = table.replace_schema_metadata(existing)
            pq.write_table(table, path)

        result: Dict[str, Any] = {
            "pr_url":       None,
            "commit_url":   None,
            "file_path":    final_rel,
            "n_uploaded":   len(valid_df),
            "n_rejected":   len(rejected_df) if rejected_df is not None else 0,
            "report":       report,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            if if_exists == "append":
                try:
                    existing_path = hf_download(
                        repo_id=repo,
                        filename=final_rel,
                        repo_type="dataset",
                        local_dir=tmpdir,
                        token=token,
                    )
                    existing_df = pd.read_parquet(existing_path)
                except Exception:
                    existing_df = None
                if existing_df is not None and len(existing_df) > 0:
                    valid_df = _safe_concat(
                        [existing_df, valid_df], ignore_index=True,
                    )
                    result["n_uploaded"] = len(valid_df)

            # Valid file
            local_main = os.path.join(tmpdir, os.path.basename(final_rel))
            _write_with_metadata(valid_df, local_main, arrow_meta)

            # Optional rejected file
            local_rejected = None
            rejected_rel   = None
            if rejected_df is not None and len(rejected_df) > 0:
                rej_name = (
                    os.path.splitext(os.path.basename(final_rel))[0]
                    + ".rejected" + ext
                )
                rejected_rel = f"{model_dir_remote}/community/_rejected/{rej_name}"
                local_rejected = os.path.join(tmpdir, rej_name)
                rej_meta = dict(arrow_meta)
                rej_meta["status"] = "rejected"
                _write_with_metadata(rejected_df, local_rejected, rej_meta)

            # Upload
            commit_msg = f"add: {final_rel} (label={label}, by={committed_by})"
            operations = []
            operations.append(dict(
                local_path=local_main, path_in_repo=final_rel,
            ))
            if local_rejected is not None:
                operations.append(dict(
                    local_path=local_rejected, path_in_repo=rejected_rel,
                ))

            uploaded_paths = []
            for op in operations:
                ci = hf_api.upload_file(
                    path_or_fileobj=op["local_path"],
                    path_in_repo=op["path_in_repo"],
                    repo_id=repo,
                    repo_type="dataset",
                    commit_message=commit_msg,
                    token=token,
                    create_pr=create_pr,
                )
                uploaded_paths.append(op["path_in_repo"])
                # huggingface_hub returns a CommitInfo for direct uploads
                # or a PR-related response when create_pr=True.
                if create_pr and result["pr_url"] is None:
                    pr_url = getattr(ci, "pr_url", None) or getattr(ci, "url", None)
                    result["pr_url"] = pr_url
                if not create_pr and result["commit_url"] is None:
                    result["commit_url"] = getattr(ci, "commit_url", None) or getattr(ci, "oid", None)

            result["file_path"]    = uploaded_paths[0]
            result["rejected_path"] = uploaded_paths[1] if len(uploaded_paths) > 1 else None

        return result
    def fetch_vacua_from_hub(
        self,
        model: Any = None,
        ks_id: Optional[int] = None,
        triang_id: Optional[int] = None,
        cicy_id: Optional[int] = None,
        label: Optional[str] = None,
        include_community: bool = False,
        include_retracted: bool = False,
        cache: bool = True,
        repo_id: Optional[str] = None,
        token: Optional[str] = None,
    ) -> Any:
        r"""
        **Description:**
        Download curated (and optionally community) vacuum solutions
        for a given model from the HuggingFace vacua repo.

        Args:
            model: Optional JAXVacua model for identity extraction.
            ks_id, triang_id, cicy_id: Explicit identity overrides.
            label (str | None): If given, match only files whose name
                starts with this label (before any ``_v{n}`` suffix).
            include_community (bool): Also fetch files under
                ``community/``.  Defaults to False.
            include_retracted (bool): Include rows flagged
                ``retracted=True`` in the returned DataFrame.
            cache (bool): Keep downloads in the HF cache (reserved for
                future use; ``hf_hub_download`` always caches).
            repo_id (str | None): Target dataset repo.  Defaults to
                :func:`_resolve_vault_repo`.
            token (str | None): Optional HF access token.

        Returns:
            pandas.DataFrame: Concatenated rows from all matching
            files, with an extra ``_source_path`` column identifying
            the origin file.  Empty DataFrame if nothing matches.
        """
        if self.offline:
            raise RuntimeError(
                "fetch_vacua_from_hub requires network access "
                "(self.offline=True)."
            )
        pd = _require_pandas()
        hf_download = _require_hf_hub()
        hf_api = _require_hf_api()
        repo = repo_id or _resolve_vault_repo()

        identity = _extract_model_identity(
            model=model, ks_id=ks_id, triang_id=triang_id,
            cicy_id=cicy_id,
        )
        model_dir = self._remote_model_dir(identity)
        if model_dir is None:
            raise ValueError(
                "Could not derive a remote model directory from the "
                "given model/identity."
            )

        # List remote files under the model directory
        try:
            all_files = hf_api.list_repo_files(
                repo_id=repo, repo_type="dataset", token=token,
            )
        except Exception as e:
            raise RuntimeError(
                f"Could not list files in {repo}: {e}"
            ) from e

        def _matches(path: str) -> bool:
            if not path.startswith(model_dir + "/"):
                return False
            if not path.endswith(".parquet"):
                return False
            # Curated files live directly in model_dir; community files
            # live in model_dir/community/; _rejected/ always excluded.
            rel = path[len(model_dir) + 1:]
            if rel.startswith("_rejected/"):
                return False
            is_community = rel.startswith("community/")
            if is_community and not include_community:
                return False
            if not is_community and "/" in rel:
                return False  # nested file (shouldn't happen)
            # Label filter
            if label is not None:
                name = os.path.splitext(os.path.basename(path))[0]
                # Community files carry a "{hf_username}_" prefix; strip it
                if is_community and "_" in name:
                    # Find the first underscore; the rest is {label}[_v{n}]
                    name = name.split("_", 1)[1]
                # Accept exact label or label_v{n}
                import re
                if not re.match(rf"^{re.escape(label)}(_v\d+)?$", name):
                    return False
            return True

        matching = sorted(p for p in all_files if _matches(p))
        if not matching:
            return pd.DataFrame()

        dfs = []
        for rp in matching:
            try:
                local = hf_download(
                    repo_id=repo, filename=rp,
                    repo_type="dataset", token=token,
                )
            except Exception as e:
                print(f"[fetch_vacua_from_hub] failed to download {rp}: {e}")
                continue
            try:
                df = pd.read_parquet(local)
                df["_source_path"] = rp
                dfs.append(df)
            except Exception as e:
                print(f"[fetch_vacua_from_hub] failed to parse {rp}: {e}")

        if not dfs:
            return pd.DataFrame()
        out = _safe_concat(dfs, ignore_index=True)
        if not include_retracted and "retracted" in out.columns:
            out = out[~out["retracted"].fillna(False).astype(bool)].reset_index(drop=True)
        return out
    def list_hub_vacua(
        self,
        h11: Optional[int] = None,
        h12: Optional[int] = None,
        ks_id: Optional[int] = None,
        triang_id: Optional[int] = None,
        cicy_id: Optional[int] = None,
        label: Optional[str] = None,
        committed_by: Optional[str] = None,
        include_retracted: bool = False,
        limit: Optional[int] = None,
        repo_id: Optional[str] = None,
        token: Optional[str] = None,
    ) -> Any:
        r"""
        **Description:**
        Browse the remote vacua-vault catalogue.  Downloads
        ``catalog.parquet`` from the target repo and applies boolean
        masks following the same filter API as
        :meth:`query_designated`.

        Args:
            h11, h12, ks_id, triang_id, cicy_id: Identity filters.
            label (str | None): Filter by dataset label.
            committed_by (str | None): Filter by contributor.
            include_retracted (bool): Include retracted entries.
            limit (int | None): Return at most this many rows.
            repo_id (str | None): Target dataset repo.
            token (str | None): Optional HF access token.

        Returns:
            pandas.DataFrame: Catalog rows matching the filters.
            Empty DataFrame if the remote catalog is empty or missing.
        """
        if self.offline:
            raise RuntimeError(
                "list_hub_vacua requires network access (offline=True)."
            )
        pd = _require_pandas()
        hf_download = _require_hf_hub()
        repo = repo_id or _resolve_vault_repo()

        try:
            local = hf_download(
                repo_id=repo, filename="catalog.parquet",
                repo_type="dataset", token=token,
            )
        except Exception as e:
            # Catalog may not exist yet on a fresh repo.
            print(f"[list_hub_vacua] could not fetch remote catalog: {e}")
            return pd.DataFrame()

        cat = pd.read_parquet(local)
        filters = {
            "h11": h11, "h12": h12, "ks_id": ks_id,
            "triang_id": triang_id, "cicy_id": cicy_id,
            "label": label, "committed_by": committed_by,
        }
        for col, val in filters.items():
            if val is not None and col in cat.columns:
                cat = cat[cat[col] == val]

        if not include_retracted and "retracted" in cat.columns:
            cat = cat[~cat["retracted"].fillna(False).astype(bool)]

        if limit is not None:
            cat = cat.head(limit)

        return cat.reset_index(drop=True)
    @staticmethod
    def _remote_model_dir(identity: Dict[str, Any]) -> Optional[str]:
        r"""
        Derive the per-model subdirectory on the remote vault repo from
        an identity dict.  Returns ``None`` if the identity doesn't
        provide enough information.
        """
        ks  = identity.get("ks_id", -1)
        tr  = identity.get("triang_id", -1)
        cic = identity.get("cicy_id", -1)
        h12 = identity.get("h12")
        if ks >= 0 and tr >= 0 and h12 is not None:
            return f"tdf/h12_{h12}/ks_{ks}_tri_{tr}"
        if cic >= 0:
            return f"cicy/cicy_{cic}"
        return None
    @staticmethod
    def _resolve_remote_filename(
        hf_api: Any,
        hf_download: Any,
        repo: str,
        target_rel: str,
        if_exists: str,
        token: Optional[str] = None,
    ) -> str:
        r"""
        Implement the ``if_exists`` collision policy for a target remote
        filename.  Returns the final ``path_in_repo`` to use.

        - ``"error"``: raise if the file already exists.
        - ``"append"``: return the existing path (caller is responsible
          for appending — this helper just signals).
        - ``"new_version"``: auto-suffix ``_v2``, ``_v3``, ... until a
          free name is found.
        """
        # Probe existence by trying to download with a head-only call
        # (hf_hub_download returns file path if exists).  List files
        # via HfApi for a cleaner existence check.
        try:
            files = set(hf_api.list_repo_files(repo_id=repo, repo_type="dataset",
                                                token=token))
        except Exception:
            # Could be empty / unreachable repo; treat as "no collision".
            files = set()

        if target_rel not in files:
            return target_rel

        if if_exists == "error":
            raise FileExistsError(
                f"{target_rel!r} already exists in {repo!r}. "
                f"Pick a different label or use "
                f"if_exists='new_version' (auto-suffix) or "
                f"if_exists='append' (merge)."
            )
        if if_exists == "append":
            return target_rel
        # "new_version" → find the first free _vN suffix
        import re
        stem, ext = os.path.splitext(target_rel)
        base_stem = re.sub(r"_v\d+$", "", stem)  # strip any existing _vN
        n = 2
        while f"{base_stem}_v{n}{ext}" in files:
            n += 1
        return f"{base_stem}_v{n}{ext}"
    def designate_vacua(
        self,
        vacua_df: Any,
        label: str,
        committed_by: str,
        model: Any = None,
        h11: Optional[int] = None,
        h12: Optional[int] = None,
        ks_id: Optional[int] = None,
        triang_id: Optional[int] = None,
        conifold_id: int = -1,
        cicy_id: Optional[int] = None,
        model_name: Optional[str] = None,
        tags: Optional[Any] = None,
        notes: str = "",
        validate: bool = True,
        force: bool = False,
        F_term_tol: float = 1e-8,
    ) -> List[int]:
        r"""
        **Description:**
        Designate vacuum solutions as permanent / curated.  Solutions are
        written to the ``designated_vacua/`` directory with provenance
        metadata and added to the designated catalog.

        By default, all solutions are validated before writing.  Validation
        failures raise :class:`ValidationError` unless ``force=True``.

        The input *vacua_df* can be a ``DataFrame`` from
        :meth:`query_vacua`, or any DataFrame with the standard vacuum
        columns (``flux``, ``moduli_re``, ``moduli_im``, ``tau_re``,
        ``tau_im``).

        Args:
            vacua_df: ``pandas.DataFrame`` of vacuum rows.
            label (str): Human-readable label (e.g. ``"benchmark_ISD"``,
                ``"paper:2506.xxxxx"``).
            committed_by (str): Author identifier (name or ORCID).
            model: Optional JAXVacua model for identity extraction and
                physics validation.
            h11, h12, ks_id, triang_id, conifold_id, cicy_id: Explicit
                model identity overrides.
            model_name (str | None): Human-readable label for custom models.
            tags: String or list of tag strings applied to all solutions.
            notes (str): Free-text annotation.
            validate (bool): Run validation before designating.
            force (bool): Designate even if validation fails (failed
                solutions are skipped).
            F_term_tol (float): F-term tolerance for validation.

        Returns:
            List[int]: ``designated_id`` values assigned to the new
            solutions.

        Raises:
            ValidationError: If validation fails and ``force=False``.
            ValueError: If ``label`` or ``committed_by`` is empty.
        """
        pd = _require_pandas()

        if not label or not label.strip():
            raise ValueError("label must be non-empty")
        if not committed_by or not committed_by.strip():
            raise ValueError("committed_by must be non-empty")

        # Resolve model identity
        identity = _extract_model_identity(
            model=model, h11=h11, h12=h12, ks_id=ks_id,
            triang_id=triang_id, conifold_id=conifold_id,
            cicy_id=cicy_id, model_name=model_name,
        )

        # Validation
        if validate:
            report = self.validate_vacua(
                vacua_df, model=model, F_term_tol=F_term_tol,
                identity=identity,
            )
            n_failed = sum(1 for r in report if not r["passed"])
            if n_failed > 0 and not force:
                msg_lines = [f"{n_failed}/{len(report)} solutions failed validation:"]
                for r in report:
                    if not r["passed"]:
                        msg_lines.append(f"  row {r['index']}: {'; '.join(r['errors'])}")
                raise ValidationError("\n".join(msg_lines), report=report)
            # If force, keep only passing rows
            if force and n_failed > 0:
                keep_idx = [r["index"] for r in report if r["passed"]]
                vacua_df = vacua_df.iloc[keep_idx].reset_index(drop=True)

        if len(vacua_df) == 0:
            return []

        # Normalise tags
        if tags is None:
            tags_json = "[]"
        elif isinstance(tags, str):
            tags_json = json.dumps([tags])
        else:
            tags_json = json.dumps(list(tags))

        try:
            _jv_version = version("jaxvacua")
        except PackageNotFoundError:
            _jv_version = "unknown"

        now   = datetime.datetime.now(datetime.timezone.utc)
        today = now.date().isoformat()

        self._designated_dir.mkdir(parents=True, exist_ok=True)
        catalog_path = (
            self._designated_dir / "designated_vacua_catalog.parquet"
        )

        # ``designate_vacua`` assigns a contiguous range of integer IDs
        # starting from ``max(existing) + 1``.  Serialise the read of
        # the catalog, the ID allocation, and the write so concurrent
        # cluster workers cannot collide.
        with _file_lock(catalog_path):
            self._designated_catalog = None
            self._ensure_designated_catalog()
            dcat = self._designated_catalog
            if dcat is not None and len(dcat) > 0:
                next_id = int(dcat["designated_id"].max()) + 1
            else:
                next_id = 0

            catalog_rows: List[Dict[str, Any]] = []
            data_rows:    List[Dict[str, Any]] = []
            assigned_ids: List[int]            = []

            for i in range(len(vacua_df)):
                row = vacua_df.iloc[i]
                did = next_id + i
                assigned_ids.append(did)

                existing_tags: List[str] = []
                if "tags" in vacua_df.columns:
                    try:
                        et = row["tags"]
                        if isinstance(et, str):
                            existing_tags = json.loads(et)
                    except (json.JSONDecodeError, TypeError):
                        pass
                new_tags    = json.loads(tags_json)
                merged_tags = list(dict.fromkeys(existing_tags + new_tags))

                source_run = (
                    str(row.get("run_id", ""))
                    if "run_id" in vacua_df.columns else ""
                )

                catalog_rows.append({
                    "designated_id": did,
                    "h11":           identity["h11"],
                    "h12":           identity["h12"],
                    "ks_id":         identity["ks_id"],
                    "triang_id":     identity["triang_id"],
                    "conifold_id":   identity["conifold_id"],
                    "cicy_id":       identity["cicy_id"],
                    "model_hash":    identity["model_hash"],
                    "model_name":    identity.get("model_name"),
                    "flux":          list(int(x) for x in row["flux"]),
                    "N_flux":        row.get("N_flux"),
                    "is_susy":       bool(row.get("is_susy", False)),
                    "tags":          json.dumps(merged_tags),
                    "label":         label,
                    "notes":         notes,
                    "committed_by":  committed_by,
                    "commit_date":   today,
                    "jaxvacua_version": _jv_version,
                    "source_run_id": source_run,
                    "retracted":     False,
                    "retraction_reason": None,
                })

                data_rows.append({
                    "designated_id": did,
                    "flux":          list(int(x) for x in row["flux"]),
                    "moduli_re":     list(row["moduli_re"]) if "moduli_re" in vacua_df.columns else None,
                    "moduli_im":     list(row["moduli_im"]) if "moduli_im" in vacua_df.columns else None,
                    "tau_re":        row.get("tau_re"),
                    "tau_im":        row.get("tau_im"),
                    "W_re":          row.get("W_re"),
                    "W_im":          row.get("W_im"),
                    "F_terms_re":    row.get("F_terms_re"),
                    "F_terms_im":    row.get("F_terms_im"),
                    "N_flux":        row.get("N_flux"),
                    "residual":      row.get("residual"),
                    "is_susy":       bool(row.get("is_susy", False)),
                    "tags":          json.dumps(merged_tags),
                    "extra_data":    row.get("extra_data"),
                    "label":         label,
                    "committed_by":  committed_by,
                    "commit_date":   today,
                })

            shard_dir = self._resolve_vacua_dir(model=model, **{
                k: identity[k] for k in (
                    "h11", "h12", "ks_id", "triang_id",
                    "conifold_id", "cicy_id", "model_name",
                ) if k in identity
            })
            shard_dir.mkdir(parents=True, exist_ok=True)
            shard_path = shard_dir / "shard_0.parquet"

            data_df = pd.DataFrame(data_rows)
            if shard_path.exists():
                existing = pd.read_parquet(shard_path)
                data_df  = _safe_concat([existing, data_df], ignore_index=True)
            data_df.to_parquet(shard_path, index=False)

            catalog_df = pd.DataFrame(catalog_rows)
            if dcat is not None and len(dcat) > 0:
                catalog_df = _safe_concat(
                    [dcat, catalog_df], ignore_index=True,
                )
            catalog_df.to_parquet(catalog_path, index=False)
            self._designated_catalog = catalog_df

        return assigned_ids
    def query_designated(
        self,
        h11: Optional[int] = None,
        h12: Optional[int] = None,
        ks_id: Optional[int] = None,
        triang_id: Optional[int] = None,
        conifold_id: Optional[int] = None,
        cicy_id: Optional[int] = None,
        model_hash: Optional[str] = None,
        label: Optional[str] = None,
        committed_by: Optional[str] = None,
        is_susy: Optional[bool] = None,
        N_flux_max: Optional[float] = None,
        tags: Optional[Any] = None,
        include_retracted: bool = False,
        limit: Optional[int] = None,
    ) -> Any:
        r"""
        **Description:**
        Query designated (permanent) vacua from the catalog.

        All filtering is done at the catalog level (one row per solution),
        so this is fast even for large collections.

        Args:
            h11, h12, ks_id, triang_id, conifold_id, cicy_id: Model
                identity filters (exact match).
            model_hash (str | None): Filter by geometric fingerprint.
            label (str | None): Filter by label (exact match).
            committed_by (str | None): Filter by author.
            is_susy (bool | None): Filter by SUSY flag.
            N_flux_max (float | None): Maximum tadpole value.
            tags: String or list of strings.  Rows must contain **all**
                specified tags.
            include_retracted (bool): Include retracted solutions.
                Defaults to ``False``.
            limit (int | None): Maximum number of rows to return.

        Returns:
            pandas.DataFrame: Matching rows from the designated catalog.
        """
        pd = _require_pandas()
        self._ensure_designated_catalog()
        dcat = self._designated_catalog
        if dcat is None or len(dcat) == 0:
            return pd.DataFrame()

        mask = np.ones(len(dcat), dtype=bool)

        # Exclude retracted by default
        if not include_retracted and "retracted" in dcat.columns:
            mask &= ~dcat["retracted"].fillna(False).astype(bool).values

        if h11 is not None:
            mask &= (dcat["h11"] == h11).values
        if h12 is not None:
            mask &= (dcat["h12"] == h12).values
        if ks_id is not None:
            mask &= (dcat["ks_id"] == ks_id).values
        if triang_id is not None:
            mask &= (dcat["triang_id"] == triang_id).values
        if conifold_id is not None:
            mask &= (dcat["conifold_id"] == conifold_id).values
        if cicy_id is not None:
            mask &= (dcat["cicy_id"] == cicy_id).values
        if model_hash is not None:
            mask &= (dcat["model_hash"] == model_hash).values
        if label is not None:
            mask &= (dcat["label"] == label).values
        if committed_by is not None:
            mask &= (dcat["committed_by"] == committed_by).values
        if is_susy is not None:
            mask &= (dcat["is_susy"] == is_susy).values
        if N_flux_max is not None and "N_flux" in dcat.columns:
            mask &= dcat["N_flux"].fillna(float("inf")).values <= N_flux_max
        if tags is not None:
            if isinstance(tags, str):
                tags = [tags]
            def _has_all_tags(tags_json, required=tags):
                """Check whether tags_json contains all required tags."""
                try:
                    stored = json.loads(tags_json) if isinstance(tags_json, str) else []
                except (json.JSONDecodeError, TypeError):
                    stored = []
                return all(t in stored for t in required)
            if "tags" in dcat.columns:
                mask &= dcat["tags"].apply(_has_all_tags).values

        result = dcat[mask]
        if limit is not None:
            result = result.head(limit)
        return result.reset_index(drop=True)
    def load_designated(
        self,
        designated_ids: Optional[List[int]] = None,
        h11: Optional[int] = None,
        h12: Optional[int] = None,
        ks_id: Optional[int] = None,
        triang_id: Optional[int] = None,
        model_hash: Optional[str] = None,
    ) -> Any:
        r"""
        **Description:**
        Load full data (moduli, flux, derived quantities) for designated
        solutions.

        Either specify ``designated_ids`` directly, or use model identity
        filters to select all designated solutions for a model.

        Args:
            designated_ids (list[int] | None): Specific designated IDs.
            h11, h12, ks_id, triang_id: Model identity filters.
            model_hash (str | None): Filter by geometric fingerprint.

        Returns:
            pandas.DataFrame: Full solution data with ``designated_id``.
        """
        pd = _require_pandas()
        self._ensure_designated_catalog()
        dcat = self._designated_catalog
        if dcat is None or len(dcat) == 0:
            return pd.DataFrame()

        # Determine which shard directories to load
        if designated_ids is not None:
            subset = dcat[dcat["designated_id"].isin(designated_ids)]
        else:
            mask = np.ones(len(dcat), dtype=bool)
            if h11 is not None:
                mask &= (dcat["h11"] == h11).values
            if h12 is not None:
                mask &= (dcat["h12"] == h12).values
            if ks_id is not None:
                mask &= (dcat["ks_id"] == ks_id).values
            if triang_id is not None:
                mask &= (dcat["triang_id"] == triang_id).values
            if model_hash is not None:
                mask &= (dcat["model_hash"] == model_hash).values
            subset = dcat[mask]

        if len(subset) == 0:
            return pd.DataFrame()

        # Group by model to find shard paths
        target_ids = set(subset["designated_id"].values)
        frames = []

        # Scan all shard files under designated_vacua/
        if self._designated_dir.exists():
            for shard_path in self._designated_dir.rglob("shard_*.parquet"):
                try:
                    df = pd.read_parquet(shard_path)
                    if "designated_id" in df.columns:
                        matched = df[df["designated_id"].isin(target_ids)]
                        if len(matched) > 0:
                            frames.append(matched)
                except Exception:
                    continue

        if not frames:
            return pd.DataFrame()
        return _safe_concat(frames, ignore_index=True)
    def designated_info(self) -> None:
        r"""
        **Description:**
        Print a summary of designated (permanent) vacua.
        """
        self._ensure_designated_catalog()
        dcat = self._designated_catalog
        if dcat is None or len(dcat) == 0:
            print("No designated vacua stored.")
            return

        # Filter active (non-retracted)
        if "retracted" in dcat.columns:
            active = dcat[~dcat["retracted"].fillna(False).astype(bool)]
            n_retracted = len(dcat) - len(active)
        else:
            active = dcat
            n_retracted = 0

        n_total  = len(active)
        n_models = len(active.groupby(["ks_id", "triang_id", "model_hash"]))
        labels   = active["label"].value_counts().to_dict() if "label" in active.columns else {}
        authors  = active["committed_by"].nunique() if "committed_by" in active.columns else 0

        print(f"Designated vacua: {n_total} solution(s) across "
              f"{n_models} model(s), {authors} contributor(s)")
        if n_retracted > 0:
            print(f"  ({n_retracted} retracted, excluded from counts)")
        if labels:
            print(f"  Labels: {labels}")

        # Per-model summary
        groups = active.groupby(["h11", "ks_id", "triang_id"]).agg(
            n_vacua=("designated_id", "count"),
        ).reset_index()
        if len(groups) <= 20:
            print(f"\n{'h11':>4} {'ks_id':>6} {'trid':>5} {'vacua':>7}")
            for _, g in groups.iterrows():
                print(f"{g['h11']:4d} {g['ks_id']:6d} {g['triang_id']:5d} "
                      f"{g['n_vacua']:7d}")
        else:
            print(f"  ({len(groups)} models — use query_designated() for details)")
    def load_local_vacua(
        self,
        model: Any = None,
        label: Optional[str] = None,
        include_retracted: bool = False,
        **identity_kwargs: Any,
    ) -> Any:
        r"""
        **Description:**
        Load designated vacua stored in the local ``vacua_vault/`` for a
        given model.

        Convenience wrapper around :meth:`query_designated` that
        resolves the per-model vault subdirectory via
        :meth:`_resolve_vacua_dir` and returns any solutions previously
        designated for that model.

        Args:
            model: Optional JAXVacua model or ``lcs_tree`` used for
                identity extraction.
            label (str | None): If given, filter by designation label.
            include_retracted (bool): Include retracted entries.
                Defaults to False.
            **identity_kwargs: Explicit identity overrides
                (``h11``, ``h12``, ``ks_id``, ``triang_id``,
                ``cicy_id``, ``model_name``).

        Returns:
            pandas.DataFrame: Solutions with the standard vacuum
            columns.  Empty DataFrame if no vacua are stored.
        """
        ident = _extract_model_identity(model=model, **identity_kwargs)
        query = {}
        if ident.get("h11") is not None:     query["h11"]       = ident["h11"]
        if ident.get("h12") is not None:     query["h12"]       = ident["h12"]
        if ident.get("ks_id", -1) >= 0:      query["ks_id"]     = ident["ks_id"]
        if ident.get("triang_id", -1) >= 0:  query["triang_id"] = ident["triang_id"]
        if ident.get("cicy_id", -1) >= 0:    query["cicy_id"]   = ident["cicy_id"]
        if label is not None:                query["label"]     = label

        try:
            df = self.query_designated(**query)
        except TypeError:
            # Fallback for a reduced query_designated signature: query
            # by label (if any) and filter remaining columns manually.
            df = self.query_designated(label=label) if label else self.query_designated()
            if df is not None and len(df) > 0:
                for col, val in query.items():
                    if col in df.columns:
                        df = df[df[col] == val]
                df = df.reset_index(drop=True)

        if df is not None and len(df) > 0 and not include_retracted:
            if "retracted" in df.columns:
                df = df[~df["retracted"].fillna(False).astype(bool)].reset_index(drop=True)
        return df
    def retract_designated(
        self,
        designated_id: int,
        reason: str,
        retracted_by: Optional[str] = None,
    ) -> None:
        r"""
        **Description:**
        Retract a designated solution.  The solution is **not** deleted —
        it remains in the catalog and shard with ``retracted=True``,
        ``retraction_reason``, and ``retracted_at`` (timestamp) set.
        Retracted solutions are excluded from :meth:`query_designated` by
        default.

        A row is also appended to ``<vault>/retractions.parquet`` as a
        permanent audit trail that survives any future catalog rebuild.

        Args:
            designated_id (int): The ID of the solution to retract.
            reason (str): Explanation for the retraction (e.g.
                ``"Bug in jaxvacua v0.3.1 period computation"``).
            retracted_by (str | None): Optional identifier of the
                person/agent performing the retraction (ORCID, HF
                username, email).  Recorded in the audit trail only.

        Raises:
            KeyError: If the designated_id is not found.
            ValueError: If *reason* is empty.
        """
        pd = _require_pandas()
        if not reason or not reason.strip():
            raise ValueError("retraction reason must be non-empty")

        self._ensure_designated_catalog()
        dcat = self._designated_catalog
        if dcat is None or len(dcat) == 0:
            raise KeyError(f"Designated ID {designated_id} not found — catalog is empty.")

        mask = dcat["designated_id"] == designated_id
        if not mask.any():
            raise KeyError(f"Designated ID {designated_id} not found in catalog.")

        now = pd.Timestamp.now(tz="UTC")
        dcat.loc[mask, "retracted"]         = True
        dcat.loc[mask, "retraction_reason"] = reason
        dcat.loc[mask, "retracted_at"]      = now

        catalog_path = self._designated_dir / "designated_vacua_catalog.parquet"
        dcat.to_parquet(catalog_path, index=False)
        self._designated_catalog = dcat

        # Append to <vault>/retractions.parquet (audit trail)
        audit_row = {
            "designated_id": int(designated_id),
            "reason":        reason,
            "retracted_at":  now,
            "retracted_by":  retracted_by or "",
        }
        audit_path = _resolve_vault_dir() / "retractions.parquet"
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        if audit_path.exists():
            existing = pd.read_parquet(audit_path)
            audit_df = _safe_concat([existing, pd.DataFrame([audit_row])],
                                    ignore_index=True)
        else:
            audit_df = pd.DataFrame([audit_row])
        audit_df.to_parquet(audit_path, index=False)
    def purge_retracted(
        self,
        older_than: Optional[str] = "30d",
        dry_run: bool = True,
        confirm: bool = False,
        archive: bool = False,
    ) -> list:
        r"""
        **Description:**
        Irreversibly delete retracted designated solutions.

        Selects rows in the designated catalog with ``retracted=True``
        and (if *older_than* is given) ``retracted_at`` older than the
        threshold.  Prints a summary of what would be deleted; only acts
        if both ``dry_run=False`` and ``confirm=True``.

        Args:
            older_than (str | None): Pandas timedelta string (e.g.
                ``"30d"``, ``"90d"``, ``"6h"``) or ``None`` to ignore
                age.  Defaults to ``"30d"``.
            dry_run (bool): If True (default), print what would be
                deleted and return the catalog subset — no files are
                touched.
            confirm (bool): Must be True alongside ``dry_run=False`` to
                actually delete.  Guardrail against accidental data
                loss.
            archive (bool): If True, copy each shard to
                ``<vault>/archive/`` before deletion.  Defaults to
                False.

        Returns:
            list: List of ``designated_id`` values selected for (or,
            actually, deleted in) the purge.
        """
        pd = _require_pandas()
        import shutil

        self._ensure_designated_catalog()
        dcat = self._designated_catalog
        if dcat is None or len(dcat) == 0:
            print("[purge_retracted] No designated catalog entries.")
            return []

        if "retracted" not in dcat.columns:
            print("[purge_retracted] Catalog has no 'retracted' column — nothing to purge.")
            return []

        mask = dcat["retracted"].fillna(False).astype(bool)
        if older_than is not None and "retracted_at" in dcat.columns:
            cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(older_than)
            ts = pd.to_datetime(dcat["retracted_at"], errors="coerce", utc=True)
            mask = mask & (ts < cutoff)

        selected = dcat[mask]
        if len(selected) == 0:
            print("[purge_retracted] No retracted entries match the criteria.")
            return []

        ids = [int(x) for x in selected["designated_id"].tolist()]
        print(f"[purge_retracted] {len(selected)} entries selected for purge:")
        for _, row in selected.iterrows():
            print(f"  id={row.get('designated_id')} label={row.get('label')!r} "
                  f"retracted_at={row.get('retracted_at')}")

        if dry_run:
            print("[purge_retracted] dry_run=True — no files touched. "
                  "Pass dry_run=False, confirm=True to actually delete.")
            return ids
        if not confirm:
            raise RuntimeError(
                "purge_retracted refuses to run with dry_run=False "
                "unless confirm=True is passed explicitly."
            )

        # Each shard may contain multiple designated rows; surgically
        # remove just the purged rows and rewrite (or delete) each shard.
        vault = _resolve_vault_dir()
        archive_dir = vault / "archive"
        if archive:
            archive_dir.mkdir(parents=True, exist_ok=True)

        shard_to_ids: Dict[Path, List[int]] = {}
        for _, row in selected.iterrows():
            try:
                shard_dir = self._resolve_vacua_dir(
                    h11       = row.get("h11"),
                    h12       = row.get("h12"),
                    ks_id     = row.get("ks_id", -1),
                    triang_id = row.get("triang_id", -1),
                    cicy_id   = row.get("cicy_id", -1),
                    model_name= row.get("model_name"),
                )
            except Exception as e:
                print(f"[purge_retracted] could not resolve shard for id={row['designated_id']}: {e}")
                continue
            shard_path = shard_dir / "shard_0.parquet"
            shard_to_ids.setdefault(shard_path, []).append(int(row["designated_id"]))

        deleted = []
        for shard_path, ids in shard_to_ids.items():
            if not shard_path.exists():
                continue
            if archive:
                rel = f"{shard_path.parent.name}__{shard_path.name}"
                shutil.copy2(shard_path, archive_dir / rel)
            shard_df = pd.read_parquet(shard_path)
            keep = ~shard_df["designated_id"].isin(ids)
            shard_df = shard_df[keep]
            if len(shard_df) > 0:
                shard_df.to_parquet(shard_path, index=False)
            else:
                shard_path.unlink()
            deleted.extend(ids)

        # Drop purged rows from catalog
        dcat = dcat[~mask].reset_index(drop=True)
        catalog_path = self._designated_dir / "designated_vacua_catalog.parquet"
        dcat.to_parquet(catalog_path, index=False)
        self._designated_catalog = dcat

        print(f"[purge_retracted] Purged {len(deleted)} entry(ies) across "
              f"{len(shard_to_ids)} shard(s). Audit trail remains in "
              f"{vault}/retractions.parquet")
        return deleted


__all__ = ['VacuaWriter', '_VacuaStreamWriter']
