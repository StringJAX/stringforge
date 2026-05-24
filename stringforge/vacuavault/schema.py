r"""
Schema constants and file-level validation helpers for the vault.

**Homogeneity invariant.**  Each parquet file in the vault must
contain rows describing **exactly one model** — i.e. every row
shares the same ``(ks_id, triang_id)`` (TDF) or ``cicy_id`` (CICY).
The vault layout (``<dataset>/h12_N/<identifier>/<label>.parquet``)
makes one-file-per-model the natural unit.  Mixed-model files
break the catalogue's identity assumptions and the auto-load path
in :func:`validate_parquet_file`; they are rejected with a
top-level error.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Schema version of the vault parquet layout.  Defined here as the
# canonical source; ``cy_io`` re-exports the same constant so callers
# can import either path.  Bump alongside any change to
# ``REQUIRED_COLUMNS`` or to the parquet kv-metadata contract.
SCHEMA_VERSION: int = 1

#: Top-level filenames the vault treats as metadata (CI rejects uploads
#: whose basename matches one of these).
RESERVED_NAMES = {
    "catalog.parquet",
    "retractions.parquet",
    "schema.json",
    "README.md",
}

#: Slug regex for free-form labels (Option A in the vacua-storage plan).
LABEL_SLUG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*(?:_v\d+)?$")

#: Required columns for every vault parquet file.
REQUIRED_COLUMNS = (
    "flux", "moduli_re", "moduli_im", "tau_re", "tau_im",
)


def _basename_is_reserved(path: str | Path) -> bool:
    name = os.path.basename(str(path))
    if name in RESERVED_NAMES:
        return True
    if name.startswith("_"):
        return True
    return False


def _parse_label_from_filename(filename: str) -> Optional[str]:
    """Extract the label from a filename.  Strips the optional
    ``{hf_username}_`` community prefix when the caller passes a
    community-path filename — but since we can't tell from the name
    alone whether a prefix is present, this just returns the stem
    (caller strips prefixes as needed).
    """
    stem = os.path.splitext(os.path.basename(filename))[0]
    return stem or None


def _extract_identity_from_row(row: Any) -> Optional[Dict[str, Any]]:
    r"""
    **Description:**
    Extract a canonical model-identity dict from a single vacua row.

    Returns ``{"dataset": "tdf", "kwargs": {"ks_id": …, "triang_id": …}}``
    when the row carries non-``-1`` ``ks_id`` and ``triang_id`` columns,
    or ``{"dataset": "cicy", "kwargs": {"cicy_id": …}}`` for a non-``-1``
    ``cicy_id``.  Returns ``None`` for "custom" / "unrooted" models
    (all identity columns are ``-1`` or missing) — auto-load can't help
    those and the caller should skip the physics path with a warning.

    Args:
        row: A pandas Series (one row of the vacua table).

    Returns:
        dict | None: Identity record, or ``None`` if no identifier
        is present.
    """
    def _get(name: str, default: int = -1) -> int:
        try:
            v = row.get(name, default)
        except AttributeError:
            v = row[name] if name in row else default
        try:
            return int(v)
        except (TypeError, ValueError):
            return default

    ks  = _get("ks_id")
    tr  = _get("triang_id")
    cic = _get("cicy_id")

    if ks >= 0 and tr >= 0:
        return {"dataset": "tdf",
                "kwargs":  {"ks_id": ks, "triang_id": tr}}
    if cic >= 0:
        return {"dataset": "cicy", "kwargs": {"cicy_id": cic}}
    return None


def _identity_from_row(df: Any) -> Optional[Dict[str, Any]]:
    r"""
    **Description:**
    Extract a single canonical identity record from a homogeneous
    vacua DataFrame.  Reads the **first** row; the homogeneity
    invariant (enforced upstream by
    :func:`_check_identity_consistency`) guarantees the rest agree.

    Returns ``None`` when the DataFrame is empty or its rows lack
    identity columns.

    Args:
        df: A pandas DataFrame with vacua rows.
    """
    if df is None or len(df) == 0:
        return None
    return _extract_identity_from_row(df.iloc[0])


def _check_identity_consistency(df: Any) -> List[str]:
    r"""
    **Description:**
    Verify the per-file homogeneity invariant: every row must
    describe the same model.  See the module docstring for context.

    Returns a list of top-level error strings (empty list = OK).

    Args:
        df: A pandas DataFrame with vacua rows.
    """
    errors: List[str] = []
    if df is None or len(df) == 0:
        return errors
    id_cols = [c for c in ("ks_id", "triang_id", "cicy_id") if c in df.columns]
    if not id_cols:
        # No identity columns at all — handled at auto-load stage as a
        # warning, not as an inconsistency error.
        return errors
    distinct = df[id_cols].drop_duplicates()
    if len(distinct) > 1:
        errors.append(
            f"file contains rows for {len(distinct)} distinct models "
            f"(violates the one-model-per-file invariant). Identifier "
            f"tuples found: {distinct.values.tolist()[:5]}"
            f"{' …' if len(distinct) > 5 else ''}"
        )
    return errors


def validate_parquet_file(
    path: str | Path,
    *,
    model: Any = None,
    db: Any = None,
    model_hash_fn: Optional[Any] = None,
    physics_checks: str = "auto",
    F_term_tol: float = 1e-6,
    check_tadpole: bool = True,
    expected_schema_version: int = SCHEMA_VERSION,
) -> Dict[str, Any]:
    r"""
    **Description:**
    Load a parquet file and validate it against the vault schema.

    Performs the same row-level checks as
    :meth:`stringforge.cy_io.CYDatabase.validate_vacua`, plus an
    identity-homogeneity check (one model per file — see the module
    docstring for context) and an optional auto-load path that
    reconstructs the model from row identity when physics checks
    are requested.

    Three modes via ``physics_checks``:

    * ``"off"``      — schema-only validation (no identity check, no
      tadpole, no F-term).
    * ``"auto"`` (default) — schema + identity homogeneity always.
      If ``model`` is supplied, runs physics with it.  Else if rows
      carry identity AND ``db`` is supplied, attempts to auto-load
      via ``db.load_model(...)``.  Missing ``db`` / unrooted rows /
      auto-load failure all degrade to **warnings**.
    * ``"explicit"`` — same as ``"auto"`` but every degradation is a
      top-level error.  For users who *require* physics validation.

    **Pure dependency injection**: this function never imports
    jaxvacua (or any other downstream package).  Callers that want
    physics-aware validation must supply ``db=`` and (optionally)
    ``model_hash_fn=`` themselves.

    Args:
        path:                    Path to the parquet file.
        model:                   Optional pre-built ``FluxVacuaFinder``
            for F-term + tadpole verification.  Wins over ``db`` /
            ``physics_checks`` auto-load: if supplied, no auto-load is
            attempted.
        db:                      Optional pre-built CYDatabase (or
            subclass) instance with a ``load_model(**ident)`` method
            taking ``ks_id``/``triang_id`` (TDF) or ``cicy_id`` (CICY)
            kwargs.  Required for the auto-load path; if ``None`` and
            ``model`` is also missing, auto-load is skipped with a
            warning (in ``"auto"`` mode) or error (``"explicit"``).
        model_hash_fn:           Optional callable
            ``(lcs_tree) -> str`` that recomputes the model hash from
            an ``lcs_tree``-like object.  When supplied alongside a
            ``model``, the rebuilt hash is compared to the row's
            stored ``model_hash`` and a mismatch raises a top-level
            error (catches incompatible jaxvacua-version uploads).
            When ``None``, the check is silently skipped.
        physics_checks:          ``"off"`` | ``"auto"`` (default) |
            ``"explicit"``.  See above.
        F_term_tol:              F-term tolerance for SUSY rows.
        check_tadpole:           Verify ``|f^T σ h| ≤ D3`` (requires
            ``model``).
        expected_schema_version: Expected ``schema_version`` from
            parquet key-value metadata.

    Returns:
        dict: ``{"passed": bool, "errors": list, "warnings": list,
        "n_rows": int, "n_failed": int, "per_row_report": list,
        "metadata": dict}``.
    """
    if physics_checks not in ("off", "auto", "explicit"):
        raise ValueError(
            f"physics_checks must be 'off', 'auto', or 'explicit'; "
            f"got {physics_checks!r}"
        )
    import pandas as pd
    import pyarrow.parquet as pq

    errors: List[str] = []
    warnings: List[str] = []
    report: List[Dict[str, Any]] = []
    p = Path(path)

    # 1. Filename checks
    if _basename_is_reserved(p):
        errors.append(f"Reserved filename: {p.name}")

    stem = os.path.splitext(p.name)[0]
    # Strip a leading {hf_username}_ prefix for community files (best-effort).
    parent = p.parent.name
    if parent == "community":
        # {hf_username}_{label}[_v{n}]: require at least one underscore.
        if "_" not in stem:
            errors.append(
                f"community file {p.name!r} lacks a '{{hf_username}}_' prefix"
            )
    if not LABEL_SLUG_RE.match(stem):
        errors.append(f"filename stem {stem!r} is not a valid slug")

    # 2. Load parquet + metadata
    try:
        table = pq.read_table(p)
    except Exception as e:
        return {
            "passed": False,
            "errors": [f"Failed to read parquet: {e}"],
            "warnings": warnings,
            "n_rows": 0,
            "n_failed": 0,
            "per_row_report": [],
            "metadata": {},
        }

    md_raw = table.schema.metadata or {}
    metadata = {k.decode(): v.decode() for k, v in md_raw.items()
                if isinstance(k, (bytes, bytearray))}

    sv = metadata.get("schema_version")
    if sv is None:
        warnings.append("parquet has no 'schema_version' metadata")
    else:
        try:
            sv_int = int(sv)
            if sv_int != expected_schema_version:
                errors.append(
                    f"schema_version mismatch: file={sv_int}, "
                    f"expected={expected_schema_version}"
                )
        except ValueError:
            errors.append(f"unparseable schema_version: {sv!r}")

    # 3. Required columns
    df = table.to_pandas()
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        errors.append(f"missing required columns: {missing}")

    n_rows = len(df)
    if n_rows == 0:
        warnings.append("file has 0 rows")

    # 3.5 Identity homogeneity (only relevant when physics_checks ≠ "off"
    # so users who explicitly request schema-only get it).
    if physics_checks != "off":
        errors.extend(_check_identity_consistency(df))

    # 3.7 Auto-load: pure dependency injection.  We never import the
    # downstream model layer ourselves — the caller supplies a
    # ``db`` instance whose ``load_model(**ident)`` returns a model
    # carrying the necessary lcs_tree.  Missing identity / missing db
    # / load failure all degrade to a warning in ``"auto"`` mode and
    # to a top-level error in ``"explicit"``.
    if model is None and physics_checks in ("auto", "explicit") and n_rows > 0:
        ident = _identity_from_row(df)
        if ident is None:
            msg = ("rows carry no model identity; cannot auto-load. "
                   "Pass `model=` explicitly or set "
                   "physics_checks='off' to skip physics.")
            (errors if physics_checks == "explicit" else warnings).append(msg)
        elif db is None:
            msg = ("auto-load requires `db=`; pass an LCSDatabase "
                   "(or any CYDatabase subclass with a "
                   "`load_model(**ident)` method) instance.  Pass "
                   "`physics_checks='off'` to skip physics validation.")
            (errors if physics_checks == "explicit" else warnings).append(msg)
        else:
            try:
                model = db.load_model(**ident["kwargs"])
            except Exception as e:                                # noqa: BLE001
                msg = (f"auto-load failed for "
                       f"{ident['dataset']}/{ident['kwargs']}: "
                       f"{type(e).__name__}: {e}")
                (errors if physics_checks == "explicit" else warnings).append(msg)
                model = None

    # 3.8 Model-hash round-trip — when we have a model, a caller-supplied
    # ``model_hash_fn``, AND a stored ``model_hash`` on the rows, the
    # reconstructed tree's hash must match.  Catches incompatible
    # uploads (basis-change drift, conifold convention changes, etc.).
    # When ``model_hash_fn`` is ``None``, the check is silently
    # skipped — this keeps schema.py free of jaxvacua-specific
    # imports.
    if (model is not None and physics_checks != "off"
            and model_hash_fn is not None
            and "model_hash" in df.columns and n_rows > 0):
        stored = str(df.iloc[0].get("model_hash") or "")
        if stored:
            try:
                tree = getattr(model, "lcs_tree", None) or getattr(
                    getattr(model, "periods", None), "lcs_tree", None,
                )
                if tree is None:
                    warnings.append("model has no lcs_tree; "
                                    "skipping model_hash check")
                else:
                    rebuilt = model_hash_fn(tree)
                    if rebuilt != stored:
                        errors.append(
                            f"model_hash mismatch: row={stored[:12]}…, "
                            f"reconstructed={rebuilt[:12]}…  "
                            f"(incompatible package version / basis / "
                            f"conifold convention)"
                        )
            except Exception as e:                                # noqa: BLE001
                warnings.append(f"model_hash check failed: {e}")

    # 4. Per-row validation: integer flux, tadpole, F-terms
    n_failed = 0
    for i, row in df.iterrows():
        row_errors: List[str] = []
        if "flux" in df.columns:
            try:
                fl = row["flux"]
                ints = [int(x) for x in fl]
                if any(abs(x - v) > 1e-6 for x, v in zip(ints, fl)):
                    row_errors.append("flux vector contains non-integer entries")
            except (TypeError, ValueError):
                row_errors.append("flux vector is not iterable of ints")

        # Tadpole check (requires model)
        if check_tadpole and model is not None and "flux" in df.columns:
            try:
                import jax.numpy as jnp
                tad = abs(float(
                    (model.tadpole(jnp.array(list(row["flux"])))).real
                    if hasattr(model.tadpole(jnp.array(list(row["flux"]))), "real")
                    else model.tadpole(jnp.array(list(row["flux"])))
                ))
                D3 = getattr(
                    getattr(model, "lcs_tree", None), "D3_tadpole", None,
                )
                if D3 is not None and tad > float(D3) + 1e-9:
                    row_errors.append(
                        f"tadpole {tad:.2f} > D3={float(D3):.2f}"
                    )
            except Exception as e:
                row_errors.append(f"tadpole check failed: {e}")

        # F-term check for SUSY-flagged rows
        if model is not None and bool(row.get("is_susy", False)):
            try:
                import jax.numpy as jnp
                flux   = jnp.array(list(row["flux"]), dtype=float)
                mod    = jnp.array(
                    [r + 1j * im for r, im in zip(row["moduli_re"], row["moduli_im"])]
                )
                tau    = complex(row["tau_re"] + 1j * row["tau_im"])
                DW = model.DW_x(
                    model._convert_complex_to_real(mod, tau), flux,
                )
                maxDW = float(abs(DW).max())
                if maxDW > F_term_tol:
                    row_errors.append(
                        f"|DW|={maxDW:.2e} > tol={F_term_tol:.0e}"
                    )
            except Exception as e:
                row_errors.append(f"F-term check failed: {e}")

        passed = not row_errors
        if not passed:
            n_failed += 1
        report.append({
            "index": int(i),
            "passed": passed,
            "errors": row_errors,
        })

    if n_failed > 0:
        errors.append(f"{n_failed}/{n_rows} rows failed row-level validation")

    return {
        "passed":         len(errors) == 0,
        "errors":         errors,
        "warnings":       warnings,
        "n_rows":         n_rows,
        "n_failed":       n_failed,
        "per_row_report": report,
        "metadata":       metadata,
    }


def split_by_validation(
    df: Any, per_row_report: List[Dict[str, Any]],
) -> Tuple[Any, Any]:
    r"""
    **Description:**
    Split a DataFrame into (valid, rejected) subsets based on a
    per-row validation report.  The rejected DataFrame gets an extra
    ``error`` column containing the joined error strings.
    """
    import pandas as pd
    ok_idx  = [r["index"] for r in per_row_report if r["passed"]]
    bad_idx = [r["index"] for r in per_row_report if not r["passed"]]
    valid = df.iloc[ok_idx].reset_index(drop=True) if ok_idx else df.iloc[:0].copy()
    if bad_idx:
        rej = df.iloc[bad_idx].copy().reset_index(drop=True)
        errs = []
        for r in per_row_report:
            if not r["passed"]:
                errs.append("; ".join(r["errors"]))
        rej["error"] = errs
    else:
        rej = df.iloc[:0].copy()
    return valid, rej
