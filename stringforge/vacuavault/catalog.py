r"""
Catalog regeneration for the HuggingFace ``vacua_vault`` repo.

Scans the repo for ``**/*.parquet`` (excluding ``catalog.parquet``,
``retractions.parquet``, and ``_rejected/**``) and writes a fresh
``catalog.parquet`` at the repo root with one row per file.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .schema import SCHEMA_VERSION, RESERVED_NAMES


#: Columns in the regenerated catalog.
CATALOG_COLUMNS = (
    "file_path", "basename", "status",
    "h12", "ks_id", "triang_id", "cicy_id",
    "susy", "method", "nmax", "qualifiers",
    "classification_source",
    "label", "version", "committed_by",
    "n_rows", "schema_version",
    "checksum", "superseded_by",
    "retracted", "retraction_reason", "retracted_at",
    "uploaded_at",
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _parse_path(repo_root: Path, p: Path) -> Dict[str, Any]:
    """Extract identity + status from a parquet file's path."""
    rel = p.relative_to(repo_root)
    parts = rel.parts
    info: Dict[str, Any] = {
        "file_path": str(rel),
        "basename":  p.name,
        "h12":       None,
        "ks_id":     None,
        "triang_id": None,
        "cicy_id":   None,
        "status":    "curated",
        "label":     None,
        "version":   1,
    }
    # tdf/h12_N/ks_X_tri_Y/[community/_rejected/]<name>.parquet
    if len(parts) >= 4 and parts[0] == "tdf":
        m = re.match(r"h12_(\d+)", parts[1])
        if m:
            info["h12"] = int(m.group(1))
        m = re.match(r"ks_(\d+)_tri_(\d+)", parts[2])
        if m:
            info["ks_id"] = int(m.group(1))
            info["triang_id"] = int(m.group(2))
        rest = parts[3:]
        if rest and rest[0] == "community":
            info["status"] = "pending"
            rest = rest[1:]
        if rest and rest[0] == "_rejected":
            info["status"] = "rejected"
            rest = rest[1:]
    # cicy/cicy_X/...
    elif len(parts) >= 3 and parts[0] == "cicy":
        m = re.match(r"cicy_(\d+)", parts[1])
        if m:
            info["cicy_id"] = int(m.group(1))
        rest = parts[2:]
        if rest and rest[0] == "community":
            info["status"] = "pending"
            rest = rest[1:]
    # Parse label + optional _v{n}
    stem = os.path.splitext(p.name)[0]
    # Community files are {hf_username}_{label}[_v{n}]
    if info["status"] == "pending" and "_" in stem:
        # Best-effort: split once, assume first segment is username
        _hfu, lbl_stem = stem.split("_", 1)
    else:
        lbl_stem = stem
    vm = re.match(r"^(?P<label>.+?)(?:_v(?P<ver>\d+))?$", lbl_stem)
    if vm:
        info["label"] = vm.group("label")
        if vm.group("ver"):
            info["version"] = int(vm.group("ver"))
    return info


def _read_metadata(p: Path) -> Dict[str, str]:
    """Read parquet key-value metadata; return empty dict on error."""
    try:
        import pyarrow.parquet as pq
        tbl = pq.read_table(p)
        raw = tbl.schema.metadata or {}
        return {
            k.decode(): v.decode() for k, v in raw.items()
            if isinstance(k, (bytes, bytearray))
        }
    except Exception:
        return {}


def _guess_susy_from_filename(stem: str) -> str:
    s = stem.lower()
    if "nonsusy" in s or "non_susy" in s or "non-susy" in s:
        return "nonSUSY"
    if "susy" in s:
        return "SUSY"
    return "unknown"


def _count_rows(p: Path) -> int:
    try:
        import pyarrow.parquet as pq
        return pq.ParquetFile(p).metadata.num_rows
    except Exception:
        return -1


def rebuild_catalog(
    repo_path: str | Path,
    output: Optional[str] = None,
    verbose: bool = True,
) -> Path:
    r"""
    **Description:**
    Scan *repo_path* for vault parquet files and regenerate
    ``catalog.parquet``.

    Args:
        repo_path: Path to the local vault repo checkout.
        output: Optional output path.  Defaults to
            ``<repo_path>/catalog.parquet``.
        verbose: Print progress.

    Returns:
        pathlib.Path: Path to the written catalog.
    """
    import pandas as pd

    root = Path(repo_path).resolve()
    if not root.is_dir():
        raise ValueError(f"repo_path {root} is not a directory")

    rows: List[Dict[str, Any]] = []
    for p in sorted(root.rglob("*.parquet")):
        rel = p.relative_to(root)
        # Skip top-level metadata parquets and rejected split-offs
        if p.name in RESERVED_NAMES:
            continue
        if any(part == "_rejected" for part in rel.parts):
            # Still include in the catalogue, but flag status.
            pass  # fall through; _parse_path will set status="rejected"

        info = _parse_path(root, p)
        md = _read_metadata(p)

        # Determine classification source
        has_md_classification = any(k in md for k in ("susy", "method", "nmax", "qualifiers"))
        if has_md_classification:
            info["classification_source"] = "parquet_metadata"
        else:
            info["classification_source"] = "filename_guess"

        info["susy"]       = md.get("susy") or _guess_susy_from_filename(p.stem)
        info["method"]     = md.get("method", "unknown")
        info["nmax"]       = (int(md["nmax"]) if md.get("nmax") and md["nmax"].isdigit() else None)
        info["qualifiers"] = md.get("qualifiers", "{}")
        info["committed_by"] = md.get("committed_by", "")
        info["uploaded_at"]  = md.get("uploaded_at", "")

        info["n_rows"]         = _count_rows(p)
        info["schema_version"] = int(md.get("schema_version", SCHEMA_VERSION))
        info["checksum"]       = _sha256(p)
        info["superseded_by"]  = md.get("superseded_by", "")
        info["retracted"]          = (md.get("retracted", "false").lower() == "true")
        info["retraction_reason"]  = md.get("retraction_reason", "")
        info["retracted_at"]       = md.get("retracted_at", "")

        # Fill any missing catalog columns with defaults
        for c in CATALOG_COLUMNS:
            info.setdefault(c, None)

        rows.append(info)
        if verbose:
            print(f"[catalog] {rel}  rows={info['n_rows']} "
                  f"susy={info['susy']} status={info['status']}")

    df = pd.DataFrame(rows, columns=list(CATALOG_COLUMNS))
    out = Path(output) if output else (root / "catalog.parquet")
    df.to_parquet(out, index=False)
    if verbose:
        print(f"[catalog] wrote {len(df)} rows to {out}")
    return out
