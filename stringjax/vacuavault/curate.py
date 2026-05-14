r"""
Promote a community submission from ``community/`` up to the model-level
directory.

Called by the maintainer after a PR has been merged and the submission
has passed review.  Strips the ``{hf_username}_`` filename prefix and
records the original contributor in the file's parquet metadata as
``original_contributor`` so attribution survives the rename.
"""
from __future__ import annotations

import os
import re
import shutil
from pathlib import Path
from typing import Optional


def _strip_username_prefix(stem: str) -> tuple[Optional[str], str]:
    """Return (hf_username, remaining_label).  If the stem has no
    underscore, treat it as a label with no username prefix."""
    if "_" not in stem:
        return None, stem
    hfu, rest = stem.split("_", 1)
    return hfu, rest


def curate_submission(
    community_path: str | Path,
    repo_root: str | Path,
    new_label: Optional[str] = None,
    dry_run: bool = False,
    verbose: bool = True,
) -> Path:
    r"""
    **Description:**
    Promote a parquet file from a model's ``community/`` directory up
    to the model-level directory.

    Operations:

    1. Verify the file lives under ``.../{model_dir}/community/``.
    2. Compute the target path as
       ``.../{model_dir}/{new_label or stripped_stem}.parquet``.
    3. Move the file, updating its parquet key-value metadata with
       ``original_contributor`` (the stripped HF-username prefix) and
       ``status="curated"``.
    4. If the target name already exists, append a ``_v{n}`` suffix.

    Args:
        community_path: Absolute path to the file under ``community/``.
        repo_root: Absolute path to the vault repo root.  Used for
            sanity checks only.
        new_label: Optional override for the target filename stem
            (without extension, without the ``_v{n}`` suffix).
        dry_run: If True, print what would happen and return the
            projected target path without moving anything.
        verbose: Print progress.

    Returns:
        pathlib.Path: Path of the moved file (or the projected path
        when ``dry_run=True``).

    Raises:
        ValueError: If the path is not under a ``community/`` folder
            of the vault repo.
    """
    import pyarrow.parquet as pq
    import pyarrow as pa

    src = Path(community_path).resolve()
    root = Path(repo_root).resolve()
    try:
        rel = src.relative_to(root)
    except ValueError:
        raise ValueError(
            f"{src} is not under repo_root={root}"
        ) from None

    parts = rel.parts
    if "community" not in parts:
        raise ValueError(
            f"{rel} is not under a community/ directory"
        )
    ci = parts.index("community")
    model_dir = root.joinpath(*parts[:ci])

    stem = os.path.splitext(src.name)[0]
    ext  = os.path.splitext(src.name)[1]
    hfu, label_stem = _strip_username_prefix(stem)
    target_stem = new_label if new_label else label_stem
    target = model_dir / f"{target_stem}{ext}"

    # Collision → auto-version
    n = 2
    base_stem = re.sub(r"_v\d+$", "", target_stem)
    while target.exists():
        target = model_dir / f"{base_stem}_v{n}{ext}"
        n += 1

    if verbose:
        print(f"[curate] {rel} → {target.relative_to(root)}")
        if hfu:
            print(f"[curate] original_contributor = {hfu!r}")

    if dry_run:
        return target

    # Read, update metadata, write to new location
    table = pq.read_table(src)
    existing = dict(table.schema.metadata or {})
    existing[b"status"] = b"curated"
    if hfu:
        existing[b"original_contributor"] = hfu.encode()
    table = table.replace_schema_metadata(existing)
    pq.write_table(table, target)
    src.unlink()

    return target
