# Copyright 2026 Andreas Schachner
#
# This file is part of StringForge.
#
# StringForge is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

r"""
Read path for Vulcan: catalogue scan, lazy shard fetch, and query.

The reader builds a lightweight catalogue from parquet kv-metadata
(see :data:`stringforge.vulcan.writer.PROVENANCE_METADATA_KEY`) and
caches loaded shards in a small LRU.  Two backends are supported:

* **Local** -- the staging directory's ``synced/`` subtree on the
  current host.  No network calls, no authentication.  Used by the
  same workflow that produced the shards.
* **HuggingFace** -- the production repo published by
  :meth:`stringforge.vulcan.Vulcan.sync`.  Shard manifests are
  fetched on-demand via :func:`huggingface_hub.HfApi.list_repo_files`
  and individual parquet files via :func:`huggingface_hub.hf_hub_download`.

The reader complements :mod:`~stringforge.vulcan.writer` /
:mod:`~stringforge.vulcan.sync` -- it is the inverse traversal across
the same on-disk layout.
"""
from __future__ import annotations

import json
import os
import threading
import warnings
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Mapping, Optional, Tuple, Union

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .schema import GEOMETRY_KEYS, GEOMETRY_KEY_SENTINEL, SCHEMA_VERSION, geometry_id as _compute_geometry_id
from .writer import (
    PROVENANCE_METADATA_KEY,
    SCHEMA_VERSION_METADATA_KEY,
    SYNCED_DIRNAME,
)

#: Columns produced by :meth:`VulcanReader.catalog` (one row per
#: shard).  This is the read-side catalogue contract; downstream code
#: can rely on these columns being present whenever the catalogue
#: scan succeeds.
CATALOG_COLUMNS: Tuple[str, ...] = (
    "path_in_repo",
    "run_id",
    "geometry_id",
    "h11", "h12", "ks_id", "triang_id", "conifold_id", "cicy_id",
    "n_rows",
    "schema_version",
)

#: Default LRU cache size, in shards.
DEFAULT_CACHE_SIZE: int = 32


# ── catalogue scan helpers ────────────────────────────────────────────────


def _read_parquet_metadata(path: Path) -> dict[str, Any]:
    r"""
    Extract Vulcan kv-metadata from a parquet file without loading
    any row groups.

    Returns a dict combining the provenance blob (parsed JSON) and
    the schema-version integer; an empty dict when the parquet has no
    Vulcan kv-metadata.
    """
    pf = pq.ParquetFile(path)
    raw = pf.schema_arrow.metadata or {}
    out: dict[str, Any] = {}
    blob = raw.get(PROVENANCE_METADATA_KEY)
    if blob is not None:
        try:
            out.update(json.loads(blob.decode("utf-8")))
        except (UnicodeDecodeError, json.JSONDecodeError):
            pass
    ver = raw.get(SCHEMA_VERSION_METADATA_KEY)
    if ver is not None:
        try:
            out["schema_version"] = int(ver.decode("ascii"))
        except (UnicodeDecodeError, ValueError):
            pass
    out["n_rows"] = pf.metadata.num_rows
    return out


def _row_from_metadata(path_in_repo: str, meta: Mapping[str, Any]) -> dict[str, Any]:
    r"""Project a parquet-metadata blob into a catalogue row."""
    geometry = dict(meta.get("geometry", {}))
    row: dict[str, Any] = {
        "path_in_repo": path_in_repo,
        "run_id": meta.get("run_id"),
        "geometry_id": _compute_geometry_id(geometry) if geometry else None,
        "n_rows": int(meta.get("n_rows", 0)),
        "schema_version": int(meta.get("schema_version", 0)),
    }
    for key in GEOMETRY_KEYS:
        row[key] = int(geometry.get(key, GEOMETRY_KEY_SENTINEL))
    return row


# ── LRU shard cache ──────────────────────────────────────────────────────


class _LRUShardCache:
    r"""
    **Description:**
    Thread-safe in-memory LRU cache for loaded parquet shards.

    Modelled on :class:`stringforge.cy_io._LRUShardCache`.  Used by the
    Vulcan reader to keep repeated queries against the same shards
    fast without growing memory unbounded.

    Args:
        capacity: Maximum number of shards retained.
    """

    def __init__(self, capacity: int = DEFAULT_CACHE_SIZE) -> None:
        if capacity < 1:
            raise ValueError("LRU capacity must be >= 1")
        self.capacity = int(capacity)
        self._store: OrderedDict[str, pd.DataFrame] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[pd.DataFrame]:
        r"""
        **Description:**
        Look up a shard in the cache.

        Args:
            key: Cache key (typically the repo-relative path).

        Returns:
            Optional[pd.DataFrame]: The cached DataFrame, or ``None``
            on miss.  A hit moves the entry to the front of the LRU.
        """
        with self._lock:
            if key not in self._store:
                return None
            self._store.move_to_end(key)
            return self._store[key]

    def put(self, key: str, df: pd.DataFrame) -> None:
        r"""
        **Description:**
        Insert a shard into the cache, evicting the oldest entry if
        the capacity would be exceeded.

        Args:
            key: Cache key.
            df: DataFrame to cache.
        """
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
                self._store[key] = df
                return
            self._store[key] = df
            if len(self._store) > self.capacity:
                self._store.popitem(last=False)

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)

    def clear(self) -> None:
        r"""**Description:** Remove every cached entry."""
        with self._lock:
            self._store.clear()


# ── reader ───────────────────────────────────────────────────────────────


@dataclass
class _Source:
    r"""Internal: tagged-union view of the reader's data source."""
    local_root: Optional[Path] = None
    hf_repo: Optional[str] = None
    #: HuggingFace revision (branch, tag, or commit SHA) to pin reads
    #: to. ``None`` reads the repo's default branch (the rolling
    #: ``main``). A frozen snapshot tag (see
    #: :func:`stringforge.vulcan.snapshot.freeze_snapshot`) is passed
    #: here so a paper can read an immutable, citeable revision.
    revision: Optional[str] = None


class VulcanReader:
    r"""
    **Description:**
    Read-side counterpart to :class:`stringforge.vulcan.Vulcan`.

    Builds a Vulcan-shard catalogue by scanning either a local
    ``synced/`` tree or a HuggingFace dataset repo.  Loaded shards
    are cached in a small in-memory LRU.

    A typical local usage pattern:

    .. code-block:: python

        reader = VulcanReader.from_local(forge.staging_dir)
        cat    = reader.catalog()
        susy   = reader.query(h12=2, solver_name="newton")

    A HuggingFace usage pattern:

    .. code-block:: python

        reader = VulcanReader.from_hf("user/vacua_forge")
        run    = reader.fetch_run("2026-06-01-120000_fluxscan_...")

    Args:
        source: Internal source tag (use the factories instead).
        cache_size: LRU capacity in shards.
        token: HuggingFace read token; falls back to ``HF_TOKEN``.
            Ignored for local sources.
    """

    def __init__(
        self,
        *,
        source: _Source,
        cache_size: int = DEFAULT_CACHE_SIZE,
        token: Optional[str] = None,
    ) -> None:
        self._source = source
        self._cache = _LRUShardCache(capacity=cache_size)
        self._catalog: Optional[pd.DataFrame] = None
        self._catalog_lock = threading.Lock()
        self._token = token

    # ── factories ─────────────────────────────────────────────────

    @classmethod
    def from_local(
        cls,
        staging_dir: Union[str, Path],
        *,
        cache_size: int = DEFAULT_CACHE_SIZE,
    ) -> "VulcanReader":
        r"""
        **Description:**
        Construct a reader that scans ``staging_dir/synced/`` for
        already-committed shards.

        Args:
            staging_dir: Vulcan staging root.
            cache_size: LRU capacity.

        Returns:
            VulcanReader: A reader bound to the local synced tree.
        """
        root = Path(staging_dir).resolve() / SYNCED_DIRNAME
        return cls(source=_Source(local_root=root), cache_size=cache_size)

    @classmethod
    def from_hf(
        cls,
        repo: str,
        *,
        revision: Optional[str] = None,
        cache_size: int = DEFAULT_CACHE_SIZE,
        token: Optional[str] = None,
    ) -> "VulcanReader":
        r"""
        **Description:**
        Construct a reader that streams shards from a HuggingFace
        dataset repo on demand.

        Args:
            repo: ``"user/repo"`` on the Hub.
            revision: Optional branch, tag, or commit SHA to pin reads
                to.  Pass a frozen-snapshot tag (e.g. ``"v2026.06"``,
                produced by
                :func:`stringforge.vulcan.snapshot.freeze_snapshot`) to
                read an immutable, citeable revision; ``None`` reads the
                rolling default branch.
            cache_size: LRU capacity.
            token: HuggingFace read token; falls back to ``HF_TOKEN``.

        Returns:
            VulcanReader: A reader bound to the remote repo (and revision).
        """
        return cls(
            source=_Source(hf_repo=repo, revision=revision),
            cache_size=cache_size,
            token=token,
        )

    @property
    def revision(self) -> Optional[str]:
        r"""The pinned HuggingFace revision, or ``None`` for local / default-branch reads."""
        return self._source.revision

    # ── catalogue ─────────────────────────────────────────────────

    def catalog(self, *, refresh: bool = False, strict: bool = False) -> pd.DataFrame:
        r"""
        **Description:**
        Return a one-row-per-shard catalogue of the configured source.

        The first call scans every shard's kv-metadata and caches the
        result; subsequent calls return the cached view.  Pass
        ``refresh=True`` to force a rescan -- useful after a
        :meth:`~stringforge.vulcan.Vulcan.sync` adds new shards
        upstream.

        Shard metadata is read in parallel via a thread pool because the
        per-shard cost is dominated by I/O (opening the parquet file and
        reading its footer); ordering is irrelevant because the final
        DataFrame is sorted by ``path_in_repo``.

        Unreadable shards (corrupt file, missing footer, etc.) are
        skipped with a :class:`RuntimeWarning`; the number of skipped
        shards is exposed on ``result.attrs["n_skipped"]``.

        Shards whose embedded ``schema_version`` differs from the
        current :data:`stringforge.vulcan.schema.SCHEMA_VERSION` emit a
        :class:`RuntimeWarning` per shard.  When ``strict=True`` such a
        mismatch is promoted to :class:`ValueError`.

        Args:
            refresh: Force a fresh scan even if a cached catalogue
                exists.
            strict: When ``True``, raise :class:`ValueError` on any
                shard whose ``schema_version`` differs from
                :data:`stringforge.vulcan.schema.SCHEMA_VERSION`.

        Returns:
            pd.DataFrame: One row per shard with columns from
            :data:`CATALOG_COLUMNS`.

        Raises:
            ValueError: When ``strict=True`` and a shard's
                ``schema_version`` does not match
                :data:`stringforge.vulcan.schema.SCHEMA_VERSION`.
        """
        with self._catalog_lock:
            if self._catalog is not None and not refresh:
                return self._catalog
            rows: list[dict[str, Any]] = []
            n_skipped = 0
            shards = list(self._iter_shards())
            max_workers = min(32, (os.cpu_count() or 1) * 4)
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                future_to_path = {
                    pool.submit(_read_parquet_metadata, local_path): path_in_repo
                    for path_in_repo, local_path in shards
                }
                for future in as_completed(future_to_path):
                    path_in_repo = future_to_path[future]
                    try:
                        meta = future.result()
                    except (OSError, pa.ArrowInvalid) as exc:
                        warnings.warn(
                            f"VulcanReader.catalog: skipping unreadable shard "
                            f"{path_in_repo!r}: {type(exc).__name__}: {exc}",
                            RuntimeWarning,
                            stacklevel=2,
                        )
                        n_skipped += 1
                        continue
                    row = _row_from_metadata(path_in_repo, meta)
                    rows.append(row)
            df = pd.DataFrame(rows, columns=list(CATALOG_COLUMNS))
            if len(df) > 0:
                df = df.sort_values("path_in_repo").reset_index(drop=True)
            df.attrs["n_skipped"] = n_skipped
            # Surface forward-incompatible shards.
            mismatched = [
                (row["path_in_repo"], row["schema_version"])
                for row in rows
                if row.get("schema_version") != SCHEMA_VERSION
            ]
            if mismatched and strict:
                raise ValueError(
                    "VulcanReader.catalog: schema_version mismatch in shards "
                    f"(expected {SCHEMA_VERSION}): {mismatched}"
                )
            for path_in_repo, ver in mismatched:
                warnings.warn(
                    f"VulcanReader.catalog: shard {path_in_repo!r} has "
                    f"schema_version={ver}, expected {SCHEMA_VERSION}",
                    RuntimeWarning,
                    stacklevel=2,
                )
            self._catalog = df
            return self._catalog

    # ── query / fetch ─────────────────────────────────────────────

    def query(
        self,
        *,
        run_id: Optional[str] = None,
        geometry_id: Optional[str] = None,
        h11: Optional[int] = None,
        h12: Optional[int] = None,
        ks_id: Optional[int] = None,
        triang_id: Optional[int] = None,
        solver_name: Optional[str] = None,
        is_susy: Optional[bool] = None,
        is_isd: Optional[bool] = None,
        columns: Optional[Iterable[str]] = None,
    ) -> pd.DataFrame:
        r"""
        **Description:**
        Filter Vulcan rows by catalogue + row-level predicates.

        Catalogue predicates (``h11``, ``h12``, ``ks_id``,
        ``triang_id``, ``run_id``, ``geometry_id``) narrow the set of
        shards that need to be read.  Row-level predicates
        (``solver_name``, ``is_susy``, ``is_isd``) apply post-load.
        ``columns`` projects the result down to a subset of columns
        before returning.

        ``h11``, ``h12`` are interpreted in mirror (jaxvacua /
        ``lcs_tree``) convention, matching
        :class:`stringforge.lcs.LCSDatabase`.

        Args:
            run_id: Exact match.
            geometry_id: Exact match.
            h11, h12, ks_id, triang_id: Exact match per geometry key.
            solver_name: Row-level filter on the ``solver_name``
                column.
            is_susy: Row-level filter on the ``is_susy`` column.
            is_isd: Row-level filter on the ``is_isd`` column.
            columns: Optional column projection.

        Returns:
            pd.DataFrame: Concatenated rows from matching shards.
            Empty when no shard satisfies the catalogue predicates.
        """
        cat = self.catalog()
        for col, val in (("run_id", run_id), ("geometry_id", geometry_id),
                         ("h11", h11), ("h12", h12),
                         ("ks_id", ks_id), ("triang_id", triang_id)):
            if val is None:
                continue
            cat = cat[cat[col] == val]

        if len(cat) == 0:
            return pd.DataFrame(columns=list(columns) if columns else [])

        frames: list[pd.DataFrame] = []
        for path in cat["path_in_repo"]:
            df = self.fetch_shard(path)
            if solver_name is not None and "solver_name" in df.columns:
                df = df[df["solver_name"] == solver_name]
            if is_susy is not None and "is_susy" in df.columns:
                df = df[df["is_susy"] == is_susy]
            if is_isd is not None and "is_isd" in df.columns:
                df = df[df["is_isd"] == is_isd]
            if columns:
                missing = [c for c in columns if c not in df.columns]
                if missing:
                    raise KeyError(
                        f"requested columns not in shard {path!r}: {missing}"
                    )
                df = df[list(columns)]
            frames.append(df)

        if not frames:
            return pd.DataFrame(columns=list(columns) if columns else [])
        return pd.concat(frames, ignore_index=True)

    def fetch_run(self, run_id: str) -> pd.DataFrame:
        r"""
        **Description:**
        Load every row produced by a specific ``run_id``.

        Args:
            run_id: The identifier carried on each row of the target
                shards.

        Returns:
            pd.DataFrame: Concatenated rows.  Empty when no shard
            matches.
        """
        return self.query(run_id=run_id)

    def fetch_shard(self, path_in_repo: str) -> pd.DataFrame:
        r"""
        **Description:**
        Load (and cache) a single shard by its HF-repo-relative path.

        Args:
            path_in_repo: The path produced by
                :func:`stringforge.vulcan.schema.shard_relpath`.

        Returns:
            pd.DataFrame: The shard's rows.

        Raises:
            FileNotFoundError: When the shard is absent both locally
                and (where applicable) on the Hub.
        """
        hit = self._cache.get(path_in_repo)
        if hit is not None:
            return hit
        local = self._resolve_local_path(path_in_repo)
        df = pd.read_parquet(local)
        self._cache.put(path_in_repo, df)
        return df

    def read_columns(self, path_in_repo: str, columns: Iterable[str]) -> pd.DataFrame:
        r"""
        **Description:**
        Read only the named columns from one shard, skipping any that
        the shard does not carry.

        A columnar read used by snapshot manifest generation: tallying
        ``verifier_id`` / ``cert_status`` over a large dataset must not
        pay the cost of materialising every column.  Does **not**
        populate the LRU (it returns a column projection, not the full
        shard).

        Args:
            path_in_repo: The shard's repo-relative path.
            columns: Column names to read; missing ones are skipped.

        Returns:
            pd.DataFrame: The projection (empty-column frame if none of
            the requested columns are present).
        """
        local = self._resolve_local_path(path_in_repo)
        present = set(pq.ParquetFile(local).schema_arrow.names)
        wanted = [c for c in columns if c in present]
        if not wanted:
            n = pq.ParquetFile(local).metadata.num_rows
            return pd.DataFrame(index=range(n))
        return pq.read_table(local, columns=wanted).to_pandas()

    def clear_cache(self) -> None:
        r"""**Description:** Drop every cached shard from the in-memory LRU."""
        self._cache.clear()

    # ── private: backend dispatch ─────────────────────────────────

    def _iter_shards(self) -> Iterable[Tuple[str, Path]]:
        r"""
        Yield ``(path_in_repo, local_path)`` for every shard in the
        configured source.  ``local_path`` is downloaded if needed.
        """
        if self._source.local_root is not None:
            root = self._source.local_root
            if not root.is_dir():
                return
            for parquet in sorted(root.rglob("*.parquet")):
                # ``path_in_repo`` is reconstructed from the local
                # filename (writer.py replaces "/" with "__").
                path_in_repo = parquet.name.replace("__", "/")
                yield path_in_repo, parquet
            return

        # HF source -- list, download lazily.
        if self._source.hf_repo is None:
            return
        from huggingface_hub import HfApi, hf_hub_download  # type: ignore

        api = HfApi(token=self._token or os.environ.get("HF_TOKEN"))
        files = api.list_repo_files(
            repo_id=self._source.hf_repo, repo_type="dataset",
            revision=self._source.revision,
        )
        for path_in_repo in files:
            if not path_in_repo.endswith(".parquet"):
                continue
            local = Path(hf_hub_download(
                repo_id=self._source.hf_repo,
                filename=path_in_repo,
                repo_type="dataset",
                revision=self._source.revision,
                token=self._token or os.environ.get("HF_TOKEN"),
            ))
            yield path_in_repo, local

    def _resolve_local_path(self, path_in_repo: str) -> Path:
        r"""
        Map a repo-relative path back to a concrete on-disk file,
        downloading from HF if necessary.
        """
        if self._source.local_root is not None:
            local = self._source.local_root / path_in_repo.replace("/", "__")
            if not local.is_file():
                raise FileNotFoundError(
                    f"Vulcan shard not found locally: {local}"
                )
            return local

        if self._source.hf_repo is None:
            raise FileNotFoundError(
                "Vulcan reader has no configured source for shard "
                f"{path_in_repo!r}"
            )
        from huggingface_hub import hf_hub_download  # type: ignore
        return Path(hf_hub_download(
            repo_id=self._source.hf_repo,
            filename=path_in_repo,
            repo_type="dataset",
            revision=self._source.revision,
            token=self._token or os.environ.get("HF_TOKEN"),
        ))


__all__ = (
    "VulcanReader",
    "CATALOG_COLUMNS",
    "DEFAULT_CACHE_SIZE",
)
