# ==============================================================================
# stringforge / kklt_database
#
# KKLTDatabase — a curated subset of the TDF dataset, indexed by conifold
# class.  Does not duplicate Calabi-Yau geometry data: every row links back
# to the full TDF database via the logical key
# ``(ks_id, triang_id, tdf_conifold_id)`` and the wrapped
# :class:`~stringforge.lcs_database.LCSDatabase` (``dataset="tdf"``)
# resolves physical shard coordinates on the fly.
#
# Three local catalogs (under ``<cache_dir>/kklt/``):
#
#   - ``catalog.parquet``                  (polytope-grain;
#                                           one row per ks_id;
#                                           columns include n_rigids_dual, Q,
#                                           n_coni_classes)
#   - ``conifold_class_catalog.parquet``   (one row per (ks_id, coni_class_id);
#                                           carries one_face_divisor)
#   - ``conifold_catalog.parquet``         (one row per
#                                           (ks_id, coni_class_id, coni_id);
#                                           carries (triang_id, tdf_conifold_id))
#
# Optional local run logs and solution-stage fragments are run artefacts.  Public
# designated vacua belong in the shared ``vacua_vault/kklt_vacua`` namespace,
# not in the static ``cy-database/kklt`` index.
#
# Subset criterion (applied at build time): ``n_rigids_dual > h12``.
# ``Q := h11 + h12 + 2`` is a column on ``catalog.parquet`` but not a
# build-time filter; the historically well-studied ``Q >= 100`` subset is
# recovered by ``db.query_polytopes(Q_min=100)``.
# ==============================================================================

from __future__ import annotations

import datetime as _dt
import json
import socket
import uuid
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

import numpy as np

from .cy_io import (
    HF_REPO_ID,
    _DATASET_CONFIGS,
    _require_pandas,
)
from .lcs_database import LCSDatabase, _swap_h_columns


_RUN_STATUS_VALUES = {"running", "done", "failed", "cancelled"}
_RUN_SCOPES        = {"class", "conifold"}


# KKLT solution-stage registry: stage name → tuple of valid "kind"
# sub-directories under that stage, or ``None`` if the stage has no
# sub-kind.  Used by :meth:`KKLTDatabase._kklt_stage_dir` and friends.
_KKLT_STAGES: Dict[str, Optional[tuple]] = {
    "pfv":       ("harvested", "promoted"),
    "precursor": ("derived",   "proxy"),
    "vacua":     None,
}


def _kklt_polytope_dirname(h11: int, h12: int, ks_id: int) -> str:
    r"""Return ``h11_<n>_h12_<m>_ksid_<id>`` — the polytope-level
    directory name under a KKLT stage.  ``(h11, h12)`` is required to
    fully disambiguate ``ks_id``."""
    return f"h11_{int(h11)}_h12_{int(h12)}_ksid_{int(ks_id)}"


def _kklt_conifold_dirname(coni_class_id: int, coni_id: int) -> str:
    r"""Return ``coniclass_<c>_coni_<k>`` — the conifold-level
    directory name under a KKLT polytope directory."""
    return f"coniclass_{int(coni_class_id)}_coni_{int(coni_id)}"


def _utcnow() -> _dt.datetime:
    r"""Return the current UTC time as a ``datetime``."""
    return _dt.datetime.now(_dt.timezone.utc)


def _normalise_tags_arg(tags: Optional[Union[str, List[str]]]) -> set:
    r"""Normalise a tag filter argument to a set of strings."""
    if tags is None:
        return set()
    if isinstance(tags, str):
        return {tags}
    return {str(tag) for tag in tags}


def _tags_from_cell(value: Any) -> set:
    r"""Return a set of tag labels from a scalar or list-like parquet cell."""
    if value is None:
        return set()
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        value = value.tolist()
    if isinstance(value, str):
        raw = value.replace(";", ",").replace("|", ",").split(",")
        return {item.strip() for item in raw if item.strip()}
    if isinstance(value, (list, tuple, set)):
        return {
            str(item).strip()
            for item in value
            if item is not None and str(item).strip()
        }
    try:
        if bool(np.isnan(value)):
            return set()
    except (TypeError, ValueError):
        pass
    return {str(value).strip()} if str(value).strip() else set()


def _apply_tag_filters(
    frame: Any,
    mask: Any,
    *,
    tags_include: Optional[Union[str, List[str]]] = None,
    tags_exclude: Optional[Union[str, List[str]]] = None,
    tag_column: str = "tags",
    tag_mode: str = "all",
) -> Any:
    r"""Apply include/exclude tag filters to a boolean pandas/numpy mask."""
    include = _normalise_tags_arg(tags_include)
    exclude = _normalise_tags_arg(tags_exclude)
    if not include and not exclude:
        return mask
    if tag_mode not in {"all", "any"}:
        raise ValueError("tag_mode must be 'all' or 'any'.")
    if tag_column not in frame.columns:
        message = (
            f"Tag column '{tag_column}' not found in KKLT catalog; "
            "include-tag filter matched no rows."
            if include
            else
            f"Tag column '{tag_column}' not found in KKLT catalog; "
            "tag filter ignored."
        )
        warnings.warn(message, stacklevel=3)
        if include:
            return mask & np.zeros(len(frame), dtype=bool)
        return mask

    row_tags = frame[tag_column].map(_tags_from_cell)
    if include:
        if tag_mode == "all":
            mask &= row_tags.map(lambda tags: include.issubset(tags)).values
        else:
            mask &= row_tags.map(lambda tags: bool(include & tags)).values
    if exclude:
        mask &= row_tags.map(lambda tags: exclude.isdisjoint(tags)).values
    return mask


@contextmanager
def _file_lock(path: Path):
    r"""Advisory exclusive flock on a sibling lockfile.

    Used to serialise concurrent appends to ``run_log.parquet`` from
    parallel cluster workers writing into a shared local cache.  Falls
    back to a no-op on platforms without :mod:`fcntl` (Windows).
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


# Column schema for the append-only run log (``run_log.parquet``).
_RUN_LOG_COLUMNS = [
    "run_id",        "scope",          "ks_id",        "coni_class_id",
    "coni_id",       "attempt_index",  "status",
    "started_at",    "finished_at",
    "job_id",        "host",           "worker",       "task",
    "n_solutions",   "n_attempted",
    "payload_uri",   "payload_hash",
    "error_summary", "notes",
]


# ----------------------------------------------------------------------------
# Module-level constants and env-var helpers
# ----------------------------------------------------------------------------

DEFAULT_KKLT_HF_REPO: str = HF_REPO_ID


def _resolve_kklt_hf_repo() -> str:
    r"""
    **Description:**
    Return the HuggingFace dataset repo ID for the ``kklt`` sub-dataset.

    KKLT index data now live inside the general-purpose
    ``aschachner/cy-database`` repository.  This helper is kept for
    compatibility with older code paths and therefore resolves to the
    normal ``STRINGFORGE_HF_REPO`` value, not to a separate KKLT
    repository.

    Returns:
        str: ``"user/repo"`` HF Hub identifier.
    """
    return DEFAULT_KKLT_HF_REPO


# ----------------------------------------------------------------------------
# KKLTDatabase
# ----------------------------------------------------------------------------


class KKLTDatabase(LCSDatabase):
    r"""
    **Description:**
    Advanced KKLT index interface: a curated subset of
    :class:`~stringforge.lcs_database.LCSDatabase` (``dataset="tdf"``).

    This is specialised infrastructure for curated KKLT-style searches, not the default user entry point.
    The KKLT database does **not** duplicate Calabi-Yau geometry data.
    Every row in the local catalogs carries a *logical* link
    ``(ks_id, triang_id, tdf_conifold_id)`` into the full TDF dataset;
    physical shard coordinates are resolved on demand by the wrapped TDF
    database held on :attr:`tdf`.

    Models are indexed by ``(ks_id, coni_class_id, coni_id)``.  Conifolds
    sharing the same *one-face divisor* on a given polytope form a single
    ``coni_class``; this grouping cuts across triangulations of the same
    polytope.

    Three local catalogs live under ``<cache_dir>/kklt/``:

    - ``catalog.parquet`` (polytope-grain): one row per ``ks_id`` with
      ``h11``, ``h12``, ``chi``, ``n_rigids_dual``, ``Q``,
      ``n_coni_classes``.  All KKLT polytopes satisfy
      ``n_rigids_dual > h12``; the historically well-studied subset
      ``Q >= 100`` is a downstream DataFrame filter, not a build-time cut.
    - ``conifold_class_catalog.parquet``: one row per
      ``(ks_id, coni_class_id)``.
    - ``conifold_catalog.parquet``: one row per
      ``(ks_id, coni_class_id, coni_id)``, carrying the logical TDF link.

    Optional local ``run_log.parquet`` records cluster runs
    (``scope="class"`` or ``scope="conifold"``) for private/local
    run-tracking.  Public designated vacuum records belong in the
    shared ``vacua_vault/kklt_vacua`` area.

    A *TDF-compat fingerprint* (``tdf_schema_version``,
    ``tdf_catalog_sha256``) is stored in ``schema.json`` at build time
    and re-checked on every instantiation.  A mismatch raises a
    :class:`UserWarning` pointing the user at :meth:`rebuild_links`.

    Example usage::

        from stringforge.kklt_database import KKLTDatabase
        db = KKLTDatabase()
        polys   = db.query_polytopes(Q_min=100)
        classes = db.query_classes(ks_id=int(polys.iloc[0]["ks_id"]))
        cfs     = db.query_conifolds(
            ks_id=int(polys.iloc[0]["ks_id"]),
            coni_class_id=int(classes.iloc[0]["coni_class_id"]),
        )
    """

    # Consumed by :meth:`CYDatabase.from_local` so that
    # ``KKLTDatabase.from_local(path)`` infers ``dataset="kklt"``.
    _DATASET: str = "kklt"

    @classmethod
    def from_local(  # type: ignore[override]
        cls,
        path: Union[str, Path],
        dataset: Optional[str] = None,
        shard_cache_size: int = 32,
        *,
        tdf_db: Optional[LCSDatabase] = None,
        tdf_cache_dir: Optional[str] = None,
    ) -> "KKLTDatabase":
        r"""
        **Description:**
        Create an offline :class:`KKLTDatabase` from a local build,
        allowing the wrapped TDF database to be supplied explicitly.

        Args:
            path (str | Path): KKLT cache root (the directory
                *containing* ``kklt/``, or the ``kklt/`` directory
                itself).
            dataset (str | None): Sub-dataset id; must be
                ``"kklt"`` if supplied.  Defaults to ``cls._DATASET``.
            shard_cache_size (int): LRU shard cache size for KKLT
                (does not affect the wrapped TDF).
            tdf_db (LCSDatabase | None): Pre-built TDF database.
                Recommended when your TDF cache is not a sibling of
                the KKLT cache.
            tdf_cache_dir (str | None): If ``tdf_db`` is not supplied
                but the TDF cache is in a non-default location, set
                ``tdf_cache_dir`` and a fresh offline
                ``LCSDatabase(dataset="tdf", cache_dir=tdf_cache_dir,
                offline=True)`` is built and wrapped.

        Returns:
            KKLTDatabase: Offline instance with the right wrapped TDF.

        Example::

            db = KKLTDatabase.from_local(".", tdf_cache_dir="./cy-database/")
            # or
            tdf = LCSDatabase(dataset="tdf", cache_dir="./cy-database/", offline=True)
            db  = KKLTDatabase.from_local(".", tdf_db=tdf)
        """
        if dataset is None:
            dataset = cls._DATASET
        if dataset != cls._DATASET:
            raise ValueError(
                f"KKLTDatabase.from_local requires dataset={cls._DATASET!r}; "
                f"got {dataset!r}."
            )
        path = Path(path)
        if path.name == dataset and (path / "catalog.parquet").exists():
            path = path.parent
        if tdf_db is None and tdf_cache_dir is not None:
            tdf_db = LCSDatabase(
                dataset    = "tdf",
                cache_dir  = str(tdf_cache_dir),
                offline    = True,
            )
        return cls(
            tdf_db           = tdf_db,
            cache_dir        = str(path),
            offline          = True,
            shard_cache_size = shard_cache_size,
        )

    def __init__(
        self,
        tdf_db: Optional[LCSDatabase] = None,
        hf_repo: Optional[str] = None,
        cache_dir: Optional[str] = None,
        offline: bool = False,
        cache_mode: str = "persistent",
        shard_cache_size: int = 32,
        dataset: Optional[str] = None,
    ) -> None:
        r"""Initialise a :class:`KKLTDatabase` instance.

        Args:
            tdf_db (LCSDatabase | None): Pre-built TDF
                :class:`~stringforge.lcs_database.LCSDatabase` to delegate
                shard fetches and model loads to.  If ``None``, a fresh
                ``LCSDatabase(dataset="tdf", ...)`` is constructed lazily
                on first use with matching ``offline`` / ``cache_mode`` /
                ``shard_cache_size`` settings and the same parent
                ``cache_dir``.
            hf_repo (str | None): HuggingFace repository ID for the
                ``kklt`` parquet shards.  Defaults to the general
                cy-database repository.
            cache_dir (str | None): Local cache directory.  Defaults to
                the global ``stringforge.data_dir``.
            offline (bool): If ``True``, no network access is attempted.
                The wrapped TDF database inherits this setting.
            cache_mode (str): Shard cache mode.  See
                :class:`~stringforge.cy_io.CYDatabase`.
            shard_cache_size (int): LRU shard cache size.  See
                :class:`~stringforge.cy_io.CYDatabase`.
            dataset (str | None): Accepted for compatibility with
                :meth:`~stringforge.cy_io.CYDatabase.from_local`; must
                equal ``"kklt"`` if supplied.

        Raises:
            ValueError: If ``dataset`` is supplied and is not
                ``"kklt"``.
        """
        if dataset is not None and dataset != self._DATASET:
            raise ValueError(
                f"KKLTDatabase requires dataset={self._DATASET!r}; "
                f"got {dataset!r}"
            )

        super().__init__(
            dataset=self._DATASET,
            hf_repo=hf_repo or _resolve_kklt_hf_repo(),
            cache_dir=cache_dir,
            offline=offline,
            shard_cache_size=shard_cache_size,
            cache_mode=cache_mode,
        )

        # Wrapped TDF database — owns the raw geometry shards.  Built
        # lazily on first access via :attr:`tdf` so a user can construct
        # a KKLTDatabase pointing at a local build without paying the
        # TDF-init cost upfront.
        self._tdf_seed: Optional[LCSDatabase]  = tdf_db
        self._tdf_built: Optional[LCSDatabase] = None

        # KKLT-specific catalogs (loaded lazily by Phase-B _ensure_*).
        self._class_catalog: Any = None
        self._run_log: Any       = None

        # Fingerprint check (no-op when schema.json carries no TDF
        # fingerprint, e.g. for a freshly bootstrapped local build).
        self._check_tdf_compat()

    # ------------------------------------------------------------------
    # Wrapped TDF database (lazy)
    # ------------------------------------------------------------------

    @property
    def tdf(self) -> LCSDatabase:
        r"""
        **Description:**
        The wrapped TDF :class:`~stringforge.lcs_database.LCSDatabase`
        instance.  Built lazily on first access from the seed passed to
        ``__init__`` (if any), or freshly with matching cache / offline
        settings, using the parent of ``self.cache_dir`` as the TDF cache
        root.
        """
        if self._tdf_built is not None:
            return self._tdf_built
        if self._tdf_seed is not None:
            self._tdf_built = self._tdf_seed
            return self._tdf_built
        tdf_cache = str(self.cache_dir.parent)
        warnings.warn(
            "KKLTDatabase: no `tdf_db` or `tdf_cache_dir` was provided; "
            f"defaulting the wrapped TDF database to cache_dir={tdf_cache!r} "
            "(the parent of this KKLT cache). Pass `tdf_cache_dir=` to "
            "`from_local` to make this explicit.",
            UserWarning,
            stacklevel=2,
        )
        self._tdf_built = LCSDatabase(
            dataset="tdf",
            cache_dir=tdf_cache,
            offline=self.offline,
            cache_mode=self.cache_mode,
        )
        return self._tdf_built

    # ------------------------------------------------------------------
    # TDF-compat fingerprint check
    # ------------------------------------------------------------------

    def _check_tdf_compat(self) -> None:
        r"""
        **Description:**
        Verify that the local KKLT cache was built against the same TDF
        version that the wrapped :attr:`tdf` instance is currently
        serving.

        Reads two optional fields from ``<cache_dir>/schema.json``:

        - ``tdf_schema_version`` (int) — TDF schema version at KKLT build
          time;
        - ``tdf_catalog_sha256`` (hex string) — sha256 over the TDF
          catalog columns KKLT depends on.

        If either field is absent the check is silently skipped (so a
        freshly bootstrapped local build does not trip the warning).  On
        mismatch a :class:`UserWarning` is emitted pointing the user at
        :meth:`rebuild_links`.

        This method is called from :meth:`__init__` and is also exposed
        so callers can re-run the check after replacing :attr:`tdf`.
        """
        import json

        path = self.cache_dir / "schema.json"
        if not path.exists():
            # No local schema.json — the cache has not been populated yet.
            return
        try:
            with open(path) as f:
                stored = json.load(f)
        except (OSError, ValueError):
            return

        expected_version = stored.get("tdf_schema_version")
        expected_hash    = stored.get("tdf_catalog_sha256")
        if expected_version is None and expected_hash is None:
            # No fingerprint recorded — pre-fingerprint local build.
            return

        # Avoid forcing a TDF catalog download just to run the check; if
        # the wrapped TDF cannot be inspected cheaply, skip silently.
        try:
            current_version = _read_tdf_schema_version(self.tdf)
            current_hash    = _compute_tdf_fingerprint(self.tdf)
        except Exception:
            return

        if (expected_version is not None and current_version != expected_version) \
           or (expected_hash is not None and current_hash != expected_hash):
            warnings.warn(
                "KKLTDatabase: TDF fingerprint mismatch.\n"
                f"  Built against:  schema v{expected_version!s}, "
                f"sha256={(expected_hash or '')[:12]}...\n"
                f"  Current TDF:    schema v{current_version!s}, "
                f"sha256={(current_hash  or '')[:12]}...\n"
                "Logical TDF keys may no longer resolve.  "
                "Run db.rebuild_links() to refresh.",
                stacklevel=2,
            )

    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        r"""String representation."""
        status = "catalog loaded" if self._catalog is not None else "catalog not yet loaded"
        return (
            f"KKLTDatabase(dataset='{self.dataset}', "
            f"hf_repo='{self.hf_repo}', {status})"
        )

    # ------------------------------------------------------------------
    # Catalog ensure-loaders
    # ------------------------------------------------------------------

    def _ensure_class_catalog(self) -> None:
        r"""
        **Description:**
        Download ``conifold_class_catalog.parquet`` if not already cached,
        then load it into :attr:`_class_catalog`.

        One row per ``(ks_id, coni_class_id)`` with columns
        ``ks_id``, ``coni_class_id``, ``h11``, ``h12``, ``one_face_divisor``,
        ``n_conifolds_in_class``.

        Raises:
            FileNotFoundError: If ``offline=True`` and the file is not in
                the local cache.
        """
        if self._class_catalog is not None:
            return

        pd = _require_pandas()
        path = self.cache_dir / "conifold_class_catalog.parquet"

        if not path.exists():
            if self.offline:
                raise FileNotFoundError(
                    f"Conifold-class catalog not found in cache ({path}) and "
                    "offline=True.  Run with offline=False first to download "
                    "it, or rebuild the database to generate "
                    "conifold_class_catalog.parquet."
                )
            self._download_file(
                f"{self.dataset}/conifold_class_catalog.parquet", path
            )

        self._class_catalog = pd.read_parquet(path)

    def _ensure_run_log(self) -> None:
        r"""
        **Description:**
        Load ``run_log.parquet`` into :attr:`_run_log`.

        The run log is append-only and stores one row per
        :meth:`start_run` / :meth:`finish_run` / :meth:`cancel_run`
        invocation; see :data:`_RUN_LOG_COLUMNS` for the schema.

        If the file is missing the log is initialised in memory as an
        empty DataFrame.  This is the standard state for a freshly-built
        database (the build script writes an empty parquet, but the file
        may be absent if the user has never invoked run-tracking).
        """
        if self._run_log is not None:
            return

        pd = _require_pandas()
        path = self.cache_dir / "run_log.parquet"

        if path.exists():
            self._run_log = pd.read_parquet(path)
            return

        if not self.offline:
            # Try a best-effort download; the run log may not exist on
            # the Hub yet for a freshly-published dataset.
            try:
                self._download_file(f"{self.dataset}/run_log.parquet", path)
                self._run_log = pd.read_parquet(path)
                return
            except RuntimeError:
                pass

        # Fall back to an empty in-memory log.
        self._run_log = pd.DataFrame(columns=_RUN_LOG_COLUMNS)

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def info(self) -> None:  # type: ignore[override]
        r"""Print a summary of the KKLT database.

        Includes the number of polytopes, conifold classes, and
        individual conifolds, plus the Hodge-number ranges (in mirror
        convention) and tadpole range.
        """
        self._ensure_catalog()
        self._ensure_class_catalog()
        self._ensure_conifold_catalog()
        cat       = self._catalog
        class_cat = self._class_catalog
        cf        = self._conifold_catalog

        n_polys   = len(cat)
        n_classes = len(class_cat)
        n_cfs     = len(cf)

        h11_min, h11_max = int(cat["h11"].min()), int(cat["h11"].max())
        h12_min, h12_max = int(cat["h12"].min()), int(cat["h12"].max())
        Q_min, Q_max = (
            (int(cat["Q"].min()), int(cat["Q"].max()))
            if "Q" in cat.columns and len(cat)
            else (None, None)
        )

        print(f"KKLTDatabase — sub-dataset: '{self.dataset}' (mirror convention)")
        print(f"  Polytopes              : {n_polys:,}")
        print(f"  Conifold classes       : {n_classes:,}")
        print(f"  Conifolds              : {n_cfs:,}")
        # Mirror convention: catalog h12 reported as h11, and vice versa.
        print(f"  h11 range (mirror)     : {h12_min} - {h12_max}")
        print(f"  h12 range (mirror)     : {h11_min} - {h11_max}")
        if Q_min is not None:
            print(f"  Q range                : {Q_min} - {Q_max}")
        print(f"  Cache dir              : {self.cache_dir}")
        print(f"  Wrapped TDF hf_repo    : {self.tdf.hf_repo}")

    def query_polytopes(
        self,
        h11: Optional[int] = None,
        h12: Optional[int] = None,
        Q_min: Optional[int] = None,
        Q_max: Optional[int] = None,
        n_rigids_dual_min: Optional[int] = None,
        n_coni_classes_min: Optional[int] = None,
        n_coni_classes_max: Optional[int] = None,
        tags_include: Optional[Union[str, List[str]]] = None,
        tags_exclude: Optional[Union[str, List[str]]] = None,
        tag_mode: str = "all",
        **filters: Any,
    ) -> Any:
        r"""
        **Description:**
        Filter the polytope-grain :attr:`_catalog` and return matching
        rows as a ``pandas.DataFrame``.  Operates entirely on the
        in-memory catalog; no geometry data is fetched.

        Hodge-number arguments are interpreted in *mirror convention*
        (i.e. ``h12`` here = catalog ``h11``), matching the rest of the
        :class:`~stringforge.lcs_database.LCSDatabase` surface.

        Args:
            h11 (int | None): Mirror :math:`h^{1,1}` (= catalog
                :math:`h^{1,2}`).
            h12 (int | None): Mirror :math:`h^{1,2}` (= catalog
                :math:`h^{1,1}`).
            Q_min (int | None): Lower bound on the D3 tadpole
                :math:`Q = h^{1,1} + h^{1,2} + 2`.
            Q_max (int | None): Upper bound on the D3 tadpole.
            n_rigids_dual_min (int | None): Minimum number of rigid
                divisors on the dual side.  (All KKLT polytopes already
                satisfy ``n_rigids_dual > h12``.)
            n_coni_classes_min (int | None): Minimum number of conifold
                classes on the polytope.
            n_coni_classes_max (int | None): Maximum number of conifold
                classes on the polytope.
            tags_include (str | list[str] | None): Require rows to carry
                these tag labels.  Tags are read from the ``tags`` column.
            tags_exclude (str | list[str] | None): Reject rows carrying
                any of these tag labels.
            tag_mode (str): ``"all"`` requires every include tag;
                ``"any"`` requires at least one include tag.
            **filters: Additional exact-match filters on catalog columns.

        Returns:
            pandas.DataFrame: Matching polytope rows, with ``h11`` and
            ``h12`` columns relabelled to mirror convention.  The
            catalog ``chi = 2(h11_cat - h12_cat)`` column is dropped
            because its sign is inverted under the mirror swap;
            recompute as ``chi = 2 * (h11 - h12)`` from the mirrored
            columns if needed.
        """
        self._ensure_catalog()
        cat = self._catalog
        mask = np.ones(len(cat), dtype=bool)

        # Mirror swap at the boundary.
        if h11 is not None:
            mask &= (cat["h12"] == h11).values
        if h12 is not None:
            mask &= (cat["h11"] == h12).values
        if Q_min is not None and "Q" in cat.columns:
            mask &= (cat["Q"] >= Q_min).values
        if Q_max is not None and "Q" in cat.columns:
            mask &= (cat["Q"] <= Q_max).values
        if n_rigids_dual_min is not None and "n_rigids_dual" in cat.columns:
            mask &= (cat["n_rigids_dual"] >= n_rigids_dual_min).values
        if n_coni_classes_min is not None and "n_coni_classes" in cat.columns:
            mask &= (cat["n_coni_classes"] >= n_coni_classes_min).values
        if n_coni_classes_max is not None and "n_coni_classes" in cat.columns:
            mask &= (cat["n_coni_classes"] <= n_coni_classes_max).values
        for col, val in filters.items():
            if col in cat.columns:
                mask &= (cat[col] == val).values
            else:
                warnings.warn(
                    f"Column '{col}' not found in KKLT polytope catalog; "
                    "filter ignored."
                )

        mask = _apply_tag_filters(
            cat,
            mask,
            tags_include=tags_include,
            tags_exclude=tags_exclude,
            tag_mode=tag_mode,
        )
        out = _swap_h_columns(cat[mask].reset_index(drop=True))
        return out.drop(columns=["chi"], errors="ignore")

    def query(  # type: ignore[override]
        self,
        h11: Optional[int] = None,
        h12: Optional[int] = None,
        **filters: Any,
    ) -> Any:
        r"""Alias for :meth:`query_polytopes`.

        Inherited :class:`~stringforge.lcs_database.LCSDatabase` filters
        (``has_conifolds``, ``has_gv``, ``D3_tadpole_max``,
        ``n_conifolds_min``, ``n_conifolds_max``) are silently dropped:
        the KKLT polytope-grain catalog does not carry the columns those
        filters rely on.  Use :meth:`query_conifolds` or
        :meth:`query_classes` for per-conifold / per-class views.
        """
        drop_keys = {
            "has_conifolds", "n_conifolds_min", "n_conifolds_max",
            "has_gv", "D3_tadpole_max",
        }
        forwarded = {k: v for k, v in filters.items() if k not in drop_keys}
        return self.query_polytopes(h11=h11, h12=h12, **forwarded)

    def query_classes(
        self,
        ks_id: Optional[int] = None,
        coni_class_id: Optional[int] = None,
        h11: Optional[int] = None,
        h12: Optional[int] = None,
        n_conifolds_in_class_min: Optional[int] = None,
        n_conifolds_in_class_max: Optional[int] = None,
        tags_include: Optional[Union[str, List[str]]] = None,
        tags_exclude: Optional[Union[str, List[str]]] = None,
        tag_mode: str = "all",
        **filters: Any,
    ) -> Any:
        r"""
        **Description:**
        Filter the per-class catalog and return matching rows.

        One row per ``(ks_id, coni_class_id)`` — i.e. per conifold class
        on a given polytope.  Hodge numbers are in mirror convention.

        Args:
            ks_id (int | None): Filter by polytope ID.
            coni_class_id (int | None): Filter by conifold-class ID.
            h11 (int | None): Mirror :math:`h^{1,1}` filter.
            h12 (int | None): Mirror :math:`h^{1,2}` filter.
            n_conifolds_in_class_min (int | None): Minimum number of
                individual conifolds in the class.
            n_conifolds_in_class_max (int | None): Maximum number.
            tags_include (str | list[str] | None): Require rows to carry
                these tag labels.  Tags are read from the ``tags`` column.
            tags_exclude (str | list[str] | None): Reject rows carrying
                any of these tag labels.
            tag_mode (str): ``"all"`` requires every include tag;
                ``"any"`` requires at least one include tag.
            **filters: Additional exact-match filters.

        Returns:
            pandas.DataFrame: Matching class rows.
        """
        self._ensure_class_catalog()
        cat = self._class_catalog
        mask = np.ones(len(cat), dtype=bool)

        if ks_id is not None:
            mask &= (cat["ks_id"] == ks_id).values
        if coni_class_id is not None:
            mask &= (cat["coni_class_id"] == coni_class_id).values
        if h11 is not None and "h12" in cat.columns:
            mask &= (cat["h12"] == h11).values
        if h12 is not None and "h11" in cat.columns:
            mask &= (cat["h11"] == h12).values
        if n_conifolds_in_class_min is not None \
                and "n_conifolds_in_class" in cat.columns:
            mask &= (cat["n_conifolds_in_class"] >= n_conifolds_in_class_min).values
        if n_conifolds_in_class_max is not None \
                and "n_conifolds_in_class" in cat.columns:
            mask &= (cat["n_conifolds_in_class"] <= n_conifolds_in_class_max).values
        for col, val in filters.items():
            if col in cat.columns:
                mask &= (cat[col] == val).values
            else:
                warnings.warn(
                    f"Column '{col}' not found in KKLT class catalog; "
                    "filter ignored."
                )

        mask = _apply_tag_filters(
            cat,
            mask,
            tags_include=tags_include,
            tags_exclude=tags_exclude,
            tag_mode=tag_mode,
        )
        df = cat[mask].reset_index(drop=True)
        if "h11" in df.columns and "h12" in df.columns:
            df = _swap_h_columns(df)
        return df.drop(columns=["chi"], errors="ignore")

    def query_conifolds(  # type: ignore[override]
        self,
        ks_id: Optional[int] = None,
        coni_class_id: Optional[int] = None,
        coni_id: Optional[int] = None,
        triang_id: Optional[int] = None,
        h11: Optional[int] = None,
        h12: Optional[int] = None,
        tdf_status: Optional[str] = None,
        tags_include: Optional[Union[str, List[str]]] = None,
        tags_exclude: Optional[Union[str, List[str]]] = None,
        tag_mode: str = "all",
        **filters: Any,
    ) -> Any:
        r"""
        **Description:**
        Filter the per-conifold catalog (one row per
        ``(ks_id, coni_class_id, coni_id)``) and return matching rows.

        Each row carries the *logical* TDF link
        ``(ks_id, triang_id, tdf_conifold_id)`` — pass that triple to
        :attr:`tdf` to fetch the underlying geometry.

        Args:
            ks_id (int | None): Filter by polytope ID.
            coni_class_id (int | None): Filter by conifold-class ID.
            coni_id (int | None): Filter by conifold ID within the class.
            triang_id (int | None): Filter by linked TDF triangulation ID.
            h11 (int | None): Mirror :math:`h^{1,1}` filter.
            h12 (int | None): Mirror :math:`h^{1,2}` filter.
            tdf_status (str | None): Filter by TDF-link status
                (``"ok"`` or ``"orphaned"``).  Populated by
                :meth:`rebuild_links` after a TDF rebuild.
            tags_include (str | list[str] | None): Require rows to carry
                these tag labels.  Tags are read from the ``tags`` column.
            tags_exclude (str | list[str] | None): Reject rows carrying
                any of these tag labels.
            tag_mode (str): ``"all"`` requires every include tag;
                ``"any"`` requires at least one include tag.
            **filters: Additional exact-match filters.

        Returns:
            pandas.DataFrame: Matching conifold rows, with h-columns
            relabelled to mirror convention when present.
        """
        self._ensure_conifold_catalog()
        cat = self._conifold_catalog
        mask = np.ones(len(cat), dtype=bool)

        if ks_id is not None:
            mask &= (cat["ks_id"] == ks_id).values
        if coni_class_id is not None and "coni_class_id" in cat.columns:
            mask &= (cat["coni_class_id"] == coni_class_id).values
        if coni_id is not None and "coni_id" in cat.columns:
            mask &= (cat["coni_id"] == coni_id).values
        if triang_id is not None and "triang_id" in cat.columns:
            mask &= (cat["triang_id"] == triang_id).values
        if h11 is not None and "h12" in cat.columns:
            mask &= (cat["h12"] == h11).values
        if h12 is not None and "h11" in cat.columns:
            mask &= (cat["h11"] == h12).values
        if tdf_status is not None and "tdf_status" in cat.columns:
            mask &= (cat["tdf_status"] == tdf_status).values
        for col, val in filters.items():
            if col in cat.columns:
                mask &= (cat[col] == val).values
            else:
                warnings.warn(
                    f"Column '{col}' not found in KKLT conifold catalog; "
                    "filter ignored."
                )

        mask = _apply_tag_filters(
            cat,
            mask,
            tags_include=tags_include,
            tags_exclude=tags_exclude,
            tag_mode=tag_mode,
        )
        df = cat[mask].reset_index(drop=True)
        if "h11" in df.columns and "h12" in df.columns:
            df = _swap_h_columns(df)
        return df.drop(columns=["chi"], errors="ignore")

    # ------------------------------------------------------------------
    # load / load_model — delegation to the wrapped TDF database
    # ------------------------------------------------------------------

    def _resolve_kklt_row(
        self,
        h11: int,
        ks_id: int,
        coni_class_id: int,
        coni_id: int,
        *,
        h12: Optional[int] = None,
    ) -> Any:
        r"""Filter the conifold catalog by the KKLT primary key and
        return the (unique) matching row as a :class:`pandas.Series`.

        **All h11/h12 args are in MIRROR convention** (jaxvacua /
        :class:`LCSDatabase` convention), matching the rest of the
        ``KKLTDatabase`` public API.  Internally the on-disk catalog
        stores h11/h12 in catalog convention; this method swaps at the
        boundary (``mirror h11 == catalog h12``, ``mirror h12 ==
        catalog h11``) before filtering.

        ``ks_id`` is only unique within a fixed ``(h11, h12)``, so
        ``h11`` is required and ``h12`` is recommended for full
        disambiguation.  When ``h12`` is omitted, the filter uses
        ``(h11, ks_id, coni_class_id, coni_id)`` alone and raises
        :class:`ValueError` if more than one row matches.

        Raises:
            KeyError:  No matching row.
            ValueError: Multiple matches when ``h12`` was not supplied,
                or zero/multiple matches even with ``h12`` given.
        """
        self._ensure_conifold_catalog()
        cf = self._conifold_catalog
        # Mirror → catalog swap.  cf columns are catalog convention
        # (cf["h11"] is small; cf["h12"] is large) but the user passes
        # mirror values (h11 large, h12 small).
        mask = (
            (cf["h12"]            == h11)             # mirror h11 ↔ catalog h12
            & (cf["ks_id"]         == ks_id)
            & (cf["coni_class_id"] == coni_class_id)
            & (cf["coni_id"]       == coni_id)
        )
        if h12 is not None:
            mask = mask & (cf["h11"] == h12)          # mirror h12 ↔ catalog h11
        match = cf[mask]
        if len(match) == 0:
            key_repr = (
                f"(h11={h11}, h12={h12}, ks_id={ks_id}, "
                f"coni_class_id={coni_class_id}, coni_id={coni_id})"
            )
            raise KeyError(f"No KKLT conifold for {key_repr}.")
        if len(match) > 1:
            if h12 is None:
                # ``match["h11"]`` is catalog h11 == mirror h12; show
                # the user the mirror values they care about.
                seen_mirror_h12 = sorted(set(int(x) for x in match["h11"]))
                raise ValueError(
                    f"Ambiguous KKLT lookup: {len(match)} rows match "
                    f"(h11={h11}, ks_id={ks_id}, "
                    f"coni_class_id={coni_class_id}, coni_id={coni_id}) "
                    f"across mirror h12 ∈ {seen_mirror_h12}.  "
                    "Pass h12= to disambiguate."
                )
            raise ValueError(
                f"Ambiguous KKLT lookup: {len(match)} rows match "
                f"(h11={h11}, h12={h12}, ks_id={ks_id}, "
                f"coni_class_id={coni_class_id}, coni_id={coni_id}) "
                "— the catalog has duplicate rows for this key."
            )
        return match.iloc[0]

    def _resolve_kklt_to_tdf(
        self,
        h11: int,
        ks_id: int,
        coni_class_id: int,
        coni_id: int,
        *,
        h12: Optional[int] = None,
    ) -> Dict[str, int]:
        r"""Resolve a KKLT key to its logical TDF coordinates.

        Args:
            h11 (int): :math:`h^{1,1}` in **mirror convention**
                (matches :class:`LCSDatabase` and the rest of the KKLT
                public API).  Required — ``ks_id`` is only unique
                within a fixed ``(h11, h12)``.
            ks_id (int): Polytope ID.
            coni_class_id (int): Conifold-class ID.
            coni_id (int): Conifold ID within the class.
            h12 (int | None): Optional :math:`h^{1,2}` (mirror).
                Required only when the
                ``(h11, ks_id, coni_class_id, coni_id)`` quad alone is
                ambiguous; in that case :meth:`_resolve_kklt_row` raises
                a ``ValueError`` advising you to pass it.

        Returns:
            Dict[str, int]: ``{"ks_id", "triang_id", "tdf_conifold_id",
            "h11_mirror", "h12_mirror", "h11_cat", "h12_cat"}`` — the
            row from the local ``conifold_catalog.parquet`` reduced to
            the columns relevant for a TDF lookup.  Both mirror and
            catalog Hodge numbers are returned so callers can pick the
            convention they need.

        Raises:
            KeyError: No matching row in the conifold catalog.
            ValueError: Row exists but ``tdf_status="orphaned"`` (the TDF
                model it pointed at is gone — see :meth:`rebuild_links`),
                or the key is ambiguous and ``h12`` was not supplied.
        """
        row = self._resolve_kklt_row(
            h11, ks_id, coni_class_id, coni_id, h12=h12,
        )
        # cf row stores catalog convention.  Compute the mirror values
        # by swapping so callers have both available.
        cat_h11 = int(row["h11"])
        cat_h12 = int(row["h12"])
        mir_h11 = cat_h12
        mir_h12 = cat_h11
        if str(row.get("tdf_status", "ok")) == "orphaned":
            raise ValueError(
                f"KKLT row (h11={mir_h11}, h12={mir_h12} [mirror], "
                f"ks_id={ks_id}, coni_class_id={coni_class_id}, "
                f"coni_id={coni_id}) references a TDF model that no "
                "longer exists.  Run db.rebuild_links() or pick a "
                "different row."
            )
        return {
            "ks_id":           int(row["ks_id"]),
            "triang_id":       int(row["triang_id"]),
            "tdf_conifold_id": int(row["tdf_conifold_id"]),
            "h11_mirror":      mir_h11,
            "h12_mirror":      mir_h12,
            "h11_cat":         cat_h11,
            "h12_cat":         cat_h12,
        }

    def _has_kklt_gv(
        self,
        h11: int,
        ks_id: int,
        coni_class_id: int,
        coni_id: int,
        *,
        h12: Optional[int] = None,
    ) -> bool:
        r"""Return ``True`` iff the KKLT conifold-catalog row has populated
        ``gv_kklt_shard_id`` / ``gv_kklt_row_index`` pointers (i.e.
        KKLT-native GVs are available for this conifold).

        ``h11`` is required for disambiguation; ``h12`` is optional and
        only needed when ``(h11, ks_id, coni_class_id, coni_id)`` alone
        is ambiguous (which raises ``ValueError`` via
        :meth:`_resolve_kklt_row`).  Returns ``False`` rather than
        raising when the key simply doesn't exist."""
        pd = _require_pandas()
        self._ensure_conifold_catalog()
        if ("gv_kklt_shard_id" not in self._conifold_catalog.columns
                or "gv_kklt_row_index" not in self._conifold_catalog.columns):
            return False
        try:
            row = self._resolve_kklt_row(
                h11, ks_id, coni_class_id, coni_id, h12=h12,
            )
        except KeyError:
            return False
        sh = row["gv_kklt_shard_id"]
        rw = row["gv_kklt_row_index"]
        return not (pd.isna(sh) or pd.isna(rw))

    def _fetch_kklt_gv(
        self,
        h11: int,
        ks_id: int,
        coni_class_id: int,
        coni_id: int,
        *,
        h12: Optional[int] = None,
        maximum_degree: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        r"""
        **Description:**
        Fetch the KKLT-native GV row for a given conifold and return it
        in the same dict shape as :func:`stringforge.cy_io._parse_gv_row`
        (so it's plug-compatible with TDF's GV loader).

        Args:
            h11 (int): Hodge number :math:`h^{1,1}` in mirror
                convention.  Required.
            ks_id (int): Polytope ID.
            coni_class_id (int): Conifold-class ID.
            coni_id (int): Conifold ID within the class.
            h12 (int | None): Optional :math:`h^{1,2}`.  Required only
                when the ``(h11, ks_id, coni_class_id, coni_id)`` quad
                is ambiguous.
            maximum_degree (int | None): If supplied, truncate the
                returned GVs/GWs to entries whose grading-vector
                dot-product is at most this degree.

        Returns:
            dict | None: ``{"GVs": {...}, "GWs": {...}, "grading_vector":
            ndarray}`` matching :func:`_parse_gv_row`; ``None`` if no
            KKLT GV pointer is populated for this conifold.

        Raises:
            KeyError: No matching row in the conifold catalog.
            ValueError: Ambiguous key without ``h12``.
        """
        from .cy_io import _parse_gv_row

        pd = _require_pandas()
        row = self._resolve_kklt_row(
            h11, ks_id, coni_class_id, coni_id, h12=h12,
        )
        sh_id  = row.get("gv_kklt_shard_id")
        rw_idx = row.get("gv_kklt_row_index")
        if sh_id is None or rw_idx is None or pd.isna(sh_id) or pd.isna(rw_idx):
            return None

        cat_h11 = int(row["h11"])
        split   = f"gv/h11_{cat_h11}"

        if self.cache_mode == "persistent":
            shard = self._fetch_shard(split, int(sh_id))
            gv_row = shard.iloc[int(rw_idx)]
        else:
            gv_row = self._fetch_row(split, int(sh_id), int(rw_idx))

        gv_data = _parse_gv_row(gv_row)

        if maximum_degree is not None and gv_data.get("grading_vector") is not None:
            gvs, gws = self._truncate_gv(
                gv_data["GVs"], gv_data["GWs"],
                gv_data["grading_vector"], maximum_degree,
            )
            gv_data["GVs"], gv_data["GWs"] = gvs, gws

        return gv_data

    def _effective_gv_source(
        self,
        h11: int,
        ks_id: int,
        coni_class_id: int,
        coni_id: int,
        requested: Optional[str],
        *,
        h12: Optional[int] = None,
    ) -> str:
        r"""Resolve ``gv_source`` to a concrete ``"kklt"`` or ``"tdf"``.

        - If *requested* is ``"kklt"`` or ``"tdf"``, return it.
        - If *requested* is ``None``, return ``"kklt"`` whenever
          :meth:`_has_kklt_gv` is true, else ``"tdf"``.

        Args:
            h11, ks_id, coni_class_id, coni_id, h12:  KKLT key (see
                :meth:`_resolve_kklt_row`).
            requested:  Explicit ``gv_source`` argument from the public
                API, or ``None`` for auto-select.
        """
        if requested in ("kklt", "tdf"):
            return requested
        if requested is not None:
            raise ValueError(
                f"gv_source must be 'kklt', 'tdf', or None; "
                f"got {requested!r}."
            )
        return (
            "kklt"
            if self._has_kklt_gv(h11, ks_id, coni_class_id, coni_id, h12=h12)
            else "tdf"
        )

    def load(  # type: ignore[override]
        self,
        h11: Optional[int] = None,
        ks_id: Optional[int] = None,
        coni_class_id: Optional[int] = None,
        coni_id: Optional[int] = None,
        *,
        h12: Optional[int] = None,
        include_gv: bool = False,
        gv_source: Optional[str] = None,
        conifold_basis: bool = True,
        include_extra_data: bool = True,
        include_polytope: bool = False,
        extra_fields: Optional[List[str]] = None,
        maximum_degree: Optional[int] = None,
    ) -> Any:
        r"""
        **Description:**
        Load a KKLT model as an :class:`~jaxvacua.lcs.lcs_tree`,
        resolving the logical TDF link from this database's
        ``conifold_catalog.parquet`` and delegating to the wrapped
        :attr:`tdf` database for the underlying geometry.

        GV invariants can come from either the KKLT-native ``gv/`` split
        (typically computed at higher truncation degrees and/or in a
        class-adapted basis) or from the wrapped TDF database's ``gv/``
        split.  See ``gv_source``.

        Args:
            h11 (int): Hodge number :math:`h^{1,1}` in **mirror
                convention** (matches :class:`LCSDatabase` and the rest
                of the KKLT public API).  Required — ``ks_id`` is only
                unique within a fixed ``(h11, h12)``.
            ks_id (int): Polytope ID.  Required.
            coni_class_id (int): Conifold-class ID.  Required.
            coni_id (int): Conifold ID within the class.  Required.
            h12 (int | None): Optional :math:`h^{1,2}` (mirror).  Pass
                when the ``(h11, ks_id, coni_class_id, coni_id)`` quad
                is ambiguous (the resolver raises a clear ``ValueError``
                in that case and tells you which ``h12`` values are
                available).
            include_gv (bool): If ``True``, fetch and attach
                Gopakumar-Vafa and Gromov-Witten invariants.
            gv_source (str | None): Which GV split to use.  One of
                ``"kklt"`` (KKLT-native), ``"tdf"`` (wrapped TDF), or
                ``None`` (default; prefer KKLT when its pointer is
                populated, else fall back to TDF).  Ignored when
                ``include_gv=False``.
            conifold_basis (bool): If ``True``, rotate to the basis
                defined by the conifold curve.  Passed through.
            include_extra_data (bool): Passed through.
            include_polytope (bool): Passed through.
            extra_fields (list[str] | None): Passed through.
            maximum_degree (int | None): Passed through.  **Known
                limitation**: ``gv_source="kklt"`` does not currently
                support truncation because the KKLT GV split stores
                ``grading_vector=NULL``.  Use ``gv_source="tdf"`` if you
                need ``maximum_degree>0``.

        Returns:
            lcs_tree: The model in mirror convention, with
            ``lcs_tree.conifold`` populated from the linked TDF conifold
            and ``gv_charges`` / ``gv_invariants`` / ``grading_vector``
            populated from the resolved GV source.

        Raises:
            KeyError: If the KKLT key is not in the catalog.
            ValueError: If required key args are missing, the row is
                ``tdf_status="orphaned"``, ``gv_source`` is invalid, or
                the ``(h11, ks_id, coni_class_id, coni_id)`` quad is
                ambiguous and ``h12`` was not supplied.
        """
        if (h11 is None or ks_id is None
                or coni_class_id is None or coni_id is None):
            raise ValueError(
                "KKLTDatabase.load requires h11, ks_id, coni_class_id, "
                "coni_id.  Pass h12= too if multiple polytopes share "
                "the same (h11, ks_id) — see _resolve_kklt_row."
            )
        link = self._resolve_kklt_to_tdf(
            h11, ks_id, coni_class_id, coni_id, h12=h12,
        )

        effective = (
            self._effective_gv_source(
                h11, ks_id, coni_class_id, coni_id, gv_source, h12=h12,
            )
            if include_gv
            else "tdf"   # value irrelevant when include_gv=False
        )

        # Path A: include_gv=False, or include_gv=True with gv_source=tdf.
        # Either way, the wrapped TDF database produces the tree directly.
        if not include_gv or effective == "tdf":
            return self.tdf.load(
                ks_id              = link["ks_id"],
                triang_id          = link["triang_id"],
                include_gv         = include_gv,
                include_conifolds  = True,
                conifold_id        = link["tdf_conifold_id"],
                conifold_basis     = conifold_basis,
                include_extra_data = include_extra_data,
                include_polytope   = include_polytope,
                extra_fields       = extra_fields,
                maximum_degree     = maximum_degree,
            )

        # Path B: include_gv=True with gv_source=kklt.
        # Build the tree without GVs, then attach the KKLT GVs.
        tree = self.tdf.load(
            ks_id              = link["ks_id"],
            triang_id          = link["triang_id"],
            include_gv         = False,
            include_conifolds  = True,
            conifold_id        = link["tdf_conifold_id"],
            conifold_basis     = conifold_basis,
            include_extra_data = include_extra_data,
            include_polytope   = include_polytope,
            extra_fields       = extra_fields,
            maximum_degree     = maximum_degree,
        )
        gv_data = self._fetch_kklt_gv(
            h11, ks_id, coni_class_id, coni_id,
            h12=h12, maximum_degree=maximum_degree,
        )
        if gv_data is not None:
            # Unpack the GV / GW dict into the tree's
            # ``gv_charges`` / ``gv_invariants`` / ``gv_degrees`` (and
            # GW counterparts) attributes.  Empty dicts (the KKLT
            # build carries no GW data) map to ``(None, None, None)``.
            def _safe_unpack(d: Any):
                if (not isinstance(d, dict)
                        or d.get("charges") is None
                        or d.get("invariants") is None):
                    return (None, None, None)
                return tree._charges_from_gv_gw(d)

            (tree.gv_charges,
             tree.gv_invariants,
             tree.gv_degrees) = _safe_unpack(gv_data["GVs"])
            (tree.gw_charges,
             tree.gw_invariants,
             tree.gw_degrees) = _safe_unpack(gv_data["GWs"])
            if gv_data.get("grading_vector") is not None:
                import jax.numpy as jnp
                tree.grading_vector = jnp.asarray(gv_data["grading_vector"])
        return tree

    def load_model(  # type: ignore[override]
        self,
        h11: Optional[int] = None,
        ks_id: Optional[int] = None,
        coni_class_id: Optional[int] = None,
        coni_id: Optional[int] = None,
        *,
        h12: Optional[int] = None,
        include_gv: bool = False,
        gv_source: Optional[str] = None,
        conifold_basis: bool = True,
        maximum_degree: Optional[int] = None,
        include_polytope: bool = False,
        **kwargs: Any,
    ) -> Any:
        r"""
        **Description:**
        Load a KKLT model as a
        :class:`~jaxvacua.flux_vacua_finder.FluxVacuaFinder`, with
        topology fetched from the wrapped :attr:`tdf` database and GVs
        from the KKLT-native split (or TDF — see ``gv_source``).

        See :meth:`load` for argument semantics; additional ``**kwargs``
        are forwarded to ``LCSDatabase.load_model``.

        Example::

            db = KKLTDatabase.from_local(".", tdf_cache_dir="./cy-database/")
            model = db.load_model(
                h11=95, ks_id=384564, coni_class_id=0, coni_id=75,
                include_gv=True, gv_source="kklt",
            )

        ``h11`` is mirror convention (matches :class:`LCSDatabase`); the
        on-disk catalog convention is swapped at the boundary.
        """
        if (h11 is None or ks_id is None
                or coni_class_id is None or coni_id is None):
            raise ValueError(
                "KKLTDatabase.load_model requires "
                "h11, ks_id, coni_class_id, coni_id."
            )
        from jaxvacua.flux_vacua_finder import FluxVacuaFinder

        tree = self.load(
            h11                = h11,
            ks_id              = ks_id,
            coni_class_id      = coni_class_id,
            coni_id            = coni_id,
            h12                = h12,
            include_gv         = include_gv,
            gv_source          = gv_source,
            conifold_basis     = conifold_basis,
            include_extra_data = True,
            include_polytope   = include_polytope,
            maximum_degree     = maximum_degree,
        )
        return FluxVacuaFinder(lcs_tree=tree, **kwargs)

    def load_dataframes(
        self,
        h11: int,
        ks_id: int,
        coni_class_id: int,
        coni_id: int,
        *,
        h12: Optional[int] = None,
        include_gv: bool = True,
        gv_source: Optional[str] = None,
        include_conifold: bool = True,
        include_polytope: bool = False,
    ) -> Dict[str, Any]:
        r"""
        **Description:**
        Load raw parquet rows for a KKLT model as
        ``pandas.Series`` / ``pandas.DataFrame`` objects — **no
        jaxvacua** dependency, no ``lcs_tree`` construction.

        Useful for tabular inspection, conversion to other formats, or
        any workflow that wants the on-disk data without the cost of
        building a full model.

        Args:
            h11 (int): :math:`h^{1,1}` in mirror convention.
                Required — ``ks_id`` is only unique within ``(h11, h12)``.
            ks_id (int): Polytope ID (KKLT-aligned, 1-based).
            coni_class_id (int): Conifold-class ID.
            coni_id (int): Conifold ID within the class.
            h12 (int | None): Optional :math:`h^{1,2}`.  Pass when the
                ``(h11, ks_id, coni_class_id, coni_id)`` quad is
                ambiguous.
            include_gv (bool): If ``True`` (default), include the GV row
                in the returned dict under key ``"gv"``.
            gv_source (str | None): ``"kklt"`` / ``"tdf"`` / ``None``
                (auto-select).  Ignored when ``include_gv=False``.
            include_conifold (bool): If ``True`` (default), include all
                conifold-shard rows for the underlying TDF triangulation
                under key ``"conifold"``.
            include_polytope (bool): If ``True``, include the TDF
                polytope row under key ``"polytope"``.  Defaults to
                ``False`` (the polytope shard can be large).

        Returns:
            Dict[str, Any]: A dict with the following keys (omitted when
            not requested or not available):

            - ``"kklt_polytope"``  (``pandas.Series``): row from
              ``kklt/catalog.parquet``;
            - ``"kklt_class"``     (``pandas.Series``): row from
              ``kklt/conifold_class_catalog.parquet``;
            - ``"kklt_conifold"``  (``pandas.Series``): row from
              ``kklt/conifold_catalog.parquet``;
            - ``"lcs"``            (``pandas.Series``): row from
              ``tdf/lcs_data/h11_<n>/data-XXXXX.parquet``;
            - ``"gv"``             (``pandas.Series``): row from the
              resolved GV shard;
            - ``"gv_source"``      (``str``): ``"kklt"`` or ``"tdf"``,
              showing which GV split the ``"gv"`` row came from;
            - ``"conifold"``       (``pandas.DataFrame``): every
              conifold row for the linked TDF triangulation;
            - ``"polytope"``       (``pandas.Series``): row from
              ``tdf/polytope/data-XXXXX.parquet``.

        Raises:
            KeyError: If the KKLT key is not in the conifold catalog.
            ValueError: If the row is ``tdf_status="orphaned"`` or the
                key is ambiguous and ``h12`` was not supplied.
        """
        pd = _require_pandas()
        link = self._resolve_kklt_to_tdf(
            h11, ks_id, coni_class_id, coni_id, h12=h12,
        )
        # Catalog-convention Hodge numbers (used to filter the on-disk
        # KKLT catalogs, which are stored in catalog convention).
        cat_h11 = link["h11_cat"]
        cat_h12 = link["h12_cat"]

        out: Dict[str, Any] = {}
        self._ensure_catalog()
        self._ensure_class_catalog()

        poly_mask = (
            (self._catalog["h11"]   == cat_h11)
            & (self._catalog["h12"]  == cat_h12)
            & (self._catalog["ks_id"] == ks_id)
        )
        if poly_mask.any():
            out["kklt_polytope"] = self._catalog[poly_mask].iloc[0]

        cls_mask = (
            (self._class_catalog["h11"]            == cat_h11)
            & (self._class_catalog["h12"]           == cat_h12)
            & (self._class_catalog["ks_id"]         == ks_id)
            & (self._class_catalog["coni_class_id"] == coni_class_id)
        )
        if cls_mask.any():
            out["kklt_class"] = self._class_catalog[cls_mask].iloc[0]

        cf_row = self._resolve_kklt_row(
            h11, ks_id, coni_class_id, coni_id, h12=link["h12_mirror"],
        )
        out["kklt_conifold"] = cf_row

        # ----- LCS data row (from TDF) --------------------------------
        tdf = self.tdf
        tdf._ensure_catalog()
        tdf_cat = tdf._catalog
        # `(ks_id, triang_id)` is only unique within a fixed h11, so we
        # filter on the catalog-convention h11 carried by ``link``.
        tdf_mask = (
            (tdf_cat["h11"]       == link["h11_cat"])
            & (tdf_cat["ks_id"]     == link["ks_id"])
            & (tdf_cat["triang_id"] == link["triang_id"])
        )
        if not tdf_mask.any():
            raise KeyError(
                f"TDF catalog has no row for (h11_cat={link['h11_cat']}, "
                f"ks_id={link['ks_id']}, triang_id={link['triang_id']})."
            )
        tdf_row    = tdf_cat[tdf_mask].iloc[0]
        cat_h11    = int(tdf_row["h11"])
        lcs_sh_id  = int(tdf_row["lcs_shard_id"])
        lcs_rw_idx = int(tdf_row["lcs_row_index"])
        lcs_split  = f"lcs_data/h11_{cat_h11}"
        if self.cache_mode == "persistent":
            lcs_shard = tdf._fetch_shard(lcs_split, lcs_sh_id)
            out["lcs"] = lcs_shard.iloc[lcs_rw_idx]
        else:
            out["lcs"] = tdf._fetch_row(lcs_split, lcs_sh_id, lcs_rw_idx)

        # ----- GV row -------------------------------------------------
        if include_gv:
            effective = self._effective_gv_source(
                h11, ks_id, coni_class_id, coni_id, gv_source,
                h12=link["h12_mirror"],
            )
            out["gv_source"] = effective
            if effective == "kklt":
                gv_split   = f"gv/h11_{cat_h11}"
                gv_sh_id   = int(cf_row["gv_kklt_shard_id"])
                gv_rw_idx  = int(cf_row["gv_kklt_row_index"])
                if self.cache_mode == "persistent":
                    gv_shard = self._fetch_shard(gv_split, gv_sh_id)
                    out["gv"] = gv_shard.iloc[gv_rw_idx]
                else:
                    out["gv"] = self._fetch_row(gv_split, gv_sh_id, gv_rw_idx)
            else:  # "tdf"
                gv_sh_id  = tdf_row.get("gv_shard_id")
                gv_rw_idx = tdf_row.get("gv_row_index")
                if (gv_sh_id is None or gv_rw_idx is None
                        or pd.isna(gv_sh_id) or pd.isna(gv_rw_idx)):
                    out["gv"] = None
                else:
                    tdf_gv_split = f"gv/h11_{cat_h11}"
                    if self.cache_mode == "persistent":
                        gv_shard = tdf._fetch_shard(
                            tdf_gv_split, int(gv_sh_id),
                        )
                        out["gv"] = gv_shard.iloc[int(gv_rw_idx)]
                    else:
                        out["gv"] = tdf._fetch_row(
                            tdf_gv_split, int(gv_sh_id), int(gv_rw_idx),
                        )

        # ----- Conifold rows (per-triangulation) -----------------------
        if include_conifold:
            cf_sh_id = tdf_row.get("conifold_shard_id")
            if cf_sh_id is not None and not pd.isna(cf_sh_id):
                cf_split = f"conifolds/h11_{cat_h11}"
                cf_shard = tdf._fetch_shard(cf_split, int(cf_sh_id))
                # Filter to this triangulation only
                cf_mask_tdf = (
                    (cf_shard["ks_id"]     == link["ks_id"])
                    & (cf_shard["triang_id"] == link["triang_id"])
                )
                out["conifold"] = cf_shard[cf_mask_tdf].reset_index(drop=True)

        # ----- Polytope row -------------------------------------------
        if include_polytope:
            poly_sh_id  = tdf_row.get("polytope_shard_id")
            poly_rw_idx = tdf_row.get("polytope_row_index")
            if (poly_sh_id is not None and poly_rw_idx is not None
                    and not pd.isna(poly_sh_id) and not pd.isna(poly_rw_idx)):
                if self.cache_mode == "persistent":
                    poly_shard = tdf._fetch_shard("polytope", int(poly_sh_id))
                    out["polytope"] = poly_shard.iloc[int(poly_rw_idx)]
                else:
                    out["polytope"] = tdf._fetch_row(
                        "polytope", int(poly_sh_id), int(poly_rw_idx),
                    )

        return out

    def load_from_conifold_row(  # type: ignore[override]
        self,
        row: Any,
        **kwargs: Any,
    ) -> Any:
        r"""
        **Description:**
        Load a KKLT model from a row of :meth:`query_conifolds` output.

        Convenience wrapper over :meth:`load_model` that pulls the
        full primary key (``h11``, ``h12``, ``ks_id``, ``coni_class_id``,
        ``coni_id``) from the row.  Both the row and :meth:`load_model`
        use mirror convention, so the values pass straight through.

        Args:
            row: A ``pandas.Series`` (one row from
                :meth:`query_conifolds`).
            **kwargs: Forwarded to :meth:`load_model`.

        Returns:
            FluxVacuaFinder: The constructed model.
        """
        return self.load_model(
            h11           = int(row["h11"]),
            h12           = int(row["h12"]),
            ks_id         = int(row["ks_id"]),
            coni_class_id = int(row["coni_class_id"]),
            coni_id       = int(row["coni_id"]),
            **kwargs,
        )

    def load_batch(  # type: ignore[override]
        self,
        identifiers: Any,
        *,
        as_models: bool = False,
        include_gv: bool = False,
        gv_source: Optional[str] = None,
        conifold_basis: bool = True,
        maximum_degree: Optional[int] = None,
        include_polytope: bool = False,
        **kwargs: Any,
    ) -> List[Any]:
        r"""
        **Description:**
        Load a batch of KKLT models.

        Args:
            identifiers: One of
                * a ``pandas.DataFrame`` with columns ``h11``,
                  ``ks_id``, ``coni_class_id``, ``coni_id`` (and
                  optionally ``h12``) — typically the output of
                  :meth:`query_conifolds`;
                * a list of dicts with the same keys;
                * a list of 4- or 5-tuples
                  ``(h11, ks_id, coni_class_id, coni_id[, h12])``.
            as_models (bool): If ``True``, return ``FluxVacuaFinder``
                instances; otherwise return bare ``lcs_tree`` objects.
            include_gv, gv_source, conifold_basis, maximum_degree,
            include_polytope, **kwargs:  Forwarded to :meth:`load` or
                :meth:`load_model`.

        Returns:
            list: Loaded objects in the same order as *identifiers*.
        """
        records  = self._identifiers_to_kklt_records(identifiers)
        loader   = self.load_model if as_models else self.load
        return [
            loader(
                h11           = rec["h11"],
                ks_id         = rec["ks_id"],
                coni_class_id = rec["coni_class_id"],
                coni_id       = rec["coni_id"],
                h12           = rec.get("h12"),
                include_gv    = include_gv,
                gv_source     = gv_source,
                conifold_basis= conifold_basis,
                maximum_degree= maximum_degree,
                include_polytope = include_polytope,
                **kwargs,
            )
            for rec in records
        ]

    def iter_batch(  # type: ignore[override]
        self,
        identifiers: Any,
        *,
        as_models: bool = False,
        include_gv: bool = False,
        gv_source: Optional[str] = None,
        conifold_basis: bool = True,
        maximum_degree: Optional[int] = None,
        include_polytope: bool = False,
        **kwargs: Any,
    ) -> Iterator[Any]:
        r"""
        **Description:**
        Lazy generator counterpart of :meth:`load_batch`.  Yields one
        model at a time so a large iteration does not have to fit in
        memory.

        See :meth:`load_batch` for argument semantics.
        """
        records = self._identifiers_to_kklt_records(identifiers)
        loader  = self.load_model if as_models else self.load
        for rec in records:
            yield loader(
                h11           = rec["h11"],
                ks_id         = rec["ks_id"],
                coni_class_id = rec["coni_class_id"],
                coni_id       = rec["coni_id"],
                h12           = rec.get("h12"),
                include_gv    = include_gv,
                gv_source     = gv_source,
                conifold_basis= conifold_basis,
                maximum_degree= maximum_degree,
                include_polytope = include_polytope,
                **kwargs,
            )

    def sample(  # type: ignore[override]
        self,
        n: int = 1,
        *,
        seed: Optional[int] = None,
        as_models: bool = False,
        query: Optional[Dict[str, Any]] = None,
        include_gv: bool = False,
        gv_source: Optional[str] = None,
        conifold_basis: bool = True,
        maximum_degree: Optional[int] = None,
        include_polytope: bool = False,
        **kwargs: Any,
    ) -> List[Any]:
        r"""
        **Description:**
        Draw *n* random conifolds from this database and load them.

        Args:
            n (int): Number of conifolds to draw.
            seed (int | None): RNG seed for reproducibility.
            as_models (bool): Return ``FluxVacuaFinder`` instances
                instead of bare ``lcs_tree`` objects.
            query (dict | None): Optional filter dict forwarded to
                :meth:`query_conifolds` (e.g. ``{"h11": 95}``) so the
                sample is drawn from a subset.
            include_gv, gv_source, conifold_basis, maximum_degree,
            include_polytope, **kwargs:  Forwarded to the loader.

        Returns:
            list: Loaded objects.
        """
        pool = self.query_conifolds(**(query or {}))
        if len(pool) == 0:
            raise ValueError("query_conifolds returned no rows; nothing to sample.")
        sampled = pool.sample(n=min(int(n), len(pool)), random_state=seed)
        return self.load_batch(
            sampled,
            as_models        = as_models,
            include_gv       = include_gv,
            gv_source        = gv_source,
            conifold_basis   = conifold_basis,
            maximum_degree   = maximum_degree,
            include_polytope = include_polytope,
            **kwargs,
        )

    def _identifiers_to_kklt_records(
        self, identifiers: Any,
    ) -> List[Dict[str, int]]:
        r"""Normalise *identifiers* to a list of dicts with keys
        ``h11``, ``ks_id``, ``coni_class_id``, ``coni_id`` and
        (optionally) ``h12``.

        Accepts:
          * a ``pandas.DataFrame`` with those columns;
          * an iterable of dicts;
          * an iterable of 4- or 5-tuples
            ``(h11, ks_id, coni_class_id, coni_id[, h12])``.
        """
        pd = _require_pandas()
        out: List[Dict[str, int]] = []

        if isinstance(identifiers, pd.DataFrame):
            for _, row in identifiers.iterrows():
                rec = {
                    "h11":           int(row["h11"]),
                    "ks_id":         int(row["ks_id"]),
                    "coni_class_id": int(row["coni_class_id"]),
                    "coni_id":       int(row["coni_id"]),
                }
                if "h12" in row.index:
                    rec["h12"] = int(row["h12"])
                out.append(rec)
            return out

        for item in identifiers:
            if isinstance(item, dict):
                rec = {k: int(item[k]) for k in
                       ("h11", "ks_id", "coni_class_id", "coni_id")}
                if "h12" in item:
                    rec["h12"] = int(item["h12"])
                out.append(rec)
            elif isinstance(item, (list, tuple)) and len(item) in (4, 5):
                rec = {
                    "h11":           int(item[0]),
                    "ks_id":         int(item[1]),
                    "coni_class_id": int(item[2]),
                    "coni_id":       int(item[3]),
                }
                if len(item) == 5:
                    rec["h12"] = int(item[4])
                out.append(rec)
            else:
                raise TypeError(
                    f"Cannot interpret identifier {item!r}; pass a "
                    "DataFrame, a dict, or a "
                    "(h11, ks_id, coni_class_id, coni_id[, h12]) tuple."
                )
        return out

    # ------------------------------------------------------------------
    # Solution stages: pfv / precursor / vacua
    #
    # Layout (per stage):
    #
    #   <cache_dir>/kklt/<stage>/<kind?>/
    #       h11_<n>_h12_<m>_ksid_<id>/
    #           coniclass_<c>_coni_<k>/
    #               run_<run_id>.parquet
    #
    # Each cluster run writes its own ``run_<run_id>.parquet`` fragment;
    # readers concatenate every matching fragment in the per-conifold
    # directory.  Identity columns (h11, h12, ks_id, coni_class_id,
    # coni_id, run_id) are stamped on every row; everything else the
    # caller supplies in ``rows`` is preserved verbatim.
    # ------------------------------------------------------------------

    def _kklt_stage_dir(
        self,
        stage: str,
        kind: Optional[str] = None,
    ) -> Path:
        r"""Return the absolute root directory for a KKLT solution stage.

        Args:
            stage (str): One of the keys of :data:`_KKLT_STAGES`.
            kind (str | None): A valid sub-kind for the stage, or
                ``None`` for stages that have no sub-kind (``"vacua"``).

        Raises:
            ValueError: If *stage* is unknown or *kind* is invalid for
                the stage.
        """
        if stage not in _KKLT_STAGES:
            raise ValueError(
                f"Unknown KKLT stage {stage!r}; "
                f"valid: {sorted(_KKLT_STAGES)}."
            )
        valid_kinds = _KKLT_STAGES[stage]
        if valid_kinds is None:
            if kind is not None:
                raise ValueError(
                    f"Stage {stage!r} accepts no sub-kind; got kind={kind!r}."
                )
            return self.cache_dir / stage
        if kind is None or kind not in valid_kinds:
            raise ValueError(
                f"Stage {stage!r} requires kind in {list(valid_kinds)!r}; "
                f"got kind={kind!r}."
            )
        return self.cache_dir / stage / kind

    def _kklt_fragment_path(
        self,
        stage: str,
        kind: Optional[str],
        *,
        h11: int,
        h12: int,
        ks_id: int,
        coni_class_id: int,
        coni_id: int,
        run_id: str,
    ) -> Path:
        r"""Return the absolute path for a KKLT stage's per-run
        fragment parquet file."""
        return (
            self._kklt_stage_dir(stage, kind)
            / _kklt_polytope_dirname(h11, h12, ks_id)
            / _kklt_conifold_dirname(coni_class_id, coni_id)
            / f"run_{run_id}.parquet"
        )

    def _write_stage_fragment(
        self,
        rows: Any,
        stage: str,
        kind: Optional[str],
        *,
        h11: int,
        h12: int,
        ks_id: int,
        coni_class_id: int,
        coni_id: int,
        run_id: Optional[str] = None,
        extra_columns: Optional[Dict[str, Any]] = None,
    ) -> Path:
        r"""Write one KKLT stage fragment.

        ``rows`` may be a ``pandas.DataFrame``, a list of dicts, or any
        object the :class:`pandas.DataFrame` constructor accepts.  The
        five identity columns (``h11, h12, ks_id, coni_class_id,
        coni_id``) and ``run_id`` are stamped on every row, overwriting
        any preexisting values.  Additional columns supplied by the
        caller (via ``rows`` or ``extra_columns``) pass through
        verbatim.

        Args:
            rows: Open-ended row payload.
            stage (str): KKLT stage name (``"pfv"`` / ``"precursor"`` /
                ``"vacua"``).
            kind (str | None): Sub-kind for the stage (or ``None``).
            h11, h12, ks_id, coni_class_id, coni_id (int): KKLT
                identity.
            run_id (str | None): UUID-style run identifier.  Auto-
                generated if ``None``.
            extra_columns (dict | None): Extra columns to stamp on
                every row (useful for run-level scalars like ``method``,
                ``solver_version``, ...).

        Returns:
            Path: Absolute path of the written fragment file.
        """
        pd = _require_pandas()
        if run_id is None:
            run_id = uuid.uuid4().hex

        if isinstance(rows, pd.DataFrame):
            df = rows.copy()
        else:
            df = pd.DataFrame(rows)

        df["h11"]            = int(h11)
        df["h12"]            = int(h12)
        df["ks_id"]          = int(ks_id)
        df["coni_class_id"]  = int(coni_class_id)
        df["coni_id"]        = int(coni_id)
        df["run_id"]         = run_id
        if extra_columns:
            n = len(df)
            for k, v in extra_columns.items():
                # Broadcast non-scalar values (arrays, lists, dicts) as
                # object-typed cells — pandas otherwise interprets a
                # length-K array as one-value-per-row.  Scalars (int,
                # float, str, complex, np.number) assign directly.
                is_scalar = (
                    v is None
                    or isinstance(v, (bool, int, float, complex, str, bytes,
                                      np.bool_, np.integer, np.floating,
                                      np.complexfloating))
                )
                if is_scalar:
                    df[k] = v
                else:
                    df[k] = pd.Series([v] * n, index=df.index, dtype=object)

        path = self._kklt_fragment_path(
            stage, kind,
            h11=h11, h12=h12, ks_id=ks_id,
            coni_class_id=coni_class_id, coni_id=coni_id,
            run_id=run_id,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        return path

    def _read_stage_fragments(
        self,
        stage: str,
        kind: Optional[str] = None,
        *,
        h11: Optional[int] = None,
        h12: Optional[int] = None,
        ks_id: Optional[int] = None,
        coni_class_id: Optional[int] = None,
        coni_id: Optional[int] = None,
        run_id: Optional[str] = None,
    ) -> Any:
        r"""Read and concatenate every matching fragment parquet.

        Each filter argument left as ``None`` becomes a glob wildcard.
        Missing directories silently yield an empty DataFrame.

        For stages with sub-kinds, ``kind=None`` fans in across every
        sub-kind under the stage.
        """
        pd = _require_pandas()
        valid_kinds = _KKLT_STAGES.get(stage)
        if valid_kinds is None:
            kind_dirs = [self._kklt_stage_dir(stage, None)]
        elif kind is None:
            kind_dirs = [self._kklt_stage_dir(stage, k) for k in valid_kinds]
        else:
            kind_dirs = [self._kklt_stage_dir(stage, kind)]

        # Build directory-name glob fragments
        h11_pat  = h11  if h11  is not None else "*"
        h12_pat  = h12  if h12  is not None else "*"
        ks_pat   = ks_id if ks_id is not None else "*"
        cls_pat  = coni_class_id if coni_class_id is not None else "*"
        coni_pat = coni_id       if coni_id       is not None else "*"
        run_pat  = run_id        if run_id        is not None else "*"

        poly_glob = f"h11_{h11_pat}_h12_{h12_pat}_ksid_{ks_pat}"
        cf_glob   = f"coniclass_{cls_pat}_coni_{coni_pat}"
        file_glob = f"run_{run_pat}.parquet"

        files: List[Path] = []
        for kd in kind_dirs:
            if not kd.exists():
                continue
            files.extend(sorted(kd.glob(f"{poly_glob}/{cf_glob}/{file_glob}")))

        if not files:
            return pd.DataFrame()

        dfs = [pd.read_parquet(f) for f in files]
        return pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]

    # ----- pfv ----------------------------------------------------------

    def write_pfvs(
        self,
        rows: Any,
        *,
        kind: str,
        h11: int,
        h12: int,
        ks_id: int,
        coni_class_id: int,
        coni_id: int,
        run_id: Optional[str] = None,
        extra_columns: Optional[Dict[str, Any]] = None,
    ) -> Path:
        r"""Append a PFV fragment under
        ``<cache>/kklt/pfv/<kind>/.../run_<run_id>.parquet``.

        Args:
            rows: Open-ended row payload (DataFrame or list of dicts).
            kind (str): ``"harvested"`` (raw flux scan output) or
                ``"promoted"`` (post-Newton, integer flux).
            h11, h12, ks_id, coni_class_id, coni_id (int): KKLT
                identity.  All required (``h11`` and ``h12`` together
                disambiguate ``ks_id``).
            run_id (str | None): Run identifier; auto-generated if
                ``None``.  Pass an explicit ``run_id`` (e.g. from
                :meth:`start_run`) to tie the fragment to a run-log
                entry.
            extra_columns (dict | None): Extra columns to stamp on
                every row.

        Returns:
            Path: Absolute path of the written fragment.
        """
        return self._write_stage_fragment(
            rows, "pfv", kind,
            h11=h11, h12=h12, ks_id=ks_id,
            coni_class_id=coni_class_id, coni_id=coni_id,
            run_id=run_id, extra_columns=extra_columns,
        )

    def load_pfvs(
        self,
        *,
        kind: Optional[str] = None,
        h11: Optional[int] = None,
        h12: Optional[int] = None,
        ks_id: Optional[int] = None,
        coni_class_id: Optional[int] = None,
        coni_id: Optional[int] = None,
        run_id: Optional[str] = None,
    ) -> Any:
        r"""Read every matching PFV fragment, concatenated.

        Each argument left as ``None`` becomes a wildcard (e.g.
        ``kind=None`` fans in across both ``"harvested"`` and
        ``"promoted"``; ``ks_id=None`` returns all polytopes).
        """
        return self._read_stage_fragments(
            "pfv", kind,
            h11=h11, h12=h12, ks_id=ks_id,
            coni_class_id=coni_class_id, coni_id=coni_id,
            run_id=run_id,
        )

    def query_pfvs(self, **filters: Any) -> Any:
        r"""Alias for :meth:`load_pfvs` (returned DataFrame is then the
        result of any further pandas filtering)."""
        return self.load_pfvs(**filters)

    # ----- precursor ----------------------------------------------------

    def write_precursor(
        self,
        rows: Any,
        *,
        kind: str,
        h11: int,
        h12: int,
        ks_id: int,
        coni_class_id: int,
        coni_id: int,
        run_id: Optional[str] = None,
        extra_columns: Optional[Dict[str, Any]] = None,
    ) -> Path:
        r"""Append a precursor-stage fragment under
        ``<cache>/kklt/precursor/<kind>/.../run_<run_id>.parquet``.

        Args:
            rows: Open-ended row payload.  May carry any solver-specific
                columns the caller needs (e.g. ``W0``, ``gs``,
                eigenvalues, fluxes, derived quantities).
            kind (str): ``"derived"`` (W0, gs derived from a full SUSY
                solution) or ``"proxy"`` (SUSY vacua at pre-specified
                ``W0``, ``gs``).
            h11, h12, ks_id, coni_class_id, coni_id (int): KKLT
                identity.
            run_id (str | None): Run identifier; auto-generated if
                ``None``.
            extra_columns (dict | None): Extra columns to stamp on
                every row (useful for run-level scalars like
                ``method``, ``solver_version``, ``solver_tolerance``).

        Returns:
            Path: Absolute path of the written fragment.
        """
        return self._write_stage_fragment(
            rows, "precursor", kind,
            h11=h11, h12=h12, ks_id=ks_id,
            coni_class_id=coni_class_id, coni_id=coni_id,
            run_id=run_id, extra_columns=extra_columns,
        )

    def load_precursor(
        self,
        *,
        kind: Optional[str] = None,
        h11: Optional[int] = None,
        h12: Optional[int] = None,
        ks_id: Optional[int] = None,
        coni_class_id: Optional[int] = None,
        coni_id: Optional[int] = None,
        run_id: Optional[str] = None,
    ) -> Any:
        r"""Read every matching precursor fragment, concatenated."""
        return self._read_stage_fragments(
            "precursor", kind,
            h11=h11, h12=h12, ks_id=ks_id,
            coni_class_id=coni_class_id, coni_id=coni_id,
            run_id=run_id,
        )

    def query_precursor(self, **filters: Any) -> Any:
        r"""Alias for :meth:`load_precursor`."""
        return self.load_precursor(**filters)

    # ----- vacua (full stabilisation) ----------------------------------

    def write_vacua(  # type: ignore[override]
        self,
        rows: Any,
        *,
        h11: int,
        h12: int,
        ks_id: int,
        coni_class_id: int,
        coni_id: int,
        run_id: Optional[str] = None,
        extra_columns: Optional[Dict[str, Any]] = None,
    ) -> Path:
        r"""Append a full-stabilisation-vacua fragment under
        ``<cache>/kklt/vacua/.../run_<run_id>.parquet``.

        Args:
            rows: Open-ended row payload — any solver-specific columns
                the caller needs (fluxes, moduli, W0, gs, mass spectrum,
                Kähler moduli, uplift parameters, derived quantities,
                ...) pass through verbatim.
            h11, h12, ks_id, coni_class_id, coni_id (int): KKLT
                identity.
            run_id (str | None): Run identifier; auto-generated if
                ``None``.
            extra_columns (dict | None): Extra columns to stamp on
                every row.

        Returns:
            Path: Absolute path of the written fragment.
        """
        return self._write_stage_fragment(
            rows, "vacua", None,
            h11=h11, h12=h12, ks_id=ks_id,
            coni_class_id=coni_class_id, coni_id=coni_id,
            run_id=run_id, extra_columns=extra_columns,
        )

    def load_vacua(  # type: ignore[override]
        self,
        *,
        h11: Optional[int] = None,
        h12: Optional[int] = None,
        ks_id: Optional[int] = None,
        coni_class_id: Optional[int] = None,
        coni_id: Optional[int] = None,
        run_id: Optional[str] = None,
    ) -> Any:
        r"""Read every matching vacua fragment, concatenated."""
        return self._read_stage_fragments(
            "vacua", None,
            h11=h11, h12=h12, ks_id=ks_id,
            coni_class_id=coni_class_id, coni_id=coni_id,
            run_id=run_id,
        )

    def query_vacua(self, **filters: Any) -> Any:  # type: ignore[override]
        r"""Alias for :meth:`load_vacua`."""
        return self.load_vacua(**filters)

    # ------------------------------------------------------------------
    # Run-tracking API
    # ------------------------------------------------------------------

    def _validate_run_args(
        self,
        scope: str,
        ks_id: int,
        coni_class_id: int,
        coni_id: int,
    ) -> None:
        r"""Internal: validate run-tracking arguments."""
        if scope not in _RUN_SCOPES:
            raise ValueError(
                f"scope must be one of {sorted(_RUN_SCOPES)}; got {scope!r}."
            )
        if scope == "conifold" and coni_id < 0:
            raise ValueError(
                "scope='conifold' requires a non-negative coni_id; "
                f"got coni_id={coni_id}."
            )

    def _append_run_log_row(self, row: Dict[str, Any]) -> None:
        r"""Internal: file-lock-guarded append-and-write of a run-log row.

        If ``row["attempt_index"]`` is ``None``, it is allocated under
        the file lock from the *on-disk* run log so concurrent workers
        on the same ``(scope, ks_id, coni_class_id, coni_id)`` key get
        distinct attempt indices.

        The local in-memory cache (:attr:`_run_log`) is updated in-place
        so subsequent :meth:`run_status` and :meth:`list_runs` calls see
        the new row without re-reading the parquet.
        """
        pd = _require_pandas()
        path = self.cache_dir / "run_log.parquet"
        with _file_lock(path):
            if path.exists():
                existing = pd.read_parquet(path)
            else:
                existing = pd.DataFrame(columns=_RUN_LOG_COLUMNS)

            if row.get("attempt_index") is None:
                if len(existing):
                    mask = (
                        (existing["scope"]         == row["scope"])
                        & (existing["ks_id"]         == row["ks_id"])
                        & (existing["coni_class_id"] == row["coni_class_id"])
                        & (existing["coni_id"]       == row["coni_id"])
                    )
                    attempts = existing.loc[mask, "attempt_index"].dropna()
                    row["attempt_index"] = (
                        int(attempts.astype(int).max()) + 1
                        if len(attempts) > 0 else 1
                    )
                else:
                    row["attempt_index"] = 1

            new_row_df = pd.DataFrame([row], columns=_RUN_LOG_COLUMNS)
            if len(existing) == 0:
                new = new_row_df.copy()
            else:
                # Concat with a deprecation warning suppressed locally:
                # the run log carries nullable columns (finished_at,
                # n_solutions, ...) so pandas warns about empty/all-NA
                # entries even when we explicitly provide column dtypes.
                # The resulting parquet is correct on disk; ignore.
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=FutureWarning)
                    new = pd.concat(
                        [existing, new_row_df], ignore_index=True
                    )
            new.to_parquet(path, index=False)
        self._run_log = new

    def start_run(
        self,
        scope: str,
        ks_id: int,
        coni_class_id: int,
        coni_id: int = -1,
        *,
        task: str = "",
        job_id: str = "",
        worker: str = "",
        host: Optional[str] = None,
        notes: str = "",
    ) -> str:
        r"""
        **Description:**
        Open a new run-log entry.  Appends a ``"running"`` row to
        ``run_log.parquet`` and returns the generated ``run_id``.

        Args:
            scope (str): ``"class"`` or ``"conifold"``.
            ks_id (int): Polytope ID.
            coni_class_id (int): Conifold-class ID.
            coni_id (int): Conifold ID within the class (only meaningful
                when ``scope="conifold"``; pass ``-1`` for
                ``scope="class"``).
            task (str): Free-form task tag (e.g. ``"basis_fit"``,
                ``"flux_scan"``, ``"gv_recompute"``).
            job_id (str): Cluster job ID / SLURM array index.
            worker (str): Free-form worker identifier, e.g.
                ``"rank 12 of 64"``.
            host (str | None): Hostname.  Defaults to the local
                ``socket.gethostname()``.
            notes (str): Free-form notes.

        Returns:
            str: The newly generated ``run_id`` (32-character hex
            string).  Pass back to :meth:`finish_run` /
            :meth:`cancel_run` to close the entry.
        """
        self._validate_run_args(scope, ks_id, coni_class_id, coni_id)
        self._ensure_run_log()

        row = {
            "run_id":         uuid.uuid4().hex,
            "scope":          scope,
            "ks_id":          int(ks_id),
            "coni_class_id":  int(coni_class_id),
            "coni_id":        int(coni_id) if scope == "conifold" else -1,
            # Allocated under the file lock in `_append_run_log_row`
            # so concurrent workers on the same key get distinct values.
            "attempt_index":  None,
            "status":         "running",
            "started_at":     _utcnow(),
            "finished_at":    None,
            "job_id":         str(job_id),
            "host":           str(host or socket.gethostname()),
            "worker":         str(worker),
            "task":           str(task),
            "n_solutions":    None,
            "n_attempted":    None,
            "payload_uri":    "",
            "payload_hash":   "",
            "error_summary":  "",
            "notes":          str(notes),
        }
        self._append_run_log_row(row)
        return row["run_id"]

    def finish_run(
        self,
        run_id: str,
        status: str,
        *,
        n_solutions: Optional[int] = None,
        n_attempted: Optional[int] = None,
        payload_uri: str = "",
        payload_hash: str = "",
        error_summary: str = "",
        notes: str = "",
    ) -> None:
        r"""
        **Description:**
        Close a previously-opened run-log entry by appending a closing
        row (status ``"done"``, ``"failed"``, or ``"cancelled"``).

        The log is append-only — the original ``"running"`` row remains.
        The derived status view (:meth:`run_status`) picks the latest
        row per ``(scope, ks_id, coni_class_id, coni_id)``.

        Args:
            run_id (str): Run ID returned by :meth:`start_run`.
            status (str): Terminal status.  Must be one of ``"done"``,
                ``"failed"``, ``"cancelled"``.
            n_solutions (int | None): Number of solutions produced by
                the run (if applicable).
            n_attempted (int | None): Number of trials attempted.
            payload_uri (str): URI to the results blob, if any.
            payload_hash (str): sha256 of the payload, for de-duplication.
            error_summary (str): One-line error message, if the run
                failed.
            notes (str): Free-form notes.

        Raises:
            ValueError: If ``status`` is not a terminal status, or if
                ``run_id`` does not match any open run-log row.
        """
        if status not in {"done", "failed", "cancelled"}:
            raise ValueError(
                f"status must be one of 'done', 'failed', 'cancelled'; "
                f"got {status!r}."
            )
        self._ensure_run_log()
        rl = self._run_log
        matches = rl[rl["run_id"] == run_id]
        if len(matches) == 0:
            raise ValueError(f"No run_log row found for run_id={run_id!r}.")
        head = matches.iloc[0]

        row = {
            "run_id":         run_id,
            "scope":          head["scope"],
            "ks_id":          int(head["ks_id"]),
            "coni_class_id":  int(head["coni_class_id"]),
            "coni_id":        int(head["coni_id"]),
            "attempt_index":  int(head["attempt_index"]),
            "status":         status,
            "started_at":     head["started_at"],
            "finished_at":    _utcnow(),
            "job_id":         head["job_id"],
            "host":           head["host"],
            "worker":         head["worker"],
            "task":           head["task"],
            "n_solutions":    int(n_solutions) if n_solutions is not None else None,
            "n_attempted":    int(n_attempted) if n_attempted is not None else None,
            "payload_uri":    str(payload_uri),
            "payload_hash":   str(payload_hash),
            "error_summary":  str(error_summary),
            "notes":          str(notes),
        }
        self._append_run_log_row(row)

    def cancel_run(self, run_id: str, *, notes: str = "") -> None:
        r"""Convenience wrapper for :meth:`finish_run` with
        ``status="cancelled"``."""
        self.finish_run(run_id, status="cancelled", notes=notes)

    def list_runs(
        self,
        scope: Optional[str] = None,
        ks_id: Optional[int] = None,
        coni_class_id: Optional[int] = None,
        coni_id: Optional[int] = None,
        status: Optional[str] = None,
        task: Optional[str] = None,
        **filters: Any,
    ) -> Any:
        r"""
        **Description:**
        Filter the run log and return matching rows as a
        ``pandas.DataFrame``.

        Args:
            scope (str | None): Filter by run scope.
            ks_id (int | None): Filter by polytope ID.
            coni_class_id (int | None): Filter by conifold-class ID.
            coni_id (int | None): Filter by conifold ID.
            status (str | None): Filter by status.
            task (str | None): Filter by task tag.
            **filters: Additional exact-match filters.

        Returns:
            pandas.DataFrame: Matching run-log rows.
        """
        self._ensure_run_log()
        rl = self._run_log
        mask = np.ones(len(rl), dtype=bool)
        if scope is not None:
            mask &= (rl["scope"] == scope).values
        if ks_id is not None:
            mask &= (rl["ks_id"] == ks_id).values
        if coni_class_id is not None:
            mask &= (rl["coni_class_id"] == coni_class_id).values
        if coni_id is not None:
            mask &= (rl["coni_id"] == coni_id).values
        if status is not None:
            mask &= (rl["status"] == status).values
        if task is not None:
            mask &= (rl["task"] == task).values
        for col, val in filters.items():
            if col in rl.columns:
                mask &= (rl[col] == val).values
            else:
                warnings.warn(
                    f"Column '{col}' not found in run log; filter ignored."
                )
        return rl[mask].reset_index(drop=True)

    def run_status(
        self,
        ks_id: int,
        coni_class_id: Optional[int] = None,
        coni_id: Optional[int] = None,
    ) -> Optional[str]:
        r"""
        **Description:**
        Return the latest run-log status for the given key.

        - If ``coni_class_id`` is ``None``, returns the latest status
          across ALL runs (any scope) on the polytope.
        - If ``coni_class_id`` is set and ``coni_id`` is ``None``,
          returns the latest status at class scope.
        - If both are set, returns the latest status at conifold scope.

        Returns:
            str | None: Latest status, or ``None`` if no runs match.
        """
        self._ensure_run_log()
        rl = self._run_log
        mask = (rl["ks_id"] == ks_id)
        if coni_class_id is not None:
            mask &= (rl["coni_class_id"] == coni_class_id)
            if coni_id is None:
                mask &= (rl["scope"] == "class")
            else:
                mask &= (rl["scope"] == "conifold")
                mask &= (rl["coni_id"] == coni_id)
        matched = rl[mask]
        if len(matched) == 0:
            return None
        # The latest entry — break ties by row order (append-only ⇒
        # later rows are later in time).
        return str(matched.iloc[-1]["status"])

    # ------------------------------------------------------------------
    # Hub sync (run log)
    # ------------------------------------------------------------------

    def push_run_log(self, repo_id: Optional[str] = None) -> None:
        r"""
        **Description:**
        Upload the local ``run_log.parquet`` to the HuggingFace dataset
        repository.  Mirrors :meth:`VacuaWriter.push_vacua_to_hub` —
        local-only writes plus periodic push is the v1 concurrency
        story for cluster workers.

        Args:
            repo_id (str | None): Target HF dataset repo.  Defaults to
                ``self.hf_repo``.
        """
        from .cy_io import _require_hf_api

        api = _require_hf_api()
        repo_id = repo_id or self.hf_repo
        path = self.cache_dir / "run_log.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"No local run_log.parquet to push at {path}."
            )
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=f"{self.dataset}/run_log.parquet",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"update run_log.parquet ({_utcnow().isoformat()})",
        )

    def fetch_run_log(
        self,
        repo_id: Optional[str] = None,
        merge: bool = True,
    ) -> None:
        r"""
        **Description:**
        Download the latest ``run_log.parquet`` from the Hub.  If
        ``merge=True`` (default), local rows are kept and merged with
        the remote: rows with the same ``run_id`` are deduplicated,
        preferring the row with the latest ``finished_at`` (falling
        back to ``started_at``).  If ``merge=False``, the remote
        replaces the local log entirely.

        Args:
            repo_id (str | None): Source HF dataset repo.  Defaults to
                ``self.hf_repo``.
            merge (bool): Merge local + remote (``True``) or replace
                (``False``).
        """
        pd = _require_pandas()
        repo_id = repo_id or self.hf_repo
        path = self.cache_dir / "run_log.parquet"

        # Download into a temporary location to avoid clobbering local
        # state until we're sure the download succeeded.
        try:
            from .cy_io import _require_hf_hub
            import tempfile

            hf_download = _require_hf_hub()
            with tempfile.TemporaryDirectory() as tmpdir:
                downloaded = hf_download(
                    repo_id=repo_id,
                    filename=f"{self.dataset}/run_log.parquet",
                    repo_type="dataset",
                    local_dir=tmpdir,
                )
                remote = pd.read_parquet(downloaded)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to fetch run log from {repo_id}: {exc}"
            ) from exc

        if not merge or not path.exists():
            remote.to_parquet(path, index=False)
            self._run_log = remote
            return

        local = pd.read_parquet(path)
        combined = pd.concat([local, remote], ignore_index=True)
        # Sort so that, for each run_id, the row with the latest
        # finished_at (or started_at, when finished_at is null) wins.
        combined = combined.sort_values(
            by=["finished_at", "started_at"], na_position="first"
        )
        deduped = combined.drop_duplicates(subset=["run_id"], keep="last")
        deduped = deduped.reset_index(drop=True)
        deduped.to_parquet(path, index=False)
        self._run_log = deduped

    # ------------------------------------------------------------------
    # TDF-link maintenance
    # ------------------------------------------------------------------

    def rebuild_links(
        self,
        tdf_db: Optional[LCSDatabase] = None,
        pick_up_new_conifolds: bool = False,
        verbose: bool = True,
    ) -> Dict[str, List[Any]]:
        r"""
        **Description:**
        Refresh the KKLT catalogs against a (possibly rebuilt) TDF
        database.

        Steps:

        1. Take a fresh :class:`LCSDatabase` (or the one passed in).
        2. For every row of ``conifold_catalog.parquet``, look up the
           logical key ``(ks_id, triang_id, tdf_conifold_id)`` in the
           new TDF ``conifold_catalog.parquet``.  Rows whose key is
           missing in the new TDF are marked ``tdf_status="orphaned"``
           rather than deleted (so associated vacua + run history are
           preserved).
        3. Refresh polytope-derived denormalised columns in
           ``catalog.parquet`` (``chi``, ``n_rigids_dual``, ``Q``).
        4. If ``pick_up_new_conifolds=True``: scan the new TDF for
           conifolds belonging to existing ``(ks_id, coni_class_id)``
           pairs but absent from KKLT.  (Anchored by
           ``one_face_divisor`` — only new conifolds sharing an
           existing anchor are picked up.)
        5. Recompute and write the new TDF fingerprint into
           ``schema.json``.

        Args:
            tdf_db (LCSDatabase | None): The TDF database to align
                against.  Defaults to a fresh
                ``LCSDatabase(dataset="tdf")``.
            pick_up_new_conifolds (bool): If ``True``, append newly
                discovered conifolds in existing classes to the
                catalog.  Defaults to ``False``.
            verbose (bool): Print progress markers.

        Returns:
            Dict[str, List[Any]]: Diff report with keys
            ``"orphaned"``, ``"added"``, ``"denorm_changed"``.
        """
        pd = _require_pandas()
        if tdf_db is None:
            tdf_db = self.tdf
        else:
            self._tdf_built = tdf_db

        report: Dict[str, List[Any]] = {
            "orphaned":       [],
            "added":          [],
            "denorm_changed": [],
        }

        self._ensure_catalog()
        self._ensure_conifold_catalog()
        tdf_db._ensure_catalog()
        tdf_db._ensure_conifold_catalog()

        # ---- orphan detection -------------------------------------------
        # Logical keys span ``(h11, h12, ks_id, triang_id, conifold_id)``;
        # ks_id alone is not unique across Hodge sectors.
        join_cols_tdf  = ["h11", "h12", "ks_id", "triang_id", "conifold_id"]
        join_cols_kklt = ["h11", "h12", "ks_id", "triang_id", "tdf_conifold_id"]

        new_keys_df = (
            tdf_db._conifold_catalog[join_cols_tdf]
            .drop_duplicates()
            .copy()
        )
        new_keys_df["__present"] = True

        cf = self._conifold_catalog.copy()
        if "tdf_status" not in cf.columns:
            cf["tdf_status"] = "ok"
        merged = cf.merge(
            new_keys_df.rename(columns={"conifold_id": "tdf_conifold_id"}),
            left_on  = join_cols_kklt,
            right_on = join_cols_kklt,
            how      = "left",
        )
        assert len(merged) == len(cf), (
            f"rebuild_links: merge produced {len(merged)} rows for a "
            f"KKLT catalog of {len(cf)} rows — check for duplicate "
            "(h11, h12, ks_id, triang_id, conifold_id) tuples in TDF."
        )
        ok_mask = merged["__present"].fillna(False).astype(bool).values
        cf["tdf_status"] = np.where(ok_mask, "ok", "orphaned")
        orphaned = cf.loc[
            ~ok_mask, ["ks_id", "coni_class_id", "coni_id"]
        ].astype(int)
        report["orphaned"] = list(map(tuple, orphaned.values.tolist()))

        cf.to_parquet(self.cache_dir / "conifold_catalog.parquet",
                      index=False)
        self._conifold_catalog = cf
        if verbose:
            print(f"  orphaned: {len(report['orphaned'])}")

        # ---- denormalised columns ----------------------------------------
        # Q and chi are derivable from (h11, h12); n_rigids_dual comes
        # from a per-polytope TDF fetch (LRU-cached).
        cat       = self._catalog.copy()
        old_rigid = cat["n_rigids_dual"].tolist()
        ks_arr    = cat["ks_id"].tolist()
        h11_arr   = cat["h11"].tolist()
        h12_arr   = cat["h12"].tolist()
        new_rigid: List[int] = []
        for i, (ks_id, h11_v, h12_v) in enumerate(
                zip(ks_arr, h11_arr, h12_arr)):
            try:
                poly = tdf_db.get_polytope(
                    int(ks_id), h11=int(h12_v), h12=int(h11_v),
                )
                data = poly.get("polytope_data", {}) or {}
                new_rigid.append(int(data.get("n_rigids_dual", -1)))
            except Exception:
                new_rigid.append(int(old_rigid[i]))

        new_rigid_s = pd.Series(new_rigid, index=cat.index, dtype=int)
        new_Q       = (cat["h11"] + cat["h12"] + 2).astype(int)
        new_chi     = (2 * (cat["h11"] - cat["h12"])).astype(int)

        changed_mask = (
            (new_rigid_s != cat["n_rigids_dual"])
            | (new_Q   != cat["Q"])
            | (new_chi != cat["chi"])
        )
        report["denorm_changed"] = [
            int(k) for k in cat.loc[changed_mask, "ks_id"].tolist()
        ]

        cat["n_rigids_dual"] = new_rigid_s
        cat["Q"]             = new_Q
        cat["chi"]           = new_chi
        cat.to_parquet(self.cache_dir / "catalog.parquet", index=False)
        self._catalog = cat
        if verbose:
            print(f"  denorm refreshed: "
                  f"{len(report['denorm_changed'])} polytopes updated")

        if pick_up_new_conifolds:
            warnings.warn(
                "pick_up_new_conifolds=True is not supported; "
                "no new conifolds added.",
                stacklevel=2,
            )

        # ----------------------------------------------- fingerprint -----
        new_version = _read_tdf_schema_version(tdf_db)
        new_hash    = _compute_tdf_fingerprint(tdf_db)
        schema_path = self.cache_dir / "schema.json"
        if schema_path.exists():
            with open(schema_path) as f:
                schema = json.load(f)
        else:
            schema = {"schema_version": 1, "dataset": self._DATASET}
        schema["tdf_schema_version"] = new_version
        schema["tdf_catalog_sha256"] = new_hash
        with open(schema_path, "w") as f:
            json.dump(schema, f, indent=2)
        if verbose:
            print(
                f"  new TDF fingerprint: schema v{new_version}, "
                f"sha256={(new_hash or '')[:12]}..."
            )

        return report


# ----------------------------------------------------------------------------
# TDF-fingerprint helpers (used by _check_tdf_compat and the build script)
# ----------------------------------------------------------------------------


def _read_tdf_schema_version(tdf_db: LCSDatabase) -> Optional[int]:
    r"""
    **Description:**
    Return the schema version recorded in the wrapped TDF database's
    local ``schema.json``, or ``None`` if no such file is present.

    Args:
        tdf_db (LCSDatabase): Wrapped TDF database.

    Returns:
        int | None: Schema version, or ``None`` if unavailable.
    """
    import json

    path = Path(tdf_db.cache_dir) / "schema.json"
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f).get("schema_version")
    except (OSError, ValueError):
        return None


def _compute_tdf_fingerprint(tdf_db: LCSDatabase) -> Optional[str]:
    r"""
    **Description:**
    Compute a sha256 fingerprint of the TDF catalog columns KKLT
    depends on.  Returns ``None`` if the catalog is not available.

    The fingerprint covers the logical-key columns of the main catalog
    (``ks_id, triang_id``) and of the conifold sub-catalog
    (``ks_id, triang_id, conifold_id``).  Any change to these columns
    will trigger a TDF-fingerprint mismatch on next instantiation of a
    :class:`KKLTDatabase` built against the previous version.

    Args:
        tdf_db (LCSDatabase): Wrapped TDF database.

    Returns:
        str | None: Hexadecimal sha256 digest, or ``None`` if any
        required catalog is unavailable.
    """
    import hashlib

    pd = _require_pandas()

    try:
        tdf_db._ensure_catalog()
        tdf_db._ensure_conifold_catalog()
    except Exception:
        return None

    cat = tdf_db._catalog
    cf  = tdf_db._conifold_catalog
    if cat is None or cf is None:
        return None

    # Sort to make the fingerprint order-independent.  Use only the
    # logical-key columns so a TDF rebuild that re-shards but keeps the
    # logical keys does not appear as a mismatch.
    main = cat[["ks_id", "triang_id"]].sort_values(["ks_id", "triang_id"])
    conf = cf[["ks_id", "triang_id", "conifold_id"]].sort_values(
        ["ks_id", "triang_id", "conifold_id"]
    )

    digest = hashlib.sha256()
    digest.update(pd.util.hash_pandas_object(main, index=False).values.tobytes())
    digest.update(pd.util.hash_pandas_object(conf, index=False).values.tobytes())
    return digest.hexdigest()


__all__ = [
    "DEFAULT_KKLT_HF_REPO",
    "KKLTDatabase",
    "_RUN_LOG_COLUMNS",
    "_compute_tdf_fingerprint",
    "_read_tdf_schema_version",
    "_resolve_kklt_hf_repo",
]
