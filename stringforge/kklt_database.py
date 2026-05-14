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
# Three local catalogs (under ``<cache_dir>/kklt_vacua/``):
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
# Plus an append-only ``run_log.parquet`` for cluster-run provenance.
#
# Subset criterion (applied at build time): ``n_rigids_dual > h12``.
# ``Q := h11 + h12 + 2`` is a column on ``catalog.parquet`` but not a
# build-time filter; the historically well-studied ``Q >= 100`` subset is
# recovered by ``db.query_polytopes(Q_min=100)``.
# ==============================================================================

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Any, Optional

from .cy_io import (
    _DATASET_CONFIGS,
    _require_pandas,
)
from .lcs_database import LCSDatabase


# ----------------------------------------------------------------------------
# Module-level constants and env-var helpers
# ----------------------------------------------------------------------------

DEFAULT_KKLT_HF_REPO: str = "aschachner/kklt-vacua-database"


def _resolve_kklt_hf_repo() -> str:
    r"""
    **Description:**
    Return the HuggingFace dataset repo ID for the ``kklt_vacua``
    sub-dataset.  Honours the ``STRINGFORGE_KKLT_HF_REPO`` environment
    variable, falling back to :data:`DEFAULT_KKLT_HF_REPO`.

    Returns:
        str: ``"user/repo"`` HF Hub identifier.
    """
    return os.environ.get("STRINGFORGE_KKLT_HF_REPO", DEFAULT_KKLT_HF_REPO)


# ----------------------------------------------------------------------------
# KKLTDatabase
# ----------------------------------------------------------------------------


class KKLTDatabase(LCSDatabase):
    r"""
    **Description:**
    KKLT-vacua sub-dataset interface: a curated subset of
    :class:`~stringforge.lcs_database.LCSDatabase` (``dataset="tdf"``).

    The KKLT database does **not** duplicate Calabi-Yau geometry data.
    Every row in the local catalogs carries a *logical* link
    ``(ks_id, triang_id, tdf_conifold_id)`` into the full TDF dataset;
    physical shard coordinates are resolved on demand by the wrapped TDF
    database held on :attr:`tdf`.

    Models are indexed by ``(ks_id, coni_class_id, coni_id)``.  Conifolds
    sharing the same *one-face divisor* on a given polytope form a single
    ``coni_class``; this grouping cuts across triangulations of the same
    polytope.

    Three local catalogs live under ``<cache_dir>/kklt_vacua/``:

    - ``catalog.parquet`` (polytope-grain): one row per ``ks_id`` with
      ``h11``, ``h12``, ``chi``, ``n_rigids_dual``, ``Q``,
      ``n_coni_classes``.  All KKLT polytopes satisfy
      ``n_rigids_dual > h12``; the historically well-studied subset
      ``Q >= 100`` is a downstream DataFrame filter, not a build-time cut.
    - ``conifold_class_catalog.parquet``: one row per
      ``(ks_id, coni_class_id)``.
    - ``conifold_catalog.parquet``: one row per
      ``(ks_id, coni_class_id, coni_id)``, carrying the logical TDF link.

    An append-only ``run_log.parquet`` records cluster runs
    (``scope="class"`` or ``scope="conifold"``) for run-tracking and
    provenance.  See :meth:`start_run`, :meth:`finish_run`,
    :meth:`run_status`.

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
    # ``KKLTDatabase.from_local(path)`` infers ``dataset="kklt_vacua"``.
    _DATASET: str = "kklt_vacua"

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
                ``kklt_vacua`` parquet shards.  Defaults to the value
                returned by :func:`_resolve_kklt_hf_repo`.
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
                equal ``"kklt_vacua"`` if supplied.

        Raises:
            ValueError: If ``dataset`` is supplied and is not
                ``"kklt_vacua"``.
        """
        if dataset is not None and dataset != "kklt_vacua":
            raise ValueError(
                f"KKLTDatabase requires dataset='kklt_vacua'; got {dataset!r}"
            )

        super().__init__(
            dataset="kklt_vacua",
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
        self._tdf_built = LCSDatabase(
            dataset="tdf",
            cache_dir=str(self.cache_dir.parent),
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
    "_resolve_kklt_hf_repo",
]
