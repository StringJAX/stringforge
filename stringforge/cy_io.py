# ==============================================================================
# stringforge / cy_io
#
# Pure-I/O layer for the HuggingFace-hosted cy-database.  Contains the core
# CYDatabase class and the two sub-dataset convenience subclasses (TDFDatabase,
# CICYDatabase) along with parquet / HF download plumbing, caching, catalog
# queries, and schema versioning.
#
# Sits at the dependency root of the stringforge package — vacua_writer and
# lcs_database import from this module, not the other way round.  Also free of
# jaxvacua imports, so the sibling package can read/write the cache without
# circular dependencies.
# ==============================================================================

import datetime
import hashlib
import os
import json
import threading
import uuid
import warnings
from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union
import numpy as np
HF_REPO_ID: str = os.environ.get("STRINGFORGE_HF_REPO", "aschachner/cy-database")

_CACHE_SIZE_WARNING_MB: int = 500

def _get_default_data_dir() -> str:
    r"""
    **Description:**
    Return the current global data directory from ``stringforge.data_dir``.

    Falls back to ``~/.stringforge_cache`` if the package-level global
    isn't set yet (i.e. when ``cy_io`` is imported before
    ``stringforge/__init__.py`` finishes executing, or when the module
    is used standalone for some reason).

    Returns:
        str: Path to the data directory.
    """
    try:
        from . import data_dir
        return data_dir
    except ImportError:
        return os.path.join(Path.home(), ".stringforge_cache")


DEFAULT_CACHE_DIR: str = _get_default_data_dir()

VAULT_DIRNAME: str = "vacua_vault"

DEFAULT_VAULT_REPO: str = "aschachner/vacua_vault"

def _resolve_vault_repo() -> str:
    r"""
    **Description:**
    Return the HuggingFace dataset repo ID for vacua uploads.  Honours
    ``STRINGFORGE_VAULT_REPO`` env var, falling back to
    :data:`DEFAULT_VAULT_REPO`.
    """
    return os.environ.get("STRINGFORGE_VAULT_REPO", DEFAULT_VAULT_REPO)


def _find_stringforge_repo_root(start: Optional[Path] = None) -> Optional[Path]:
    r"""
    **Description:**
    Walk up from *start* (default: cwd) looking for a stringforge source
    checkout.  A directory counts as the repo root when it contains both
    a ``setup.py`` with ``name="stringforge"`` and a ``stringforge/``
    package directory.

    Args:
        start (Path | None): Starting directory.  Defaults to cwd.

    Returns:
        Path | None: Absolute path to the repo root, or None if not
        found.
    """
    current = Path(start or os.getcwd()).resolve()
    for candidate in [current, *current.parents]:
        setup_py = candidate / "setup.py"
        pkg_dir  = candidate / "stringforge"
        if setup_py.is_file() and pkg_dir.is_dir():
            try:
                text = setup_py.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if 'name="stringforge"' in text or "name='stringforge'" in text:
                return candidate
    return None


def _resolve_vault_dir() -> Path:
    r"""
    **Description:**
    Resolve the location of the permanent vacuum-solutions directory
    (``vacua_vault/``).

    Resolution order — uses whichever stage succeeds first:

    1. **``STRINGFORGE_VAULT`` env var** — explicit override. Recommended
       for any real workflow.  Set via ``export STRINGFORGE_VAULT=...``,
       ``os.environ["STRINGFORGE_VAULT"] = ...``, or
       :func:`stringforge.set_vault_dir`.
    2. **stringforge source checkout** — walks up from cwd looking for a
       directory containing ``setup.py`` (with ``name="stringforge"``) and
       a ``stringforge/`` package directory.  Returns
       ``<repo_root>/vacua_vault/``.
    3. **No fallback.**  Raises :class:`LookupError` with explicit
       instructions for how to configure the vault.  This is
       **deliberate**: silently creating ``vacua_vault/`` next to the
       current working directory used to scatter empty vault folders
       across users' filesystems.  Forcing an explicit choice prevents
       fragmentation.

    The directory is **not** created by this function; callers create
    it on demand (typically inside :class:`VacuaWriter` writer methods
    via ``mkdir(parents=True, exist_ok=True)``).

    Returns:
        Path: Absolute path to the vault directory (may not exist yet).

    Raises:
        LookupError: If neither ``STRINGFORGE_VAULT`` is set nor cwd is
            inside a stringforge source checkout.
    """
    override = os.environ.get("STRINGFORGE_VAULT")
    if override:
        return Path(override).expanduser().resolve()

    repo_root = _find_stringforge_repo_root()
    if repo_root is not None:
        return repo_root / VAULT_DIRNAME

    raise LookupError(
        "Could not resolve a vault directory.\n\n"
        "  STRINGFORGE_VAULT env var:  not set\n"
        "  stringforge repo root:      not found in cwd or any parent\n\n"
        "Set the vault explicitly:\n"
        "  - export STRINGFORGE_VAULT=/path/to/vault\n"
        "  - stringforge.set_vault_dir('/path/to/vault')\n"
        "  - os.environ['STRINGFORGE_VAULT'] = '/path/to/vault'\n\n"
        "To opt into a cwd-local vault for a quick demo, set\n"
        "STRINGFORGE_VAULT=. (or any explicit path)."
    )


_DATASET_CONFIGS: Dict[str, str] = {
    "tdf":        "KS",
    "cicy":       "CICY",
    "kklt":       "KKLT",
}

SCHEMA_VERSION: int = 1

SCHEMA_CHANGELOG: Dict[int, str] = {
    1: "Initial versioned schema. Conifold basis-change matrix stored as "
       "'basis_change' column; 'n_conifolds' and 'D3_tadpole' added to "
       "catalog.parquet; conifold_catalog.parquet introduced.  "
       "The 'kklt' sub-dataset adopts v1 with an extended catalog "
       "layout: polytope-grain catalog.parquet (one row per ks_id) with "
       "n_rigids_dual / Q / n_coni_classes; conifold_class_catalog.parquet "
       "keyed by (ks_id, coni_class_id); conifold_catalog.parquet keyed by "
       "(ks_id, coni_class_id, coni_id) carrying the logical TDF link "
       "(triang_id, tdf_conifold_id); optional tag columns record "
       "curation and release-readiness metadata; schema.json carries a "
       "TDF-compat fingerprint (tdf_schema_version, tdf_catalog_sha256).",
}

class SchemaVersionError(RuntimeError):
    r"""
    Raised when the schema version of the local cache does not match the
    version expected by this release of the code.
    """


class ValidationError(RuntimeError):
    r"""
    Raised when vacuum solutions fail validation checks before designation.
    The ``report`` attribute contains per-solution diagnostics.
    """

    def __init__(self, message: str, report: Optional[List[dict]] = None):
        """Initialise with message and optional per-solution diagnostic report."""
        super().__init__(message)
        self.report = report or []


def _require_pandas() -> Any:
    r"""
    **Description:**
    Import and return the ``pandas`` module, raising a clear error if absent.
    """
    try:
        import pandas as pd
        return pd
    except ImportError:
        raise ImportError(
            "The 'pandas' package is required to use CYDatabase.  "
            "Install it with:  pip install pandas"
        )


def _require_pyarrow() -> Any:
    r"""
    **Description:**
    Import and return the ``pyarrow`` module, raising a clear error if absent.
    """
    try:
        import pyarrow.parquet as pq
        return pq
    except ImportError:
        raise ImportError(
            "The 'pyarrow' package is required to use CYDatabase.  "
            "Install it with:  pip install pyarrow"
        )


def _require_hf_hub() -> Any:
    r"""
    **Description:**
    Import and return the ``hf_hub_download`` function, raising a clear
    error if ``huggingface_hub`` is absent.
    """
    try:
        from huggingface_hub import hf_hub_download
        return hf_hub_download
    except ImportError:
        raise ImportError(
            "The 'huggingface_hub' package is required to use CYDatabase.  "
            "Install it with:  pip install huggingface-hub"
        )


def _require_hf_api() -> Any:
    r"""
    **Description:**
    Import and return a fresh ``HfApi`` instance, raising a clear error
    if ``huggingface_hub`` is absent.  Used for write operations
    (uploading files, opening pull requests, identifying the
    authenticated user).
    """
    try:
        from huggingface_hub import HfApi
        return HfApi()
    except ImportError:
        raise ImportError(
            "The 'huggingface_hub' package is required for upload "
            "operations. Install it with:  pip install huggingface-hub"
        )


def _safe_concat(frames: List[Any], **kwargs: Any) -> Any:
    r"""
    Concatenate DataFrames, filtering out empty frames to avoid the pandas
    ``FutureWarning`` about all-NA columns during concatenation.
    """
    pd = _require_pandas()
    non_empty = [f for f in frames if len(f) > 0]
    if not non_empty:
        return pd.DataFrame()
    return pd.concat(non_empty, **kwargs)


def _decode_array(value: Any) -> Optional[np.ndarray]:
    r"""
    **Description:**
    Convert a parquet column entry to a numpy array.

    Parquet stores multi-dimensional arrays either as nested lists or as
    bytes-serialised numpy arrays (depending on the serialisation scheme used
    when the dataset was written).  This helper normalises both cases.

    Args:
        value: Raw column value from a parquet row.

    Returns:
        ``np.ndarray`` if the value is array-like, ``None`` if the value is
        ``None`` or a pandas ``NA``.
    """
    if value is None:
        return None
    try:
        import pandas as pd
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, np.ndarray):
        return value
    return np.array(value)


def _decode_invariants(value: Any) -> Optional[np.ndarray]:
    r"""
    **Description:**
    Decode a GV/GW ``invariants`` column entry.  Like :func:`_decode_array`,
    but additionally parses **decimal-string** encodings back to exact Python
    ints: BPS invariants can exceed ``int64`` (which parquet cannot store), so
    the KKLT GV split records them as strings.  Numeric (``int64`` / float)
    encodings -- e.g. the TDF split -- pass through unchanged.

    Returns an ``object``-dtype array of Python ints for the string case (so
    arbitrary-precision values survive), otherwise the plain numeric array.
    """
    arr = _decode_array(value)
    if arr is None:
        return None
    if arr.dtype.kind in ("U", "S", "O"):
        try:
            return np.array([int(x) for x in arr.tolist()], dtype=object)
        except (TypeError, ValueError):
            return arr
    return arr


def _parse_gv_row(row: Any) -> Dict[str, Any]:
    r"""
    **Description:**
    Convert a single row from the ``gv`` parquet split into a dictionary of
    Gopakumar-Vafa / Gromov-Witten data.

    Expected columns: ``gv_charges``, ``gv_invariants``, ``gw_charges``,
    ``gw_invariants``.

    Args:
        row: A ``pandas.Series`` from the GV parquet file.

    Not every sub-dataset carries both families: the ``cicy`` ``gv`` split has no
    ``gw_charges``/``gw_invariants`` columns at all, whereas ``tdf``'s does.  A missing
    family is reported as ``None`` rather than as a dict of ``None``s, because
    :meth:`jaxvacua.lcs.lcs_tree._charges_from_gv_gw` guards on ``x is not None`` — handing
    it ``{"charges": None, "invariants": None}`` slipped past that guard and raised
    ``AttributeError: 'NoneType' object has no attribute 'shape'``.

    Returns:
        Dict[str, Any]: Dictionary with keys ``"GVs"``, ``"GWs"`` and
        ``"grading_vector"``.  ``"GVs"``/``"GWs"`` are each a dict with ``"charges"``
        and ``"invariants"`` arrays, or ``None`` when that family is absent.
    """
    def _family(charge_col: str, invariant_col: str) -> Optional[Dict[str, Any]]:
        charges = _decode_array(row.get(charge_col))
        invariants = _decode_invariants(row.get(invariant_col))
        if charges is None or invariants is None:
            return None
        return {"charges": charges, "invariants": invariants}

    return {
        "GVs": _family("gv_charges", "gv_invariants"),
        "GWs": _family("gw_charges", "gw_invariants"),
        "grading_vector": _decode_array(row.get("grading_vector")),
    }


def _parse_conifold_rows(rows: Any) -> List[Dict[str, Any]]:
    r"""
    **Description:**
    Convert all rows belonging to one model from the ``conifolds`` parquet
    split into a list of conifold-data dictionaries.

    Expected columns per row: ``conifold_id``, ``conifold_curve``, ``ncf``,
    ``basis_change``, ``flop_edge``, ``one_face_divisors``.

    Args:
        rows: A ``pandas.DataFrame`` containing all conifold rows for one model
            (i.e. all rows sharing the same ``ks_id`` and ``triang_id``).

    Returns:
        List[Dict[str, Any]]: One dict per conifold, sorted by ``conifold_id``.
    """
    result = []
    for _, row in rows.sort_values("conifold_id").iterrows():
        result.append({
            "conifold_id":       int(row["conifold_id"]),
            "conifold_curve":    _decode_array(row.get("conifold_curve")),
            "ncf":               int(row["ncf"]) if ("ncf" in row.index and row["ncf"] is not None) else 2,
            "basis_change":      _decode_array(row.get("basis_change")),
            "flop_edge":         _decode_array(row.get("flop_edge")),
            "one_face_divisors": _decode_array(row.get("one_face_divisors")),
        })
    return result


def _parse_polytope_row(row: Any) -> Dict[str, Any]:
    r"""
    **Description:**
    Convert a single row from the ``polytope`` parquet split into a dict.

    The mandatory column is ``polytope_points``.  Any additional columns
    produced by ``process_polytope`` (e.g. ``is_favorable``, ``volume``,
    ``glsm_charge_matrix``) are collected into a ``"polytope_data"`` sub-dict
    so they do not collide with ``lcs_tree`` constructor arguments.

    Args:
        row: A ``pandas.Series`` from the polytope parquet file.

    Returns:
        Dict[str, Any]: Contains ``"polytope_points"`` and optionally
        ``"polytope_data"`` with any extra fields.
    """
    _RESERVED = {"ks_id", "h11", "h12", "polytope_points"}
    result: Dict[str, Any] = {
        "polytope_points": _decode_array(row.get("polytope_points")),
    }
    extra = {
        k: (v.tolist() if hasattr(v, "tolist") else v)
        for k, v in row.items()
        if k not in _RESERVED and v is not None
    }
    if extra:
        result["polytope_data"] = extra
    return result


class _LRUShardCache:
    r"""
    Thread-safe LRU cache for parquet shard DataFrames.

    Keeps the ``maxsize`` most-recently-used ``(split, shard_id)`` shards in
    memory so that repeated loads within the same h11 bucket avoid redundant
    disk reads.

    Args:
        maxsize (int): Maximum number of shards to keep in memory.
    """

    def __init__(self, maxsize: int = 32) -> None:
        """Initialise the LRU shard cache with a given maximum size."""
        self._cache: OrderedDict = OrderedDict()
        self._maxsize = maxsize
        self._lock = threading.Lock()

    def get(self, key: tuple) -> Any:
        """Return the cached DataFrame for *key*, or ``None`` if not cached."""
        with self._lock:
            if key not in self._cache:
                return None
            self._cache.move_to_end(key)
            return self._cache[key]

    def put(self, key: tuple, value: Any) -> None:
        """Insert *value* under *key*, evicting the LRU entry if full."""
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            else:
                if len(self._cache) >= self._maxsize:
                    self._cache.popitem(last=False)
                self._cache[key] = value

    def clear(self) -> None:
        """Evict all cached shards."""
        with self._lock:
            self._cache.clear()

    def __len__(self) -> int:
        return len(self._cache)


class CYDatabase:
    r"""
    **Description:**
    Pure-I/O interface for Calabi-Yau geometry data stored in the
    HuggingFace-hosted ``cy-database``.

    The database is organised into sub-datasets (``dataset`` parameter):

    - ``"tdf"``: toric models from the Kreuzer-Skarke list, identified by
      ``(ks_id, triang_id)`` in catalogue convention.
    - ``"cicy"``: Complete Intersection Calabi-Yau models, identified by
      ``cicy_id``.
    - ``"kklt"``: curated KKLT search index with logical links back to TDF.

    ``CYDatabase`` downloads catalogues and parquet shards lazily, caches them
    locally, supports offline operation, and exposes tabular query helpers.  It
    intentionally has no JAXVacua dependency and does not construct
    ``lcs_tree`` or ``FluxVacuaFinder`` objects.  Use
    :class:`stringforge.lcs_database.LCSDatabase` for that model-loading bridge.

    Example usage::

        db = CYDatabase(dataset="tdf")
        rows = db.query(h11=2, has_conifolds=True)
        poly = db.get_polytope(ks_id=int(rows.iloc[0]["ks_id"]), h11=2)

        offline_db = CYDatabase.from_local("./.stringforge_cache", dataset="tdf")

    """


    def __init__(
        self,
        dataset: str = "tdf",
        hf_repo: str = HF_REPO_ID,
        cache_dir: Optional[str] = None,
        offline: bool = False,
        shard_cache_size: int = 32,
        cache_mode: str = "persistent",
    ) -> None:
        r"""
        **Description:**
        Initialise a :class:`CYDatabase` instance.

        The catalog parquet file is downloaded on first use and cached to
        ``cache_dir``.  Subsequent calls are served from disk.  Set
        ``offline=True`` to suppress any network access.

        Args:
            dataset (str): Sub-dataset identifier.  Must be one of
                ``"tdf"`` (Kreuzer-Skarke toric models), ``"cicy"``
                (Complete Intersection CY models), or ``"kklt"``
                (curated KKLT index).  Defaults to ``"tdf"``.
            hf_repo (str): HuggingFace repository identifier of the form
                ``"org/repo-name"``.  Defaults to :data:`HF_REPO_ID`.
            cache_dir (str | None): Local directory for cached parquet
                files and vacua storage.  If ``None`` (default), uses the
                global ``stringforge.data_dir`` (which defaults to
                ``{cwd}/.stringforge_cache`` and can be overridden via
                :func:`stringforge.set_data_dir` or the
                ``STRINGFORGE_DATA_DIR`` environment variable).
            offline (bool): If ``True``, only serve data from the local
                cache and raise an error on any cache miss.  Defaults to
                ``False``.
            shard_cache_size (int): Number of parquet shards to keep in
                memory after loading.  Repeated accesses to the same shard
                (common when iterating over models with the same h11) are
                served from memory without a disk read.  Set to ``0`` to
                disable the in-memory cache.  Defaults to ``32``.
            cache_mode (str): Controls how downloaded shard files are
                managed.  ``"persistent"`` (default) keeps files on disk
                and in the LRU cache for repeated use.  ``"none"``
                downloads each shard, reads the needed row, then deletes
                the file — ideal for scanning millions of models without
                accumulating disk usage.

        Raises:
            ValueError: If ``dataset`` is not a recognised identifier.
        """
        if cache_dir is None:
            cache_dir = _get_default_data_dir()

        if dataset not in _DATASET_CONFIGS:
            raise ValueError(
                f"Unknown dataset '{dataset}'.  "
                f"Choose one of: {list(_DATASET_CONFIGS)}"
            )

        self.dataset    = dataset
        self.hf_repo    = hf_repo
        self.cache_dir  = Path(cache_dir) / dataset
        self.offline    = offline
        self.cache_mode = cache_mode

        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Catalog DataFrames — loaded lazily on first use.
        self._catalog              = None
        self._conifold_catalog     = None
        self._vacua_catalog        = None
        self._designated_catalog   = None

        # In-memory LRU cache for recently loaded parquet shards.
        self._shard_cache = _LRUShardCache(maxsize=shard_cache_size)

        # One-time flag for the cache-size warning.
        self._size_warned = False

        # Lock that serialises file downloads so concurrent threads never
        # race to download the same shard simultaneously.
        self._download_lock = threading.Lock()

    @classmethod
    def from_local(
        cls,
        path: Union[str, "Path"],
        dataset: Optional[str] = None,
        shard_cache_size: int = 32,
    ) -> "CYDatabase":
        r"""
        **Description:**
        Create a :class:`CYDatabase` backed by a local directory, with no
        network access.

        This is the recommended way to work with a database that was built
        locally (e.g. via ``build_tdf_database``).

        The ``path`` should point to the directory that **contains** the
        sub-dataset folder.  For example, if the catalog lives at
        ``/data/mydb/tdf/catalog.parquet``, pass ``path="/data/mydb"``.
        If ``path`` itself already ends with the dataset name (e.g.
        ``/data/mydb/tdf``), the parent is used automatically.

        Example::

            db = CYDatabase.from_local("/data/mydb")
            db = CYDatabase.from_local("/data/mydb/tdf")   # also works
            cat = db.query(h11=3)

        Args:
            path (str | Path): Root directory of the local database.
            dataset (str): Sub-dataset identifier (``"tdf"``, ``"cicy"``,
                or ``"kklt"``).
            shard_cache_size (int): In-memory shard cache size.

        Returns:
            CYDatabase: A fully offline database instance.
        """
        # Dataset resolution:
        #   1. explicit `dataset=` kwarg wins
        #   2. subclass default (e.g. TDFDatabase._DATASET = "tdf")
        #   3. fallback to "tdf"
        if dataset is None:
            dataset = getattr(cls, "_DATASET", "tdf")
        path = Path(path)
        # If the user passed the sub-dataset directory itself, step up.
        if path.name == dataset and (path / "catalog.parquet").exists():
            path = path.parent
        return cls(
            dataset=dataset,
            cache_dir=str(path),
            offline=True,
            shard_cache_size=shard_cache_size,
        )

    def __repr__(self) -> str:
        r"""
        **Description:**
        Returns a string representation of the database object.

        Returns:
            str: Description of the database.
        """
        status = "catalog loaded" if self._catalog is not None else "catalog not yet loaded"
        return (
            f"CYDatabase(dataset='{self.dataset}', "
            f"hf_repo='{self.hf_repo}', {status})"
        )

    def _check_schema(self) -> None:
        r"""
        **Description:**
        Verify that the locally cached database was built against the same
        schema version that this release of the code expects.

        Downloads ``schema.json`` from the HuggingFace repository on first
        call (or reads it from the local cache).  Compares the stored
        ``schema_version`` against :data:`SCHEMA_VERSION` and raises
        :exc:`SchemaVersionError` if they differ, with a message that
        explains exactly what changed and how to fix the problem.

        When ``offline=True`` and no ``schema.json`` is cached, the check is
        skipped with a warning rather than blocking the user.

        Raises:
            SchemaVersionError: If the cached schema version does not match
                :data:`SCHEMA_VERSION`.
        """
        path = self.cache_dir / "schema.json"

        if not path.exists():
            if self.offline:
                warnings.warn(
                    "schema.json not found in local cache and offline=True; "
                    "schema version check skipped.",
                    stacklevel=3,
                )
                return
            try:
                self._download_file(f"{self.dataset}/schema.json", path)
            except RuntimeError:
                # schema.json may not exist yet for databases built before
                # versioning was introduced — skip the check gracefully.
                warnings.warn(
                    "schema.json not found in the remote database; "
                    "schema version check skipped.  Consider rebuilding "
                    "the database to enable versioning.",
                    stacklevel=3,
                )
                return

        with open(path) as f:
            stored = json.load(f).get("schema_version")

        if stored is None or stored == SCHEMA_VERSION:
            return

        if stored < SCHEMA_VERSION:
            changes = "\n".join(
                f"  v{v}: {msg}"
                for v, msg in SCHEMA_CHANGELOG.items()
                if v > stored
            )
            raise SchemaVersionError(
                f"Local cache has schema v{stored}, but stringforge expects "
                f"v{SCHEMA_VERSION}.\n"
                f"Changes since your cached version:\n{changes}\n"
                f"Run db.clear_cache() to delete the stale cache and "
                f"re-download the latest database."
            )

        raise SchemaVersionError(
            f"Local cache has schema v{stored}, but this version of stringforge "
            f"only supports up to v{SCHEMA_VERSION}.  "
            f"Please upgrade stringforge."
        )

    def clear_cache(
        self,
        include_vacua: bool = False,
        include_designated: bool = False,
        keep_vault: bool = True,
        remove_dir: bool = False,
    ) -> None:
        r"""
        **Description:**
        Delete all locally cached files for this dataset and reset the
        in-memory catalog.

        After calling this method, the next access will re-download the
        catalog, schema, and any requested shards from the HuggingFace
        repository.  Useful after a schema version mismatch or when the
        remote database has been updated.

        The **permanent vacua vault** (``vacua_vault/`` — see
        :func:`_resolve_vault_dir`) is **never** touched by this method:
        it lives outside the cache directory and is protected by design.
        The *include_designated* flag only controls the legacy
        ``<cache_dir>/designated_vacua/`` location (used as a fallback
        for pre-vault installations).

        By default, session vacua (``<cache_dir>/vacua/``) and the
        legacy designated dir are preserved.  Pass
        ``include_vacua=True`` / ``include_designated=True`` to delete
        them as well.

        By default the **directory itself is recreated empty** so the
        same database instance can keep working — the next
        :meth:`load` call expects the directory to exist.  Pass
        ``remove_dir=True`` to also remove the parent directory itself
        (use this when you're done with the database for good and want
        ``.stringforge_cache/`` gone from your filesystem).

        Args:
            include_vacua (bool): If True, also delete the
                ``<cache_dir>/vacua/`` subdirectory containing session
                vacuum solutions.  Defaults to ``False``.
            include_designated (bool): If True, also delete the legacy
                ``<cache_dir>/designated_vacua/`` subdirectory.  The
                modern vault (``vacua_vault/``) is unaffected and must
                be removed manually if desired.  Defaults to ``False``.
            keep_vault (bool): Reserved for symmetry; ``vacua_vault/``
                is always protected and this flag has no effect in
                normal use.  Defaults to ``True``.
            remove_dir (bool): If True, also delete ``self.cache_dir``
                itself rather than recreating it empty.  Implies
                ``include_vacua=True`` and ``include_designated=True``
                (you cannot remove the parent while keeping its
                children).  After this call, any further use of the
                database instance will recreate the directory on first
                write/download.  Defaults to ``False``.

        Raises:
            RuntimeError: If ``offline=True``, since clearing the cache
                would leave the database inaccessible.
        """
        if self.offline:
            raise RuntimeError(
                "Cannot clear cache while offline=True — doing so would "
                "leave the database inaccessible.  Set offline=False first."
            )
        # keep_vault is documented for API symmetry; the vault lives
        # outside self.cache_dir so this method cannot touch it anyway.
        del keep_vault
        if remove_dir:
            # Removing the parent implies wiping any "protected" children too.
            include_vacua = True
            include_designated = True
        import shutil
        vacua_dir      = self.cache_dir / "vacua"
        designated_dir = self.cache_dir / "designated_vacua"
        protected: set = set()
        if not include_vacua and vacua_dir.exists():
            protected.add(vacua_dir)
        if not include_designated and designated_dir.exists():
            protected.add(designated_dir)

        if protected:
            for item in self.cache_dir.iterdir():
                if item in protected:
                    continue
                if item.is_dir():
                    shutil.rmtree(item, ignore_errors=True)
                else:
                    item.unlink(missing_ok=True)
        else:
            shutil.rmtree(self.cache_dir, ignore_errors=True)
        # Recreate the directory unless the caller asked to nuke it
        # outright.  When ``remove_dir=True``, leave it gone — the
        # directory will be recreated on demand by the next write or
        # download.  Also prune empty parent directories *one level up*
        # (e.g. when ``cache_dir`` is ``.stringforge_cache/tdf``, take
        # ``.stringforge_cache`` down too if it now sits empty).  Never
        # cross above one level — protects against accidental cleanup
        # of unrelated directories the user happened to put in the
        # parent.
        if not remove_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        else:
            parent = self.cache_dir.parent
            try:
                parent.rmdir()           # only succeeds if empty
            except OSError:
                pass                     # parent has other content; leave it
        self._catalog              = None
        self._conifold_catalog     = None
        self._vacua_catalog        = None
        self._designated_catalog   = None
        self._shard_cache.clear()

    def _check_cache_size(self) -> None:
        r"""
        **Description:**
        Issue a one-time warning if the data directory exceeds
        :data:`_CACHE_SIZE_WARNING_MB` (default 500 MB).

        Called automatically on the first :meth:`load` or
        :meth:`vacua_writer` invocation.

        Returns:
            None
        """
        if self._size_warned:
            return
        try:
            total = sum(
                f.stat().st_size
                for f in self.cache_dir.rglob("*")
                if f.is_file()
            )
        except OSError:
            return
        if total > _CACHE_SIZE_WARNING_MB * 1024 * 1024:
            warnings.warn(
                f"stringforge data directory at '{self.cache_dir}' is "
                f"{total / 1024**2:.0f} MB. "
                f"Use db.clear_cache() to free space.",
                stacklevel=3,
            )
        self._size_warned = True

    def _ensure_catalog(self) -> None:
        r"""
        **Description:**
        Download the catalog parquet file if not already cached, then load it
        into ``self._catalog`` as a ``pandas.DataFrame``.

        The catalog is the central index: it maps every ``(ks_id, triang_id)``
        (or ``cicy_id``) to shard pointers for all data splits, enabling O(1)
        lookups without scanning the full dataset.

        Raises:
            FileNotFoundError: If ``offline=True`` and the catalog is not in
                the local cache.
            SchemaVersionError: If the cached database schema does not match
                :data:`SCHEMA_VERSION`.
        """
        if self._catalog is not None:
            return

        self._check_schema()

        pd = _require_pandas()
        catalog_path = self.cache_dir / "catalog.parquet"

        if not catalog_path.exists():
            if self.offline:
                raise FileNotFoundError(
                    f"Catalog not found in cache ({catalog_path}) and "
                    "offline=True.  Run with offline=False first to "
                    "download it."
                )
            self._download_file(f"{self.dataset}/catalog.parquet", catalog_path)

        self._catalog = pd.read_parquet(catalog_path)

    def _ensure_conifold_catalog(self) -> None:
        r"""
        **Description:**
        Download ``conifold_catalog.parquet`` if not already cached, then load
        it into ``self._conifold_catalog``.  One row per conifold limit with
        columns ``ks_id``, ``triang_id``, ``conifold_id``, ``h11``, ``h12``,
        ``ncf``, ``conifold_curve``, ``conifold_shard_id``,
        ``conifold_row_index``.

        Raises:
            FileNotFoundError: If ``offline=True`` and the file is not cached,
                or if the database was built without conifold data.
        """
        if self._conifold_catalog is not None:
            return

        pd = _require_pandas()
        path = self.cache_dir / "conifold_catalog.parquet"

        if not path.exists():
            if self.offline:
                raise FileNotFoundError(
                    f"Conifold catalog not found in cache ({path}) and "
                    "offline=True.  Run with offline=False first to download it, "
                    "or rebuild the database to generate conifold_catalog.parquet."
                )
            self._download_file(f"{self.dataset}/conifold_catalog.parquet", path)

        self._conifold_catalog = pd.read_parquet(path)

    def _download_file(self, hf_path: str, local_path: Path) -> None:
        r"""
        **Description:**
        Download a single file from the HuggingFace repository to a local path.

        Args:
            hf_path (str): Path within the HuggingFace repository.
            local_path (Path): Destination path on the local filesystem.

        Raises:
            RuntimeError: If the download fails.
        """
        hf_hub_download = _require_hf_hub()
        local_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            # Use a no-op tqdm class to avoid ipywidgets/threading crashes
            # when called from a thread pool inside Jupyter notebooks.
            _quiet = type("_QuietTqdm", (), {
                "__init__": lambda s, *a, **kw: None,
                "__enter__": lambda s: s,
                "__exit__": lambda s, *a: None,
                "update": lambda s, *a, **kw: None,
                "close": lambda s: None,
            })
            downloaded = hf_hub_download(
                repo_id=self.hf_repo,
                filename=hf_path,
                repo_type="dataset",
                local_dir=str(local_path.parent),
                tqdm_class=_quiet,
            )
            # huggingface_hub may place the file in a subdirectory; move if needed.
            if Path(downloaded) != local_path:
                import shutil
                shutil.move(downloaded, local_path)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to download '{hf_path}' from '{self.hf_repo}': {exc}"
            ) from exc

    def _fetch_shard(self, split: str, shard_id: int) -> Any:
        r"""
        **Description:**
        Fetch a single parquet shard from the given split.

        Checks the in-memory LRU shard cache first.  On a cache miss the
        shard is read from disk (downloading it from HuggingFace if not yet
        cached locally), then stored in the LRU cache for future calls.

        Args:
            split (str): Name of the split directory, e.g. ``"lcs_data/h11_2"``
                or ``"gv"``.
            shard_id (int): Integer shard index.

        Returns:
            pandas.DataFrame: Contents of the shard.
        """
        key = (split, shard_id)

        cached = self._shard_cache.get(key)
        if cached is not None:
            return cached

        pd = _require_pandas()
        shard_filename = f"data-{shard_id:05d}.parquet"
        local_path = self.cache_dir / split / shard_filename

        if not local_path.exists():
            with self._download_lock:
                # Re-check inside the lock in case another thread downloaded
                # the file while this thread was waiting.
                if not local_path.exists():
                    if self.offline:
                        raise FileNotFoundError(
                            f"Shard '{split}/{shard_filename}' not found in cache "
                            f"({local_path}) and offline=True."
                        )
                    hf_path = f"{self.dataset}/{split}/{shard_filename}"
                    self._download_file(hf_path, local_path)

        df = pd.read_parquet(local_path)
        self._shard_cache.put(key, df)
        return df

    def _cleanup_shard(self, local_path: Path) -> None:
        r"""
        **Description:**
        Delete a shard file and remove directories left by
        ``hf_hub_download``.

        ``hf_hub_download`` creates ``.cache/huggingface/`` metadata
        trees and nested ``tdf/…`` subdirectories inside each split
        folder.  This helper removes the parquet file itself, any
        ``.cache`` tree in the same split directory, and prunes empty
        parent directories up to (but not including) ``self.cache_dir``.

        Args:
            local_path (Path): Path to the parquet shard file on disk.

        Returns:
            None
        """
        import shutil

        # Delete the parquet file.
        local_path.unlink(missing_ok=True)

        # Remove the .cache/huggingface tree that hf_hub_download creates
        # inside the split directory.
        split_dir = local_path.parent
        hf_cache = split_dir / ".cache"
        if hf_cache.exists():
            shutil.rmtree(hf_cache, ignore_errors=True)

        # Remove the nested tdf/… directories that hf_hub_download creates.
        nested = split_dir / self.dataset
        if nested.exists():
            shutil.rmtree(nested, ignore_errors=True)

        # Prune empty parent directories up to cache_dir.
        d = split_dir
        while d != self.cache_dir and d.exists():
            try:
                d.rmdir()          # only succeeds if empty
                d = d.parent
            except OSError:
                break              # not empty — stop climbing

    def _fetch_row(self, split: str, shard_id: int, row_index: int) -> Any:
        r"""
        **Description:**
        Fetch a single row from a parquet shard without caching the full
        DataFrame in memory.

        The shard file is downloaded to the local cache if not already
        present.  Only the requested row is read into memory using
        PyArrow.  When ``cache_mode="none"``, the shard file is deleted
        from disk after the row is extracted.

        Args:
            split (str): Name of the split directory, e.g.
                ``"lcs_data/h11_2"`` or ``"gv"``.
            shard_id (int): Integer shard index.
            row_index (int): Row offset within the shard.

        Returns:
            pandas.Series: The requested row.
        """
        import pyarrow.parquet as pq

        shard_filename = f"data-{shard_id:05d}.parquet"
        local_path = self.cache_dir / split / shard_filename

        if not local_path.exists():
            with self._download_lock:
                if not local_path.exists():
                    if self.offline:
                        raise FileNotFoundError(
                            f"Shard '{split}/{shard_filename}' not found "
                            f"in cache ({local_path}) and offline=True."
                        )
                    hf_path = f"{self.dataset}/{split}/{shard_filename}"
                    self._download_file(hf_path, local_path)

        table = pq.read_table(str(local_path))
        row = table.slice(row_index, 1).to_pandas().iloc[0]

        if self.cache_mode == "none":
            self._cleanup_shard(local_path)

        return row

    def _fetch_rows(self, split: str, shard_id: int, mask_fn: Any) -> Any:
        r"""
        **Description:**
        Fetch rows matching a filter from a parquet shard without caching
        the full DataFrame in memory.

        Like :meth:`_fetch_row` but applies a callable ``mask_fn`` to the
        loaded DataFrame and returns the matching subset.  Used for
        conifold data where multiple rows may belong to the same model.

        Args:
            split (str): Name of the split directory.
            shard_id (int): Integer shard index.
            mask_fn (callable): Function that takes a DataFrame and returns
                a boolean mask.

        Returns:
            pandas.DataFrame: Matching rows.
        """
        import pyarrow.parquet as pq

        shard_filename = f"data-{shard_id:05d}.parquet"
        local_path = self.cache_dir / split / shard_filename

        if not local_path.exists():
            with self._download_lock:
                if not local_path.exists():
                    if self.offline:
                        raise FileNotFoundError(
                            f"Shard '{split}/{shard_filename}' not found "
                            f"in cache ({local_path}) and offline=True."
                        )
                    hf_path = f"{self.dataset}/{split}/{shard_filename}"
                    self._download_file(hf_path, local_path)

        df = pq.read_table(str(local_path)).to_pandas()
        result = df[mask_fn(df)]

        if self.cache_mode == "none":
            self._cleanup_shard(local_path)

        return result

    def _lookup(
        self,
        ks_id: Optional[int],
        triang_id: Optional[int],
        cicy_id: Optional[int],
        h11: Optional[int] = None,
        h12: Optional[int] = None,
    ) -> Any:
        r"""
        **Description:**
        Look up the catalog row for a given model primary key.

        For ``tdf`` models the full primary key is
        ``(h11, h12, ks_id, triang_id)``.  When ``h11`` and ``h12`` are
        omitted, only ``(ks_id, triang_id)`` is used; if this yields multiple
        matches an error is raised asking the caller to disambiguate.

        Args:
            ks_id (int | None): Kreuzer-Skarke polytope index (tdf only).
            triang_id (int | None): Triangulation index (tdf only).
            cicy_id (int | None): CICY model index (cicy only).
            h11 (int | None): Hodge number :math:`h^{1,1}` (tdf only).
            h12 (int | None): Hodge number :math:`h^{1,2}` (tdf only).

        Returns:
            pandas.Series: The matching catalog row.

        Raises:
            KeyError: If no matching row is found.
            ValueError: If multiple rows match — supply ``h11`` and ``h12``
                to disambiguate.
        """
        self._ensure_catalog()
        cat = self._catalog

        if self.dataset == "tdf":
            mask = (cat["ks_id"] == ks_id) & (cat["triang_id"] == triang_id)
            if h11 is not None:
                mask = mask & (cat["h11"] == h11)
            if h12 is not None:
                mask = mask & (cat["h12"] == h12)
        else:
            mask = cat["cicy_id"] == cicy_id

        matches = cat[mask]
        if len(matches) == 0:
            key = f"ks_id={ks_id}, triang_id={triang_id}" if self.dataset == "tdf" \
                  else f"cicy_id={cicy_id}"
            if h11 is not None or h12 is not None:
                key += f", h11={h11}, h12={h12}"
            raise KeyError(
                f"No model found for {key} in the '{self.dataset}' dataset."
            )
        if len(matches) > 1:
            h_values = matches[["h11", "h12"]].drop_duplicates().values.tolist()
            raise ValueError(
                f"Multiple models match ks_id={ks_id}, triang_id={triang_id} "
                f"with (h11, h12) values {h_values}.  "
                f"Please pass h11 and h12 to disambiguate."
            )
        return matches.iloc[0]

    def info(self) -> None:
        r"""
        **Description:**
        Print a summary of the available models in this sub-dataset.

        The summary includes total model count, Hodge-number ranges, and the
        fraction of models with GV invariants and conifold data available.
        """
        self._ensure_catalog()
        cat = self._catalog

        n_total = len(cat)
        h11_min, h11_max = int(cat["h11"].min()), int(cat["h11"].max())
        h12_min, h12_max = int(cat["h12"].min()), int(cat["h12"].max())
        n_gv = int(cat["has_gv"].sum()) if "has_gv" in cat.columns else -1
        n_cf = int((cat["n_conifolds"] > 0).sum()) if "n_conifolds" in cat.columns else -1

        print(f"CYDatabase — sub-dataset: '{self.dataset}'")
        print(f"  Total models : {n_total:,}")
        print(f"  h11 range    : {h11_min} – {h11_max}")
        print(f"  h12 range    : {h12_min} – {h12_max}")
        if n_gv >= 0:
            print(f"  With GV data : {n_gv:,}  ({100*n_gv/n_total:.1f}%)")
        if n_cf >= 0:
            print(f"  With conifolds: {n_cf:,}  ({100*n_cf/n_total:.1f}%)")
        print(f"  Cache dir    : {self.cache_dir}")

    def query_conifolds(
        self,
        h11: Optional[int] = None,
        h12: Optional[int] = None,
        ncf: Optional[int] = None,
        conifold_id: Optional[int] = None,
        **filters: Any,
    ) -> Any:
        r"""
        **Description:**
        Filter the conifold sub-catalog and return a ``pandas.DataFrame`` of
        matching conifold limits.  Each row represents one conifold limit and
        carries enough information to load the corresponding model directly via
        :meth:`load_from_conifold_row`.

        This method operates entirely on the in-memory conifold catalog and
        does not load any geometry data.

        For filters not covered by the keyword arguments, use pandas on the
        returned DataFrame::

            df = db.query_conifolds()
            df[df["conifold_curve"].apply(lambda c: c == [0, 1])]

        Args:
            h11 (int | None): Filter by :math:`h^{1,1}`.
            h12 (int | None): Filter by :math:`h^{1,2}`.
            ncf (int | None): Filter by the GV invariant of the conifold node.
            conifold_id (int | None): Filter by conifold index within a model
                (0-based).
            **filters: Additional exact-match filters on any catalog column.

        Returns:
            pandas.DataFrame: Matching rows from the conifold sub-catalog.
        """
        self._ensure_conifold_catalog()
        cat = self._conifold_catalog
        mask = np.ones(len(cat), dtype=bool)

        if h11 is not None:
            mask &= (cat["h11"] == h11).values
        if h12 is not None:
            mask &= (cat["h12"] == h12).values
        if ncf is not None:
            mask &= (cat["ncf"] == ncf).values
        if conifold_id is not None:
            mask &= (cat["conifold_id"] == conifold_id).values
        for col, val in filters.items():
            if col in cat.columns:
                mask &= (cat[col] == val).values
            else:
                warnings.warn(f"Column '{col}' not found in conifold catalog; filter ignored.")

        return cat[mask].reset_index(drop=True)

    def query(
        self,
        h11: Optional[int] = None,
        h12: Optional[int] = None,
        has_conifolds: Optional[bool] = None,
        n_conifolds_min: Optional[int] = None,
        n_conifolds_max: Optional[int] = None,
        has_gv: Optional[bool] = None,
        D3_tadpole_max: Optional[int] = None,
        **filters: Any,
    ) -> Any:
        r"""
        **Description:**
        Filter the catalog and return a ``pandas.DataFrame`` of matching models.

        This method operates entirely on the in-memory catalog and does not
        download any geometry data.

        For filters not covered by the keyword arguments, operate on the
        returned DataFrame directly, e.g.::

            df = db.query()
            df[df["chi"] < -200]

        Args:
            h11 (int | None): Filter by :math:`h^{1,1}`.
            h12 (int | None): Filter by :math:`h^{1,2}`.
            has_conifolds (bool | None): If ``True``, return only models with
                at least one conifold limit available.
            n_conifolds_min (int | None): Return only models with at least this
                many conifold limits, e.g. ``n_conifolds_min=2``.
            n_conifolds_max (int | None): Return only models with at most this
                many conifold limits.
            has_gv (bool | None): If ``True``, return only models for which
                Gopakumar-Vafa invariants are stored.
            D3_tadpole_max (int | None): Return only models whose D3-tadpole
                is at most this value.
            **filters: Additional exact-match filters on catalog columns,
                e.g. ``chi=-240``.

        Returns:
            pandas.DataFrame: Matching rows from the catalog.
        """
        self._ensure_catalog()
        mask = np.ones(len(self._catalog), dtype=bool)

        if h11 is not None:
            mask &= (self._catalog["h11"] == h11).values
        if h12 is not None:
            mask &= (self._catalog["h12"] == h12).values
        if "n_conifolds" in self._catalog.columns:
            if has_conifolds is not None:
                if has_conifolds:
                    mask &= (self._catalog["n_conifolds"] > 0).values
                else:
                    mask &= (self._catalog["n_conifolds"] == 0).values
            if n_conifolds_min is not None:
                mask &= (self._catalog["n_conifolds"] >= n_conifolds_min).values
            if n_conifolds_max is not None:
                mask &= (self._catalog["n_conifolds"] <= n_conifolds_max).values
        if has_gv is not None and "has_gv" in self._catalog.columns:
            mask &= (self._catalog["has_gv"] == has_gv).values
        if D3_tadpole_max is not None and "D3_tadpole" in self._catalog.columns:
            mask &= (self._catalog["D3_tadpole"] <= D3_tadpole_max).values
        for col, val in filters.items():
            if col in self._catalog.columns:
                mask &= (self._catalog[col] == val).values
            else:
                warnings.warn(f"Column '{col}' not found in catalog; filter ignored.")

        return self._catalog[mask].reset_index(drop=True)

    def get_polytope(
        self,
        ks_id: int,
        h11: int,
        h12: Optional[int] = None,
    ) -> Dict[str, Any]:
        r"""
        **Description:**
        Fetch the polytope data for a model identified by ``(h11, ks_id)``,
        optionally disambiguated by ``h12``.

        Polytope data is stored per-polytope (not per-triangulation), so
        ``triang_id`` is not required: every triangulation of the same
        polytope shares one row in the ``polytope`` parquet split.  When
        multiple polytopes share the same ``(h11, ks_id)`` (which is rare),
        pass ``h12`` to disambiguate.

        Args:
            ks_id (int): Kreuzer-Skarke polytope index.
            h11 (int): Hodge number :math:`h^{1,1}`.
            h12 (int | None): Hodge number :math:`h^{1,2}`.  Optional.

        Returns:
            Dict[str, Any]: Contains ``"polytope_points"`` (an ``np.ndarray``
            of lattice points) and optionally ``"polytope_data"`` with any
            extra polytope-level fields stored by ``process_polytope`` such
            as ``is_favorable``, ``volume``, ``glsm_charge_matrix``.

        Raises:
            ValueError: If this database is not the ``tdf`` dataset, or the
                catalog has no ``polytope`` split, or the matching row has
                no polytope shard pointer.
            KeyError: If no row matches the supplied filters.
        """
        if self.dataset != "tdf":
            raise ValueError(
                f"get_polytope is only available for the 'tdf' dataset; "
                f"got dataset='{self.dataset}'."
            )

        self._ensure_catalog()
        cat = self._catalog
        mask = (cat["ks_id"] == ks_id) & (cat["h11"] == h11)
        if h12 is not None:
            mask = mask & (cat["h12"] == h12)
        matches = cat[mask]
        if len(matches) == 0:
            key = f"ks_id={ks_id}, h11={h11}"
            if h12 is not None:
                key += f", h12={h12}"
            raise KeyError(
                f"No polytope found for {key} in the '{self.dataset}' dataset."
            )

        if "polytope_shard_id" not in matches.columns:
            raise ValueError(
                "Catalog has no 'polytope_shard_id' column — this dataset does "
                "not provide a polytope split."
            )

        row = matches.iloc[0]
        psid = row["polytope_shard_id"]
        try:
            import pandas as pd
            _has_poly = psid is not None and not pd.isna(psid)
        except (TypeError, ValueError):
            _has_poly = psid is not None
        if not _has_poly:
            key = f"ks_id={ks_id}, h11={h11}"
            if h12 is not None:
                key += f", h12={h12}"
            raise ValueError(f"No polytope data available for {key}.")

        prid = row["polytope_row_index"]
        if self.cache_mode == "persistent":
            poly_shard = self._fetch_shard("polytope", int(psid))
            poly_row = poly_shard.iloc[int(prid)]
        else:
            poly_row = self._fetch_row("polytope", int(psid), int(prid))
            
        poly_data = _parse_polytope_row(poly_row)
        points    = poly_data["polytope_points"]            # np.ndarray
        points    = np.array([list(x) for x in points])
        poly_data["polytope_points"] = points
        return poly_data

    def _validate_key(
        self,
        ks_id: Optional[int],
        triang_id: Optional[int],
        cicy_id: Optional[int],
    ) -> None:
        r"""
        **Description:**
        Validate that the correct primary-key arguments are supplied for the
        active sub-dataset.

        Raises:
            ValueError: On missing or unexpected key arguments.
        """
        if self.dataset == "tdf":
            if ks_id is None or triang_id is None:
                raise ValueError(
                    "Both 'ks_id' and 'triang_id' are required for dataset='tdf'."
                )
        else:
            if cicy_id is None:
                raise ValueError("'cicy_id' is required for dataset='cicy'.")

    def _identifiers_to_list(
        self, identifiers: Any
    ) -> List[Tuple[Optional[int], Optional[int], Optional[int],
                     Optional[int], Optional[int]]]:
        r"""
        **Description:**
        Normalise ``identifiers`` to a list of
        ``(ks_id, triang_id, cicy_id, h11, h12)`` quintuples.

        Accepted input formats:

        - ``pandas.DataFrame`` with columns ``ks_id, triang_id`` (and
          optionally ``h11``, ``h12``) or ``cicy_id``.
        - List of ``(ks_id, triang_id)`` 2-tuples (for ``"tdf"``).
        - List of ``(h11, h12, ks_id, triang_id)`` 4-tuples (for ``"tdf"``,
          recommended to avoid ambiguity).
        - List of ``int`` values (for ``"cicy"``).

        Args:
            identifiers: See above.

        Returns:
            list of ``(ks_id, triang_id, cicy_id, h11, h12)`` quintuples.
        """
        pd = _require_pandas()
        result = []
        if isinstance(identifiers, pd.DataFrame):
            for _, row in identifiers.iterrows():
                if self.dataset == "tdf":
                    h11 = int(row["h11"]) if "h11" in row.index else None
                    h12 = int(row["h12"]) if "h12" in row.index else None
                    result.append((int(row["ks_id"]), int(row["triang_id"]),
                                   None, h11, h12))
                else:
                    result.append((None, None, int(row["cicy_id"]), None, None))
        elif isinstance(identifiers, (list, tuple)):
            for item in identifiers:
                if isinstance(item, (list, tuple)) and len(item) == 4:
                    # (h11, h12, ks_id, triang_id)
                    result.append((int(item[2]), int(item[3]), None,
                                   int(item[0]), int(item[1])))
                elif isinstance(item, (list, tuple)) and len(item) == 2:
                    # (ks_id, triang_id) — legacy format
                    result.append((int(item[0]), int(item[1]), None, None, None))
                elif isinstance(item, (int, np.integer)):
                    result.append((None, None, int(item), None, None))
                else:
                    raise ValueError(f"Cannot interpret identifier: {item!r}")
        else:
            raise TypeError(
                f"'identifiers' must be a DataFrame or a list of tuples/ints, "
                f"got {type(identifiers).__name__}."
            )
        return result

    @property
    def _vacua_dir(self) -> Path:
        r"""Root directory for vacua storage."""
        return self.cache_dir / "vacua"

    def _ensure_vacua_catalog(self) -> None:
        r"""Load (or reload) the vacua catalog from disk.

        Reloads when the on-disk catalog has been modified since the
        cached copy was loaded — keeps freshly-written runs visible to
        readers in the same Python session.  Tracks the file's
        modification time on ``self._vacua_catalog_mtime``.
        """
        pd = _require_pandas()
        catalog_path = self._vacua_dir / "vacua_catalog.parquet"
        if not catalog_path.exists():
            if self._vacua_catalog is None:
                self._vacua_catalog = pd.DataFrame()
            return
        cur_mtime = catalog_path.stat().st_mtime
        cached_mtime = getattr(self, "_vacua_catalog_mtime", None)
        if self._vacua_catalog is not None and cached_mtime == cur_mtime:
            return
        self._vacua_catalog = pd.read_parquet(catalog_path)
        self._vacua_catalog_mtime = cur_mtime

    @property
    def _designated_dir(self) -> Path:
        r"""
        Description:
        Root directory for designated (permanent) vacua storage.

        Resolves to the project-local ``vacua_vault/`` directory (see
        :func:`_resolve_vault_dir`).  Falls back to the legacy
        ``<cache_dir>/designated_vacua/`` location only when a user has
        pre-existing data there and the vault directory is empty — this
        preserves backwards compatibility without silent duplication.
        """
        vault = _resolve_vault_dir()
        legacy = self.cache_dir / "designated_vacua"
        # Prefer vault unless only legacy has data (one-time migration hint).
        if (not vault.exists()) and legacy.exists() and any(legacy.iterdir()):
            return legacy
        return vault

    # `_resolve_vacua_dir` was moved to stringforge.vacua_writer.VacuaWriter —
    # it depends on `_extract_model_identity`, which reads `lcs_tree`
    # attributes and therefore cannot live in this jaxvacua-dep-free layer.
    # Callers that used to do `db._resolve_vacua_dir(model=...)` should use
    # `VacuaWriter(db)._resolve_vacua_dir(model=...)` instead.  The delegated
    # method on LCSDatabase (stringforge/lcs_database.py) still exposes it
    # for convenience.

    def _ensure_designated_catalog(self) -> None:
        r"""Load designated vacua catalog from disk if not yet in memory."""
        if self._designated_catalog is not None:
            return
        pd = _require_pandas()
        catalog_path = self._designated_dir / "designated_vacua_catalog.parquet"
        if catalog_path.exists():
            self._designated_catalog = pd.read_parquet(catalog_path)
        else:
            self._designated_catalog = pd.DataFrame()


class TDFDatabase(CYDatabase):
    r"""
    **Description:**
    Convenience subclass of :class:`CYDatabase` pre-configured for the
    ``"tdf"`` sub-dataset (toric models from the Kreuzer-Skarke list).

    Models are identified by ``(ks_id, triang_id)`` in catalogue convention.

    Example usage::

        db = TDFDatabase()
        rows = db.query(h11=2, has_conifolds=True)
        poly = db.get_polytope(ks_id=int(rows.iloc[0]["ks_id"]), h11=2)
    """

    # Consumed by :meth:`CYDatabase.from_local` so that
    # ``TDFDatabase.from_local(path)`` infers ``dataset="tdf"``.
    _DATASET = "tdf"

    def __init__(
        self,
        hf_repo: str = HF_REPO_ID,
        cache_dir: Optional[str] = None,
        offline: bool = False,
        cache_mode: str = "persistent",
        shard_cache_size: int = 32,
        dataset: Optional[str] = None,
    ) -> None:
        r"""Initialise a :class:`TDFDatabase` instance.

        The ``dataset`` kwarg is accepted but must equal ``"tdf"`` if
        provided — it exists so that :meth:`CYDatabase.from_local` (which
        forwards ``dataset=...`` generically) can be called as
        ``TDFDatabase.from_local(...)``.
        """
        if dataset is not None and dataset != "tdf":
            raise ValueError(
                f"TDFDatabase requires dataset='tdf'; got {dataset!r}"
            )
        super().__init__(
            dataset="tdf",
            hf_repo=hf_repo,
            cache_dir=cache_dir,
            offline=offline,
            shard_cache_size=shard_cache_size,
            cache_mode=cache_mode,
        )


class CICYDatabase(CYDatabase):
    r"""
    **Description:**
    Convenience subclass of :class:`CYDatabase` pre-configured for the
    ``"cicy"`` sub-dataset (Complete Intersection Calabi-Yau models).

    Models are identified by ``cicy_id`` in catalogue convention.

    Example usage::

        db = CICYDatabase()
        rows = db.query(h11=1)
    """

    # Consumed by :meth:`CYDatabase.from_local` so that
    # ``CICYDatabase.from_local(path)`` infers ``dataset="cicy"``.
    _DATASET = "cicy"

    def __init__(
        self,
        hf_repo: str = HF_REPO_ID,
        cache_dir: Optional[str] = None,
        offline: bool = False,
        cache_mode: str = "persistent",
        shard_cache_size: int = 32,
        dataset: Optional[str] = None,
    ) -> None:
        r"""Initialise a :class:`CICYDatabase` instance.

        The ``dataset`` kwarg is accepted but must equal ``"cicy"`` if
        provided — it exists so that :meth:`CYDatabase.from_local` (which
        forwards ``dataset=...`` generically) can be called as
        ``CICYDatabase.from_local(...)``.
        """
        if dataset is not None and dataset != "cicy":
            raise ValueError(
                f"CICYDatabase requires dataset='cicy'; got {dataset!r}"
            )
        super().__init__(
            dataset="cicy",
            hf_repo=hf_repo,
            cache_dir=cache_dir,
            offline=offline,
            shard_cache_size=shard_cache_size,
            cache_mode=cache_mode,
        )


def query_models(
    dataset: str = "tdf",
    h11: Optional[int] = None,
    h12: Optional[int] = None,
    has_conifolds: Optional[bool] = None,
    n_conifolds_min: Optional[int] = None,
    n_conifolds_max: Optional[int] = None,
    has_gv: Optional[bool] = None,
    D3_tadpole_max: Optional[int] = None,
    cache_dir: str = DEFAULT_CACHE_DIR,
    **filters: Any,
) -> Any:
    r"""
    **Description:**
    Query the catalog of a sub-dataset and return matching rows as a
    ``pandas.DataFrame``, without instantiating a :class:`CYDatabase`
    explicitly.

    Args:
        dataset (str): Sub-dataset identifier (``"tdf"`` or ``"cicy"``).
        h11 (int | None): Filter by :math:`h^{1,1}`.
        h12 (int | None): Filter by :math:`h^{1,2}`.
        has_conifolds (bool | None): Filter by conifold availability.
        n_conifolds_min (int | None): Minimum number of conifold limits.
        n_conifolds_max (int | None): Maximum number of conifold limits.
        has_gv (bool | None): Filter by GV availability.
        D3_tadpole_max (int | None): Upper bound on the D3-tadpole.
        cache_dir (str): Local cache directory.
        **filters: Additional exact-match column filters.

    Returns:
        pandas.DataFrame: Matching rows from the catalog.
    """
    db = CYDatabase(dataset=dataset, cache_dir=cache_dir)
    return db.query(
        h11=h11, h12=h12,
        has_conifolds=has_conifolds,
        n_conifolds_min=n_conifolds_min,
        n_conifolds_max=n_conifolds_max,
        has_gv=has_gv,
        D3_tadpole_max=D3_tadpole_max,
        **filters,
    )


def load_catalog(
    path: Union[str, "Path", None] = None,
    dataset: str = "tdf",
    catalog: str = "catalog",
    cache_dir: str = DEFAULT_CACHE_DIR,
) -> Any:
    r"""
    **Description:**
    Load a catalog as a ``pandas.DataFrame`` from either a local database
    directory or the default HuggingFace cache.

    This is a lightweight entry point — it reads only the catalog parquet
    file, not any geometry data.

    Args:
        path (str | Path | None): Path to a local database.  Can point to
            the root (containing the ``tdf/`` or ``cicy/`` sub-folder) or
            directly to the sub-dataset folder.  If ``None``, the catalog
            is loaded from (or downloaded to) the default HuggingFace cache.
        dataset (str): Sub-dataset identifier (``"tdf"`` or ``"cicy"``).
            Defaults to ``"tdf"``.
        catalog (str): Which catalog to load.  One of:

            - ``"catalog"`` — main model catalog (default)
            - ``"conifold_catalog"`` — conifold sub-catalog
            - ``"designated_vacua_catalog"`` — designated (Tier 2) solutions

        cache_dir (str): Cache directory when ``path`` is ``None``.

    Returns:
        pandas.DataFrame: The requested catalog.

    Example::

        # From a local build
        cat = load_catalog("/data/mydb")
        cf  = load_catalog("/data/mydb", catalog="conifold_catalog")

        # From the online / cached HuggingFace repo
        cat = load_catalog()
    """
    if path is not None:
        db = CYDatabase.from_local(path, dataset=dataset)
    else:
        db = CYDatabase(dataset=dataset, cache_dir=cache_dir)

    if catalog == "catalog":
        db._ensure_catalog()
        return db._catalog
    elif catalog == "conifold_catalog":
        db._ensure_conifold_catalog()
        return db._conifold_catalog
    elif catalog == "designated_vacua_catalog":
        db._ensure_designated_catalog()
        return db._designated_catalog
    else:
        raise ValueError(
            f"Unknown catalog '{catalog}'.  Choose one of: "
            "'catalog', 'conifold_catalog', 'designated_vacua_catalog'."
        )




__all__ = [
    # ----- constants & exceptions --------------------------------------------
    "HF_REPO_ID", "DEFAULT_CACHE_DIR", "VAULT_DIRNAME", "DEFAULT_VAULT_REPO",
    "SCHEMA_VERSION", "SCHEMA_CHANGELOG",
    "SchemaVersionError", "ValidationError",
    # ----- parsing helpers ---------------------------------------------------
    "_decode_array", "_parse_gv_row", "_parse_conifold_rows", "_parse_polytope_row",
    # ----- classes -----------------------------------------------------------
    "CYDatabase", "TDFDatabase", "CICYDatabase",
    # ----- module-level convenience ------------------------------------------
    "load_catalog", "query_models",
]
