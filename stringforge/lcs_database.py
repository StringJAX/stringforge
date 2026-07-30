# ==============================================================================
# stringforge / lcs_database
#
# LCSDatabase — extends cy_io.CYDatabase with:
#
#   (a) methods that construct :class:`jaxvacua.lcs.lcs_tree` objects and
#       :class:`jaxvacua.flux_vacua_finder.FluxVacuaFinder` instances from
#       parquet-row data (load, load_from_conifold_row, load_model, load_batch,
#       iter_batch, sample, plus the batch-model variants load_model_batch
#       and iter_model_batch);
#
#   (b) a filter-keyed cache of the last constructed batch of FluxVacuaFinder
#       instances (cached_models property, clear_cached_models method);
#
#   (c) vacua-persistence delegation — `db.designate_vacua(...)` and friends
#       forward to a lazily-built :class:`stringforge.vacua_writer.VacuaWriter`
#       held on the instance.
# ==============================================================================

from __future__ import annotations

import hashlib
import json
import os
import warnings
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np

from .cy_io import (
    CYDatabase,
    SCHEMA_VERSION,
    SchemaVersionError,
    ValidationError,
    DEFAULT_CACHE_DIR,
    _DATASET_CONFIGS,
    _decode_array,
    _parse_gv_row,
    _parse_conifold_rows,
    _parse_polytope_row,
    _safe_concat,
    _require_pandas,
    _require_pyarrow,
)

from .vacua_writer import VacuaWriter


def _swap_h_columns(df: Any) -> Any:
    r"""Relabel the ``h11`` and ``h12`` columns of *df*.

    Used by :class:`LCSDatabase` to convert between the catalog-convention
    DataFrames returned by :class:`CYDatabase` and the mirror-convention
    surface presented by :class:`LCSDatabase`.  The swap is symmetric, so
    applying it twice returns the original DataFrame.
    """
    return df.rename(columns={"h11": "h12", "h12": "h11"})


#: Sub-datasets whose catalog already stores Hodge numbers in the **mirror**
#: convention that :class:`~jaxvacua.lcs.lcs_tree` expects, and which must
#: therefore *not* be swapped on the way in or out.
#:
#: ``cicy`` is the case: its catalog stores ``h11`` = the CY's :math:`h^{2,1}`
#: and ``h12`` = the CY's :math:`h^{1,1}` — e.g. ``cicy_id=7890`` (the quintic in
#: :math:`\mathbb{P}^4`, :math:`h^{1,1}=1`, :math:`h^{2,1}=101`) is stored as
#: ``h11=101, h12=1``.  Swapping it handed ``lcs_tree`` ``h12=101`` where the
#: geometry has rank 1, and since ``lcs.py`` dimensions the intersection tensor by
#: ``h12`` the result was a ``(101, 101, 101)`` tensor alongside a length-1 ``c2``.
#: ``tdf``/``kklt`` do store the catalog convention (small ``h11`` = rank) and are
#: unaffected — which is why the bug only ever showed up for ``cicy``.
_NO_H_SWAP_DATASETS = frozenset({"cicy"})


def _needs_h_swap(dataset: str) -> bool:
    r"""Whether *dataset*'s catalog Hodge numbers need the mirror swap."""
    return dataset not in _NO_H_SWAP_DATASETS


#: Sub-datasets whose ``gv/`` split is **flat** (a single bucket) rather than
#: bucketed as ``gv/h11_{N}/``.  Kept separate from
#: :data:`_NO_H_SWAP_DATASETS` on purpose: the GV layout and the Hodge-number
#: convention are independent properties that merely coincide for ``cicy``, and
#: conflating them would silently mis-route a future sub-dataset that had one but
#: not the other.
_FLAT_GV_DATASETS = frozenset({"cicy"})


def _parse_lcs_row(row: Any, dataset: str) -> Dict[str, Any]:
    r"""
    **Description:**
    Convert a single row from the ``lcs_data`` parquet split into a keyword-
    argument dictionary suitable for constructing an :class:`~jaxvacua.lcs.lcs_tree`.

    The expected column names in the parquet row are:

    .. code-block:: text

        h11, h12, chi,
        intnums_coo_i, intnums_coo_j, intnums_coo_k, intnums_coo_v,
        c2, a_matrix,
        hyperplanes, kahler_generators, kahler_rays, mori_rays,
        polytope_points,  # tdf only
        heights,          # tdf only
        extra_data        # JSON string

    Args:
        row: A ``pandas.Series`` corresponding to one row of the parquet file.
        dataset (str): Sub-dataset identifier (``"tdf"`` or ``"cicy"``).

    Returns:
        Dict[str, Any]: Keyword arguments for ``lcs_tree.__init__``.
    """
    # ---- Reconstruct sparse intersection-number array (COO format) ----------
    coo_i = _decode_array(row.get("intnums_coo_i"))
    coo_j = _decode_array(row.get("intnums_coo_j"))
    coo_k = _decode_array(row.get("intnums_coo_k"))
    coo_v = _decode_array(row.get("intnums_coo_v"))

    if all(x is not None for x in (coo_i, coo_j, coo_k, coo_v)):
        intnums_coo = np.stack([coo_i, coo_j, coo_k, coo_v], axis=1)
    else:
        intnums_coo = None

    # ---- Parse extra_data JSON string ---------------------------------------
    raw_extra = row.get("extra_data")
    if isinstance(raw_extra, str):
        try:
            extra_data = json.loads(raw_extra)
        except json.JSONDecodeError:
            extra_data = {}
    elif isinstance(raw_extra, dict):
        extra_data = raw_extra
    else:
        extra_data = {}

    # ---- Inject primary-key fields into extra_data -------------------------
    if dataset == "tdf":
        extra_data.setdefault("ks_id",     int(row.get("ks_id",     -1)))
        extra_data.setdefault("triang_id", int(row.get("triang_id",  0)))
        model_ID = int(row.get("ks_id", -1))
    else:
        extra_data.setdefault("cicy_id", int(row.get("cicy_id", -1)))
        model_ID = int(row.get("cicy_id", -1))

    kwargs = dict(
        h11          = int(row["h11"]),
        h12          = int(row["h12"]),
        #chi          = int(row["chi"])  if "chi"  in row.index else None,
        intnums_coo  = intnums_coo,
        c2           = _decode_array(row.get("c2")),
        a_matrix     = _decode_array(row.get("a_matrix")),
        hyperplanes  = _decode_array(row.get("hyperplanes")),
        kahler_generators = _decode_array(row.get("kahler_generators")),
        kahler_rays  = _decode_array(row.get("kahler_rays")),
        mori_rays    = _decode_array(row.get("mori_rays")),
        tip_skc    = _decode_array(row.get("tip_skc")),
        model_type   = _DATASET_CONFIGS[dataset],
        model_ID     = model_ID,
        extra_data   = extra_data or None,
    )

    # KS-specific fields
    if dataset == "tdf":
        kwargs["polytope_points"] = _decode_array(row.get("polytope_points"))
        kwargs["heights"]         = _decode_array(row.get("heights"))

    return kwargs



class LCSDatabase(CYDatabase):
    """Database layer that bridges cached geometry rows into JAXVacua.

    ``LCSDatabase`` extends the pure-I/O :class:`CYDatabase` with
    ``lcs_tree`` and ``FluxVacuaFinder`` construction, batch loading, sampling,
    and delegation to :class:`stringforge.vacua_writer.VacuaWriter` for vacua
    persistence.  It owns the data/convention boundary; the physical
    computations performed by returned models are owned by JAXVacua.
    """

    def __new__(cls, *args: Any, **kwargs: Any) -> "LCSDatabase":
        r"""Factory dispatch.

        When the user constructs ``LCSDatabase(dataset="kklt")`` or
        ``LCSDatabase("kklt")`` directly, return a
        :class:`~stringforge.kklt_database.KKLTDatabase` instance instead.
        Python then calls
        :meth:`KKLTDatabase.__init__` on the returned object (because
        ``isinstance(returned, LCSDatabase)``), so the dispatch is
        transparent to callers.
        """
        dataset = kwargs.get("dataset")
        if dataset is None and args:
            dataset = args[0]
        if cls is LCSDatabase and dataset == "kklt":
            from .kklt_database import KKLTDatabase
            return object.__new__(KKLTDatabase)
        return super().__new__(cls)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Filter-keyed cache populated by load_model_batch
        self._cached_models: Optional[List[Any]] = None
        self._cached_filter_key: Optional[Tuple[Any, ...]] = None

    # ------------------------------------------------------------------
    # Mirror-convention wrappers
    #
    # `LCSDatabase` exposes h11/h12 in the *mirror* convention used by
    # `lcs_tree` / jaxvacua, whereas the underlying cy-database catalog
    # stores them in the *catalog* convention (typically small h11, large
    # h12).  Every public method that accepts or returns h11/h12 swaps at
    # the boundary so the inside of the class continues to operate on
    # catalog-convention values.  See :func:`_swap_h_columns`.
    # ------------------------------------------------------------------
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
        r"""Filter the catalog and return matching models in mirror convention.

        Same semantics as :meth:`CYDatabase.query`, except both the
        ``h11`` / ``h12`` *filter kwargs* and the ``h11`` / ``h12``
        *columns of the returned DataFrame* are interpreted in the mirror
        (jaxvacua) convention.  In particular::

            LCSDatabase(...).query(h12=2)

        returns the same rows as ``CYDatabase(...).query(h11=2)``, with
        the two Hodge-number columns relabelled accordingly.

        The catalog ``chi`` column is dropped from the returned DataFrame:
        the catalog's :math:`\chi = 2(h^{1,1}_{\text{cat}} - h^{1,2}_{\text{cat}})`
        flips sign under the mirror swap, so exposing it here would be
        misleading.  This mirrors the ``chi=None`` line in :meth:`load`.
        """
        df = super().query(
            h11=h12,
            h12=h11,
            has_conifolds=has_conifolds,
            n_conifolds_min=n_conifolds_min,
            n_conifolds_max=n_conifolds_max,
            has_gv=has_gv,
            D3_tadpole_max=D3_tadpole_max,
            **filters,
        )
        return _swap_h_columns(df).drop(columns=["chi"], errors="ignore")

    def query_conifolds(
        self,
        h11: Optional[int] = None,
        h12: Optional[int] = None,
        ncf: Optional[int] = None,
        conifold_id: Optional[int] = None,
        **filters: Any,
    ) -> Any:
        r"""Filter the conifold sub-catalog in mirror convention.

        Mirror-convention wrapper around :meth:`CYDatabase.query_conifolds`.
        See :meth:`query` for the convention.  The catalog ``chi`` column
        is dropped from the returned DataFrame for the same reason as in
        :meth:`query` (sign flip under the mirror swap).
        """
        df = super().query_conifolds(
            h11=h12,
            h12=h11,
            ncf=ncf,
            conifold_id=conifold_id,
            **filters,
        )
        return _swap_h_columns(df).drop(columns=["chi"], errors="ignore")

    def get_polytope(  # type: ignore[override]
        self,
        ks_id: int,
        h11: Optional[int] = None,
        h12: Optional[int] = None,
    ) -> Dict[str, Any]:
        r"""Fetch polytope data for a model in mirror convention.

        Mirror-convention wrapper around :meth:`CYDatabase.get_polytope`.
        The parent's primary discriminator is the *catalog* ``h11``
        (typically small).  In mirror convention this is the user-supplied
        ``h12``, which is therefore required here even though both
        Hodge-number arguments are typed ``Optional`` to keep the keyword
        order consistent with the rest of the mirror-convention surface.

        Raises:
            ValueError: If ``h12`` is omitted.
        """
        if h12 is None:
            raise ValueError(
                "LCSDatabase.get_polytope requires `h12` "
                "(= catalog h11, the primary discriminator)."
            )
        return super().get_polytope(ks_id, h11=h12, h12=h11)

    def info(self) -> None:
        r"""Print summary of available models, with Hodge ranges in mirror convention."""
        self._ensure_catalog()
        cat = self._catalog
        assert cat is not None  # populated by _ensure_catalog
        n_total = len(cat)
        h11_min, h11_max = int(cat["h11"].min()), int(cat["h11"].max())
        h12_min, h12_max = int(cat["h12"].min()), int(cat["h12"].max())
        n_gv = int(cat["has_gv"].sum()) if "has_gv" in cat.columns else -1
        n_cf = int((cat["n_conifolds"] > 0).sum()) if "n_conifolds" in cat.columns else -1

        print(f"LCSDatabase — sub-dataset: '{self.dataset}' (mirror convention)")
        print(f"  Total models : {n_total:,}")
        # Mirror convention: catalog h12 is reported as h11, and vice versa.
        print(f"  h11 range    : {h12_min} – {h12_max}")
        print(f"  h12 range    : {h11_min} – {h11_max}")
        if n_gv >= 0:
            print(f"  With GV data : {n_gv:,}  ({100*n_gv/n_total:.1f}%)")
        if n_cf >= 0:
            print(f"  With conifolds: {n_cf:,}  ({100*n_cf/n_total:.1f}%)")
        print(f"  Cache dir    : {self.cache_dir}")

    # ------------------------------------------------------------------
    # lcs_tree / FluxVacuaFinder construction — method bodies below were
    # copied verbatim from the original CYDatabase (except `load_model`
    # which now returns FluxVacuaFinder instead of FluxEFT).
    # ------------------------------------------------------------------
    def load(
        self,
        ks_id: Optional[int] = None,
        triang_id: Optional[int] = None,
        cicy_id: Optional[int] = None,
        h11: Optional[int] = None,
        h12: Optional[int] = None,
        include_gv: bool = False,
        include_conifolds: Union[bool, str] = False,
        conifold_id: Optional[int] = None,
        conifold_basis: bool = True,
        include_extra_data: bool = True,
        include_polytope: bool = False,
        extra_fields: Optional[List[str]] = None,
        maximum_degree: Optional[int] = None,
    ) -> Any:
        r"""
        **Description:**
        Load a single Calabi-Yau model from the database and return it as an
        :class:`~jaxvacua.lcs.lcs_tree` object.

        For ``tdf`` models the full primary key is
        ``(h11, h12, ks_id, triang_id)``.  When ``h11`` and ``h12`` are
        omitted the lookup falls back to ``(ks_id, triang_id)``; if that
        combination is ambiguous a ``ValueError`` is raised asking you to
        supply the Hodge numbers.

        For ``cicy`` models, only ``cicy_id`` is required.

        Args:
            ks_id (int | None): Kreuzer-Skarke polytope index (required for
                ``dataset="tdf"``).
            triang_id (int | None): Triangulation index (required for
                ``dataset="tdf"``).
            cicy_id (int | None): CICY model index (required for
                ``dataset="cicy"``).
            h11 (int | None): Mirror :math:`h^{1,1}` (= catalog
                :math:`h^{1,2}`).  Recommended for ``dataset="tdf"`` to
                ensure a unique match.
            h12 (int | None): Mirror :math:`h^{1,2}` (= catalog
                :math:`h^{1,1}`).  Recommended for ``dataset="tdf"`` to
                ensure a unique match.
            include_gv (bool): If ``True``, fetch and attach Gopakumar-Vafa
                and Gromov-Witten invariants.  Defaults to ``False``.
            include_conifolds (bool | str): If ``True``, fetch all available
                conifold limits and populate ``lcs_tree.conifold`` (the
                active :class:`~jaxvacua.conifold_utils.Conifold`) from the
                selected conifold (controlled by ``conifold_id``; defaults
                to 0).  Pass ``"all"`` to obtain a list of trees, one per
                conifold.  Defaults to ``False``.
            conifold_id (int | None): Index of the conifold limit to use when
                ``include_conifolds=True``.  Ignored when
                ``include_conifolds="all"``.  Defaults to ``0``.
            conifold_basis (bool): If ``True``, rotate to the basis defined by
                the conifold curve when ``include_conifolds`` is active.
                Defaults to ``True``.
            include_extra_data (bool): If ``True``, populate
                ``lcs_tree.extra_data`` from the ``extra`` split (merged with
                the identifier fields always stored there).  Defaults to
                ``True``.
            include_polytope (bool): If ``True``, fetch the ``polytope`` split
                and attach ``polytope_points`` (and any extra polytope-level
                fields) to the returned tree.  Only meaningful for
                ``dataset="tdf"``.  Defaults to ``False``.
            extra_fields (list[str] | None): Subset of columns to load from
                the ``extra`` split when ``include_extra_data=True``.  If
                ``None``, all available columns are loaded.
            maximum_degree (int | None): If ``include_gv=True``, truncate GV
                invariants to at most this degree.  ``None`` keeps all stored
                invariants.

        Returns:
            lcs_tree | list[lcs_tree]: A single tree, or a list when
            ``include_conifolds="all"``.

        Raises:
            KeyError: If the requested model is not in the catalog.
            ValueError: If required primary-key arguments are missing, or if
                the ``(ks_id, triang_id)`` pair matches multiple models and
                ``h11``/``h12`` were not supplied.

        Notes:
            ``h11`` / ``h12`` are interpreted in the *mirror* (jaxvacua)
            convention.  They are translated to the underlying catalog
            convention here at the API boundary; the post-fetch swap that
            puts the values back into mirror convention for ``lcs_tree``
            is done further down (see the ``# Flip Hodge numbers for input``
            block).
        """
        # Mirror -> catalog: the underlying catalog and `_lookup`/shard paths
        # use the catalog convention, so swap the user-supplied Hodge numbers
        # before any further processing.  The post-fetch swap further down
        # restores mirror convention for the kwargs handed to `lcs_tree`.
        # Skipped for datasets already stored mirror-convention (see
        # :data:`_NO_H_SWAP_DATASETS`).
        if _needs_h_swap(self.dataset):
            h11, h12 = h12, h11

        self._check_cache_size()
        self._validate_key(ks_id, triang_id, cicy_id)
        entry = self._lookup(ks_id, triang_id, cicy_id, h11=h11, h12=h12)

        # ---- Fetch LCS geometry data ----------------------------------------
        h11 = int(entry["h11"])
        if self.cache_mode == "persistent":
            lcs_shard = self._fetch_shard(
                f"lcs_data/h11_{h11}", int(entry["lcs_shard_id"])
            )
            lcs_row = lcs_shard.iloc[int(entry["lcs_row_index"])]
        else:
            lcs_row = self._fetch_row(
                f"lcs_data/h11_{h11}",
                int(entry["lcs_shard_id"]),
                int(entry["lcs_row_index"]),
            )
        kwargs = _parse_lcs_row(lcs_row, self.dataset)

        # ---- Optionally fetch extra data ------------------------------------
        if include_extra_data and "extra_shard_id" in entry.index:
            if self.cache_mode == "persistent":
                extra_shard = self._fetch_shard("extra", int(entry["extra_shard_id"]))
                extra_row   = extra_shard.iloc[int(entry["extra_row_index"])]
            else:
                extra_row = self._fetch_row(
                    "extra", int(entry["extra_shard_id"]),
                    int(entry["extra_row_index"]),
                )
            extra = dict(extra_row)
            if extra_fields is not None:
                extra = {k: v for k, v in extra.items() if k in extra_fields}
            existing = kwargs.get("extra_data") or {}
            kwargs["extra_data"] = {**existing, **extra}

        # ---- Optionally fetch polytope data (tdf only) ----------------------
        if include_polytope and "polytope_shard_id" in entry.index:
            _psid = entry["polytope_shard_id"]
            _prid = entry["polytope_row_index"]
            _has_poly = False
            try:
                import pandas as pd
                _has_poly = _psid is not None and not pd.isna(_psid)
            except (TypeError, ValueError):
                _has_poly = _psid is not None
            if _has_poly:
                if self.cache_mode == "persistent":
                    poly_shard = self._fetch_shard("polytope", int(_psid))
                    poly_row   = poly_shard.iloc[int(_prid)]
                else:
                    poly_row = self._fetch_row("polytope", int(_psid), int(_prid))
                poly_data  = _parse_polytope_row(poly_row)
                kwargs["polytope_points"] = poly_data["polytope_points"]
                if "polytope_data" in poly_data:
                    existing = kwargs.get("extra_data") or {}
                    kwargs["extra_data"] = {**existing, "polytope_data": poly_data["polytope_data"]}

        # ---- Optionally fetch GV invariants --------------------------------
        gvs, gws, grading_vec = None, None, None
        if include_gv and entry.get("has_gv", False):
            # `cicy`'s gv/ split is FLAT by design (a single bucket); tdf/kklt bucket
            # it by catalog h11.  Building the bucketed path unconditionally raised
            # FileNotFoundError for every cicy model, including the card's headline
            # `include_gv=True` example.
            _gv_split = "gv" if self.dataset in _FLAT_GV_DATASETS else f"gv/h11_{h11}"
            if self.cache_mode == "persistent":
                gv_shard = self._fetch_shard(_gv_split, int(entry["gv_shard_id"]))
                gv_row   = gv_shard.iloc[int(entry["gv_row_index"])]
            else:
                gv_row = self._fetch_row(
                    _gv_split, int(entry["gv_shard_id"]),
                    int(entry["gv_row_index"]),
                )
            gv_data  = _parse_gv_row(gv_row)
            gvs = gv_data["GVs"]
            gws = gv_data["GWs"]
            grading_vec = gv_data["grading_vector"]
            if maximum_degree is not None:
                gvs, gws = self._truncate_gv(gvs, gws, grading_vec,maximum_degree)
        kwargs["gvs"] = gvs
        kwargs["gws"] = gws
        kwargs["grading_vector"] = grading_vec
        
        
        # Catalog -> mirror for the kwargs handed to `lcs_tree`.  Skipped for
        # datasets already stored mirror-convention (see :data:`_NO_H_SWAP_DATASETS`);
        # `chi` is always dropped so `lcs_tree` recomputes chi = -2*(h11 - h12),
        # which is correct in either convention.
        h11 = kwargs["h11"]
        h12 = kwargs["h12"]
        if _needs_h_swap(self.dataset):
            kwargs["h11"] = h12
            kwargs["h12"] = h11
        kwargs["chi"] = None
        
        if h11 == 0 or h12 == 0:
            warnings.warn(
                "One of the Hodge numbers is zero, which may lead to "
                "unexpected behaviour in some JAXVacua computations. "
                "Proceed with caution."
            )
        
        # ---- Optionally fetch conifold data --------------------------------
        # Conifold shards are sharded by *catalog* h11 (the on-disk
        # convention), so use the entry's original h11 before the mirror
        # swap performed above.
        _cat_h11 = int(entry["h11"])
        conifold_list = []
        if include_conifolds and int(entry.get("n_conifolds", 0)) > 0:
            _cf_shard_id = int(entry["conifold_shard_id"])
            _cf_split = f"conifolds/h11_{_cat_h11}"
            if self.cache_mode == "persistent":
                cf_shard = self._fetch_shard(_cf_split, _cf_shard_id)
            else:
                # For conifolds we need the full shard to filter by model
                # ID, so use _fetch_rows with a mask function.
                if self.dataset == "tdf":
                    def _cf_mask(df):
                        m = (df["ks_id"] == ks_id) & (df["triang_id"] == triang_id)
                        if "h11" in df.columns:
                            m = m & (df["h11"] == _cat_h11)
                        return m
                else:
                    _cf_mask = lambda df: df["cicy_id"] == cicy_id
                cf_rows = self._fetch_rows(_cf_split, _cf_shard_id, _cf_mask)
                conifold_list = _parse_conifold_rows(cf_rows)

            if self.cache_mode == "persistent":
                if self.dataset == "tdf":
                    mask = (
                        (cf_shard["ks_id"]     == ks_id) &
                        (cf_shard["triang_id"] == triang_id)
                    )
                    if "h11" in cf_shard.columns:
                        mask = mask & (cf_shard["h11"] == _cat_h11)
                    cf_rows = cf_shard[mask]
                else:
                    cf_rows = cf_shard[cf_shard["cicy_id"] == cicy_id]
                conifold_list = _parse_conifold_rows(cf_rows)

        # ---- Construct lcs_tree --------------------------------------------
        from jaxvacua.lcs import lcs_tree

        _CF_EXTRA_KEYS = ("flop_edge", "one_face_divisors")

        if include_conifolds == "all" and conifold_list:
            trees = []
            for cf in conifold_list:
                kw = {
                    **kwargs,
                    "ncf":            cf["ncf"],
                    "conifold_curve": cf["conifold_curve"],
                    "conifold_basis": conifold_basis,
                    "limit":          "coniLCS",
                }
                if conifold_basis and cf.get("basis_change") is not None:
                    kw["basis_change"] = cf["basis_change"]
                if include_extra_data:
                    cf_extra = {k: cf[k] for k in _CF_EXTRA_KEYS if cf.get(k) is not None}
                    if cf_extra:
                        kw["extra_data"] = {**(kw.get("extra_data") or {}), **cf_extra}
                trees.append(lcs_tree.from_dict(kw))
            return trees

        if conifold_list:
            # Select the requested conifold; fall back to id=0 if not found.
            _target_id = conifold_id if conifold_id is not None else 0
            _matches = [cf for cf in conifold_list if cf["conifold_id"] == _target_id]
            if not _matches:
                raise KeyError(
                    f"conifold_id={_target_id} not found for this model. "
                    f"Available ids: {[cf['conifold_id'] for cf in conifold_list]}"
                )
            cf = _matches[0]
            kwargs["ncf"]            = cf["ncf"]
            kwargs["conifold_curve"] = cf["conifold_curve"]
            kwargs["conifold_basis"] = conifold_basis
            kwargs["limit"]          = "coniLCS"
            if conifold_basis and cf.get("basis_change") is not None:
                kwargs["basis_change"] = cf["basis_change"]
            if include_extra_data:
                cf_extra = {k: cf[k] for k in _CF_EXTRA_KEYS if cf.get(k) is not None}
                if cf_extra:
                    kwargs["extra_data"] = {**(kwargs.get("extra_data") or {}), **cf_extra}

        # Pass ``maximum_degree`` through as an int; 0 means no
        # instanton-degree cutoff.
        kwargs["maximum_degree"] = (
            int(maximum_degree) if maximum_degree is not None else 0
        )

        return lcs_tree.from_dict(kwargs)

    def load_from_conifold_row(
        self,
        row: Any,
        include_gv: bool = False,
        conifold_basis: bool = True,
        include_extra_data: bool = True,
        include_polytope: bool = False,
        maximum_degree: Optional[int] = None,
    ) -> Any:
        r"""
        **Description:**
        Load the model corresponding to a single row from
        :meth:`query_conifolds` and return it as an
        :class:`~jaxvacua.lcs.lcs_tree` with that conifold limit active.

        Example::

            rows = db.query_conifolds(h11=2, ncf=2)
            tree = db.load_from_conifold_row(rows.iloc[0])

        Args:
            row: A single-row result from :meth:`query_conifolds` (a
                ``pandas.Series``).
            include_gv (bool): Attach GV invariants.  Defaults to ``False``.
            conifold_basis (bool): Rotate to the conifold basis.  Defaults to
                ``True``.
            include_extra_data (bool): Populate ``extra_data``.  Defaults to
                ``True``.
            include_polytope (bool): Attach polytope data.  Defaults to
                ``False``.
            maximum_degree (int | None): GV truncation degree.

        Returns:
            lcs_tree: The requested model with the selected conifold active.
        """
        return self.load(
            ks_id=int(row["ks_id"]) if self.dataset == "tdf" else None,
            triang_id=int(row["triang_id"]) if self.dataset == "tdf" else None,
            cicy_id=int(row["cicy_id"]) if self.dataset == "cicy" else None,
            include_conifolds=True,
            conifold_id=int(row["conifold_id"]),
            conifold_basis=conifold_basis,
            include_gv=include_gv,
            include_extra_data=include_extra_data,
            include_polytope=include_polytope,
            maximum_degree=maximum_degree,
        )

    def load_model(
        self,
        ks_id: Optional[int] = None,
        triang_id: Optional[int] = None,
        cicy_id: Optional[int] = None,
        h11: Optional[int] = None,
        h12: Optional[int] = None,
        include_gv: bool = False,
        include_conifolds: Union[bool, str] = False,
        maximum_degree: int = 0,
        **model_kwargs: Any,
    ) -> Any:
        r"""
        **Description:**
        Load a model from the database and return it as a fully initialised
        :class:`~jaxvacua.flux_vacua_finder.FluxVacuaFinder` object.

        This is a convenience wrapper around :meth:`load` followed by
        construction of a
        :class:`~jaxvacua.flux_vacua_finder.FluxVacuaFinder` with
        ``lcs_tree``.  The returned object includes the period,
        complex-structure-sector, flux-EFT, and vacuum-finding layers.

        Args:
            ks_id (int | None): Kreuzer-Skarke polytope index (tdf only).
            triang_id (int | None): Triangulation index (tdf only).
            cicy_id (int | None): CICY index (cicy only).
            h11 (int | None): Mirror :math:`h^{1,1}` (= catalog
                :math:`h^{1,2}`; tdf only).
            h12 (int | None): Mirror :math:`h^{1,2}` (= catalog
                :math:`h^{1,1}`; tdf only).
            include_gv (bool): Attach GV invariants.  Defaults to ``False``.
            include_conifolds (bool): Attach conifold data.  Defaults to
                ``False``.
            maximum_degree (int): Maximum instanton degree; passed to
                :class:`~jaxvacua.flux_vacua_finder.FluxVacuaFinder`.
                Defaults to ``0`` (instanton corrections disabled).
            **model_kwargs: Additional keyword arguments forwarded to
                :class:`~jaxvacua.flux_vacua_finder.FluxVacuaFinder`,
                e.g. ``Q``, ``limit``, ``gauge_choice``.

        Returns:
            FluxVacuaFinder: Initialised JAXVacua model.

        Raises:
            KeyError: If the requested model is not in the catalog.
        """
        from jaxvacua.flux_vacua_finder import FluxVacuaFinder

        tree = self.load(
            ks_id=ks_id,
            triang_id=triang_id,
            cicy_id=cicy_id,
            h11=h11,
            h12=h12,
            include_gv=include_gv,
            include_conifolds=include_conifolds,
            maximum_degree=maximum_degree if include_gv else None,
        )
        return FluxVacuaFinder(
            lcs_tree=tree,
            maximum_degree=maximum_degree,
            **model_kwargs,
        )

    def load_batch(
        self,
        identifiers: Any = None,
        include_gv: bool = False,
        include_conifolds: Union[bool, str] = False,
        conifold_id: Optional[int] = None,
        conifold_basis: bool = True,
        include_extra_data: bool = True,
        include_polytope: bool = False,
        maximum_degree: Optional[int] = None,
        parallel: bool = True,
        n_workers: Optional[int] = None,
        **query_filters: Any,
    ) -> List[Any]:
        r"""
        **Description:**
        Load multiple models and return them as a list of
        :class:`~jaxvacua.lcs.lcs_tree` objects.

        By default shard I/O is parallelised across a thread pool so that
        models in different shards are fetched concurrently.  Results are
        always returned in the same order as ``identifiers``.

        Can be called in two ways:

        1. **With explicit identifiers:**
           ``db.load_batch(df)`` or ``db.load_batch([(ks_id, triang_id), ...])``
        2. **With query filters** (same kwargs as :meth:`query`):
           ``db.load_batch(h11=2)`` or ``db.load_batch(ks_id=12345)``

        When query filters are used, :meth:`query` is called first and
        the matching catalog entries are loaded.

        Args:
            identifiers: Either a ``pandas.DataFrame`` with columns
                ``ks_id`` and ``triang_id`` (or ``cicy_id``), or a list of
                ``(ks_id, triang_id)`` tuples for ``"tdf"``, or a list of
                ``cicy_id`` integers for ``"cicy"``.  If ``None``, the
                ``**query_filters`` are used instead.
            include_gv (bool): Attach GV invariants to each model.
            include_conifolds (bool | str): Attach conifold data.
            conifold_id (int | None): Index of the conifold limit to use.
                Defaults to ``0``. Ignored when ``include_conifolds="all"``.
            conifold_basis (bool): Rotate to the conifold basis when
                ``include_conifolds`` is active. Defaults to ``True``.
            include_extra_data (bool): Populate ``extra_data`` field.
            include_polytope (bool): Attach polytope data from the polytope split.
            maximum_degree (int | None): GV truncation degree.
            parallel (bool): If ``False``, load models sequentially (no thread
                pool).  Useful when the caller is already inside a parallel
                context or when debugging.  Defaults to ``True``.
            n_workers (int | None): Number of threads when ``parallel=True``.
                ``None`` uses ``min(len(identifiers), 8)``.
            **query_filters: Keyword arguments forwarded to :meth:`query`
                when ``identifiers`` is ``None``.  Common filters:
                ``h11``, ``h12``, ``ks_id``, ``has_conifolds``, ``has_gv``.

        Returns:
            list[lcs_tree]: Loaded trees in the same order as ``identifiers``
            (or in catalog order when using query filters).
        """
        _require_pandas()

        if identifiers is None:
            if not query_filters:
                raise ValueError(
                    "Either pass identifiers (DataFrame / list) or "
                    "query filters (e.g. h11=2, ks_id=12345)."
                )
            identifiers = self.query(**query_filters)
            if len(identifiers) == 0:
                warnings.warn("Query returned no models.")
                return []

        rows = self._identifiers_to_list(identifiers)

        if not rows:
            return []

        workers = 1 if not parallel else (min(len(rows), 8) if n_workers is None else n_workers)

        def _load_one(row):
            """Load a single model from an identifier row."""
            ks_id, triang_id, cicy_id, h11, h12 = row
            return self.load(
                ks_id=ks_id,
                triang_id=triang_id,
                cicy_id=cicy_id,
                h11=h11,
                h12=h12,
                include_gv=include_gv,
                include_conifolds=include_conifolds,
                conifold_id=conifold_id,
                conifold_basis=conifold_basis,
                include_extra_data=include_extra_data,
                include_polytope=include_polytope,
                maximum_degree=maximum_degree,
            )

        if workers == 1:
            return [_load_one(row) for row in rows]

        # Submit all jobs, preserving insertion order via index.
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_idx = {
                executor.submit(_load_one, row): i
                for i, row in enumerate(rows)
            }
            results = [None] * len(rows)
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                results[idx] = future.result()  # re-raises any exception

        return results

    def iter_batch(
        self,
        identifiers: Any,
        include_gv: bool = False,
        include_conifolds: Union[bool, str] = False,
        conifold_id: Optional[int] = None,
        conifold_basis: bool = True,
        include_extra_data: bool = True,
        include_polytope: bool = False,
        maximum_degree: Optional[int] = None,
        prefetch: int = 4,
    ) -> Iterator[Any]:
        r"""
        **Description:**
        Yield models one at a time from a (potentially large) set of
        identifiers, keeping memory usage constant regardless of the total
        number of models requested.

        Unlike :meth:`load_batch`, which loads all models into a list before
        returning, ``iter_batch`` is a generator: it yields each
        :class:`~jaxvacua.lcs.lcs_tree` as soon as it is ready, so the
        caller can begin processing immediately and only one model needs to be
        in memory at a time.

        A sliding prefetch window of size ``prefetch`` submits the next
        ``prefetch`` loads to a background thread pool while the caller
        processes the current model, keeping disk I/O overlapped with
        computation.  Set ``prefetch=1`` (or ``prefetch=0``) to disable
        parallelism and load strictly one model at a time.

        Example::

            df = db.query(h11=3)
            for tree in db.iter_batch(df, prefetch=8):
                result = analyse(tree)   # memory: ~1 model at a time

        Args:
            identifiers: Either a ``pandas.DataFrame`` with columns
                ``ks_id`` and ``triang_id`` (or ``cicy_id``), or a list of
                ``(ks_id, triang_id)`` tuples for ``"tdf"``, or a list of
                ``cicy_id`` integers for ``"cicy"``.
            include_gv (bool): Attach GV invariants to each model.
            include_conifolds (bool | str): Attach conifold data.
            conifold_id (int | None): Index of the conifold limit to use.
                Defaults to ``0``. Ignored when ``include_conifolds="all"``.
            conifold_basis (bool): Rotate to the conifold basis when
                ``include_conifolds`` is active. Defaults to ``True``.
            include_extra_data (bool): Populate ``extra_data`` field.
            include_polytope (bool): Attach polytope data.
            maximum_degree (int | None): GV truncation degree.
            prefetch (int): Number of models to load ahead in background
                threads.  Defaults to ``4``.  Set to ``0`` or ``1`` for
                strictly sequential loading.

        Yields:
            lcs_tree: One loaded model per iteration, in the same order as
            ``identifiers``.
        """
        _require_pandas()
        rows = self._identifiers_to_list(identifiers)

        if not rows:
            return

        def _load_one(row):
            """Load a single model from an identifier row for iteration."""
            ks_id, triang_id, cicy_id, h11, h12 = row
            return self.load(
                ks_id=ks_id,
                triang_id=triang_id,
                cicy_id=cicy_id,
                h11=h11,
                h12=h12,
                include_gv=include_gv,
                include_conifolds=include_conifolds,
                conifold_id=conifold_id,
                conifold_basis=conifold_basis,
                include_extra_data=include_extra_data,
                include_polytope=include_polytope,
                maximum_degree=maximum_degree,
            )

        # Sequential path — no thread pool overhead.
        if prefetch <= 1:
            for row in rows:
                yield _load_one(row)
            return

        # Prefetch path — sliding window of futures keeps I/O and computation
        # overlapped.  The window size is capped to the number of rows.
        window = min(prefetch, len(rows))
        executor = ThreadPoolExecutor(max_workers=window)
        pending: deque = deque()

        try:
            # Prime the window with the first `window` rows.
            for i in range(window):
                pending.append(executor.submit(_load_one, rows[i]))

            # For each remaining row, pop the oldest future (blocking until
            # done), yield it, then immediately submit the next row so the
            # window stays full.
            for i in range(window, len(rows)):
                pending.append(executor.submit(_load_one, rows[i]))
                yield pending.popleft().result()

            # Drain the window.
            while pending:
                yield pending.popleft().result()

        finally:
            # Cancel any still-pending futures if the caller broke out of the
            # loop early, then shut down the executor cleanly.
            for f in pending:
                f.cancel()
            executor.shutdown(wait=False)

    def sample(
        self,
        n: int,
        h11: Optional[int] = None,
        h12: Optional[int] = None,
        has_conifolds: Optional[bool] = None,
        has_gv: Optional[bool] = None,
        seed: Optional[int] = None,
        include_gv: bool = False,
        include_conifolds: Union[bool, str] = False,
        conifold_id: Optional[int] = None,
        conifold_basis: bool = True,
        include_extra_data: bool = True,
        include_polytope: bool = False,
        maximum_degree: Optional[int] = None,
        parallel: bool = True,
        n_workers: Optional[int] = None,
    ) -> List[Any]:
        r"""
        **Description:**
        Draw a random sample of models from the database and return them as a
        list of :class:`~jaxvacua.lcs.lcs_tree` objects.

        Args:
            n (int): Number of models to draw.
            h11 (int | None): Restrict sample to models with this mirror
                :math:`h^{1,1}` (= catalog :math:`h^{1,2}`).
            h12 (int | None): Restrict sample to models with this mirror
                :math:`h^{1,2}` (= catalog :math:`h^{1,1}`).  This is the
                primary discriminator of small-modulus models.
            has_conifolds (bool | None): Restrict to models with conifold data.
            has_gv (bool | None): Restrict to models with GV invariants.
            seed (int | None): Random seed for reproducibility.
            include_gv (bool): Attach GV invariants to each sampled model.
            include_conifolds (bool | str): Attach conifold data.
            conifold_id (int | None): Index of the conifold limit to use.
                Defaults to ``0``. Ignored when ``include_conifolds="all"``.
            conifold_basis (bool): Rotate to the conifold basis when
                ``include_conifolds`` is active. Defaults to ``True``.
            include_extra_data (bool): Populate ``extra_data`` field.
            include_polytope (bool): Attach polytope data from the polytope split.
            maximum_degree (int | None): GV truncation degree.
            parallel (bool): If ``False``, load models sequentially. See
                :meth:`load_batch`.
            n_workers (int | None): Number of parallel threads. See
                :meth:`load_batch`.

        Returns:
            list[lcs_tree]: ``n`` randomly drawn trees.

        Raises:
            ValueError: If the filtered catalog contains fewer than ``n``
                models.
        """
        subset = self.query(
            h11=h11, h12=h12,
            has_conifolds=has_conifolds,
            has_gv=has_gv,
        )
        if len(subset) < n:
            raise ValueError(
                f"Requested {n} models but only {len(subset)} match the filter."
            )
        sampled = subset.sample(n=n, random_state=seed)
        return self.load_batch(
            sampled,
            include_gv=include_gv,
            include_conifolds=include_conifolds,
            conifold_id=conifold_id,
            conifold_basis=conifold_basis,
            include_extra_data=include_extra_data,
            include_polytope=include_polytope,
            maximum_degree=maximum_degree,
            parallel=parallel,
            n_workers=n_workers,
        )

    @staticmethod
    def _truncate_gv(
        gvs: Dict[str, Any],
        gws: Dict[str, Any],
        grading_vec: Optional[np.ndarray],
        maximum_degree: int,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        r"""
        **Description:**
        Truncate Gopakumar-Vafa and Gromov-Witten invariants to a maximum
        total degree.

        The total degree of a curve with charge vector ``q`` is taken to be
        ``q.sum()``.  Entries with total degree greater than ``maximum_degree``
        are removed.

        Either family may be ``None`` when the sub-dataset does not carry it — the
        ``cicy`` ``gv`` split has no GW columns at all — in which case it is passed
        through untouched.

        Args:
            gvs (dict | None): GV data with keys ``"charges"`` and ``"invariants"``.
            gws (dict | None): GW data, same shape, or ``None`` if absent.
            maximum_degree (int): Maximum allowed total degree.

        Returns:
            Tuple[dict | None, dict | None]: Truncated ``(gvs, gws)``.
        """
        def _trunc(data: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
            """Truncate data arrays to entries within the maximum degree."""
            if data is None:
                return None
            charges = data.get("charges")
            invs    = data.get("invariants")
            if charges is None or invs is None or grading_vec is None:
                return data
            charges = np.asarray([np.array(c).astype(int) for c in charges])
            invs    = np.asarray(invs)
            deg = charges@grading_vec
            if np.any(deg < 0):
                raise ValueError("Found negative curve degrees. Grading vector may be incorrect.")
            mask    = deg <= maximum_degree
            return {"charges": charges[mask], "invariants": invs[mask]}
        
        return _trunc(gvs), _trunc(gws)


    # ------------------------------------------------------------------
    # NEW — batch-model methods + filter-keyed cache
    # ------------------------------------------------------------------

    @staticmethod
    def _filter_cache_key(filters: Dict[str, Any]) -> Optional[Tuple[Any, ...]]:
        """Return a hashable tuple summarising `filters`, or None if any value
        isn't hashable (e.g. a DataFrame passed as an entire filter argument).
        Callers treat a ``None`` key as "bypass cache"."""
        try:
            return tuple(sorted((k, v) for k, v in filters.items()))
        except TypeError:
            return None

    def load_model_batch(self, **filters) -> List[Any]:
        """Eager ensemble load — returns a list of FluxVacuaFinder instances.

        Filter-keyed cache: repeated calls with the same ``**filters`` return
        the same cached list object without rebuilding.  Calling with a
        different filter set rebuilds and replaces the cache.  Unhashable
        filters bypass the cache transparently (always rebuild).

        See `iter_model_batch` for a lazy alternative that does not retain
        constructed models.
        """
        from jaxvacua.flux_vacua_finder import FluxVacuaFinder
        key = self._filter_cache_key(filters)
        if key is not None and key == self._cached_filter_key:
            return self._cached_models
        trees = self.load_batch(**filters)
        models = [FluxVacuaFinder(lcs_tree=t) for t in trees]
        self._cached_models     = models
        self._cached_filter_key = key
        return models

    # kwargs `iter_batch` accepts directly (pass through); anything else is
    # treated as a catalog filter and routed through `self.query(**filters)`.
    _ITER_BATCH_KWARGS = frozenset((
        "include_gv", "include_conifolds", "conifold_id", "conifold_basis",
        "include_extra_data", "include_polytope", "maximum_degree", "prefetch",
    ))

    def iter_model_batch(self, identifiers: Any = None, **kwargs) -> Iterator[Any]:
        """Lazy ensemble iterator over FluxVacuaFinder instances.

        Yields one FluxVacuaFinder at a time without retaining prior models.
        **Intended for cluster runs and million-model scans** where holding
        the full batch in RAM is prohibitive — each yielded instance is
        constructed fresh from shard data, and the previous instance becomes
        eligible for GC once the user moves on.  No filter caching (unlike
        `load_model_batch`); re-iterating the same filter set rebuilds every
        FluxVacuaFinder from scratch.  Shard-level caching (the LRU on
        CYDatabase) still applies, so re-iteration is cheaper than a cold
        run.

        Accepts either a pre-built `identifiers` (DataFrame / list), or catalog
        filter kwargs (e.g. ``h11=3, has_conifolds=True``) — matching the
        ergonomics of ``load_batch``.  ``iter_batch``-specific kwargs
        (``include_gv``, ``include_conifolds``, ``maximum_degree`` …) are
        forwarded to the underlying :meth:`iter_batch`.

        Use `load_model_batch` instead if the batch fits in memory and you
        want cached O(1) re-access via ``db.cached_models``.
        """
        from jaxvacua.flux_vacua_finder import FluxVacuaFinder
        iter_kw = {k: kwargs.pop(k) for k in list(kwargs) if k in self._ITER_BATCH_KWARGS}
        if identifiers is None:
            if not kwargs:
                raise ValueError(
                    "iter_model_batch requires either `identifiers` or catalog filter kwargs"
                )
            identifiers = self.query(**kwargs)
        elif kwargs:
            raise TypeError(
                f"iter_model_batch got unexpected kwargs with identifiers already given: {list(kwargs)}"
            )
        for tree in self.iter_batch(identifiers, **iter_kw):
            yield FluxVacuaFinder(lcs_tree=tree)

    @property
    def cached_models(self) -> Optional[List[Any]]:
        """The FluxVacuaFinder batch most recently produced by
        ``load_model_batch``, or ``None`` if ``load_model_batch`` has never
        been called (or was just cleared).  Read-only; identical to the list
        returned by the most recent ``load_model_batch(**filters)`` call.
        """
        return self._cached_models

    def clear_cached_models(self) -> None:
        """Drop the cached model batch (releases the FluxVacuaFinder list)."""
        self._cached_models     = None
        self._cached_filter_key = None

    # ------------------------------------------------------------------
    # Vacua delegation — forwards to a lazily-built VacuaWriter(self).
    # Method names match the originals on CYDatabase.
    # ------------------------------------------------------------------

    @property
    def _vw(self) -> "VacuaWriter":
        vw = self.__dict__.get("_vw_cached")
        if vw is None:
            vw = VacuaWriter(self)
            self.__dict__["_vw_cached"] = vw
        return vw

    def designate_vacua(self, *args, **kwargs):     return self._vw.designate_vacua(*args, **kwargs)
    def retract_designated(self, *args, **kwargs):  return self._vw.retract_designated(*args, **kwargs)
    def purge_retracted(self, *args, **kwargs):     return self._vw.purge_retracted(*args, **kwargs)
    def query_designated(self, *args, **kwargs):    return self._vw.query_designated(*args, **kwargs)
    def load_designated(self, *args, **kwargs):     return self._vw.load_designated(*args, **kwargs)
    def load_local_vacua(self, *args, **kwargs):    return self._vw.load_local_vacua(*args, **kwargs)
    def push_vacua_to_hub(self, *args, **kwargs):   return self._vw.push_vacua_to_hub(*args, **kwargs)
    def fetch_vacua_from_hub(self, *args, **kwargs):return self._vw.fetch_vacua_from_hub(*args, **kwargs)
    def list_hub_vacua(self, *args, **kwargs):      return self._vw.list_hub_vacua(*args, **kwargs)
    def validate_vacua(self, *args, **kwargs):      return self._vw.validate_vacua(*args, **kwargs)
    def complete_missing(self, *args, **kwargs):    return self._vw.complete_missing(*args, **kwargs)
    def delete_vacua(self, *args, **kwargs):        return self._vw.delete_vacua(*args, **kwargs)
    def vacua_writer(self, *args, **kwargs):        return self._vw.vacua_writer(*args, **kwargs)
    def write_vacua(self, *args, **kwargs):         return self._vw.write_vacua(*args, **kwargs)
    def read_vacua(self, *args, **kwargs):          return self._vw.read_vacua(*args, **kwargs)
    def designated_info(self, *args, **kwargs):     return self._vw.designated_info(*args, **kwargs)
    def _resolve_vacua_dir(self, *args, **kwargs):   return self._vw._resolve_vacua_dir(*args, **kwargs)
    def query_vacua(self, *args, **kwargs):         return self._vw.query_vacua(*args, **kwargs)
    def load_vacua(self, *args, **kwargs):          return self._vw.load_vacua(*args, **kwargs)
    def solution_exists(self, *args, **kwargs):     return self._vw.solution_exists(*args, **kwargs)
    def find_similar_vacua(self, *args, **kwargs):  return self._vw.find_similar_vacua(*args, **kwargs)
    def vacua_info(self, *args, **kwargs):          return self._vw.vacua_info(*args, **kwargs)


# ==========================================================================
# Module-level one-shot shortcuts
# ==========================================================================

def load_tdf_model(
    ks_id: int,
    triang_id: int,
    include_gv: bool = False,
    include_conifolds: Union[bool, str] = False,
    maximum_degree: Optional[int] = None,
    cache_dir: str = DEFAULT_CACHE_DIR,
) -> Any:
    r"""Load a single TDF (toric Kreuzer-Skarke) model as an
    :class:`~jaxvacua.lcs.lcs_tree` without instantiating a database
    explicitly.  Equivalent to
    ``LCSDatabase(dataset="tdf", cache_dir=cache_dir).load(ks_id=..., triang_id=..., ...)``.
    """
    db = LCSDatabase(dataset="tdf", cache_dir=cache_dir)
    return db.load(
        ks_id=ks_id,
        triang_id=triang_id,
        include_gv=include_gv,
        include_conifolds=include_conifolds,
        maximum_degree=maximum_degree,
    )


def load_cicy_model(
    cicy_id: int,
    include_gv: bool = False,
    maximum_degree: Optional[int] = None,
    cache_dir: str = DEFAULT_CACHE_DIR,
) -> Any:
    r"""Load a single CICY model as an :class:`~jaxvacua.lcs.lcs_tree` without
    instantiating a database explicitly.  Equivalent to
    ``LCSDatabase(dataset="cicy", cache_dir=cache_dir).load(cicy_id=..., ...)``.
    """
    db = LCSDatabase(dataset="cicy", cache_dir=cache_dir)
    return db.load(
        cicy_id=cicy_id,
        include_gv=include_gv,
        maximum_degree=maximum_degree,
    )


__all__ = ["LCSDatabase", "load_tdf_model", "load_cicy_model", "_parse_lcs_row"]
