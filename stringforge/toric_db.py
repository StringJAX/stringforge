r"""
Consumer for the unified, **sharded** ``toric`` cy-database sub-dataset (FRST + VEX).

``ToricCYDatabase`` subclasses :class:`stringforge.cy_io.CYDatabase`, but note that the
inheritance is *nominal*: the base class assumes a monolithic ``{dataset}/catalog.parquet``,
which the toric sub-dataset does not have, so the whole catalogue layer is **replaced rather
than extended** and every inherited member that assumes the monolithic layout is explicitly
closed (see :meth:`query_conifolds`). What *is* reused is the download/cache machinery.

The toric layout (built by ``private/database/frst_vex_merge/build_toric_database.py``) stores
everything as per-h11 **sharded** parts, so the catalog is never loaded whole:

- **Point lookups are O(1)** via the per-h11 ``_ksid_index`` (``ks_id → (part, row0, n)``):
  :meth:`load`, :meth:`get_polytope`, and ``query(mode, h11, ks_id=…)`` read one part slice.
- **Attribute queries** (``fav_N``/``h12``/…) stream the per-h11 catalog via ``pyarrow.dataset`` +
  filter pushdown (using ``_metadata`` when present) — never materialising billions of rows.

The **thin** phase catalog omits ``polytope_hash`` (joined via ``ks_id`` from the polytope catalog on
:meth:`load`) and ``phase_id`` (derived ``"{mode}:{h11}:{ks_id}:{triang_id}"``); ``wall_hash`` is a
32-byte digest, exposed as ``.hex()``. Per-polytope VEX counts live in ``polytope_vex_counts`` and are
left-joined by :meth:`query_polytopes`.

**Access is currently local only** (:meth:`~stringforge.cy_io.CYDatabase.from_local`);
lazy download of the sharded layout from the Hub is not yet implemented. Example::

    from stringforge import ToricCYDatabase
    db = ToricCYDatabase.from_local("/path/to/build")     # the dir containing toric/, or toric/
    polys = db.query_polytopes(h11=4, fav_N=True)          # n_frst_classes AND n_vex_classes
    frst  = db.query("frst", h11=4)
    geom  = db.load("frst", h11=4, ks_id=0, triang_id=0, in_basis=True)
    poly  = db.get_polytope(h11=4, ks_id=0)
"""

from __future__ import annotations

import glob
import json
import warnings
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from . import cy_io
from . import toric_normalize as nz
_MODES = ("frst", "vex")


class ToricCYDatabase(cy_io.CYDatabase):
    """Consumer for the sharded ``toric`` sub-dataset (shared polytopes; FRST/VEX phases)."""

    _DATASET = "toric"

    def __init__(self, **kwargs):
        kwargs.setdefault("dataset", "toric")
        super().__init__(**kwargs)
        self._idx_cache: Dict[str, Dict[int, Tuple[int, int, int]]] = {}
        self._ds_cache: Dict[str, object] = {}

    def _check_schema(self, *args, **kwargs) -> None:  # type: ignore[override]
        r"""Validate the **toric** ``schema.json`` against :data:`normalize.TORIC_SCHEMA_VERSION`.

        The base implementation cannot be reused: it targets the repository-level
        :data:`stringforge.cy_io.SCHEMA_VERSION` and reaches for a monolithic
        ``{dataset}/catalog.parquet`` that the sharded toric layout does not have.

        This previously returned ``None`` unconditionally, which meant **any** toric
        ``schema.json`` was accepted — including a future, incompatible one — silently
        defeating the versioning it was supposed to enforce.  Missing/unversioned files are
        still tolerated (warn only), matching the base class's treatment of pre-versioning
        caches; a *mismatch* now raises.
        """
        path = self.cache_dir / "schema.json"
        if not path.exists():
            warnings.warn(
                f"No toric schema.json in {self.cache_dir}; skipping the version check. "
                f"Expected schema_version={nz.TORIC_SCHEMA_VERSION}.",
                stacklevel=2,
            )
            return None
        try:
            with open(path) as fh:
                stored = json.load(fh).get("schema_version")
        except (OSError, ValueError) as exc:
            warnings.warn(f"Could not read {path}: {exc!r}; skipping the version check.",
                          stacklevel=2)
            return None
        if stored is None or int(stored) == nz.TORIC_SCHEMA_VERSION:
            return None
        stored = int(stored)
        if stored < nz.TORIC_SCHEMA_VERSION:
            changes = "\n".join(
                f"  v{v}: {nz.TORIC_SCHEMA_CHANGELOG[v]}"
                for v in sorted(nz.TORIC_SCHEMA_CHANGELOG) if v > stored
            )
            raise cy_io.SchemaVersionError(
                f"The cached 'toric' sub-dataset is at schema_version={stored}, but this "
                f"stringforge expects {nz.TORIC_SCHEMA_VERSION}.  Changes since yours:\n"
                f"{changes}\nRun db.clear_cache() to re-download."
            )
        raise cy_io.SchemaVersionError(
            f"The cached 'toric' sub-dataset is at schema_version={stored}, which is NEWER "
            f"than the {nz.TORIC_SCHEMA_VERSION} this stringforge understands.  "
            f"Please upgrade stringforge."
        )

    # -- inherited surface that the sharded layout invalidates ------------- #
    #
    # ``CYDatabase`` assumes a monolithic ``{dataset}/catalog.parquet``.  The toric layout is
    # sharded (``{mode}/catalog/h11_{N}/data-*.parquet``) and has no such file, so every
    # inherited member that reaches for it is closed here.  Left inherited they raise a bare
    # ``FileNotFoundError`` (``info``, ``query_conifolds``) or demand a ``cicy_id`` — because a
    # non-``tdf`` dataset falls into the ``else``-cicy branch of ``_lookup`` /
    # ``_validate_key`` / ``_identifiers_to_list``.  Closing them turns a confusing error into
    # an explanatory one, and stops a caller silently receiving an empty catalogue.
    _SHARDED_MSG = (
        "{what} is not available on the sharded 'toric' sub-dataset: it assumes a single "
        "{dataset}/catalog.parquet, whereas toric stores one catalogue per (mode, h11) under "
        "{{mode}}/catalog/h11_{{N}}/. Use {alt} instead."
    )

    def _unsupported(self, what: str, alt: str) -> "NotImplementedError":
        return NotImplementedError(
            self._SHARDED_MSG.format(what=what, dataset=self.dataset, alt=alt)
        )

    def _ensure_catalog(self) -> None:  # type: ignore[override]
        raise self._unsupported("_ensure_catalog()", "query(mode, h11=...)")

    def _ensure_conifold_catalog(self) -> None:  # type: ignore[override]
        raise self._unsupported(
            "_ensure_conifold_catalog()", "the tdf sub-dataset (toric has no conifold data)"
        )

    def query_conifolds(self, *args, **kwargs):  # type: ignore[override]
        raise self._unsupported(
            "query_conifolds()", "the tdf sub-dataset (toric has no conifold data)"
        )

    def _lookup(self, *args, **kwargs):  # type: ignore[override]
        raise self._unsupported("_lookup()", "query(mode, h11=..., ks_id=...) or load(...)")

    def _validate_key(self, *args, **kwargs):  # type: ignore[override]
        raise self._unsupported(
            "_validate_key()", "load(mode, h11, ks_id, triang_id), which validates its own key"
        )

    def _identifiers_to_list(self, *args, **kwargs):  # type: ignore[override]
        raise self._unsupported("_identifiers_to_list()", "query(mode, h11=...)")

    def info(self) -> None:  # type: ignore[override]
        r"""
        **Description:**
        Print a summary of the locally available toric buckets.

        Replaces the inherited :meth:`~stringforge.cy_io.CYDatabase.info`, which reads the
        monolithic catalogue toric does not have.  Counts come from Parquet footers, so this
        stays cheap even at h11 = 12 (billions of rows).
        """
        import pyarrow.parquet as _pq

        print(f"ToricCYDatabase — sub-dataset: {self.dataset!r}")
        print(f"  cache_dir : {self.cache_dir}")
        rows = []
        for h11 in range(1, 13):
            pcat = self._sdir("polytope_catalog", f"h11_{h11}")
            if not pcat.is_dir():
                continue
            n_poly = sum(_pq.ParquetFile(f).metadata.num_rows
                         for f in sorted(glob.glob(str(pcat / "data-*.parquet"))))
            per_mode = {}
            for mode in _MODES:
                gdir = self._sdir(mode, "geom", f"h11_{h11}")
                if gdir.is_dir():
                    per_mode[mode] = sum(
                        _pq.ParquetFile(f).metadata.num_rows
                        for f in sorted(glob.glob(str(gdir / "data-*.parquet")))
                    )
            rows.append((h11, n_poly, per_mode))
        if not rows:
            print("  (no buckets found)")
            return
        print(f"  {'h11':>4} {'polytopes':>12} {'frst phases':>14} {'vex phases':>13}")
        for h11, n_poly, per_mode in rows:
            frst = f"{per_mode.get('frst', 0):,}" if "frst" in per_mode else "—"
            vex = f"{per_mode.get('vex', 0):,}" if "vex" in per_mode else "—"
            print(f"  {h11:>4} {n_poly:>12,} {frst:>14} {vex:>13}")
        tot_p = sum(r[1] for r in rows)
        tot_f = sum(r[2].get("frst", 0) for r in rows)
        tot_v = sum(r[2].get("vex", 0) for r in rows)
        print(f"  {'tot':>4} {tot_p:>12,} {tot_f:>14,} {tot_v:>13,}")

    @staticmethod
    def _check_mode(mode: str) -> None:
        if mode not in _MODES:
            raise ValueError(f"mode must be one of {_MODES}; got {mode!r}.")

    # -- sharded IO ------------------------------------------------------- #
    def _sdir(self, *parts: str):
        return self.cache_dir.joinpath(*parts)

    def _index(self, split_dir) -> Dict[int, Tuple[int, int, int]]:
        """Load + cache the ``_ksid_index`` (``ks_id → (part, row0, n)``) for a split."""
        key = str(split_dir)
        if key not in self._idx_cache:
            t = pq.read_table(split_dir / "_ksid_index.parquet")
            self._idx_cache[key] = {
                int(k): (int(p), int(r), int(n)) for k, p, r, n in zip(
                    t.column("ks_id").to_pylist(), t.column("part").to_pylist(),
                    t.column("row0").to_pylist(), t.column("n").to_pylist())}
        return self._idx_cache[key]

    @staticmethod
    def _part_slice(split_dir, part: int, row0: int, n: int):
        """Read rows ``[row0, row0+n)`` from a part by reading **only the row group(s)** that cover
        them — not the whole part. Locally this is a bounded disk read; over a range-read filesystem
        (e.g. ``HfFileSystem``) it range-downloads only those row groups. Per-record I/O ≈ one row
        group (``ROW_GROUP_SIZE`` rows), independent of the part/file size.
        """
        pf = pq.ParquetFile(split_dir / f"data-{part:05d}.parquet")
        if pf.num_row_groups <= 1:
            return pf.read().slice(row0, n)
        lo, hi = row0, row0 + n
        groups, base, cum = [], None, 0
        for rg in range(pf.num_row_groups):
            rg_rows = pf.metadata.row_group(rg).num_rows
            s, e = cum, cum + rg_rows
            if not (e <= lo or s >= hi):            # this row group overlaps the requested range
                groups.append(rg)
                if base is None:
                    base = s
            cum = e
        return pf.read_row_groups(groups).slice(row0 - base, n)

    def _dataset(self, split_dir):
        key = str(split_dir)
        if key not in self._ds_cache:
            import pyarrow.dataset as pds
            meta = split_dir / "_metadata"
            ds = None
            if meta.exists():
                try:
                    ds = pds.parquet_dataset(str(meta))
                except Exception:  # noqa: BLE001
                    ds = None
            if ds is None:
                files = sorted(glob.glob(str(split_dir / "data-*.parquet")))
                ds = pds.dataset(files, format="parquet")
            self._ds_cache[key] = ds
        return self._ds_cache[key]

    @staticmethod
    def _attr_filter(h12=None, fav_N=None, fav_M=None, trilayer=None):
        import pyarrow.compute as pc
        conds = [pc.field(c) == v for c, v in
                 [("h12", h12), ("fav_N", fav_N), ("fav_M", fav_M), ("trilayer", trilayer)]
                 if v is not None]
        if not conds:
            return None
        filt = conds[0]
        for c in conds[1:]:
            filt = filt & c
        return filt

    @staticmethod
    def _hexify(df: pd.DataFrame) -> pd.DataFrame:
        if "wall_hash" in df.columns:
            df["wall_hash_hex"] = [w.hex() if isinstance(w, (bytes, bytearray)) else w
                                   for w in df["wall_hash"]]
        return df

    # -- queries ---------------------------------------------------------- #
    def query(self, mode: str, h11: int, ks_id=None, triang_id=None, h12=None,
              fav_N=None, fav_M=None, trilayer=None) -> pd.DataFrame:
        """Per-phase catalog for one ``(mode, h11)``. ``ks_id`` → O(1) index slice; else dataset scan.

        Returns thin rows + a derived ``phase_id`` + ``wall_hash_hex``; ``polytope_hash`` is not
        included (use :meth:`get_polytope` / :meth:`load`).
        """
        self._check_mode(mode)
        cdir = self._sdir(mode, "catalog", f"h11_{h11}")
        if ks_id is not None:
            idx = self._index(cdir)
            if ks_id not in idx:
                return pd.DataFrame()
            part, r0, n = idx[ks_id]
            df = self._part_slice(cdir, part, r0, n).to_pandas()
        else:
            df = self._dataset(cdir).to_table(
                filter=self._attr_filter(h12, fav_N, fav_M, trilayer)).to_pandas()
        if triang_id is not None:
            df = df[df["triang_id"] == triang_id]
        df = df.reset_index(drop=True)
        if len(df):
            df["phase_id"] = [f"{mode}:{h11}:{int(k)}:{int(t)}"
                              for k, t in zip(df["ks_id"], df["triang_id"])]
            df = self._hexify(df)
        return df

    def query_polytopes(self, h11: int, ks_id=None, h12=None, fav_N=None, fav_M=None,
                        trilayer=None) -> pd.DataFrame:
        """Shared per-polytope catalog for one h11 (FRST counts + left-joined VEX counts)."""
        pdir = self._sdir("polytope_catalog", f"h11_{h11}")
        if ks_id is not None:
            idx = self._index(pdir)
            if ks_id not in idx:
                return pd.DataFrame()
            part, r0, n = idx[ks_id]
            df = self._part_slice(pdir, part, r0, n).to_pandas()
        else:
            df = self._dataset(pdir).to_table(
                filter=self._attr_filter(h12, fav_N, fav_M, trilayer)).to_pandas()
        return self._join_vex_counts(h11, df).reset_index(drop=True)

    def _join_vex_counts(self, h11: int, df: pd.DataFrame) -> pd.DataFrame:
        vdir = self._sdir("polytope_vex_counts", f"h11_{h11}")
        files = sorted(glob.glob(str(vdir / "data-*.parquet"))) if vdir.is_dir() else []
        if len(df) and files:
            v = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
            df = df.merge(v[["ks_id", "n_vex", "n_vex_classes"]], on="ks_id", how="left")
        if "n_vex" not in df.columns:
            df["n_vex"] = pd.array([pd.NA] * len(df), dtype="Int64")
            df["n_vex_classes"] = pd.array([pd.NA] * len(df), dtype="Int64")
        return df

    # -- record loaders --------------------------------------------------- #
    def get_polytope(self, *, h11: int, ks_id: int) -> dict:  # type: ignore[override]
        r"""
        **Description:**
        Shared ``polytope`` record (vertices, glsm_basis, charge matrix, hash) via the index.
        There is no ``mode``: the polytope layer is shared between FRST and VEX.

        **Keyword-only, deliberately.** The inherited
        :meth:`~stringforge.cy_io.CYDatabase.get_polytope` takes ``(ks_id, h11, ...)`` — the
        opposite order — so a positional ``db.get_polytope(6, 42)`` would mean
        ``(h11=6, ks_id=42)`` here and ``(ks_id=6, h11=42)`` on every other database in the
        package. Rather than silently flip the meaning of existing calls by adopting the base
        order, positional use is rejected outright: pass ``h11=`` and ``ks_id=``.

        Args:
            h11 (int): Hodge number :math:`h^{1,1}` (the bucket).
            ks_id (int): Canonical Kreuzer–Skarke polytope index within that bucket.

        Returns:
            dict: The polytope record.
        """
        pcat_dir = self._sdir("polytope_catalog", f"h11_{h11}")
        idx = self._index(pcat_dir)
        if ks_id not in idx:
            raise KeyError(f"no polytope with (h11={h11}, ks_id={ks_id})")
        part, r0, _ = idx[ks_id]
        r = self._part_slice(pcat_dir, part, r0, 1).to_pylist()[0]        # metadata
        g = self._part_slice(self._sdir("polytope", f"h11_{h11}"), part, r0, 1).to_pylist()[0]  # geometry
        return {
            "h11": h11, "ks_id": ks_id, "polytope_hash": r["polytope_hash"],
            "vertices": np.asarray(g["vertices"]),
            "glsm_basis": [int(x) for x in g["glsm_basis"]],
            "glsm_charge_matrix": np.asarray([list(x) for x in g["glsm_charge_matrix"]]),
            "fav_N": bool(r["fav_N"]), "fav_M": bool(r["fav_M"]), "trilayer": bool(r["trilayer"]),
            "oob_dim": int(r["oob_dim"]), "basis_dim": int(r["basis_dim"]),
        }

    def load(self, mode: str, h11: int, ks_id: int, triang_id: int, in_basis: bool = False) -> dict:
        """Load one phase's geometry from ``{mode}`` (O(1) via the index). ``in_basis`` uses the
        shared ``glsm_basis``; ``polytope_hash`` joined from the polytope catalog; ``wall_hash`` bytes.
        """
        self._check_mode(mode)
        cdir = self._sdir(mode, "catalog", f"h11_{h11}")
        idx = self._index(cdir)
        if ks_id not in idx:
            raise KeyError(f"no {mode} phases for (h11={h11}, ks_id={ks_id})")
        part, r0, n = idx[ks_id]
        cat = self._part_slice(cdir, part, r0, n).to_pandas()
        rows = cat[cat["triang_id"] == triang_id]
        if rows.empty:
            raise KeyError(f"no {mode} phase (h11={h11}, ks_id={ks_id}, triang_id={triang_id})")
        c = rows.iloc[0]
        g = self._part_slice(self._sdir(mode, "geom", f"h11_{h11}"),
                             int(c["geom_shard_id"]), int(c["geom_row_index"]), 1).to_pylist()[0]
        coo = list(zip([int(x) for x in g["intnums_coo_i"]], [int(x) for x in g["intnums_coo_j"]],
                       [int(x) for x in g["intnums_coo_k"]], [int(x) for x in g["intnums_coo_v"]]))
        c2 = [int(x) for x in g["c2"]]
        poly = self.get_polytope(h11=h11, ks_id=ks_id)
        wh = c["wall_hash"]
        out = {
            "h11": h11, "ks_id": ks_id, "triang_id": triang_id, "h12": int(c["h12"]),
            "polytope_hash": poly["polytope_hash"],
            "wall_hash": (bytes(wh) if wh is not None else None),
            "phase_id": f"{mode}:{h11}:{ks_id}:{triang_id}",
            "heights": [float(x) for x in g["heights"]],
            "intnums_coo": np.asarray(coo, dtype=int).reshape(-1, 4),
            "c2": np.asarray(c2, dtype=int),
            "c2_origin": (int(g["c2_origin"]) if g["c2_origin"] is not None else None),
        }
        if in_basis:
            ib_coo, ib_c2 = nz.in_basis_from_stored(coo, c2, poly["glsm_basis"])
            out["intnums_coo_in_basis"] = np.asarray(ib_coo, dtype=int).reshape(-1, 4)
            out["c2_in_basis"] = np.asarray(ib_c2, dtype=int)
        return out


__all__ = ["ToricCYDatabase"]
