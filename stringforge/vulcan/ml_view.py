# Copyright 2026 Andreas Schachner
#
# This file is part of StringForge.
#
# StringForge is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

r"""
ML-friendly view of a Vulcan repo.

ML training jobs need a *stable* train/validation/test split that
never leaks the same Calabi-Yau geometry across splits.  Vulcan rows
carry a :func:`~stringforge.vulcan.schema.geometry_id` -- a
content-addressed hash of the model -- which we use as the natural
shard key: every row sharing a ``geometry_id`` lands in exactly one
split.

This module exposes :class:`VulcanMLView`, a thin layer over
:class:`stringforge.vulcan.VulcanReader` that:

* assigns each geometry to a fixed split deterministically;
* projects rows down to a feature list and optionally pads variable-
  length columns (``flux``, ``moduli_re``, ``moduli_im``, ``F_terms_*``)
  to fixed-length tensors;
* hands the result back as either a :class:`pandas.DataFrame` or, when
  ``datasets`` is installed, a streaming :class:`datasets.Dataset`.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .reader import VulcanReader
from .schema import GEOMETRY_KEYS

#: Recognised splits.  Two- or three-way slicing is the standard
#: pattern; downstream code that wants a different scheme can call
#: :func:`assign_split` directly with custom fractions.
DEFAULT_SPLITS: Tuple[str, ...] = ("train", "val", "test")

#: Default split fractions (train, val, test).  Sums to 1.0.
DEFAULT_FRACTIONS: Tuple[float, ...] = (0.8, 0.1, 0.1)

#: Columns whose values are variable-length lists.  Pad to a fixed
#: ``max_len`` when requested.
_LIST_COLUMNS_BUILTIN: Tuple[str, ...] = (
    "flux", "moduli_re", "moduli_im", "F_terms_re", "F_terms_im",
)


# ── deterministic split assignment ────────────────────────────────────────


def assign_split(
    geometry_id: str,
    *,
    fractions: Sequence[float] = DEFAULT_FRACTIONS,
    splits: Sequence[str] = DEFAULT_SPLITS,
    salt: str = "",
) -> str:
    r"""
    **Description:**
    Map a ``geometry_id`` to one of the supplied splits
    deterministically.

    The function hashes ``salt + geometry_id`` to a uniform point in
    ``[0, 1)`` and uses the cumulative-fraction breakpoints to pick a
    split.  Same ``(geometry_id, salt, fractions, splits)`` always
    returns the same split.

    Args:
        geometry_id: The Vulcan content-addressed model identifier.
        fractions: Per-split fractions.  Must sum to 1.0; order must
            match ``splits``.
        splits: Split labels.
        salt: Optional salt to perturb the assignment without changing
            geometry ids -- useful when running cross-validation
            folds.

    Returns:
        str: One of the labels in ``splits``.

    Raises:
        ValueError: When ``fractions`` does not match ``splits`` in
            length, contains a negative value, or sums to something
            other than 1.0.
    """
    if len(fractions) != len(splits):
        raise ValueError(
            f"fractions and splits length mismatch: "
            f"{len(fractions)} vs {len(splits)}"
        )
    if any(f < 0 for f in fractions):
        raise ValueError("fractions must be non-negative")
    total = float(sum(fractions))
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"fractions must sum to 1.0; got {total}")

    digest = hashlib.sha256((salt + geometry_id).encode("utf-8")).digest()
    # Use 8 bytes -> uniform u in [0, 1).
    n = int.from_bytes(digest[:8], "big")
    u = n / float(1 << 64)
    cumulative = 0.0
    for name, frac in zip(splits, fractions):
        cumulative += float(frac)
        if u < cumulative:
            return name
    # Numerical fall-through.
    return splits[-1]


# ── padding helpers ──────────────────────────────────────────────────────


def _pad_list_to(values: Iterable[Any], length: int, fill: Any = 0) -> np.ndarray:
    r"""Right-pad with ``fill`` or silently truncate to a length-``length`` numpy array."""
    arr = np.asarray(list(values))
    out = np.full(length, fill, dtype=arr.dtype if arr.size else float)
    n = min(length, arr.size)
    if n:
        out[:n] = arr[:n]
    return out


def _pad_column(series: pd.Series, length: int, fill: Any = 0) -> pd.Series:
    r"""Apply :func:`_pad_list_to` row-wise.

    A row value may arrive as a Python list, a numpy array (after a
    parquet round-trip), or ``None``.  We cannot use ``lst or []``
    because numpy arrays raise on truth-value checks; treat ``None``
    explicitly instead.
    """
    def _pad(lst: Any) -> np.ndarray:
        if lst is None:
            lst = []
        return _pad_list_to(lst, length, fill=fill)
    return series.apply(_pad)


# ── feature transform ──────────────────────────────────────────────────────


@dataclass
class FeatureSpec:
    r"""
    **Description:**
    Spec for shaping Vulcan rows into ML features.

    Attributes:
        features: Column projection.  Optional; when ``None`` every
            non-geometry column is kept.
        list_max_len: Fixed length for variable-length list columns.
            ``None`` leaves them as Python lists.  Rows longer than
            ``list_max_len`` are silently right-truncated.  Callers
            needing strict-length checks should pre-check
            ``max(len(x) for x in series)`` against ``list_max_len``.
        list_fill: Pad value when ``list_max_len`` is set.
        keep_geometry_id: Whether to retain ``geometry_id`` in the
            returned DataFrame -- typically wanted for monitoring
            held-out performance per geometry.
    """
    features: Optional[Sequence[str]] = None
    list_max_len: Optional[int] = None
    list_fill: Any = 0
    keep_geometry_id: bool = True


def shape_features(df: pd.DataFrame, spec: FeatureSpec) -> pd.DataFrame:
    r"""
    **Description:**
    Apply a :class:`FeatureSpec` to a Vulcan-shaped DataFrame.

    The returned DataFrame contains exactly the projected feature
    columns (plus ``geometry_id`` when ``spec.keep_geometry_id`` is
    True) with list columns padded to ``spec.list_max_len`` where
    applicable.

    Args:
        df: Source rows (typically the output of
            :meth:`stringforge.vulcan.VulcanReader.query`).
        spec: The feature spec.

    Returns:
        pd.DataFrame: Shaped feature table.

    Raises:
        KeyError: When ``spec.features`` references a column missing
            from ``df``.
    """
    if spec.features is not None:
        missing = [c for c in spec.features if c not in df.columns]
        if missing:
            raise KeyError(f"feature spec references missing columns: {missing}")
        out = df[list(spec.features)].copy()
    else:
        drop = list(GEOMETRY_KEYS) + ["run_id", "geometry_id", "created_at"]
        out = df.drop(columns=[c for c in drop if c in df.columns]).copy()

    if spec.list_max_len is not None:
        for col in _LIST_COLUMNS_BUILTIN:
            if col in out.columns:
                out[col] = _pad_column(out[col], spec.list_max_len,
                                       fill=spec.list_fill)

    if spec.keep_geometry_id and "geometry_id" in df.columns:
        if "geometry_id" not in out.columns:
            out["geometry_id"] = df["geometry_id"].values

    return out.reset_index(drop=True)


# ── ML view ──────────────────────────────────────────────────────────────


class VulcanMLView:
    r"""
    **Description:**
    Train/val/test view of a Vulcan source.

    Wraps a :class:`stringforge.vulcan.VulcanReader` with deterministic,
    geometry-disjoint splits and a configurable
    :class:`FeatureSpec`.  Two consumption paths:

    * :meth:`as_dataframe` -- materialise a pandas DataFrame for
      light workflows.
    * :meth:`as_hf_dataset` -- materialise a streaming
      :class:`datasets.Dataset` (requires the ``datasets`` package).

    Args:
        reader: Source reader.
        fractions: Per-split fractions (must sum to 1.0).
        splits: Split labels.
        salt: Optional salt for cross-validation folds.
        feature_spec: Default :class:`FeatureSpec`; can be overridden
            per call.
    """

    def __init__(
        self,
        reader: VulcanReader,
        *,
        fractions: Sequence[float] = DEFAULT_FRACTIONS,
        splits: Sequence[str] = DEFAULT_SPLITS,
        salt: str = "",
        feature_spec: Optional[FeatureSpec] = None,
    ) -> None:
        self.reader = reader
        self.fractions = tuple(fractions)
        self.splits = tuple(splits)
        self.salt = str(salt)
        self.feature_spec = feature_spec or FeatureSpec()

    def split_of(self, geometry_id: str) -> str:
        r"""
        **Description:**
        Return the split label assigned to a specific geometry.

        Args:
            geometry_id: The model identifier to look up.

        Returns:
            str: One of :attr:`splits`.
        """
        return assign_split(
            geometry_id,
            fractions=self.fractions,
            splits=self.splits,
            salt=self.salt,
        )

    def split_assignments(self) -> pd.DataFrame:
        r"""
        **Description:**
        Return a two-column table of ``(geometry_id, split)`` for
        every geometry the reader can see.

        Returns:
            pd.DataFrame: One row per distinct geometry.
        """
        cat = self.reader.catalog()
        unique = sorted(set(cat["geometry_id"].dropna().unique()))
        return pd.DataFrame({
            "geometry_id": unique,
            "split": [self.split_of(g) for g in unique],
        })

    def as_dataframe(
        self,
        split: str,
        *,
        feature_spec: Optional[FeatureSpec] = None,
        query_filters: Optional[Mapping[str, Any]] = None,
    ) -> pd.DataFrame:
        r"""
        **Description:**
        Materialise the rows of a specific split as a pandas
        DataFrame.

        The split is determined by hashing ``geometry_id``; rows from
        the same geometry are guaranteed to land in the same split.

        Args:
            split: Label to materialise (must be in :attr:`splits`).
            feature_spec: Override the default feature spec for this
                call.
            query_filters: Optional extra filters forwarded to
                :meth:`VulcanReader.query` to scope the input rows.

        Returns:
            pd.DataFrame: The shaped feature table.

        Raises:
            ValueError: When ``split`` is not a recognised label.
        """
        if split not in self.splits:
            raise ValueError(
                f"unknown split {split!r}; expected one of {self.splits}"
            )
        spec = feature_spec or self.feature_spec
        df = self.reader.query(**(query_filters or {}))
        if df.empty:
            return df
        df = df[df["geometry_id"].apply(self.split_of) == split]
        if df.empty:
            return df
        return shape_features(df, spec)

    def as_hf_dataset(
        self,
        split: str,
        *,
        feature_spec: Optional[FeatureSpec] = None,
        query_filters: Optional[Mapping[str, Any]] = None,
    ) -> "Any":
        r"""
        **Description:**
        Materialise the rows of a specific split as a HuggingFace
        :class:`datasets.Dataset` for streaming consumption by an ML
        framework.

        ``datasets`` is an optional dependency; this method imports it
        lazily and raises a clear error when it is not installed.

        Args:
            split: Label to materialise.
            feature_spec: Override the default feature spec.
            query_filters: Optional extra :class:`VulcanReader.query`
                filters.

        Returns:
            datasets.Dataset: A HuggingFace dataset.

        Raises:
            ImportError: When the optional ``datasets`` package is
                not installed.
        """
        try:
            from datasets import Dataset  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "VulcanMLView.as_hf_dataset requires the optional "
                "`datasets` package.  Install it with "
                "`pip install datasets`."
            ) from exc
        df = self.as_dataframe(
            split,
            feature_spec=feature_spec,
            query_filters=query_filters,
        )
        return Dataset.from_pandas(df, preserve_index=False)


__all__ = (
    "VulcanMLView",
    "FeatureSpec",
    "assign_split",
    "shape_features",
    "DEFAULT_SPLITS",
    "DEFAULT_FRACTIONS",
)
