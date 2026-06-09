# Copyright 2026 Andreas Schachner
#
# This file is part of StringForge.
#
# StringForge is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

r"""
Frozen, citeable snapshots of a rolling Vulcan database.

A *rolling* database and a *citeable* database are different objects: a
paper must cite an immutable state, while the production repo keeps
moving.  This module resolves that tension the way the HuggingFace Hub
already allows -- a snapshot is

* an immutable **git tag** on the dataset repo (so
  ``VulcanReader.from_hf(repo, revision="v2026.06")`` reads a fixed
  state), plus
* a generated **manifest** (record counts per ``(h11, h12)``, the
  ``verifier_id`` histogram, the ``cert_status`` histogram, a content
  hash) committed at the tagged revision, plus
* optionally a **DOI** (HuggingFace mints dataset DOIs natively; a
  Zenodo mirror is an optional backup).

The manifest is the governance artefact: before tagging a snapshot for
citation you can read off how many records are still ``unverified`` or
``provisional``, and which verifiers certified the corpus.  A snapshot
whose manifest shows unverified records should not be cited.

This module is solver-free; manifest generation reads only the
catalogue plus the ``verifier_id`` / ``cert_status`` columns.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from .reader import CATALOG_COLUMNS, VulcanReader
from .schema import SCHEMA_VERSION, metric_pd_passed
from .verifier import VerifierRegistry

#: Filename of the manifest committed at a snapshot's tagged revision.
MANIFEST_FILENAME: str = "snapshot_manifest.json"

#: Sentinel tallied for records that carry no ``verifier_id`` /
#: ``cert_status`` -- i.e. were written before the certified-record
#: contract, or by a producer that skipped it.  Its presence in a
#: manifest is a governance red flag for a citeable snapshot.
UNVERIFIED_SENTINEL: str = "unverified"


@dataclass
class SnapshotManifest:
    r"""
    **Description:**
    Summary of a frozen snapshot.

    Attributes:
        snapshot_tag: The immutable git tag (e.g. ``"v2026.06"``).
        created_at: UTC timestamp string (passed in by the caller so
            manifest generation stays deterministic / testable).
        n_shards: Number of parquet shards in the snapshot.
        n_records: Total vacuum rows across all shards.
        records_by_hodge: ``{"h11=H,h12=H": count}`` per Hodge pair.
        verifier_id_histogram: ``{verifier_id: n_records}`` (records
            without a ``verifier_id`` tally under
            :data:`UNVERIFIED_SENTINEL`).
        cert_status_histogram: ``{cert_status: n_records}``.
        schema_version: The Vulcan parquet ``SCHEMA_VERSION``.
        content_sha256: Content hash over the sorted catalogue rows
            (a stable fingerprint of *which* shards, with their row
            counts and schema versions, the snapshot contains).  It
            deliberately covers catalogue identity ONLY -- not the cert
            columns -- so an in-place ``certified -> invalidated``
            transition does not change it; use the ``cert_status`` /
            ``verifier_id`` histograms to detect re-certification.
        metric_pd_failures: Number of records stamped ``certified``
            whose own evidence (``cert_checks`` / ``kahler_metric_min_eig``)
            shows the Kähler metric was NOT positive-definite (the
            2026-06-02 saddle signature).  Any non-zero value fails
            :meth:`fully_certified`.
        tool_version: Optional stringforge version string.
    """
    snapshot_tag: str
    created_at: str
    n_shards: int
    n_records: int
    records_by_hodge: Dict[str, int]
    verifier_id_histogram: Dict[str, int]
    cert_status_histogram: Dict[str, int]
    schema_version: int
    content_sha256: str
    metric_pd_failures: int = 0
    tool_version: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        r"""**Description:** Return a JSON-serialisable dict of the manifest."""
        return {
            "snapshot_tag": self.snapshot_tag,
            "created_at": self.created_at,
            "n_shards": self.n_shards,
            "n_records": self.n_records,
            "records_by_hodge": dict(sorted(self.records_by_hodge.items())),
            "verifier_id_histogram": dict(sorted(self.verifier_id_histogram.items())),
            "cert_status_histogram": dict(sorted(self.cert_status_histogram.items())),
            "schema_version": self.schema_version,
            "content_sha256": self.content_sha256,
            "metric_pd_failures": self.metric_pd_failures,
            "tool_version": self.tool_version,
        }

    def fully_certified(self, registry: Optional["VerifierRegistry"] = None) -> bool:
        r"""
        **Description:**
        Whether the snapshot is safe to tag for citation: a non-empty
        corpus in which every record is ``certified``, no record shows
        the 2026-06-02 metric-PD-failure signature, and (when a
        ``registry`` is supplied) every ``verifier_id`` resolves to a
        spec that ran the Kähler-metric PD check.

        Args:
            registry: Optional :class:`~stringforge.vulcan.verifier.VerifierRegistry`.
                When given, every non-sentinel ``verifier_id`` in the
                histogram must resolve to a spec whose
                ``includes_metric_pd_check()`` is ``True``; an
                unregistered or PD-lacking verifier fails the gate.

        Returns:
            bool: ``True`` iff the snapshot is citeable.
        """
        # An empty corpus is NEVER citeable (vacuous histograms would
        # otherwise pass) -- guards a mis-pointed / unsynced reader.
        if self.n_records == 0:
            return False
        # Defensive: the histograms must account for every record, else
        # a (hand-built / future) manifest could hide records from the
        # gate.  build_manifest always satisfies this, so it is a latent
        # guard, not a live path.
        if sum(self.verifier_id_histogram.values()) != self.n_records:
            return False
        if sum(self.cert_status_histogram.values()) != self.n_records:
            return False
        if UNVERIFIED_SENTINEL in self.verifier_id_histogram:
            return False
        if self.metric_pd_failures:
            return False
        non_certified = {
            k: v for k, v in self.cert_status_histogram.items() if k != "certified"
        }
        if sum(non_certified.values()) != 0:
            return False
        if registry is not None:
            for vid in self.verifier_id_histogram:
                if vid == UNVERIFIED_SENTINEL:
                    return False
                spec = registry.get(vid)
                if spec is None or not spec.includes_metric_pd_check():
                    return False
        return True


def _content_sha256(catalog: pd.DataFrame) -> str:
    r"""Stable content hash over the catalogue's identity-bearing columns.

    Intentionally covers catalogue identity only (no cert columns), so a
    re-certification that transitions ``cert_status`` in place does not
    change the hash; detect that via the histograms instead.
    """
    cols = [c for c in CATALOG_COLUMNS if c in catalog.columns]
    rows = (
        catalog[cols]
        .sort_values("path_in_repo")
        .to_dict(orient="records")
    )
    canonical = json.dumps(rows, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_manifest(
    reader: VulcanReader,
    *,
    snapshot_tag: str,
    created_at: str,
    tool_version: Optional[str] = None,
) -> SnapshotManifest:
    r"""
    **Description:**
    Build a :class:`SnapshotManifest` from a reader's current view.

    Reads the catalogue for per-Hodge and shard counts, then reads the
    ``verifier_id`` / ``cert_status`` / ``cert_checks`` /
    ``kahler_metric_min_eig`` columns of each shard to build the
    histograms and count metric-PD failures.  Records lacking the cert
    columns tally under :data:`UNVERIFIED_SENTINEL`, so an
    incompletely-certified corpus is visible rather than silently clean.

    Refuses to build a manifest if the reader's catalogue scan skipped
    any unreadable shard (``catalog.attrs['n_skipped'] > 0``): a citeable
    snapshot must not silently omit shards.

    Args:
        reader: A :class:`VulcanReader` (local or HF) over the dataset.
        snapshot_tag: The tag this manifest describes.
        created_at: UTC timestamp string (caller-supplied for
            determinism).
        tool_version: Optional stringforge version string.

    Returns:
        SnapshotManifest: The assembled manifest.

    Raises:
        ValueError: If the catalogue scan skipped any unreadable shard.
    """
    catalog = reader.catalog()
    n_skipped = int(catalog.attrs.get("n_skipped", 0))
    if n_skipped:
        raise ValueError(
            f"refusing to build a snapshot manifest: VulcanReader.catalog() "
            f"skipped {n_skipped} unreadable shard(s). A citeable snapshot "
            f"must not silently omit shards; repair or remove the corrupt "
            f"shard(s) (see the RuntimeWarning(s) above) and retry."
        )
    n_shards = int(len(catalog))
    n_records = int(catalog["n_rows"].sum()) if n_shards else 0

    records_by_hodge: Counter = Counter()
    for _, row in catalog.iterrows():
        key = f"h11={int(row['h11'])},h12={int(row['h12'])}"
        records_by_hodge[key] += int(row["n_rows"])

    verifier_hist: Counter = Counter()
    status_hist: Counter = Counter()
    metric_pd_failures = 0
    cert_cols = ["verifier_id", "cert_status", "cert_checks", "kahler_metric_min_eig"]
    for path_in_repo in catalog["path_in_repo"]:
        cols = reader.read_columns(path_in_repo, cert_cols)
        n = len(cols)
        if "verifier_id" in cols.columns:
            vids = cols["verifier_id"].fillna(UNVERIFIED_SENTINEL)
            for vid, cnt in vids.value_counts().items():
                verifier_hist[str(vid)] += int(cnt)
        else:
            verifier_hist[UNVERIFIED_SENTINEL] += n
        if "cert_status" in cols.columns:
            statuses = cols["cert_status"].fillna(UNVERIFIED_SENTINEL)
            for st, cnt in statuses.value_counts().items():
                status_hist[str(st)] += int(cnt)
        else:
            status_hist[UNVERIFIED_SENTINEL] += n
        # Count, among rows claiming `certified`, every one that does NOT
        # POSITIVELY assert a PD Kähler metric.  Using the positive
        # predicate (vs. "explicitly failed?") is what catches the literal
        # 2026-06-02 signature: a verifier that OMITTED the metric-PD
        # check (no key, NaN eig) does not positively assert it, so it is
        # counted here -- and a shard that carries cert_status but lacks
        # the cert_checks / min-eig columns is treated as "not asserted"
        # for every certified row rather than silently skipped.
        if "cert_status" in cols.columns:
            checks_col = (
                cols["cert_checks"] if "cert_checks" in cols.columns
                else pd.Series([None] * n, index=cols.index)
            )
            eig_col = (
                cols["kahler_metric_min_eig"] if "kahler_metric_min_eig" in cols.columns
                else pd.Series([float("nan")] * n, index=cols.index)
            )
            for status, cj, me in zip(cols["cert_status"], checks_col, eig_col):
                if status != "certified":
                    continue
                if not metric_pd_passed(cj, me):
                    metric_pd_failures += 1

    return SnapshotManifest(
        snapshot_tag=snapshot_tag,
        created_at=created_at,
        n_shards=n_shards,
        n_records=n_records,
        records_by_hodge=dict(records_by_hodge),
        verifier_id_histogram=dict(verifier_hist),
        cert_status_histogram=dict(status_hist),
        schema_version=SCHEMA_VERSION,
        content_sha256=_content_sha256(catalog),
        metric_pd_failures=metric_pd_failures,
        tool_version=tool_version,
    )


def freeze_snapshot(
    reader: VulcanReader,
    *,
    tag: str,
    created_at: str,
    repo: Optional[str] = None,
    token: Optional[str] = None,
    message: Optional[str] = None,
    require_fully_certified: bool = True,
    registry: Optional[VerifierRegistry] = None,
    local_manifest_dir: Optional[Path] = None,
    dry_run: bool = False,
    tool_version: Optional[str] = None,
) -> SnapshotManifest:
    r"""
    **Description:**
    Freeze a citeable snapshot: build the manifest, optionally write it
    locally, and (unless ``dry_run``) commit it to the HuggingFace repo
    and create an immutable tag.

    Args:
        reader: A :class:`VulcanReader` over the dataset to freeze.
        tag: The immutable git tag to create (e.g. ``"v2026.06"``).
        created_at: UTC timestamp string (caller-supplied).
        repo: HuggingFace ``"user/repo"`` to tag.  Required unless
            ``dry_run`` (then the manifest is built but nothing is
            pushed).
        token: HuggingFace write token; falls back to ``HF_TOKEN``.
        message: Optional commit/tag message.
        require_fully_certified: If ``True`` (default), refuse to tag a
            snapshot whose manifest is not fully certified
            (:meth:`SnapshotManifest.fully_certified`) -- the
            governance gate against citing unverified records.  The
            manifest is still returned so the caller can inspect *why*.
        registry: Optional :class:`~stringforge.vulcan.verifier.VerifierRegistry`
            passed to :meth:`SnapshotManifest.fully_certified`; when
            given, every ``verifier_id`` in the corpus must resolve to a
            registered spec that ran the metric-PD check.
        local_manifest_dir: If given, also write the manifest JSON to
            ``<dir>/<MANIFEST_FILENAME>`` (useful for review / CI).
        dry_run: Build (and optionally write locally) the manifest but
            make no network call and create no tag.
        tool_version: Optional stringforge version string.

    Returns:
        SnapshotManifest: The manifest (built regardless of ``dry_run``).

    Raises:
        ValueError: If the corpus is empty; if ``require_fully_certified``
            and the manifest is not fully certified; or if a real
            (non-dry-run) freeze is requested without a ``repo``.
    """
    manifest = build_manifest(
        reader, snapshot_tag=tag, created_at=created_at, tool_version=tool_version,
    )

    if local_manifest_dir is not None:
        d = Path(local_manifest_dir)
        d.mkdir(parents=True, exist_ok=True)
        (d / MANIFEST_FILENAME).write_text(
            json.dumps(manifest.to_dict(), indent=2, sort_keys=True)
        )

    # An empty corpus is never citeable -- guard explicitly so this
    # fires even when require_fully_certified=False.
    if manifest.n_records == 0:
        raise ValueError(
            f"refusing to freeze snapshot {tag!r}: the corpus is empty "
            f"(n_shards={manifest.n_shards}, n_records=0). Check that the "
            f"reader points at a synced corpus."
        )

    if require_fully_certified and not manifest.fully_certified(registry=registry):
        raise ValueError(
            f"refusing to freeze snapshot {tag!r}: manifest is not fully "
            f"certified (verifier histogram={manifest.verifier_id_histogram}, "
            f"cert_status histogram={manifest.cert_status_histogram}, "
            f"metric_pd_failures={manifest.metric_pd_failures}). "
            f"Pass require_fully_certified=False to override, or finish "
            f"certifying / retracting the offending records first."
        )

    if dry_run:
        return manifest

    if not repo:
        raise ValueError(
            "freeze_snapshot: a real freeze requires repo='user/repo' "
            "(pass dry_run=True to build the manifest only)."
        )

    _push_and_tag(
        repo=repo,
        tag=tag,
        manifest=manifest,
        token=token,
        message=message or f"vulcan snapshot {tag}",
    )
    return manifest


def _push_and_tag(
    *,
    repo: str,
    tag: str,
    manifest: SnapshotManifest,
    token: Optional[str],
    message: str,
) -> None:
    r"""
    Upload the manifest to the repo root and create an immutable tag at
    the resulting commit.  Lazily imports ``huggingface_hub`` so this
    module stays import-clean without the optional dependency.
    """
    import os
    try:
        from huggingface_hub import CommitOperationAdd, HfApi  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised via dependency
        raise ImportError(
            "freeze_snapshot (non-dry-run) requires `huggingface_hub`. "
            "Install it via `pip install 'stringforge[sync]'`."
        ) from exc

    api = HfApi(token=token or os.environ.get("HF_TOKEN"))
    payload = json.dumps(manifest.to_dict(), indent=2, sort_keys=True).encode("utf-8")
    api.create_commit(
        repo_id=repo,
        repo_type="dataset",
        operations=[CommitOperationAdd(path_in_repo=MANIFEST_FILENAME, path_or_fileobj=payload)],
        commit_message=message,
    )
    api.create_tag(repo_id=repo, repo_type="dataset", tag=tag, tag_message=message)


__all__ = (
    "SnapshotManifest",
    "build_manifest",
    "freeze_snapshot",
    "MANIFEST_FILENAME",
    "UNVERIFIED_SENTINEL",
)
