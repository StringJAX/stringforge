# Copyright 2026 Andreas Schachner
#
# This file is part of StringForge.
#
# StringForge is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

r"""
Production parquet schema for the Vulcan tier.

Vulcan writes uniform, streaming-friendly parquet shards to a
production HuggingFace repo.  The schema is intentionally a strict
superset of :data:`stringforge.vacuavault.schema.REQUIRED_COLUMNS` so a
Vulcan shard is loadable by the existing vault tooling after column
projection -- this is the compatibility hook that enables a future
Vulcan -> VacuaWriter promotion step.

The schema is frozen at ``SCHEMA_VERSION``.  Forward-compatible
additions (new optional columns) are allowed; column removals or type
changes require a version bump.
"""
from __future__ import annotations

import hashlib
import json
import math
import warnings
from typing import Any, Iterable, Mapping, Optional, Tuple

import pandas as pd
import pyarrow as pa

# Re-use the vault-side floor.  This is the contract Vulcan must respect
# for vault-tooling compatibility.
from stringforge.vacuavault.schema import (
    LABEL_SLUG_RE,
    REQUIRED_COLUMNS as VAULT_REQUIRED_COLUMNS,
)

#: Vulcan production schema version.  Bump on any non-additive change.
SCHEMA_VERSION: int = 1

#: Required columns on every Vulcan shard.  Strict superset of the
#: vault floor; the extras are the production telemetry every cluster
#: write must carry.
REQUIRED_COLUMNS: Tuple[str, ...] = (
    # vault-compatibility floor
    "flux", "moduli_re", "moduli_im", "tau_re", "tau_im",
    # production-tier required
    "run_id", "geometry_id", "tadpole_charge", "created_at",
)

#: Certification columns (the "certified-vacuum record" contract).
#:
#: These pin *which verifier* certified a row and *what it found*, so a
#: later verifier-bug fix can select exactly the affected records and
#: transition their status rather than silently leaving corrupt vacua
#: published.  Motivated by the 2026-06-02 incident (``worklog/ERRORS.md``):
#: the Kähler-metric positive-definiteness check was once omitted and
#: ~465 saddle points were miscounted as vacua.  Without ``verifier_id``
#: that class of bug is unrecoverable at community scale.
#:
#: * ``verifier_id`` -- content hash naming the verifier
#:   (see :mod:`stringforge.vulcan.verifier`); the join key for
#:   "which rows did this verifier certify?".
#: * ``cert_status`` -- one of :data:`CERT_STATUSES`.  A verifier-bug
#:   fix *transitions* a record (never deletes it), mirroring the
#:   ``vacua_vault`` ``retractions.parquet`` pattern.
#: * ``cert_checks`` -- JSON dict of per-check pass/fail booleans for
#:   the physicality cascade (runaway, dilaton, cone, **metric-PD**,
#:   plus the stability min-eigs).  Makes the 2026-06-02 signature
#:   ("passed cone but failed metric-PD") queryable.
#: * ``kahler_metric_min_eig`` -- the number whose *absence* was the
#:   bug; storing it makes a recurrence detectable in aggregate.
#: * ``hessian_min_eig`` / ``mass_min_eig`` -- distinguish a genuine
#:   minimum from a saddle.
#: * ``record_id`` -- unique *row* key
#:   ``sha(geometry_id ‖ rounded-flux ‖ verifier_id)`` for retraction
#:   targeting.  Physical dedup of a vacuum uses the
#:   *verifier-independent* ``(geometry_id, rounded-flux)`` separately.
CERT_COLUMNS: Tuple[str, ...] = (
    "verifier_id", "cert_status", "cert_checks",
    "kahler_metric_min_eig", "hessian_min_eig", "mass_min_eig",
    "record_id",
)

#: Recognised values of the ``cert_status`` column.  ``certified`` =
#: passed the full cascade under its ``verifier_id``; ``provisional`` =
#: community-submitted, awaiting independent server-side re-cert;
#: ``invalidated`` = a later verifier-bug fix demoted it.
CERT_STATUSES: Tuple[str, ...] = ("certified", "provisional", "invalidated")

#: Optional columns recognised by the schema.  Shards may carry any
#: subset; downstream consumers must tolerate missing optionals.
OPTIONAL_COLUMNS: Tuple[str, ...] = (
    "h11", "h12", "ks_id", "triang_id", "conifold_id", "cicy_id",
    "W_re", "W_im", "F_terms_re", "F_terms_im",
    "residual", "n_iterations", "is_susy", "is_isd",
    "solver_name", "solver_config_hash",
    "git_sha", "seed", "wall_clock_s",
    "tags", "extra_data",
    # certified-vacuum record contract (see CERT_COLUMNS)
    *CERT_COLUMNS,
)

#: Sentinel for geometry-key fields when not applicable to the model
#: (e.g. a CICY shard has ``ks_id = triang_id = -1``).  Matches the
#: convention used by :func:`stringforge.vacua_writer._extract_model_identity`.
GEOMETRY_KEY_SENTINEL: int = -1

#: Geometry keys, in canonical order, used to derive ``geometry_id``.
#: Stable ordering is critical: the hash must be reproducible across
#: cluster nodes and across stringforge versions.
#: ``h11``, ``h12`` are interpreted in mirror (jaxvacua / lcs_tree)
#: convention, matching ``LCSDatabase``.
GEOMETRY_KEYS: Tuple[str, ...] = (
    "h11", "h12", "ks_id", "triang_id", "conifold_id", "cicy_id",
)


def pyarrow_schema(extra_columns: Optional[Mapping[str, pa.DataType]] = None) -> pa.Schema:
    r"""
    **Description:**
    Return the canonical pyarrow schema for a Vulcan shard.

    Required columns appear in :data:`REQUIRED_COLUMNS` order; the
    recognised optionals follow.  ``extra_columns`` lets a caller
    extend the schema with project-specific fields without modifying
    the canonical layout.

    Args:
        extra_columns: Optional mapping ``{name: pyarrow_type}`` of
            project-specific extension columns.  Names must not
            collide with canonical fields.

    Returns:
        pa.Schema: The full schema, ready to pass to
        :func:`pyarrow.parquet.write_table`.

    Raises:
        ValueError: If an entry in ``extra_columns`` shadows a
            canonical field name.
    """
    fields: list[pa.Field] = [
        # vault floor
        pa.field("flux", pa.list_(pa.int32())),
        pa.field("moduli_re", pa.list_(pa.float64())),
        pa.field("moduli_im", pa.list_(pa.float64())),
        pa.field("tau_re", pa.float64()),
        pa.field("tau_im", pa.float64()),
        # production-tier required
        pa.field("run_id", pa.string()),
        pa.field("geometry_id", pa.string()),
        pa.field("tadpole_charge", pa.int32()),
        pa.field("created_at", pa.timestamp("us", tz="UTC")),
        # optionals
        pa.field("h11", pa.int32()),
        pa.field("h12", pa.int32()),
        pa.field("ks_id", pa.int32()),
        pa.field("triang_id", pa.int32()),
        pa.field("conifold_id", pa.int32()),
        pa.field("cicy_id", pa.int32()),
        pa.field("W_re", pa.float64()),
        pa.field("W_im", pa.float64()),
        pa.field("F_terms_re", pa.list_(pa.float64())),
        pa.field("F_terms_im", pa.list_(pa.float64())),
        pa.field("residual", pa.float64()),
        pa.field("n_iterations", pa.int32()),
        pa.field("is_susy", pa.bool_()),
        pa.field("is_isd", pa.bool_()),
        pa.field("solver_name", pa.string()),
        pa.field("solver_config_hash", pa.string()),
        pa.field("git_sha", pa.string()),
        pa.field("seed", pa.int64()),
        pa.field("wall_clock_s", pa.float32()),
        pa.field("tags", pa.string()),
        pa.field("extra_data", pa.string()),
        # certified-vacuum record contract
        pa.field("verifier_id", pa.string()),
        pa.field("cert_status", pa.string()),
        pa.field("cert_checks", pa.string()),          # JSON-encoded dict
        pa.field("kahler_metric_min_eig", pa.float64()),
        pa.field("hessian_min_eig", pa.float64()),
        pa.field("mass_min_eig", pa.float64()),
        pa.field("record_id", pa.string()),
    ]
    if extra_columns:
        existing = {f.name for f in fields}
        for name, dtype in extra_columns.items():
            if name in existing:
                raise ValueError(
                    f"Vulcan schema: extra_columns cannot override canonical "
                    f"field {name!r}"
                )
            fields.append(pa.field(name, dtype))
    return pa.schema(fields)


def geometry_id(geometry: Mapping[str, Any]) -> str:
    r"""
    **Description:**
    Compute a stable, content-addressed identifier for a Calabi-Yau
    model.

    The identifier is the first 16 hex characters of
    ``sha1(canonical-json(geometry-keys))``.  Sufficient as a join key
    against a Vulcan catalogue; not intended as a cryptographic
    digest.  Stable across processes, machines, and Python runs.
    Missing keys are filled with :data:`GEOMETRY_KEY_SENTINEL` so a
    sparse mapping yields the same id as one that lists every key with
    explicit sentinels.

    .. note::
        ``geometry`` MUST be a server-canonicalised index dict (mirror
        convention; produced by the canonical identity path), never
        derived per-row from untrusted submitter input -- otherwise the
        same physical geometry can hash two ways and dedup silently
        fails.  An *absent* key maps to the sentinel ``-1`` ("not
        applicable"); an explicit ``0`` (e.g. ``conifold_id=0``, the
        conifold-limit model) is a genuinely distinct value, NOT
        normalised to the sentinel.

    Args:
        geometry: Mapping carrying any subset of :data:`GEOMETRY_KEYS`.

    Returns:
        str: 16-hex-character identifier.
    """
    payload = {k: int(geometry.get(k, GEOMETRY_KEY_SENTINEL)) for k in GEOMETRY_KEYS}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:16]


def canonical_flux(flux: Iterable[Any]) -> Tuple[int, ...]:
    r"""
    **Description:**
    Coerce a flux vector to a canonical tuple of Python ints.

    Used as the *verifier-independent* component of both the physical
    dedup key ``(geometry_id, canonical_flux)`` and the per-row
    :func:`record_id`.  Rounding to int is intentional: the integer
    flux quanta are the physical identity of a vacuum, independent of
    any floating-point representation a producer happened to use.

    Args:
        flux: An iterable of integer-valued flux components.

    Returns:
        Tuple[int, ...]: The flux as a tuple of rounded Python ints.
    """
    return tuple(int(round(float(x))) for x in flux)


def record_id(geometry: Mapping[str, Any], flux: Iterable[Any], verifier_id: str) -> str:
    r"""
    **Description:**
    Compute the unique *row* key of a certified-vacuum record.

    The key is ``sha1`` of ``geometry_id ‖ canonical-flux ‖ verifier_id``,
    truncated to 24 hex characters.  It is the retraction-targeting
    handle: re-certifying the *same* physical vacuum under a *new*
    ``verifier_id`` produces a *different* ``record_id`` (a new row,
    preserving history) -- so this key must NOT be used for physical
    dedup.  Physical dedup uses the verifier-independent
    ``(geometry_id, canonical_flux(flux))`` pair instead.

    Args:
        geometry: Geometry mapping (see :func:`geometry_id`).
        flux: The integer flux vector of the vacuum.
        verifier_id: The certifying verifier's content hash
            (see :mod:`stringforge.vulcan.verifier`).

    Returns:
        str: 24-hex-character unique row identifier.
    """
    gid = geometry_id(geometry)
    fcanon = canonical_flux(flux)
    payload = json.dumps(
        {"geometry_id": gid, "flux": list(fcanon), "verifier_id": str(verifier_id)},
        sort_keys=True, separators=(",", ":"),
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:24]


def metric_pd_passed(cert_checks_json: Any, min_eig: Any) -> bool:
    r"""
    **Description:**
    Positive assertion that a row's *own evidence* confirms a
    positive-definite Kähler metric: ``cert_checks['kahler_metric_pd']
    is True`` **and** a finite ``kahler_metric_min_eig > 0``.

    This is the **positive** form, and the distinction is load-bearing:
    the 2026-06-02 incident was an *omitted* check (no
    ``kahler_metric_pd`` key, ``min_eig`` left ``NaN``), not a check that
    explicitly failed.  A negative test ("is it explicitly False or
    ``<= 0``?") would let an omitted check pass; this predicate treats a
    missing key, an unparseable ``cert_checks``, or a non-finite /
    non-positive eigenvalue as **not passed**.  Both the community
    ingest gate and the snapshot citation gate use it, so the two agree
    by construction.

    Args:
        cert_checks_json: The row's ``cert_checks`` value (a JSON string;
            may be ``None``).
        min_eig: The row's ``kahler_metric_min_eig`` value.

    Returns:
        bool: ``True`` iff the row positively asserts a PD Kähler metric.
    """
    try:
        checks = json.loads(cert_checks_json)
        ok = isinstance(checks, dict) and checks.get("kahler_metric_pd") is True
    except (TypeError, ValueError):
        return False
    if not ok:
        return False
    try:
        eig = float(min_eig)
    except (TypeError, ValueError):
        return False
    return math.isfinite(eig) and eig > 0.0


def validate_cert_status(status: str) -> None:
    r"""
    **Description:**
    Validate a ``cert_status`` value against :data:`CERT_STATUSES`.

    Args:
        status: The candidate status string.

    Raises:
        ValueError: If ``status`` is not a recognised certification
            status.
    """
    if status not in CERT_STATUSES:
        raise ValueError(
            f"invalid cert_status {status!r}; expected one of {CERT_STATUSES}"
        )


def validate_schema(df: pd.DataFrame, *, warn_unknown: bool = True) -> None:
    r"""
    **Description:**
    Validate that ``df`` satisfies the Vulcan schema.

    Aggregates every problem into a single :class:`ValueError` --
    callers see all schema violations, not just the first one.  Only
    cheap structural checks are performed (column presence, run-id
    slug shape, homogeneity of ``geometry_id``); no row-level
    numerical validation.

    Args:
        df: Coerced DataFrame ready to be written as a shard.

    Raises:
        ValueError: If any structural invariant is violated.  The
            message names every detected problem.
    """
    problems: list[str] = []

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        problems.append(f"missing required columns: {missing}")

    # Identity invariant: one shard = one geometry.
    if "geometry_id" in df.columns and len(df) > 0:
        unique_ids = df["geometry_id"].unique()
        if len(unique_ids) > 1:
            problems.append(
                f"mixed geometry_ids in a single shard ({len(unique_ids)} "
                f"distinct); Vulcan shards must be homogeneous per geometry"
            )

    # Run-id slug check.
    if "run_id" in df.columns and len(df) > 0:
        bad = [
            rid for rid in df["run_id"].unique()
            if not isinstance(rid, str) or not LABEL_SLUG_RE.match(rid)
        ]
        if bad:
            problems.append(
                f"run_id values violate LABEL_SLUG_RE (first 3 shown): "
                f"{bad[:3]}"
            )

    # Vault-floor list columns are non-null.
    for col in ("flux", "moduli_re", "moduli_im"):
        if col in df.columns and df[col].isna().any():
            problems.append(f"required list column {col!r} contains nulls")

    # cert_status, when present, must be a recognised value.  Skip NaN:
    # certify.py intentionally stamps rejected rows with status=None.
    if "cert_status" in df.columns and len(df) > 0:
        bad_status = [
            s for s in df["cert_status"].dropna().unique() if s not in CERT_STATUSES
        ]
        if bad_status:
            problems.append(
                f"cert_status values not in CERT_STATUSES (first 3 shown): "
                f"{bad_status[:3]}"
            )

    unknown = [
        c for c in df.columns
        if c not in REQUIRED_COLUMNS and c not in OPTIONAL_COLUMNS
    ]
    # Unknown columns are tolerated (passed through to parquet so a future
    # schema version can absorb them) -- we do NOT raise on them.  A lone
    # unknown column is often a typo (e.g. ``cert_statuss``), so we surface
    # it as a warning when validating possibly-untrusted / hand-built
    # frames.  The writer passes ``warn_unknown=False`` because arbitrary
    # extension columns are a *designed* feature there (the production ML
    # columns), so warning on every write would just cry wolf.
    if unknown and warn_unknown:
        warnings.warn(
            f"Vulcan schema: unknown columns (passed through): {sorted(unknown)}",
            stacklevel=2,
        )

    if problems:
        raise ValueError("Vulcan schema validation failed: " + "; ".join(problems))


def shard_key(geometry: Mapping[str, Any], bucket_bits: int = 4) -> Tuple[int, int, int]:
    r"""
    **Description:**
    Return the ``(h12, h11, bucket)`` key used to organise shards on
    disk.

    ``bucket`` is drawn from the top ``bucket_bits`` of
    ``sha1(geometry_id)``.  Bucketing prevents a single ``(h11, h12)``
    point from accumulating an unmanageably large directory once many
    geometries are present.

    Args:
        geometry: Geometry mapping (see :func:`geometry_id`).
        bucket_bits: Width of the bucket prefix in bits.

    Returns:
        Tuple[int, int, int]: ``(h12, h11, bucket)``.
    """
    h11 = int(geometry.get("h11", GEOMETRY_KEY_SENTINEL))
    h12 = int(geometry.get("h12", GEOMETRY_KEY_SENTINEL))
    digest = hashlib.sha1(geometry_id(geometry).encode("ascii")).digest()
    mask = (1 << bucket_bits) - 1
    bucket = digest[0] & mask
    return (h12, h11, bucket)


def shard_relpath(geometry: Mapping[str, Any], run_id: str, bucket_bits: int = 4) -> str:
    r"""
    **Description:**
    Canonical HF-repo-relative path for a Vulcan shard.

    Layout: ``h12_{H}/h11_{H}/bucket_{B}/run_{run_id}.parquet``.

    Args:
        geometry: Geometry mapping (see :func:`geometry_id`).
        run_id: Slug-safe identifier (see :func:`is_safe_run_id`).
        bucket_bits: Width of the bucket prefix.

    Returns:
        str: Forward-slash-separated path inside the repo.
    """
    h12, h11, bucket = shard_key(geometry, bucket_bits=bucket_bits)
    return f"h12_{h12}/h11_{h11}/bucket_{bucket}/run_{run_id}.parquet"


def is_safe_run_id(run_id: str) -> bool:
    r"""
    **Description:**
    Predicate: ``run_id`` is safe to use as a parquet filename
    component on POSIX, as an HF path segment, and as a vault label.

    Reuses :data:`stringforge.vacuavault.schema.LABEL_SLUG_RE` so a
    Vulcan run_id is admissible as a vault label after a future
    promotion step.  Run-ids containing ``'__'`` are rejected because
    that sequence is reserved by the staging-area encoding
    (``writer.py`` replaces ``'/'`` with ``'__'`` to produce local
    filenames).

    Args:
        run_id: Candidate identifier.

    Returns:
        bool: ``True`` if and only if ``run_id`` is a string matching
        ``LABEL_SLUG_RE`` and does not contain ``'__'``.
    """
    if not isinstance(run_id, str):
        return False
    if not LABEL_SLUG_RE.match(run_id):
        return False
    if "__" in run_id:   # reserved as the local "/" encoding separator
        return False
    return True


def vault_floor_compatible(df: pd.DataFrame) -> bool:
    r"""
    **Description:**
    Test whether ``df`` carries every column required by the curated
    vault layer (:data:`stringforge.vacuavault.schema.REQUIRED_COLUMNS`).

    Used as a precondition for a future Vulcan-to-VacuaWriter
    promotion step.

    Args:
        df: A coerced or freshly-loaded Vulcan shard.

    Returns:
        bool: ``True`` if ``df`` is loadable by the vault tooling
        after column projection.
    """
    return all(c in df.columns for c in VAULT_REQUIRED_COLUMNS)
