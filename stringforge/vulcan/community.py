# Copyright 2026 Andreas Schachner
#
# This file is part of StringForge.
#
# StringForge is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

r"""
Community submission gate for the rolling certified-vacua database.

Phase 3 of the rolling-DB plan: let external contributors propose
vacua, but **trust nothing client-side**.  A submission is a parquet of
candidate rows; the maintainer's server re-runs the *full* physics
cascade (it never believes a submitter's ``is_susy`` / ``residual`` /
``verifier_id`` / ``cert_status``), and only re-certified, de-duplicated
rows are admitted.

Two structural guards make this safe:

1. **Full-cascade requirement.**  :func:`review_submission` refuses any
   verifier whose :meth:`VerifierSpec.includes_metric_pd_check` is
   ``False``.  The current curated-vault CI runs F-term / tadpole but
   *not* the Kähler-metric positive-definiteness check; opening
   community writes without that guard would re-introduce the
   2026-06-02 bug (≈465 saddle points miscounted as vacua) at the front
   door.

2. **Client-cert stripping.**  Any certification columns a submitter
   set are *dropped before re-certification*, so a forged
   ``cert_status="certified"`` / ``verifier_id`` can never survive --
   the server stamps its own.

This module is solver-free: the server-side certifier is injected
(built via :func:`stringforge.vulcan.certify.build_jaxvacua_certifier`
in production; a stub in tests).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, List, Mapping, Optional, Set, Tuple

import pandas as pd

from .certify import Certifier, certify_shard
from .schema import canonical_flux, geometry_id, metric_pd_passed
from .verifier import VerifierSpec

#: Physical dedup key type: ``(geometry_id, canonical_flux)``.  This is
#: deliberately *verifier-independent* -- the same vacuum is one entry
#: regardless of which verifier certified it (cf. ``record_id``, which
#: includes the verifier and is the per-row key, not the dedup key).
DedupKey = Tuple[str, Tuple[int, ...]]

#: Client-supplied columns the gate must NOT trust: unverified physics /
#: telemetry flags, the orientifold ``tadpole_charge`` (a physics
#: quantity the certifier does not recompute, so a forged value would
#: otherwise survive into the DB), and provenance fields the server
#: re-derives.  All are stripped before re-certification; the writer
#: then supplies authoritative ``tadpole_charge`` / ``created_at`` at
#: write time.  (Identity columns -- ``geometry_id`` and the geometry
#: keys -- are NOT listed here because the certifier re-stamps them
#: authoritatively from the trusted ``geometry``.  ``tags`` /
#: ``extra_data`` are the designed free-form user-metadata channel and
#: are intentionally preserved.)
UNTRUSTED_CLIENT_COLUMNS: Tuple[str, ...] = (
    # cert columns the server re-stamps
    "verifier_id", "cert_status", "cert_checks",
    "kahler_metric_min_eig", "hessian_min_eig", "mass_min_eig", "record_id",
    # unverified physics / telemetry
    "is_susy", "is_isd", "residual", "n_iterations",
    "W_re", "W_im", "F_terms_re", "F_terms_im",
    "solver_name", "solver_config_hash",
    # physics quantity the certifier does not recompute -> never trust
    "tadpole_charge",
    # provenance the server re-derives / re-stamps
    "git_sha", "seed", "wall_clock_s", "created_at",
)


@dataclass
class CommunityGateResult:
    r"""
    **Description:**
    Outcome of :func:`review_submission`.

    Attributes:
        accepted: ``True`` iff at least one row survived re-certification
            and de-duplication.
        certified: The admitted rows, stamped by the *server's*
            verifier (empty when nothing survived).
        n_submitted: Number of submitted rows.
        n_certified: Rows that passed server-side re-certification.
        n_duplicates: Certified rows dropped because their physical
            dedup key was already present.
        n_admitted: Rows finally admitted (certified minus duplicates).
        reasons: Human-readable notes (rejections, dedup counts, gate
            decisions).
    """
    accepted: bool
    certified: pd.DataFrame
    n_submitted: int
    n_certified: int
    n_duplicates: int
    n_admitted: int
    reasons: List[str] = field(default_factory=list)


def dedup_key(geometry: Mapping[str, Any], flux: Iterable[Any]) -> DedupKey:
    r"""
    **Description:**
    Return the verifier-independent physical dedup key
    ``(geometry_id, canonical_flux)`` for a vacuum.

    Args:
        geometry: Geometry index dict (mirror convention).
        flux: Integer flux vector.

    Returns:
        DedupKey: The dedup key.
    """
    return (geometry_id(geometry), canonical_flux(flux))


def review_submission(
    df: pd.DataFrame,
    *,
    certifier: Certifier,
    verifier_spec: VerifierSpec,
    geometry: Mapping[str, Any],
    seen_keys: Optional[Set[DedupKey]] = None,
    require_metric_pd: bool = True,
) -> CommunityGateResult:
    r"""
    **Description:**
    Independently re-certify a community submission and admit only the
    rows that pass and are not duplicates.

    The submission's own certification columns are discarded before
    re-certification (trust nothing client-side); the server stamps its
    own ``verifier_id`` / ``cert_status`` via
    :func:`stringforge.vulcan.certify.certify_shard`.

    Args:
        df: Submitted candidate rows (untrusted).
        certifier: The *server's* per-row physics certifier.
        verifier_spec: The server's verifier description.  Must include
            the Kähler-metric PD check unless ``require_metric_pd`` is
            explicitly disabled.
        geometry: The model's geometry index dict (mirror convention).
        seen_keys: Physical dedup keys already present in the DB; any
            certified row whose key is in this set is dropped as a
            duplicate.  ``None`` treats the DB as empty.
        require_metric_pd: Refuse the submission unless the verifier
            runs the metric-PD check (default ``True`` -- the
            2026-06-02 guard).

    Returns:
        CommunityGateResult: The admission decision and admitted rows.

    Raises:
        ValueError: If ``require_metric_pd`` and the verifier does not
            run the Kähler-metric PD check.
    """
    if require_metric_pd and not verifier_spec.includes_metric_pd_check():
        raise ValueError(
            "community gate: the server-side verifier MUST run the "
            "Kähler-metric positive-definiteness check ('kahler_metric_pd'). "
            "Refusing to open the community write path with a verifier that "
            "would re-introduce the 2026-06-02 saddle-point bug. Pass "
            "require_metric_pd=False only in a deliberately-degraded test."
        )

    reasons: List[str] = []
    n_submitted = int(len(df))

    # Trust nothing client-side: drop any certification columns AND any
    # unverified physics/telemetry flags the submitter set, so a forged
    # value can never survive re-certification.  Identity columns
    # (geometry_id, geometry keys) are re-stamped authoritatively by the
    # certifier from the trusted `geometry`, so they need not be stripped
    # here -- but the physics flags have no server-recomputed counterpart
    # yet, so they MUST be removed rather than carried through unverified.
    forged = [c for c in UNTRUSTED_CLIENT_COLUMNS if c in df.columns]
    if forged:
        reasons.append(f"stripped client-supplied / unverified columns before re-cert: {forged}")
    stripped = df.drop(columns=forged, errors="ignore")

    result = certify_shard(
        stripped, certifier=certifier, verifier_spec=verifier_spec, geometry=geometry,
    )
    reasons.append(
        f"server re-certification: {result.n_certified}/{n_submitted} rows passed "
        f"under verifier {result.verifier_id}"
    )

    certified = result.certified

    # Per-row metric-PD consistency: never admit a row the certifier
    # marked passed if its own evidence (cert_checks / min-eig) shows the
    # Kähler metric was not positive-definite.  Defence-in-depth against
    # an internally-inconsistent certifier (the 2026-06-02 signature),
    # not a client exploit.
    if require_metric_pd and len(certified):
        pd_mask = [
            metric_pd_passed(cj, me)
            for cj, me in zip(certified["cert_checks"], certified["kahler_metric_min_eig"])
        ]
        n_bad = int(len(pd_mask) - sum(pd_mask))
        if n_bad:
            reasons.append(
                f"dropped {n_bad} row(s): certifier reported passed but "
                f"kahler_metric_pd != True / min-eig <= 0 (passed/checks inconsistency)"
            )
        certified = certified.loc[pd_mask].reset_index(drop=True)

    # De-duplicate against the DB AND within the submission itself: a
    # submitter must not inflate a vacuum's count by repeating the same
    # flux, which would also break the record_id uniqueness invariant.
    n_duplicates = 0
    if len(certified):
        seen = set(seen_keys) if seen_keys else set()
        keep_mask: List[bool] = []
        batch_seen: Set[DedupKey] = set()
        for flux in certified["flux"]:
            k = dedup_key(geometry, flux)
            keep = k not in seen and k not in batch_seen
            keep_mask.append(keep)
            if keep:
                batch_seen.add(k)
        n_duplicates = int(len(keep_mask) - sum(keep_mask))
        if n_duplicates:
            reasons.append(
                f"dropped {n_duplicates} duplicate row(s) (already in the DB "
                f"or repeated within the submission)"
            )
        certified = certified.loc[keep_mask].reset_index(drop=True)

    n_admitted = int(len(certified))
    accepted = n_admitted > 0
    if not accepted:
        reasons.append("submission rejected: no rows survived re-certification + dedup")

    return CommunityGateResult(
        accepted=accepted,
        certified=certified,
        n_submitted=n_submitted,
        n_certified=result.n_certified,
        n_duplicates=n_duplicates,
        n_admitted=n_admitted,
        reasons=reasons,
    )


__all__ = (
    "CommunityGateResult",
    "DedupKey",
    "dedup_key",
    "review_submission",
)
