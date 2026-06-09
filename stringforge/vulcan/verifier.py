# Copyright 2026 Andreas Schachner
#
# This file is part of StringForge.
#
# StringForge is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

r"""
Verifier identity for the certified-vacuum record contract.

Every certified vacuum carries a ``verifier_id`` (see
:data:`stringforge.vulcan.schema.CERT_COLUMNS`).  It is the content
hash of the *exact* certification procedure that accepted the row: the
``jaxvacua`` version, the ordered set of physicality checks, the
numerical tolerances, and whether the extended stability suite was
applied.

The motivation is the 2026-06-02 incident (``worklog/ERRORS.md``): the
Kähler-metric positive-definiteness check was once omitted from the
acceptance cascade, and ~465 saddle points were miscounted as vacua.
At single-researcher scale that was recoverable.  Auto-published to a
community-facing rolling database it would not be -- *unless* every
record records which verifier certified it, so a later bug-fix can
issue exactly the query "which published vacua did the buggy verifier
accept?" and transition them to ``invalidated`` rather than leaving
silent corruption in the corpus.

This module deliberately does **not** import any physics solver.  A
:class:`VerifierSpec` is a pure-data description; the certifier that
*runs* the checks (Phase 1 of the rolling-DB plan) constructs a spec
from the live ``jaxvacua`` build and hands it here for hashing.  Keeping
this module solver-free lets the curation / CI tier compute and compare
``verifier_id`` values without a JAX install.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Tuple

#: The ordered physicality-cascade check names.  The order is part of
#: the verifier identity: a verifier that runs the same checks in a
#: different order is, for provenance purposes, a different verifier.
#: These names mirror the cascade in
#: ``jaxvacua.flux_utils.is_physical`` plus Vulcan's local endpoint
#: stability diagnostics.
CANONICAL_CHECK_NAMES: Tuple[str, ...] = (
    "runaway_bound",        # max(|moduli|, |tau|) <= moduli_max
    "dilaton_floor",        # Im(tau) > s_lower
    "kahler_cone",          # cone hyperplane test
    "kahler_metric_pd",     # eig(K) > 0  -- the 2026-06-02 check
    "im_z_positive",        # fallback Im(z_i) > 0
    "hessian_min_eig",      # stability: real-Hessian min eigenvalue
    "mass_min_eig",         # stability: mass-matrix min eigenvalue
)


@dataclass(frozen=True)
class VerifierSpec:
    r"""
    **Description:**
    Pure-data description of a certification procedure.

    The frozen fields are exactly those whose change should produce a
    new :func:`verifier_id`.  Two certifications agree on
    ``verifier_id`` if and only if they would accept/reject every
    candidate identically (modulo non-determinism the spec does not
    capture, which is why ``jaxvacua_git_sha`` is included -- it pins
    the implementation, not just its declared parameters).

    Attributes:
        jaxvacua_git_sha: Short or full git SHA of the ``jaxvacua``
            build whose ``is_physical`` performed the certification.
        checks: The ordered tuple of check names that were *run* and
            *required to pass*.  Must be a subset of
            :data:`CANONICAL_CHECK_NAMES`, in canonical order.  The
            2026-06-02 bug is exactly the case where
            ``"kahler_metric_pd"`` was absent from this tuple.
        tolerances: Mapping of tolerance name -> value used by the
            checks (e.g. ``{"residual": 1e-6, "metric_tol": 1e-10,
            "hessian_tol": -1e-8, "mass_tol": -1e-8, "min_im_tau":
            1.0}``).  Values are normalised to ``repr`` strings before
            hashing so float formatting cannot perturb the id.
        stability_suite: Whether the extended Hessian / mass-matrix
            stability suite was applied (vs. ``is_physical`` alone).
        notes: Free-form human note; **not** part of the hash.
    """
    jaxvacua_git_sha: str
    checks: Tuple[str, ...]
    tolerances: Mapping[str, Any] = field(default_factory=dict)
    stability_suite: bool = True
    notes: str = ""

    def __post_init__(self) -> None:
        unknown = [c for c in self.checks if c not in CANONICAL_CHECK_NAMES]
        if unknown:
            raise ValueError(
                f"VerifierSpec.checks contains unknown check names {unknown}; "
                f"expected a subset of {CANONICAL_CHECK_NAMES}"
            )
        # Enforce canonical ordering so two specs listing the same
        # checks in different orders are rejected (rather than silently
        # producing two ids for the same procedure).
        canonical_order = [c for c in CANONICAL_CHECK_NAMES if c in self.checks]
        if list(self.checks) != canonical_order:
            raise ValueError(
                f"VerifierSpec.checks must be in canonical order "
                f"{canonical_order}; got {list(self.checks)}"
            )

    def hashable_payload(self) -> Dict[str, Any]:
        r"""
        **Description:**
        Return the canonical, JSON-serialisable dict that is hashed to
        produce :func:`verifier_id`.  Excludes ``notes`` (human-only).

        Returns:
            Dict[str, Any]: The payload, with tolerances normalised to
            ``repr`` strings for float-format stability.
        """
        return {
            "jaxvacua_git_sha": str(self.jaxvacua_git_sha),
            "checks": list(self.checks),
            "tolerances": {k: repr(self.tolerances[k]) for k in sorted(self.tolerances)},
            "stability_suite": bool(self.stability_suite),
        }

    @property
    def verifier_id(self) -> str:
        r"""
        **Description:**
        The content-addressed verifier identifier:
        ``"v1-" + sha1(canonical-json(hashable_payload))[:16]``.

        The ``v1-`` prefix is a *format* version for the hashing scheme
        itself (so the scheme can evolve without colliding), distinct
        from ``stringforge.vulcan.schema.SCHEMA_VERSION`` (the parquet
        layout version).

        Returns:
            str: The verifier id.
        """
        canonical = json.dumps(
            self.hashable_payload(), sort_keys=True, separators=(",", ":"),
        )
        digest = hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:16]
        return f"v1-{digest}"

    def includes_metric_pd_check(self) -> bool:
        r"""
        **Description:**
        Whether this verifier ran the Kähler-metric positive-definiteness
        check -- the one whose omission caused the 2026-06-02 incident.

        A community-facing certifier MUST return ``True`` here; the
        Phase-3 gate should refuse any verifier for which this is
        ``False``.

        Returns:
            bool: ``True`` iff ``"kahler_metric_pd"`` is in ``checks``.
        """
        return "kahler_metric_pd" in self.checks


def verifier_id(spec: VerifierSpec) -> str:
    r"""
    **Description:**
    Functional alias for :attr:`VerifierSpec.verifier_id`.

    Args:
        spec: The verifier description.

    Returns:
        str: The content-addressed verifier id.
    """
    return spec.verifier_id


class VerifierRegistry:
    r"""
    **Description:**
    In-memory registry mapping ``verifier_id -> VerifierSpec``.

    The registry is the committed artefact that makes the bus-factor
    risk (a single maintainer) survivable: it lets any later operator
    answer "what did ``verifier_id`` X actually check?" from a file,
    not from memory.  It round-trips losslessly through JSON so it can
    be committed alongside the dataset.

    Args:
        specs: Optional iterable of :class:`VerifierSpec` to register
            at construction.
    """

    def __init__(self, specs: Optional[Any] = None) -> None:
        self._by_id: Dict[str, VerifierSpec] = {}
        if specs:
            for spec in specs:
                self.register(spec)

    def register(self, spec: VerifierSpec) -> str:
        r"""
        **Description:**
        Register a spec and return its ``verifier_id``.  Idempotent:
        re-registering an identical spec is a no-op; registering a
        *different* spec under a colliding id raises (should never
        happen short of a hash collision).

        Args:
            spec: The verifier description.

        Returns:
            str: The registered ``verifier_id``.

        Raises:
            ValueError: On an id collision with a differing spec.
        """
        vid = spec.verifier_id
        existing = self._by_id.get(vid)
        if existing is not None and existing.hashable_payload() != spec.hashable_payload():
            raise ValueError(
                f"verifier_id collision for {vid!r}: two distinct specs hash "
                f"to the same id"
            )
        self._by_id[vid] = spec
        return vid

    def get(self, vid: str) -> Optional[VerifierSpec]:
        r"""
        **Description:**
        Look up a registered spec by id.

        Args:
            vid: The ``verifier_id`` to resolve.

        Returns:
            Optional[VerifierSpec]: The spec, or ``None`` if unknown.
        """
        return self._by_id.get(vid)

    def __contains__(self, vid: str) -> bool:
        return vid in self._by_id

    def __len__(self) -> int:
        return len(self._by_id)

    def to_dict(self) -> Dict[str, Any]:
        r"""
        **Description:**
        Serialise the registry to a JSON-able dict
        (``{verifier_id: {jaxvacua_git_sha, checks, tolerances,
        stability_suite, notes}}``).

        Returns:
            Dict[str, Any]: The serialised registry.
        """
        return {
            vid: {
                "jaxvacua_git_sha": spec.jaxvacua_git_sha,
                "checks": list(spec.checks),
                "tolerances": dict(spec.tolerances),
                "stability_suite": spec.stability_suite,
                "notes": spec.notes,
            }
            for vid, spec in sorted(self._by_id.items())
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "VerifierRegistry":
        r"""
        **Description:**
        Reconstruct a registry from :meth:`to_dict` output.  Verifies
        that each entry's key matches the recomputed ``verifier_id``
        (so a hand-edited registry cannot silently mislabel a spec).

        Args:
            data: A mapping as produced by :meth:`to_dict`.

        Returns:
            VerifierRegistry: The reconstructed registry.

        Raises:
            ValueError: If a stored key does not match the recomputed
                ``verifier_id`` of its spec.
        """
        reg = cls()
        for vid, entry in data.items():
            spec = VerifierSpec(
                jaxvacua_git_sha=entry["jaxvacua_git_sha"],
                checks=tuple(entry["checks"]),
                tolerances=dict(entry.get("tolerances", {})),
                stability_suite=bool(entry.get("stability_suite", True)),
                notes=entry.get("notes", ""),
            )
            if spec.verifier_id != vid:
                raise ValueError(
                    f"registry key {vid!r} does not match recomputed "
                    f"verifier_id {spec.verifier_id!r}; the registry has been "
                    f"corrupted or hand-edited inconsistently"
                )
            reg.register(spec)
        return reg


#: The full physicality cascade including the stability suite -- the
#: verifier a community-facing certifier (Phase 3) MUST use.  The
#: jaxvacua git SHA is a placeholder to be filled at certification time
#: from the live build; it is intentionally not pinned here so this
#: module stays solver-free.
FULL_CASCADE_CHECKS: Tuple[str, ...] = CANONICAL_CHECK_NAMES


__all__ = (
    "CANONICAL_CHECK_NAMES",
    "FULL_CASCADE_CHECKS",
    "VerifierSpec",
    "VerifierRegistry",
    "verifier_id",
)
