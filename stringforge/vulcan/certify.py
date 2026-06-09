# Copyright 2026 Andreas Schachner
#
# This file is part of StringForge.
#
# StringForge is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

r"""
Certification: stamp the certified-vacuum record contract onto a shard.

This is the heart of Phase 1 of the rolling-DB plan: take a batch of
candidate vacua, *re-certify every row* with the physics cascade, stamp
the certification provenance (``verifier_id`` / ``cert_status`` /
``cert_checks`` / the stability min-eigenvalues / ``record_id``), and
return only the rows that pass.

StringForge is deliberately solver-light (it "is not the flux-vacuum
solver"), so this module defines only the certification *contract*,
local endpoint-stability diagnostics, and record-stamping logic -- all
fully testable with a mock certifier and no JAX.  The real physics
certifier (``jaxvacua.is_physical`` plus the local Vulcan stability
check) is provided by :func:`build_jaxvacua_certifier`, which lazily
imports its JAXVacua dependency so importing this module never pulls in
a solver.

The design directly answers the 2026-06-02 incident: a certifier is
described by a :class:`stringforge.vulcan.verifier.VerifierSpec`, and
every stamped row records the metric-PD eigenvalue, so a later
verifier-bug fix can both *select* affected rows (by ``verifier_id``)
and *detect* a recurrence in aggregate (by ``kahler_metric_min_eig``).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
import inspect
from typing import Any, Callable, Dict, Mapping, Optional, Protocol

import numpy as np
import pandas as pd

from .schema import (
    CERT_COLUMNS,
    GEOMETRY_KEYS,
    GEOMETRY_KEY_SENTINEL,
    canonical_flux as _canonical_flux,
    geometry_id as _geometry_id,
    record_id as _record_id,
    validate_cert_status,
)
from .verifier import VerifierSpec


@dataclass
class CertRow:
    r"""
    **Description:**
    Per-row output of a :class:`Certifier`.

    Attributes:
        passed: Whether the row passed *all* required checks (i.e. is a
            genuine, stable physical vacuum under this verifier).
        checks: Per-check pass/fail booleans.  Keys should be drawn
            from :data:`stringforge.vulcan.verifier.CANONICAL_CHECK_NAMES`.
        kahler_metric_min_eig: Minimum eigenvalue of the Kähler metric
            (the quantity whose *absence* caused the 2026-06-02 bug).
            ``nan`` if the certifier could not compute it.
        hessian_min_eig: Minimum eigenvalue of the real Hessian.
        mass_min_eig: Minimum eigenvalue of the canonically-normalised
            mass matrix.
    """
    passed: bool
    checks: Dict[str, bool] = field(default_factory=dict)
    kahler_metric_min_eig: float = math.nan
    hessian_min_eig: float = math.nan
    mass_min_eig: float = math.nan


@dataclass
class StabilityRecord:
    r"""
    **Description:**
    Diagnostic stability result for one candidate vacuum.

    The record is intentionally small and dependency-free: callers
    provide an EFT-like object with ``kahler_metric``, ``hessian`` and
    ``mass_matrix`` methods, and Vulcan records the minimum eigenvalue
    of each matrix together with boolean threshold checks.
    """
    kahler_metric_positive: bool
    hessian_minimum: bool
    mass_minimum: bool
    stable_minimum: bool
    kahler_metric_min_eig: float
    real_hessian_min_eig: float
    mass_matrix_min_eig: float


def _call_eft_method(
    eft: Any,
    names: tuple[str, ...],
    *,
    z: Any,
    tau: Any,
    flux: Any,
) -> Any:
    r"""Call the first available EFT method using supported keywords."""
    for name in names:
        fn = getattr(eft, name, None)
        if fn is None:
            continue
        kwargs = {"z": z, "tau": tau, "flux": flux}
        try:
            sig = inspect.signature(fn)
        except (TypeError, ValueError):
            sig = None
        if sig is not None:
            params = sig.parameters
            if not any(p.kind is p.VAR_KEYWORD for p in params.values()):
                kwargs = {k: v for k, v in kwargs.items() if k in params}
        if kwargs:
            try:
                return fn(**kwargs)
            except TypeError:
                pass
        for args in ((z, tau, flux), (z, tau)):
            try:
                return fn(*args)
            except TypeError:
                continue
    raise AttributeError(
        "EFT object must provide one of "
        f"{', '.join(names)} with z/tau/flux-compatible arguments."
    )


def _min_eig(matrix: Any, *, hermitian: bool) -> float:
    r"""Return the minimum real eigenvalue of a matrix-like object."""
    arr = np.asarray(matrix)
    if arr.size == 0:
        return math.nan
    if arr.ndim == 0:
        return float(np.real(arr))
    if arr.ndim == 1:
        return float(np.min(np.real(arr)))
    if arr.shape[-1] != arr.shape[-2]:
        raise ValueError(f"Expected a square matrix; got shape {arr.shape}.")
    if hermitian:
        mat = 0.5 * (arr + np.swapaxes(np.conjugate(arr), -1, -2))
    else:
        mat = 0.5 * (np.real(arr) + np.swapaxes(np.real(arr), -1, -2))
    vals = np.linalg.eigvalsh(mat)
    return float(np.min(np.real(vals)))


def check_endpoint_stability(
    eft: Any,
    *,
    z: Any,
    tau: Any,
    flux: Any,
    source_index: int = -1,
    W0_abs: float = math.nan,
    residual: float = math.nan,
    metric_tol: float = 1e-10,
    hessian_tol: float = -1e-8,
    mass_tol: float = -1e-8,
    min_im_tau: float = 1.0,
) -> StabilityRecord:
    r"""
    **Description:**
    Compute Vulcan's local endpoint-stability diagnostics.

    The EFT object must expose matrix methods for the Kähler metric,
    real Hessian and canonically-normalised mass matrix.  Method names
    are accepted in both concise and explicit forms:
    ``kahler_metric`` / ``get_kahler_metric``, ``hessian`` /
    ``real_hessian`` / ``get_hessian``, and ``mass_matrix`` /
    ``get_mass_matrix``.  Methods may accept keyword arguments
    ``z``, ``tau`` and ``flux``; unsupported keywords are omitted.

    Args:
        eft: EFT-like object with matrix-evaluation methods.
        z: Complex-structure moduli vector.
        tau: Axio-dilaton.
        flux: Integer flux vector.
        source_index: Optional source-row index, retained for API
            compatibility with cluster diagnostics.
        W0_abs: Optional ``|W0|`` diagnostic.
        residual: Optional solver residual.
        metric_tol, hessian_tol, mass_tol: Minimum-eigenvalue
            thresholds.
        min_im_tau: Required lower bound on ``Im(tau)``.

    Returns:
        StabilityRecord: Boolean checks and diagnostic eigenvalues.
    """
    del source_index, W0_abs, residual

    metric = _call_eft_method(
        eft, ("kahler_metric", "get_kahler_metric"),
        z=z, tau=tau, flux=flux,
    )
    hessian = _call_eft_method(
        eft, ("hessian", "real_hessian", "get_hessian"),
        z=z, tau=tau, flux=flux,
    )
    mass = _call_eft_method(
        eft, ("mass_matrix", "get_mass_matrix"),
        z=z, tau=tau, flux=flux,
    )

    metric_min = _min_eig(metric, hermitian=True)
    hessian_min = _min_eig(hessian, hermitian=False)
    mass_min = _min_eig(mass, hermitian=False)

    tau_ok = float(np.imag(tau)) >= float(min_im_tau)
    metric_ok = bool(metric_min > metric_tol)
    hessian_ok = bool(hessian_min >= hessian_tol)
    mass_ok = bool(mass_min >= mass_tol)

    return StabilityRecord(
        kahler_metric_positive=metric_ok,
        hessian_minimum=hessian_ok,
        mass_minimum=mass_ok,
        stable_minimum=bool(tau_ok and metric_ok and hessian_ok and mass_ok),
        kahler_metric_min_eig=metric_min,
        real_hessian_min_eig=hessian_min,
        mass_matrix_min_eig=mass_min,
    )


class Certifier(Protocol):
    r"""
    **Description:**
    Callable contract for a per-row physics certifier.

    A certifier receives one candidate vacuum as a mapping (the row's
    columns -- at minimum ``flux``, ``moduli_re``, ``moduli_im``,
    ``tau_re``, ``tau_im``) and returns a :class:`CertRow`.  The
    real implementation wraps ``jaxvacua.is_physical`` + the stability
    suite (see :func:`build_jaxvacua_certifier`); tests use a trivial
    in-process stub.
    """

    def __call__(self, row: Mapping[str, Any]) -> CertRow:  # pragma: no cover - protocol
        ...


@dataclass
class CertifyResult:
    r"""
    **Description:**
    Outcome of :func:`certify_shard`.

    Attributes:
        certified: DataFrame of the rows that passed, stamped with the
            full certified-record contract.  Ready to hand to
            :meth:`stringforge.vulcan.Vulcan.write`.
        rejected: DataFrame of the rows that failed (stamped with their
            ``cert_checks`` / min-eigs for diagnosis), empty when
            ``keep_rejected=False``.
        n_input: Number of input rows.
        n_certified: Number of rows that passed.
        n_rejected: Number of rows that failed.
        verifier_id: The certifying verifier's id.
    """
    certified: pd.DataFrame
    rejected: pd.DataFrame
    n_input: int
    n_certified: int
    n_rejected: int
    verifier_id: str


def _stamp_row_cols(
    df: pd.DataFrame,
    *,
    geometry: Mapping[str, Any],
    verifier_id: str,
    cert_rows: list[CertRow],
    status: str,
) -> pd.DataFrame:
    r"""Attach the certified-record columns to a copy of ``df``.

    The certifier is authoritative on *identity* as well as
    certification: ``geometry_id`` and the geometry-key columns are
    (re)written from the trusted ``geometry`` argument, so a forged
    ``geometry_id`` / ``ks_id`` / ... in the input can never survive
    certification (defence for the community gate).
    """
    out = df.copy()
    # Canonicalise the stored flux to integer quanta so the persisted
    # flux matches the rounded flux used for record_id / dedup (the flux
    # quanta are integers by definition; a near-integer submitted value
    # must not produce an integer key but a non-integer stored vector).
    if "flux" in out.columns and len(out):
        out["flux"] = [list(_canonical_flux(f)) for f in out["flux"]]
    # Authoritative identity from the trusted geometry (overwrite any
    # client-supplied values).
    out["geometry_id"] = _geometry_id(geometry)
    for key in GEOMETRY_KEYS:
        out[key] = int(geometry.get(key, GEOMETRY_KEY_SENTINEL))
    out["verifier_id"] = verifier_id
    out["cert_status"] = status
    out["cert_checks"] = [json.dumps(cr.checks, sort_keys=True) for cr in cert_rows]
    out["kahler_metric_min_eig"] = [float(cr.kahler_metric_min_eig) for cr in cert_rows]
    out["hessian_min_eig"] = [float(cr.hessian_min_eig) for cr in cert_rows]
    out["mass_min_eig"] = [float(cr.mass_min_eig) for cr in cert_rows]
    out["record_id"] = [
        _record_id(geometry, flux, verifier_id) for flux in out["flux"]
    ]
    return out


def certify_shard(
    df: pd.DataFrame,
    *,
    certifier: Certifier,
    verifier_spec: VerifierSpec,
    geometry: Mapping[str, Any],
    keep_rejected: bool = False,
) -> CertifyResult:
    r"""
    **Description:**
    Re-certify every row of a candidate shard and stamp the
    certified-vacuum record contract.

    Each row is passed to ``certifier``; rows that pass are stamped
    ``cert_status="certified"`` and returned in ``result.certified``.
    Rows that fail are **excluded from the certified output** (the
    certified DB only ever contains certified rows); they are returned
    in ``result.rejected`` only when ``keep_rejected=True`` (for
    diagnosis), never with a ``certified`` status.

    Args:
        df: Candidate shard.  Must carry at least ``flux`` plus the
            moduli/tau columns the certifier needs.
        certifier: The per-row physics certifier (see
            :func:`build_jaxvacua_certifier` for the real one; tests
            inject a stub).
        verifier_spec: Description of the certifier; its
            :attr:`~stringforge.vulcan.verifier.VerifierSpec.verifier_id`
            is stamped onto every row and used in the ``record_id``.
        geometry: The model's geometry index dict (mirror convention),
            used to compute ``geometry_id`` / ``record_id``.  Must be
            the same convention enforced by the writer at ingest.
        keep_rejected: Also return the failed rows (stamped for
            diagnosis) in ``result.rejected``.

    Returns:
        CertifyResult: The certified rows plus counts.

    Raises:
        ValueError: If ``df`` lacks a ``flux`` column, or if
            ``verifier_spec`` is not a :class:`VerifierSpec`.
    """
    if not isinstance(verifier_spec, VerifierSpec):
        raise ValueError("verifier_spec must be a VerifierSpec instance")
    if "flux" not in df.columns:
        raise ValueError("certify_shard: input df must carry a 'flux' column")

    vid = verifier_spec.verifier_id
    cert_rows: list[CertRow] = [certifier(row) for _, row in df.iterrows()]
    passed_mask = [cr.passed for cr in cert_rows]

    pass_df = df.loc[passed_mask].reset_index(drop=True)
    pass_cert = [cr for cr, ok in zip(cert_rows, passed_mask) if ok]
    certified = (
        _stamp_row_cols(pass_df, geometry=geometry, verifier_id=vid,
                        cert_rows=pass_cert, status="certified")
        if len(pass_df)
        else _empty_certified_frame(df)
    )
    # Defensive: the status we wrote must be valid.
    if len(certified):
        validate_cert_status("certified")

    rejected = pd.DataFrame()
    if keep_rejected:
        fail_mask = [not ok for ok in passed_mask]
        fail_df = df.loc[fail_mask].reset_index(drop=True)
        fail_cert = [cr for cr, ok in zip(cert_rows, passed_mask) if not ok]
        if len(fail_df):
            # Rejected rows are stamped for diagnosis but NOT given a
            # 'certified' status -- they must never enter the certified
            # DB.  We omit cert_status entirely (None) so a careless
            # write cannot mistake them for accepted records.
            rejected = _stamp_row_cols(
                fail_df, geometry=geometry, verifier_id=vid,
                cert_rows=fail_cert, status=None,  # type: ignore[arg-type]
            )

    n_pass = int(sum(passed_mask))
    return CertifyResult(
        certified=certified,
        rejected=rejected,
        n_input=int(len(df)),
        n_certified=n_pass,
        n_rejected=int(len(df)) - n_pass,
        verifier_id=vid,
    )


def _empty_certified_frame(df: pd.DataFrame) -> pd.DataFrame:
    r"""An empty frame with the input columns plus the cert columns."""
    out = df.iloc[0:0].copy()
    for col in CERT_COLUMNS:
        out[col] = pd.Series(dtype="object")
    return out


# ── real physics adapter (lazily imported; not unit-tested here) ──────────


def build_jaxvacua_certifier(
    model: Any,
    sampler: Any,
    *,
    eft: Any = None,
    moduli_max: Optional[float] = None,
    metric_tol: float = 1e-10,
    hessian_tol: float = -1e-8,
    mass_tol: float = -1e-8,
    min_im_tau: float = 1.0,
    stability_fn: Optional[Callable[..., Any]] = None,
    is_physical_fn: Optional[Callable[..., Any]] = None,
) -> Certifier:
    r"""
    **Description:**
    Build a real physics :class:`Certifier` wrapping
    ``jaxvacua.flux_utils.is_physical`` (the 5-check cascade incl. the
    Kähler-metric positive-definiteness check) plus the extended
    local stability check (``check_endpoint_stability``: Kähler-metric /
    Hessian / mass-matrix minimum eigenvalues).

    Dependencies are imported lazily so that importing
    :mod:`stringforge.vulcan.certify` never requires JAX / jaxvacua.
    For testing, ``is_physical_fn`` and ``stability_fn`` may be injected
    to bypass the lazy import entirely.

    The returned certifier maps each candidate row to a
    :class:`CertRow` whose ``checks`` carries at least
    ``"kahler_metric_pd"``, ``"hessian_min_eig"``, ``"mass_min_eig"``
    (from the stability suite) and ``"is_physical"`` (the overall
    cascade verdict), and whose numeric min-eig fields come from the
    stability record.  ``passed`` is the conjunction of the overall
    ``is_physical`` verdict and the stability suite's
    ``stable_minimum``.

    Args:
        model: A jaxvacua model accepted by ``is_physical``.
        sampler: The jaxvacua sampler providing the dilaton floor etc.
        eft: The EFT object passed to ``check_endpoint_stability`` (it
            needs ``.kahler_metric`` / ``.hessian`` / ``.mass_matrix``).
            Defaults to ``model`` when not given; pass explicitly if the
            model and its EFT are distinct objects in your build.
        moduli_max: Optional runaway bound forwarded to ``is_physical``.
        metric_tol, hessian_tol, mass_tol, min_im_tau: Tolerances
            forwarded to ``check_endpoint_stability``.
        stability_fn: Override for ``check_endpoint_stability`` (tests).
        is_physical_fn: Override for ``jaxvacua.flux_utils.is_physical``
            (tests).

    Returns:
        Certifier: A per-row certifier closure.

    Raises:
        ImportError: If the physics dependencies are needed but absent.
    """
    if is_physical_fn is None:  # pragma: no cover - exercised only with jaxvacua
        try:
            from jaxvacua.flux_utils import is_physical as is_physical_fn  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "build_jaxvacua_certifier requires `jaxvacua` for the "
                "physicality cascade. Install jaxvacua, or inject "
                "is_physical_fn / stability_fn for testing."
            ) from exc
    if stability_fn is None:
        stability_fn = check_endpoint_stability

    def _certify(row: Mapping[str, Any]) -> CertRow:  # pragma: no cover - needs solver
        # The exact packing of (moduli, tau) into the vector x follows
        # jaxvacua conventions; the stability call is local to Vulcan.
        z = np.asarray(row["moduli_re"], dtype=float) + 1j * np.asarray(row["moduli_im"], dtype=float)
        tau = float(row["tau_re"]) + 1j * float(row["tau_im"])
        flux = np.asarray(row["flux"])
        # jaxvacua packs the real moduli vector INTERLEAVED: it reads
        # moduli = x[0:-2:2] + 1j*x[1:-2:2] (flux_eft.py:_convert_real_to_complex),
        # with tau at x[-2], x[-1].  A block layout
        # ([Re(z)..., Im(z)..., Re(tau), Im(tau)]) would scramble the
        # evaluation point for h12>=2 -- is_physical (and its metric-PD
        # check) would then certify the WRONG moduli point.  Pack
        # interleaved to match.
        x = np.empty(2 * z.shape[0] + 2, dtype=float)
        x[0:-2:2] = np.real(z)
        x[1:-2:2] = np.imag(z)
        x[-2] = np.real(tau)
        x[-1] = np.imag(tau)

        phys = bool(is_physical_fn(model, sampler, x, moduli_max=moduli_max))
        stab = stability_fn(
            eft if eft is not None else model,
            z=z, tau=tau, flux=flux, source_index=int(row.get("source_index", -1)),
            W0_abs=float(row.get("W0_abs", float("nan"))),
            residual=float(row.get("residual", float("nan"))),
            metric_tol=metric_tol, hessian_tol=hessian_tol, mass_tol=mass_tol,
            min_im_tau=min_im_tau,
        )
        checks = {
            "is_physical": phys,
            "kahler_metric_pd": bool(getattr(stab, "kahler_metric_positive", False)),
            "hessian_min_eig": bool(getattr(stab, "hessian_minimum", False)),
            "mass_min_eig": bool(getattr(stab, "mass_minimum", False)),
        }
        passed = phys and bool(getattr(stab, "stable_minimum", False))
        # StabilityRecord field names: kahler_metric_min_eig,
        # real_hessian_min_eig, mass_matrix_min_eig.  Reading the wrong
        # names would silently store nan, so these are pinned exactly.
        return CertRow(
            passed=passed,
            checks=checks,
            kahler_metric_min_eig=float(getattr(stab, "kahler_metric_min_eig", float("nan"))),
            hessian_min_eig=float(getattr(stab, "real_hessian_min_eig", float("nan"))),
            mass_min_eig=float(getattr(stab, "mass_matrix_min_eig", float("nan"))),
        )

    return _certify


__all__ = (
    "CertRow",
    "StabilityRecord",
    "Certifier",
    "CertifyResult",
    "check_endpoint_stability",
    "certify_shard",
    "build_jaxvacua_certifier",
)
