"""Adapter between ``jaxvacua.vacuum`` objects and the stringforge writer schema.

A ``jaxvacua.vacuum.Vacuum`` (or ``PFV`` / ``AFV``) stores the *real* coordinate
vector ``x`` (not complex ``moduli``/``tau``) and carries no geometry identity.
This module maps such an object to the writer's ``{"moduli", "tau", "flux", ...}``
row dict (the *queryable projection*) and stashes the full, authoritative
``Vacuum.to_dict()`` in the free ``extra_data`` column (the *record*), so read-back
is exact and finder-free.

Design constraints (kept deliberately):
* ``jaxvacua`` imports are **lazy / in-function** — importing this module never
  imports ``jaxvacua`` (mirrors the rest of stringforge).
* The record is stored as **JSON** (a recursive numpy/complex codec), never
  ``pickle`` — the vault is public/community, so an unpickle-on-load would be a
  remote-code-execution vector.  The codec itself lives in
  :mod:`jaxvacua.vacuum` (``encode_json`` / ``decode_json``); this module only
  wraps it.
"""
from __future__ import annotations

import json
from typing import Any, Dict, Optional, Tuple

import numpy as np

# The JSON tags (``__nd__`` / ``__c__``) are defined with the codec in
# ``jaxvacua.vacuum``; nothing here needs them directly any more.


def is_vacuum(obj: Any) -> bool:
    r"""
    **Description:**
    ``True`` iff *obj* is a ``jaxvacua.vacuum.Vacuum`` (or subclass).

    Existing writer inputs (``dict`` / ``tuple`` / ``list``) short-circuit to
    ``False`` *before* any ``jaxvacua`` import, so a plain ``append(dict)`` never
    drags in the physics stack.

    Args:
        obj: Any candidate object.

    Returns:
        bool: Whether *obj* is a ``Vacuum`` (or ``PFV`` / ``AFV``) instance.
    """
    if isinstance(obj, (dict, tuple, list)):
        return False
    try:
        from jaxvacua.vacuum import Vacuum
    except Exception:
        return False
    return isinstance(obj, Vacuum)


# --------------------------------------------------------------------------- #
# JSON codec for a ``Vacuum.to_dict()`` payload
# --------------------------------------------------------------------------- #
# The tagged-JSON codec (``{"__nd__": ...}`` / ``{"__c__": ...}``) now lives in
# ``jaxvacua.vacuum``, so this repository and jaxvacua share one implementation
# rather than each carrying its own.  These wrappers keep the local names and the
# lazy-import discipline: both are only ever reached from code paths that already
# require ``jaxvacua`` (encoding needs a ``Vacuum``; decoding is immediately
# followed by ``Vacuum.from_dict``).
def _json_encode(obj: Any) -> Any:
    r"""
    **Description:**
    Encode a ``to_dict()`` payload to JSON-safe data.

    Delegates to :func:`jaxvacua.vacuum.encode_json`.

    Args:
        obj: The (possibly nested) value to encode.

    Returns:
        A JSON-serialisable representation of *obj*.
    """
    from jaxvacua.vacuum import encode_json
    return encode_json(obj)


def _json_decode(obj: Any) -> Any:
    r"""
    **Description:**
    Inverse of :func:`_json_encode`.

    Delegates to :func:`jaxvacua.vacuum.decode_json`.

    Args:
        obj: A value produced by :func:`_json_encode`.

    Returns:
        The decoded value, with numpy arrays and ``complex`` scalars restored.
    """
    from jaxvacua.vacuum import decode_json
    return decode_json(obj)


# --------------------------------------------------------------------------- #
# Vacuum  <->  writer row
# --------------------------------------------------------------------------- #
def _decode_moduli_tau(x: Any) -> Tuple[np.ndarray, complex]:
    r"""
    **Description:**
    Decode the real coordinate vector ``x`` into complex moduli ``z`` and the
    axio-dilaton ``tau``.

    Delegates to :func:`jaxvacua.vacuum.real_to_complex`, the model-free statement
    of the ``(Re z1, Im z1, ..., Re tau, Im tau)`` layout -- previously written out
    independently here, in ``jaxvacua.vacuum`` and in
    ``FluxEFT._convert_real_to_complex``.  No finder is needed, so the queryable
    ``moduli`` / ``tau`` columns stay finder-free.

    Args:
        x (Array): Full real coordinate vector.

    Returns:
        tuple: ``(z, tau)`` -- the complex moduli vector and the axio-dilaton.
    """
    from jaxvacua.vacuum import real_to_complex
    return real_to_complex(x)


def vacuum_to_row(
    vacuum: Any,
    finder: Any = None,
    *,
    F_term_tol: float = 1e-8,
    store_trajectory: bool = False,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    r"""
    **Description:**
    Map a ``Vacuum`` to the ``(result, extra_data)`` pair the writer consumes.

    ``result`` is the ``{"moduli", "tau", "flux", "residual", "is_susy",
    "N_flux", "W", "F_terms"}`` dict fed to
    :func:`stringforge.vacua_writer._vacua_row_from_dict` (the queryable typed
    columns).  ``extra_data`` carries the authoritative JSON-encoded
    ``Vacuum.to_dict()`` under ``"vacuum"``, plus a readable ``"kind"`` and
    ``"model_name"``.

    .. admonition:: Details
        :class: dropdown

        ``is_susy`` is the *physics* flag ``residual < F_term_tol`` (F-flatness),
        **not** the solver's ``success``.  ``N_flux`` is the geometry-independent
        D3 charge and requires a finder; without one it is ``None`` (a nullable
        column, fillable later via ``VacuaWriter.complete_missing``).

    Args:
        vacuum: A ``jaxvacua.vacuum.Vacuum`` (or ``PFV`` / ``AFV``).
        finder: Optional finder, used only for ``N_flux = finder.tadpole(flux)``.
        F_term_tol (float): F-flatness threshold for the ``is_susy`` flag.
        store_trajectory (bool): Keep the verbose optimisation trajectory in the
            stored record.

    Returns:
        tuple: ``(result, extra_data)`` — the typed row dict and the extras dict
        carrying the JSON record.
    """
    z, tau = _decode_moduli_tau(vacuum.x)
    flux = np.asarray(vacuum.flux)

    n_flux: Optional[int] = None
    if finder is not None and hasattr(finder, "tadpole"):
        try:
            n_flux = int(round(float(finder.tadpole(flux))))
        except Exception:
            n_flux = None

    res = float(vacuum.residual)
    result: Dict[str, Any] = {
        "moduli": z,
        "tau": tau,
        "flux": flux,
        "residual": res,
        "is_susy": bool(np.isfinite(res) and res < F_term_tol),
        "N_flux": n_flux,
        "W": (complex(vacuum.W0) if vacuum.W0 is not None else None),
        "F_terms": (np.asarray(vacuum.DW) if vacuum.DW is not None else None),
    }

    d = vacuum.to_dict()
    if not store_trajectory:
        d.pop("trajectory", None)
    try:
        import jaxvacua as _jvc
        d["_jaxvacua_version"] = getattr(_jvc, "__version__", None)
    except Exception:
        pass

    extra: Dict[str, Any] = {
        "vacuum": _json_encode(d),
        "kind": d.get("_kind"),
        "model_name": (getattr(vacuum, "metadata", None) or {}).get("model_name"),
    }
    return result, extra


def row_to_vacuum(extra_data: Any, *, finder: Any = None) -> Optional[Any]:
    r"""
    **Description:**
    Rebuild a ``Vacuum`` from a stored row's ``extra_data`` — the inverse of
    :func:`vacuum_to_row`.

    Returns the exact original ``Vacuum`` / ``PFV`` / ``AFV`` (finder-free) when
    the ``"vacuum"`` record is present.  Rows written via the array path carry no
    record and return ``None`` (the caller may rebuild a bare ``Vacuum`` from the
    typed columns if a *finder* is available).

    Args:
        extra_data: The parsed ``extra_data`` dict or the raw JSON string.
        finder: Optional finder (reserved for rebuilding record-less rows).

    Returns:
        Vacuum | None: The reconstructed vacuum, or ``None`` when no record is
        present or *extra_data* is unparseable.
    """
    if extra_data is None:
        return None
    if isinstance(extra_data, str):
        try:
            extra_data = json.loads(extra_data)
        except (ValueError, TypeError):
            return None
    if not isinstance(extra_data, dict):
        return None
    blob = extra_data.get("vacuum")
    if blob is None:
        return None
    payload = _json_decode(blob)
    from jaxvacua.vacuum import Vacuum
    return Vacuum.from_dict(payload)
