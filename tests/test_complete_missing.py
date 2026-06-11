"""Tests for derived-quantity autocomplete on vacua tables.

``_compute_derived`` (shared by the session-flush path, ``complete_missing``
and ``designate_vacua``) fills the superpotential ``W``, F-terms, tadpole
``N_flux``, string coupling ``g_s`` and -- optionally -- the mass spectrum
(``mass2``) and gravitino mass (``m_gravitino``).  A small fake model exercises
the plumbing without a jaxvacua dependency.
"""

import numpy as np
import pandas as pd
import pytest

from stringforge.vacua_writer import _compute_derived, VacuaWriter
from stringforge.lcs_database import LCSDatabase


class _FakeModel:
    """Minimal stand-in exposing the methods ``_compute_derived`` calls."""

    def tadpole(self, flux):
        return float(np.sum(np.abs(np.asarray(flux))))

    def superpotential(self, moduli, tau, flux):
        return 2.0 + 3.0j

    def DW(self, moduli, moduli_c, tau, tau_c, flux):
        # tiny F-terms → classified SUSY
        return np.array([1e-9 + 0j, 2e-9 + 0j])

    def kahler_potential(self, moduli, moduli_c, tau, tau_c):
        return -1.0 + 0j

    def mass_matrix(self, moduli, moduli_c, tau, tau_c, flux,
                    mode=None, noscale=True):
        return np.diag([1.0, 4.0, 9.0, 16.0]).astype(complex)


def _df():
    return pd.DataFrame([{
        "flux":      [1, -2, 3, 0, 1, -1],
        "moduli_re": [1.0, 2.0],
        "moduli_im": [1.5, 0.5],
        "tau_re":    0.1,
        "tau_im":    5.0,
    }])


def test_basic_derived_quantities():
    out = _compute_derived(_df(), _FakeModel(), with_masses=True)
    row = out.iloc[0]
    assert row["N_flux"] == pytest.approx(8.0)            # sum |flux|
    assert row["W_re"] == pytest.approx(2.0)
    assert row["W_im"] == pytest.approx(3.0)
    assert row["g_s"] == pytest.approx(1.0 / 5.0)          # 1 / Im(tau)
    assert row["is_susy"] is True or bool(row["is_susy"])  # |DW| < 1e-6


def test_mass_spectrum_and_gravitino():
    out = _compute_derived(_df(), _FakeModel(), with_masses=True)
    row = out.iloc[0]
    assert list(row["mass2"]) == [1.0, 4.0, 9.0, 16.0]     # sorted eigenvalues
    # m_3/2 = e^{K/2} |W|, K = -1, |W| = sqrt(13)
    assert row["m_gravitino"] == pytest.approx(np.exp(-0.5) * np.sqrt(13.0))


def test_masses_off_by_default():
    out = _compute_derived(_df(), _FakeModel())           # with_masses=False
    assert "mass2" not in out.columns
    assert "m_gravitino" not in out.columns


def test_fill_missing_only_preserves_existing():
    df = _df()
    df["W_re"] = 99.0          # pretend an existing (different) value
    df["W_im"] = -99.0
    out = _compute_derived(df, _FakeModel(), fill_missing_only=True)
    # existing W untouched, but g_s (was absent) is filled
    assert out.iloc[0]["W_re"] == pytest.approx(99.0)
    assert out.iloc[0]["g_s"] == pytest.approx(0.2)


def test_idempotent():
    fm = _FakeModel()
    out1 = _compute_derived(_df(), fm, fill_missing_only=True, with_masses=True)
    out2 = _compute_derived(out1, fm, fill_missing_only=True, with_masses=True)
    assert out2.iloc[0]["g_s"] == pytest.approx(out1.iloc[0]["g_s"])
    assert out2.iloc[0]["m_gravitino"] == pytest.approx(out1.iloc[0]["m_gravitino"])
    assert list(out2.iloc[0]["mass2"]) == list(out1.iloc[0]["mass2"])


def test_complete_missing_requires_model():
    vw = object.__new__(VacuaWriter)          # bypass __init__ (no db needed)
    with pytest.raises(ValueError):
        vw.complete_missing(_df(), model=None)


def test_complete_missing_exposed_on_both_classes():
    assert hasattr(VacuaWriter, "complete_missing")
    assert hasattr(LCSDatabase, "complete_missing")
