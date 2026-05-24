# Copyright 2022-2026 Andreas Schachner
#
# This file is part of StringForge.
#
# StringForge is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# StringForge is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with StringForge. If not, see <https://www.gnu.org/licenses/>.

"""
StringForge: Differentiable tools for string compactifications with JAX.

An umbrella framework providing a unified computational pipeline from
Calabi-Yau compactification data to four-dimensional effective field
theories, vacuum solutions, and physical observables.

Public submodules:
    jaxvacua   - Type IIB flux vacua (complex-structure + axio-dilaton sector)
    jaxpolylog - JAX-compatible polylogarithm functions

Optional (private, install separately):
    kahlerjax  - Kahler moduli stabilisation
    jaxiverse  - Multi-axion EFT from string compactifications

────────────────────────────────────────────────────────────────────────
Common workflows — which class do I want?
────────────────────────────────────────────────────────────────────────

* **Query the on-disk catalog** (no model construction, no jaxvacua) —
  :class:`stringforge.cy_io.CYDatabase` (or its dataset-specific
  subclasses :class:`stringforge.cy_io.TDFDatabase` /
  :class:`stringforge.cy_io.CICYDatabase`)::

      from stringforge import TDFDatabase
      db   = TDFDatabase()                  # downloads catalog on first use
      cats = db.query(h11=3, has_conifolds=True)

  All catalog columns and queries on this layer use the **catalog
  convention** (small ``h11``, large ``h12``).

* **Load a TDF / CICY model as a jaxvacua object** —
  :class:`stringforge.lcs_database.LCSDatabase`::

      from stringforge import LCSDatabase
      db    = LCSDatabase(dataset="tdf")
      model = db.load_model(ks_id=12345, triang_id=0, include_gv=True)

  The ``LCSDatabase`` layer presents h11 / h12 in **mirror convention**
  (jaxvacua / lcs_tree convention) — see the docstring on
  :meth:`LCSDatabase.load` for the boundary-swap detail.

* **Work with the curated KKLT-vacua subset** (conifold-class indexed,
  cluster-run tracking, separate GV split) —
  :class:`stringforge.kklt_database.KKLTDatabase`::

      from stringforge import KKLTDatabase
      db    = KKLTDatabase.from_local(".", tdf_cache_dir="./cy-database/")
      model = db.load_model(h11=11, ks_id=384564,
                            coni_class_id=0, coni_id=75,
                            include_gv=True, gv_source="kklt")

  KKLT methods use **mirror convention** for h11/h12 (large ``h11``,
  small ``h12``) — the same convention as :class:`LCSDatabase`.  The
  on-disk KKLT catalogs use the *catalog* convention; the swap happens
  at the API boundary.  ``ks_id`` is only unique within a fixed
  ``(h11, h12)``, so ``h11`` is required; pass ``h12=`` too only when
  the resolver tells you the lookup is ambiguous.

* **Save vacuum solutions to a permanent vault** (designated solutions,
  HF Hub push/fetch) —
  :class:`stringforge.vacua_writer.VacuaWriter`::

      from stringforge import LCSDatabase
      db = LCSDatabase(dataset="tdf")
      with db.vacua_writer(model=model) as writer:
          for result in newton_search_results:
              writer.append(result, tags=["benchmark"])
      # Query the run-tagged rows back, then designate them.
      df = db.query_vacua(run_id=writer.run_id)
      db.designate_vacua(df, label="my_run", committed_by="A. Schachner")
      db.push_vacua_to_hub()                # publish to HF Hub

* **Validate vault parquet files / rebuild the catalog** —
  :mod:`stringforge.vacuavault`
  (``python -m stringforge.vacuavault validate ...``).

* **Inspect raw KKLT parquet rows without jaxvacua** —
  :meth:`KKLTDatabase.load_dataframes` returns ``pandas.Series`` /
  ``pandas.DataFrame`` objects for tabular workflows.

────────────────────────────────────────────────────────────────────────
Caching, offline mode, and the data dir
────────────────────────────────────────────────────────────────────────

Catalogs and parquet shards are downloaded **lazily on first use** and
cached under ``stringforge.data_dir`` (default ``./.stringforge_cache``;
override with the ``STRINGFORGE_DATA_DIR`` env var or
:func:`set_data_dir`).  Vacuum solutions live in a separate
``vacua_vault/`` directory (override with ``STRINGFORGE_VAULT`` /
:func:`set_vault_dir`) so :meth:`CYDatabase.clear_cache` never wipes
them.

Pass ``offline=True`` to any database constructor (or use
:meth:`CYDatabase.from_local`) to disable network access — every
``_fetch_shard`` call then raises if its file isn't already cached.
"""

__version__ = "0.1.0"

# ── Data directory ────────────────────────────────────────────────────────
# Default location for all database cache and vacua storage.  Override
# with ``STRINGFORGE_DATA_DIR`` env var or :func:`set_data_dir`.  The
# ``cy_io`` layer reads this global at import time via
# ``from . import data_dir`` so it must be defined before cy_io's
# ``DEFAULT_CACHE_DIR`` evaluates.
import os as _os
_DEFAULT_DATA_DIR = _os.path.join(_os.getcwd(), ".stringforge_cache")
data_dir = _os.environ.get("STRINGFORGE_DATA_DIR", _DEFAULT_DATA_DIR)


def set_data_dir(path):
    r"""
    **Description:**
    Set the global data directory for all stringforge database operations
    (HuggingFace cache, vacua storage, designated solutions).

    New :class:`~stringforge.cy_io.CYDatabase` (and subclass) instances
    created after this call will use the specified directory unless
    overridden by an explicit ``cache_dir`` argument.

    Args:
        path (str | Path): Absolute or relative path to the data
            directory.  The directory is created on first use.

    Returns:
        None
    """
    global data_dir
    data_dir = str(path)
# ──────────────────────────────────────────────────────────────────────────


# ── Vacua vault directory + HF repo ──────────────────────────────────────
# Permanent storage for designated vacuum solutions.  Resolution order in
# :func:`stringforge.cy_io._resolve_vault_dir`:
#   1. ``STRINGFORGE_VAULT`` env var (explicit override), or value set via
#      :func:`set_vault_dir`.
#   2. ``<repo_root>/vacua_vault/`` when cwd is inside a stringforge
#      source checkout.
#   3. ``<cwd>/vacua_vault/`` otherwise.
# The vault lives outside the cache dir so ``clear_cache()`` never wipes
# designated solutions.

def set_vault_dir(path):
    r"""
    **Description:**
    Set the vault directory for designated vacuum solutions by
    exporting ``STRINGFORGE_VAULT`` into the environment.  Takes effect
    for all subsequent database calls.

    Args:
        path (str | Path | None): Absolute or relative path to the
            vault directory.  Pass ``None`` to clear the override and
            fall back to repo-root / cwd auto-detection.

    Returns:
        None
    """
    if path is None:
        _os.environ.pop("STRINGFORGE_VAULT", None)
    else:
        _os.environ["STRINGFORGE_VAULT"] = str(path)


def set_vault_repo(repo_id):
    r"""
    **Description:**
    Set the HuggingFace dataset repo ID used for uploading / fetching
    community vacuum solutions.  Sets the ``STRINGFORGE_VAULT_REPO`` env
    var.

    Args:
        repo_id (str | None): ``"user/repo"`` on HuggingFace Hub, or
            ``None`` to clear the override and fall back to the
            package default ``aschachner/vacua_vault``.

    Returns:
        None
    """
    if repo_id is None:
        _os.environ.pop("STRINGFORGE_VAULT_REPO", None)
    else:
        _os.environ["STRINGFORGE_VAULT_REPO"] = str(repo_id)
# ──────────────────────────────────────────────────────────────────────────

from .cy_io import *
from .lcs_database import *
from .kklt_database import *
from .vacua_writer import *
from .vacuavault import *

# Optional ecosystem packages are intentionally not imported here.  Keeping the
# package import lightweight avoids solver-library side effects and lets users
# import ``stringforge`` for pure database/vault workflows.
