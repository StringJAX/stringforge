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
StringForge: shared infrastructure for string-compactification workflows.

StringForge provides reproducible access to Calabi-Yau database rows, cache and
offline workflows, model-loading bridges into physics engines such as JAXVacua,
and persistent vacua-vault infrastructure with provenance.  It is deliberately
solver-light: flux-vacuum physics lives in JAXVacua, differentiable
polylogarithms live in JAXPolyLog, and planned packages such as KahlerJAX and
JAXiverse will consume the same conventions when they are public.

Common workflows
----------------

* Query TDF/CICY catalogues without constructing physics models::

      from stringforge import TDFDatabase
      db = TDFDatabase()
      rows = db.query(h11=3, has_conifolds=True)

  This pure-I/O layer uses the catalogue convention.

* Load catalogue rows as JAXVacua-compatible objects::

      from stringforge import LCSDatabase
      db = LCSDatabase(dataset="tdf")
      model = db.load_model(ks_id=12345, triang_id=0, h11=128, h12=2)

  The LCS layer presents Hodge numbers in the mirror convention used by
  ``jaxvacua.lcs.lcs_tree``.

* Work with the advanced curated KKLT index::

      from stringforge import KKLTDatabase
      db = KKLTDatabase()
      classes = db.query_classes(ks_id=384564)

  The ``kklt`` dataset stores conifold-class indices, logical TDF links, and
  curation tags.  Actual KKLT vacuum records belong in the shared vault.  This
  is an expert workflow, not the default entry point.

* Store or curate vacuum solutions with provenance::

      from stringforge import LCSDatabase, set_vault_dir
      set_vault_dir("./vacua_vault")
      db = LCSDatabase(dataset="tdf")
      # db.designate_vacua(vacua_df, model=model, label="scan")

Caching and offline mode
------------------------

Catalogues and parquet shards are downloaded lazily and cached under
``stringforge.data_dir`` (default ``./.stringforge_cache``; override via
``STRINGFORGE_DATA_DIR`` or :func:`set_data_dir`).  Permanent designated vacua
live in a separate ``vacua_vault/`` directory (override via
``STRINGFORGE_VAULT`` or :func:`set_vault_dir`) so cache cleanup does not delete
curated results.  Pass ``offline=True`` to a database constructor, or use
``from_local(...)``, to disable network access.
"""

__version__ = "0.1.1"

# ── Data directory ────────────────────────────────────────────────────────
# Default location for database cache files.  Override
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
#   3. no fallback; an explicit path is required outside a source checkout.
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
            fall back to repo-root detection, or raise if no source checkout is found.

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


# ── Vulcan production-pipeline configuration ─────────────────────────────
# Vulcan owns the *production* / cluster-side / ML-friendly vacuum repo,
# complementary to the curated ``vacua_vault`` repo above.  Workers stage
# parquet shards in a local staging directory and a separate sync tier
# batches them into HuggingFace commits, respecting HF's 100-commits-
# per-hour cap.  See ``stringforge.vulcan`` for the full API.

def set_vulcan_repo(repo_id):
    r"""
    **Description:**
    Set the HuggingFace dataset repo ID used by Vulcan for production
    vacuum uploads.  Sets the ``STRINGFORGE_VULCAN_REPO`` env var.

    Args:
        repo_id (str | None): ``"user/repo"`` on HuggingFace Hub, or
            ``None`` to clear the override.  Vulcan has no implicit
            default repo — each ``Vulcan(repo=...)`` instance must
            be told where to write.

    Returns:
        None
    """
    if repo_id is None:
        _os.environ.pop("STRINGFORGE_VULCAN_REPO", None)
    else:
        _os.environ["STRINGFORGE_VULCAN_REPO"] = str(repo_id)


def set_vulcan_staging_dir(path):
    r"""
    **Description:**
    Set the local staging directory used by Vulcan workers for pending
    shards awaiting sync.  Sets the ``STRINGFORGE_VULCAN_STAGING_DIR``
    env var.

    Args:
        path (str | Path | None): Absolute or relative path.  Pass
            ``None`` to clear the override and fall back to
            ``<repo_root>/vulcan_staging/`` or
            ``<data_dir>/vulcan_staging/``.

    Returns:
        None
    """
    if path is None:
        _os.environ.pop("STRINGFORGE_VULCAN_STAGING_DIR", None)
    else:
        _os.environ["STRINGFORGE_VULCAN_STAGING_DIR"] = str(path)


def set_vulcan_project(project):
    r"""
    **Description:**
    Set the default ``{project}`` substitution for the Vulcan
    run-id template.  Sets the ``STRINGFORGE_VULCAN_PROJECT`` env var.

    Args:
        project (str | None): Project tag.  Pass ``None`` to clear.

    Returns:
        None
    """
    if project is None:
        _os.environ.pop("STRINGFORGE_VULCAN_PROJECT", None)
    else:
        _os.environ["STRINGFORGE_VULCAN_PROJECT"] = str(project)


def set_vulcan_token(token):
    r"""
    **Description:**
    Set the HuggingFace access token used by Vulcan's sync tier.  Sets
    the ``STRINGFORGE_VULCAN_TOKEN`` env var.

    Args:
        token (str | None): HuggingFace access token.  Pass ``None`` to
            clear the override.

    Returns:
        None
    """
    if token is None:
        _os.environ.pop("STRINGFORGE_VULCAN_TOKEN", None)
    else:
        _os.environ["STRINGFORGE_VULCAN_TOKEN"] = str(token)


def set_vulcan_budget(budget):
    r"""
    **Description:**
    Set the commits-per-hour ceiling honoured by Vulcan's sync tier.
    Sets the ``STRINGFORGE_VULCAN_BUDGET`` env var.  The hard
    HuggingFace cap is 100/hour; defaults to 90 to leave a safety
    margin.

    Args:
        budget (int | None): Commits/hour.  Pass ``None`` to clear.

    Returns:
        None

    Raises:
        ValueError: If ``budget`` is negative.
    """
    if budget is None:
        _os.environ.pop("STRINGFORGE_VULCAN_BUDGET", None)
    else:
        budget_int = int(budget)
        if budget_int < 0:
            raise ValueError(
                f"Vulcan budget must be non-negative, got {budget_int}."
            )
        from .vulcan.ratelimit import HF_COMMITS_PER_HOUR_CAP
        if budget_int > HF_COMMITS_PER_HOUR_CAP:
            import warnings as _warnings
            _warnings.warn(
                f"Vulcan budget {budget_int} exceeds the HuggingFace "
                f"commits-per-hour cap of {HF_COMMITS_PER_HOUR_CAP}; "
                "uploads above the cap will be throttled or rejected.",
                stacklevel=2,
            )
        _os.environ["STRINGFORGE_VULCAN_BUDGET"] = str(budget_int)
# ──────────────────────────────────────────────────────────────────────────

from .cy_io import *
from .toric_db import *      # ToricCYDatabase (sharded toric sub-dataset)
from .cy_phase import *      # CYPhase / ToricCYPhase (per-geometry objects)
from .lcs_database import *
from .kklt_database import *
from .vacua_writer import *
from .vacuavault import *
from .vulcan import (
    FeatureSpec,
    StagedShard,
    SyncReport,
    Vulcan,
    VulcanMLView,
    VulcanReader,
    resolve_staging_dir,
)

# Optional ecosystem packages are intentionally not imported here.  Keeping the
# package import lightweight avoids solver-library side effects and lets users
# import ``stringforge`` for pure database/vault workflows.
