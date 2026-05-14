# Copyright 2022-2026 Andreas Schachner
#
# This file is part of StringJAX.
#
# StringJAX is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# StringJAX is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with StringJAX. If not, see <https://www.gnu.org/licenses/>.

"""
StringJAX: Differentiable tools for string compactifications with JAX.

An umbrella framework providing a unified computational pipeline from
Calabi-Yau compactification data to four-dimensional effective field
theories, vacuum solutions, and physical observables.

Public submodules:
    jaxvacua   - Type IIB flux vacua (complex-structure + axio-dilaton sector)
    jaxpolylog - JAX-compatible polylogarithm functions

Optional (private, install separately):
    kahlerjax  - Kahler moduli stabilisation
    jaxiverse  - Multi-axion EFT from string compactifications
"""

__version__ = '0.0.2'

# ── Data directory ────────────────────────────────────────────────────────
# Default location for all database cache and vacua storage.  Override
# with ``STRINGJAX_DATA_DIR`` env var or :func:`set_data_dir`.  The
# ``cy_io`` layer reads this global at import time via
# ``from . import data_dir`` so it must be defined before cy_io's
# ``DEFAULT_CACHE_DIR`` evaluates.
import os as _os
_DEFAULT_DATA_DIR = _os.path.join(_os.getcwd(), ".stringjax_cache")
data_dir = _os.environ.get("STRINGJAX_DATA_DIR", _DEFAULT_DATA_DIR)


def set_data_dir(path):
    r"""
    **Description:**
    Set the global data directory for all stringjax database operations
    (HuggingFace cache, vacua storage, designated solutions).

    New :class:`~stringjax.cy_io.CYDatabase` (and subclass) instances
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
# :func:`stringjax.cy_io._resolve_vault_dir`:
#   1. ``STRINGJAX_VAULT`` env var (explicit override), or value set via
#      :func:`set_vault_dir`.
#   2. ``<repo_root>/vacua_vault/`` when cwd is inside a stringjax
#      source checkout.
#   3. ``<cwd>/vacua_vault/`` otherwise.
# The vault lives outside the cache dir so ``clear_cache()`` never wipes
# designated solutions.

def set_vault_dir(path):
    r"""
    **Description:**
    Set the vault directory for designated vacuum solutions by
    exporting ``STRINGJAX_VAULT`` into the environment.  Takes effect
    for all subsequent database calls.

    Args:
        path (str | Path | None): Absolute or relative path to the
            vault directory.  Pass ``None`` to clear the override and
            fall back to repo-root / cwd auto-detection.

    Returns:
        None
    """
    if path is None:
        _os.environ.pop("STRINGJAX_VAULT", None)
    else:
        _os.environ["STRINGJAX_VAULT"] = str(path)


def set_vault_repo(repo_id):
    r"""
    **Description:**
    Set the HuggingFace dataset repo ID used for uploading / fetching
    community vacuum solutions.  Sets the ``STRINGJAX_VAULT_REPO`` env
    var.

    Args:
        repo_id (str | None): ``"user/repo"`` on HuggingFace Hub, or
            ``None`` to clear the override and fall back to the
            package default ``aschachner/vacua_vault``.

    Returns:
        None
    """
    if repo_id is None:
        _os.environ.pop("STRINGJAX_VAULT_REPO", None)
    else:
        _os.environ["STRINGJAX_VAULT_REPO"] = str(repo_id)
# ──────────────────────────────────────────────────────────────────────────

from .cy_io import *
from .lcs_database import *
from .vacua_writer import *
from .vacuavault import *

# Re-export public submodules.
try:
    import jaxpolylog
except ImportError:
    pass

try:
    import jaxvacua
except ImportError:
    pass

# Optional private submodules — available only with `pip install .[full]`.
try:
    import kahlerjax
except ImportError:
    pass

try:
    import jaxiverse
except ImportError:
    pass