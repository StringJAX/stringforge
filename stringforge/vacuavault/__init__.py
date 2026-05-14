r"""
stringforge.vacuavault — tooling for the HuggingFace ``vacua_vault``
dataset repository.

This subpackage provides three CLI-driven operations for maintaining the
remote vault repo:

- **validate** (on PR branch): parse the git diff, re-validate added
  parquet files, post a catalog preview as a PR comment.  Implemented
  in :mod:`.ci`.
- **rebuild_catalog** (post-merge on main): scan the repo for all
  parquet files, regenerate ``catalog.parquet`` from their metadata.
  Implemented in :mod:`.catalog`.
- **curate** (maintainer operation): promote a community submission
  out of ``community/`` up to the model-level directory, stripping the
  ``{hf_username}_`` prefix and preserving attribution.  Implemented in
  :mod:`.curate`.

All three are dispatched by :mod:`.__main__` via argparse subcommands.

Module layout::

    stringforge/vacuavault/
    ├── __init__.py     — this file (re-exports public entry points)
    ├── __main__.py     — CLI dispatcher
    ├── ci.py           — validate_pr_diff()
    ├── catalog.py      — rebuild_catalog()
    ├── curate.py       — curate_submission()
    └── schema.py       — SCHEMA constants + _validate_parquet()
"""
from .ci      import validate_pr_diff
from .catalog import rebuild_catalog
from .curate  import curate_submission
from .schema  import (
    SCHEMA_VERSION,
    RESERVED_NAMES,
    LABEL_SLUG_RE,
    validate_parquet_file,
    split_by_validation,
    _extract_identity_from_row,
    _identity_from_row,
    _check_identity_consistency,
)

__all__ = [
    "validate_pr_diff",
    "rebuild_catalog",
    "curate_submission",
    "SCHEMA_VERSION",
    "RESERVED_NAMES",
    "LABEL_SLUG_RE",
    "validate_parquet_file",
    "split_by_validation",
    # Identity helpers (introduced with the auto-load path).  Underscored
    # because they're stable-but-low-level: the public API is
    # validate_parquet_file's behaviour, not the helpers themselves.
    "_extract_identity_from_row",
    "_identity_from_row",
    "_check_identity_consistency",
]
