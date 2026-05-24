stringforge.vacuavault
=======================

.. currentmodule:: stringforge.vacuavault

.. automodule:: stringforge.vacuavault


Server-side tooling for the HuggingFace ``vacua_vault`` dataset
repository.  This subpackage holds the schema definitions,
validators, and CI helpers that govern community contributions to
the public vacua datasets.  It has **no downstream-package
dependencies** — the moduli-stabilisation, identity-hashing, and
auto-load logic live in :mod:`jaxvacua` (or any sibling package
that consumes the vault).

The corresponding **HuggingFace dataset URL** is
``aschachner/vacua_vault``.  The Python module name (with no
underscore) is intentionally distinct from the on-disk vault
folder name (``vacua_vault/``) to avoid PEP 420 namespace-package
shadowing.


Schema constants
-----------------------------------

Module-level constants and regexes that define the parquet layout
and filename conventions.

* :data:`SCHEMA_VERSION`
* :data:`RESERVED_NAMES`
* :data:`LABEL_SLUG_RE`


Validation
-----------------------------------

The user-facing validator runs schema, identity, and (optional)
physics checks against a single parquet file.  Pure dependency
injection: callers wanting physics validation pass ``db=`` and
``model_hash_fn=`` themselves; ``stringforge.vacuavault`` never
imports a downstream package.

* :func:`validate_parquet_file`
* :func:`split_by_validation`


Server-side CI helpers
-----------------------------------

Run via ``python -m stringforge.vacuavault {validate | rebuild_catalog | curate}``
on the HF dataset repo.  Schema-only by default; physics-aware
variants live in downstream-package wrappers (e.g. a
``jaxvacua-vault`` CLI in :mod:`jaxvacua` that injects an
``LCSDatabase``-backed model loader).

* :func:`validate_pr_diff`
* :func:`rebuild_catalog`
* :func:`curate_submission`


Typical usage
-----------------------------------

Schema-only validation from Python:

.. code-block:: python

    from stringforge import vacuavault as vv

    result = vv.validate_parquet_file(
        "tdf/h12_2/ks_29_tri_0/SUSY_Nmax34.parquet",
        physics_checks="off",
    )
    assert result["passed"], result["errors"]

Physics-aware validation (caller supplies the database):

.. code-block:: python

    from stringforge import vacuavault as vv
    from stringforge.lcs_database import LCSDatabase
    from stringforge.vacua_writer import _compute_model_hash

    db = LCSDatabase(dataset="tdf")
    result = vv.validate_parquet_file(
        path,
        db=db,
        model_hash_fn=_compute_model_hash,
        physics_checks="auto",
    )

CLI (server-side, run from the HF dataset repo root):

.. code-block:: bash

    python -m stringforge.vacuavault validate --base-branch main
    python -m stringforge.vacuavault rebuild_catalog --repo-path .
    python -m stringforge.vacuavault curate community/alice_dS_v2.parquet


See also
-----------------------------------

* :doc:`stringforge.cy_io` — the geometry-database I/O layer that
  ``vacuavault`` reads schema constants from.
* The dataset card at ``vacua_vault/vacua_vault_dataset_card.md``
  for the public-facing schema and contribution workflow.
