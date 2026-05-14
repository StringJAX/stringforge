stringjax.lcs_database
========================

.. currentmodule:: stringjax.lcs_database

.. automodule:: stringjax.lcs_database


Database class
-----------------------------------

Extends :class:`stringjax.cy_io.CYDatabase` with model construction
(``lcs_tree`` / ``FluxVacuaFinder``) and vacua-writing delegation.  This is
the canonical user-facing database class.

.. autosummary::
    :toctree: _autosummary
    :template: custom-class-template.rst

    LCSDatabase


Loading models
-----------------------------------

.. autosummary::
    :toctree: _autosummary

    LCSDatabase.load
    LCSDatabase.load_from_conifold_row
    LCSDatabase.load_model
    LCSDatabase.load_batch
    LCSDatabase.iter_batch
    LCSDatabase.load_model_batch
    LCSDatabase.iter_model_batch
    LCSDatabase.sample


Cached model batches
-----------------------------------

.. autosummary::
    :toctree: _autosummary

    LCSDatabase.cached_models
    LCSDatabase.clear_cached_models


Module-level convenience functions
-----------------------------------

.. autosummary::
    :toctree: _autosummary

    load_tdf_model
    load_cicy_model
