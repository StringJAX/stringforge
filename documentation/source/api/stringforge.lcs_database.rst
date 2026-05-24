stringforge.lcs_database
========================

.. currentmodule:: stringforge.lcs_database

.. automodule:: stringforge.lcs_database


Database class
-----------------------------------

Extends :class:`stringforge.cy_io.CYDatabase` with model construction
(``lcs_tree`` / ``FluxVacuaFinder``) and vacua-writing delegation.  This is
the canonical user-facing database class.

.. autosummary::
    :toctree: ../_autosummary
    :template: custom-class-template.rst

    LCSDatabase


Loading models
-----------------------------------

* :meth:`LCSDatabase.load`
* :meth:`LCSDatabase.load_from_conifold_row`
* :meth:`LCSDatabase.load_model`
* :meth:`LCSDatabase.load_batch`
* :meth:`LCSDatabase.iter_batch`
* :meth:`LCSDatabase.load_model_batch`
* :meth:`LCSDatabase.iter_model_batch`
* :meth:`LCSDatabase.sample`


Cached model batches
-----------------------------------

* :meth:`LCSDatabase.cached_models`
* :meth:`LCSDatabase.clear_cached_models`


Module-level convenience functions
-----------------------------------

* :func:`load_tdf_model`
* :func:`load_cicy_model`
