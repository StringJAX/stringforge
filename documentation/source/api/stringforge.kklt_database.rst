stringforge.kklt_database
=========================

.. currentmodule:: stringforge.kklt_database

.. automodule:: stringforge.kklt_database


Database class
-----------------------------------

Extends :class:`stringforge.lcs_database.LCSDatabase` with the
``kklt_vacua`` sub-dataset — a curated subset of TDF indexed by
conifold class.  Every row carries a *logical* TDF link
``(ks_id, triang_id, tdf_conifold_id)``; physical shard coordinates
are resolved on demand from the wrapped :attr:`tdf` database.

.. autosummary::
    :toctree: ../_autosummary
    :template: custom-class-template.rst

    KKLTDatabase


Querying the catalogs
-----------------------------------

The KKLT database exposes three catalogs (polytope-grain,
conifold-class, individual-conifold) plus an append-only run log.

* :meth:`KKLTDatabase.query`
* :meth:`KKLTDatabase.query_polytopes`
* :meth:`KKLTDatabase.query_classes`
* :meth:`KKLTDatabase.query_conifolds`
* :meth:`KKLTDatabase.list_runs`
* :meth:`KKLTDatabase.info`


Loading models
-----------------------------------

Loaders take the KKLT key ``(ks_id, coni_class_id, coni_id)`` and
delegate to the wrapped :attr:`KKLTDatabase.tdf` after resolving the
logical TDF link.

* :meth:`KKLTDatabase.load`
* :meth:`KKLTDatabase.load_model`
* :meth:`KKLTDatabase.load_from_conifold_row`


Run-tracking (cluster provenance)
-----------------------------------

Append-only run log keyed by scope (``"class"`` or ``"conifold"``) +
polytope / class / conifold IDs.  Concurrent local writers are
serialised by an advisory ``flock``.

* :meth:`KKLTDatabase.start_run`
* :meth:`KKLTDatabase.finish_run`
* :meth:`KKLTDatabase.cancel_run`
* :meth:`KKLTDatabase.run_status`
* :meth:`KKLTDatabase.push_run_log`
* :meth:`KKLTDatabase.fetch_run_log`


TDF-link maintenance
-----------------------------------

* :meth:`KKLTDatabase.rebuild_links`


Module-level constants and helpers
-----------------------------------

* :data:`DEFAULT_KKLT_HF_REPO`
* :func:`_resolve_kklt_hf_repo`
* :data:`_RUN_LOG_COLUMNS`
