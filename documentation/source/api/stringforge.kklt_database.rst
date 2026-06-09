stringforge.kklt_database
=========================

.. currentmodule:: stringforge.kklt_database

.. automodule:: stringforge.kklt_database
   :no-members:

.. warning::

   The KKLT interface is an advanced curated-subset workflow.  It is public API,
   but most users should begin with ``TDFDatabase`` / ``CICYDatabase`` and the
   general ``LCSDatabase`` model-loading bridge.

Overview
--------

``KKLTDatabase`` exposes the ``kklt`` sub-dataset: a curated TDF search index
inside the general ``cy-database`` repository.  It carries logical TDF links,
optional KKLT-specific GV pointers, and scalar curation tags.  Actual
designated KKLT vacua belong in the shared ``vacua_vault/kklt_vacua`` records.

.. autosummary::
   :toctree: ../_autosummary
   :template: custom-class-template.rst

   KKLTDatabase

Curated method index
--------------------

Catalogue and conifold-class queries:

* :meth:`KKLTDatabase.query_polytopes`
* :meth:`KKLTDatabase.query_classes`
* :meth:`KKLTDatabase.query_conifolds`
* :meth:`KKLTDatabase.load_dataframes`

Model loading:

* :meth:`KKLTDatabase.load`
* :meth:`KKLTDatabase.load_model`
* :meth:`KKLTDatabase.from_local`

Run provenance:

* :meth:`KKLTDatabase.start_run`
* :meth:`KKLTDatabase.finish_run`
* :meth:`KKLTDatabase.fail_run`
* :meth:`KKLTDatabase.run_status`

Constants
---------

.. autodata:: DEFAULT_KKLT_HF_REPO
