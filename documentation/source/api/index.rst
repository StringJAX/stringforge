API reference
=============

The public API of `stringjax` is organised into four modules. All four are
auto-documented from their docstrings; click through for the full signatures
and per-method descriptions.

.. autosummary::
   :nosignatures:

   stringjax.cy_io
   stringjax.lcs_database
   stringjax.vacua_writer
   stringjax.vacuavault

Module index
------------

.. toctree::
   :maxdepth: 2

   stringjax.cy_io
   stringjax.lcs_database
   stringjax.vacua_writer
   stringjax.vacuavault

Quick reference
---------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Task
     - Entry point
   * - Filter the cy-database catalog (catalog convention).
     - :class:`stringjax.cy_io.CYDatabase` — :meth:`~stringjax.cy_io.CYDatabase.query`,
       :meth:`~stringjax.cy_io.CYDatabase.query_conifolds`, :meth:`~stringjax.cy_io.CYDatabase.info`.
   * - Build a model in mirror convention (load → ``lcs_tree``, ``FluxVacuaFinder``).
     - :class:`stringjax.lcs_database.LCSDatabase` — :meth:`~stringjax.lcs_database.LCSDatabase.load`,
       :meth:`~stringjax.lcs_database.LCSDatabase.load_model`, :meth:`~stringjax.lcs_database.LCSDatabase.load_batch`,
       :meth:`~stringjax.lcs_database.LCSDatabase.iter_batch`, :meth:`~stringjax.lcs_database.LCSDatabase.sample`.
   * - Write vacuum solutions to the local vault, designate, push to HuggingFace.
     - :class:`stringjax.vacua_writer.VacuaWriter`.
   * - Validate vault parquets, rebuild the catalog, curate community submissions.
     - :mod:`stringjax.vacuavault` — :func:`~stringjax.vacuavault.validate_parquet_file`,
       :func:`~stringjax.vacuavault.rebuild_catalog`, :func:`~stringjax.vacuavault.curate_submission`.

For the conventions that govern the boundary between catalog and mirror
representations, and the inter-module data-flow contract, see
:doc:`../ecosystem/architecture`. For the recent module rename and convention
swap, see :doc:`../ecosystem/migration_from_jaxvacua`.
