API reference
=============

The public API of `stringforge` is organised into five modules. All five are
auto-documented from their docstrings; click through for the full signatures
and per-method descriptions.

.. autosummary::
   :nosignatures:

   stringforge.cy_io
   stringforge.lcs_database
   stringforge.kklt_database
   stringforge.vacua_writer
   stringforge.vacuavault

Module index
------------

.. toctree::
   :maxdepth: 2

   stringforge.cy_io
   stringforge.lcs_database
   stringforge.kklt_database
   stringforge.vacua_writer
   stringforge.vacuavault

Quick reference
---------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Task
     - Entry point
   * - Filter the cy-database catalog (catalog convention).
     - :class:`stringforge.cy_io.CYDatabase` — :meth:`~stringforge.cy_io.CYDatabase.query`,
       :meth:`~stringforge.cy_io.CYDatabase.query_conifolds`, :meth:`~stringforge.cy_io.CYDatabase.info`.
   * - Build a model in mirror convention (load → ``lcs_tree``, ``FluxVacuaFinder``).
     - :class:`stringforge.lcs_database.LCSDatabase` — :meth:`~stringforge.lcs_database.LCSDatabase.load`,
       :meth:`~stringforge.lcs_database.LCSDatabase.load_model`, :meth:`~stringforge.lcs_database.LCSDatabase.load_batch`,
       :meth:`~stringforge.lcs_database.LCSDatabase.iter_batch`, :meth:`~stringforge.lcs_database.LCSDatabase.sample`.
   * - Work with the curated KKLT-vacua subset (indexed by conifold class) and track cluster runs.
     - :class:`stringforge.kklt_database.KKLTDatabase` — :meth:`~stringforge.kklt_database.KKLTDatabase.query_polytopes`,
       :meth:`~stringforge.kklt_database.KKLTDatabase.query_classes`,
       :meth:`~stringforge.kklt_database.KKLTDatabase.query_conifolds`,
       :meth:`~stringforge.kklt_database.KKLTDatabase.load_model`,
       :meth:`~stringforge.kklt_database.KKLTDatabase.start_run`,
       :meth:`~stringforge.kklt_database.KKLTDatabase.finish_run`,
       :meth:`~stringforge.kklt_database.KKLTDatabase.rebuild_links`.
   * - Write vacuum solutions to the local vault, designate, push to HuggingFace.
     - :class:`stringforge.vacua_writer.VacuaWriter`.
   * - Validate vault parquets, rebuild the catalog, curate community submissions.
     - :mod:`stringforge.vacuavault` — :func:`~stringforge.vacuavault.validate_parquet_file`,
       :func:`~stringforge.vacuavault.rebuild_catalog`, :func:`~stringforge.vacuavault.curate_submission`.

For the conventions that govern the boundary between catalog and mirror
representations, and the inter-module data-flow contract, see
:doc:`../ecosystem/architecture`.
