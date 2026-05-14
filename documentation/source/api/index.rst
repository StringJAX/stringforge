API reference
=============

The public API of `stringforge` is organised into four modules. All four are
auto-documented from their docstrings; click through for the full signatures
and per-method descriptions.

.. autosummary::
   :nosignatures:

   stringforge.cy_io
   stringforge.lcs_database
   stringforge.vacua_writer
   stringforge.vacuavault

Module index
------------

.. toctree::
   :maxdepth: 2

   stringforge.cy_io
   stringforge.lcs_database
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
   * - Write vacuum solutions to the local vault, designate, push to HuggingFace.
     - :class:`stringforge.vacua_writer.VacuaWriter`.
   * - Validate vault parquets, rebuild the catalog, curate community submissions.
     - :mod:`stringforge.vacuavault` — :func:`~stringforge.vacuavault.validate_parquet_file`,
       :func:`~stringforge.vacuavault.rebuild_catalog`, :func:`~stringforge.vacuavault.curate_submission`.

For the conventions that govern the boundary between catalog and mirror
representations, and the inter-module data-flow contract, see
:doc:`../ecosystem/architecture`. For the recent module rename and convention
swap, see :doc:`../ecosystem/migration_from_jaxvacua`.
