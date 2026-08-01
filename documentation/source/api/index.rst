API reference
=============

The public API of ``stringforge`` is organised around infrastructure workflows:
query data, load models, persist vacua, and validate vault submissions.  Physics
calculations live in sibling packages such as JAXVacua.

.. toctree::
   :maxdepth: 1
   :hidden:

   stringforge.cy_io
   stringforge.toric_db
   stringforge.cy_phase
   stringforge.lcs_database
   stringforge.kklt_database
   stringforge.vacua_writer
   stringforge.vacuavault
   stringforge.vulcan

Workflow entry points
---------------------

.. list-table::
   :header-rows: 1
   :widths: 36 64

   * - Goal
     - Start here
   * - Query TDF/CICY catalogues without constructing physics models.
     - :class:`stringforge.cy_io.CYDatabase`, :class:`stringforge.cy_io.TDFDatabase`, :class:`stringforge.cy_io.CICYDatabase`.
   * - Query Kreuzer–Skarke toric CY phases (FRST and VEX classes) at scale.
     - :class:`stringforge.toric_db.ToricCYDatabase` — sharded per :math:`h^{1,1}`, so point lookups stay :math:`O(1)`.
   * - Work with one Calabi–Yau phase as an object (stored geometry now, CYTools on demand).
     - :class:`stringforge.cy_phase.CYPhase`, :class:`stringforge.cy_phase.ToricCYPhase`.
   * - Load catalogue rows as JAXVacua-compatible data or models.
     - :class:`stringforge.lcs_database.LCSDatabase` — :meth:`~stringforge.lcs_database.LCSDatabase.load`, :meth:`~stringforge.lcs_database.LCSDatabase.load_model`, :meth:`~stringforge.lcs_database.LCSDatabase.load_batch`, :meth:`~stringforge.lcs_database.LCSDatabase.iter_batch`, :meth:`~stringforge.lcs_database.LCSDatabase.sample`.
   * - Work with the advanced KKLT index.
     - :class:`stringforge.kklt_database.KKLTDatabase` — conifold-class queries, logical TDF links, model loading, and curation tags.
   * - Store, designate, retract, or fetch vacuum solutions.
     - :class:`stringforge.vacua_writer.VacuaWriter`.
   * - Validate vault parquet files and rebuild vault catalogues.
     - :mod:`stringforge.vacuavault`.
   * - Forge cluster-side production vacuum runs into a HuggingFace repo.
     - :class:`stringforge.vulcan.Vulcan`, :class:`stringforge.vulcan.VulcanReader`, :class:`stringforge.vulcan.VulcanMLView`.

Module pages
------------

* :doc:`stringforge.cy_io`
* :doc:`stringforge.toric_db`
* :doc:`stringforge.cy_phase`
* :doc:`stringforge.lcs_database`
* :doc:`stringforge.kklt_database`
* :doc:`stringforge.vacua_writer`
* :doc:`stringforge.vacuavault`
* :doc:`stringforge.vulcan`
