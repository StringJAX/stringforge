StringForge
===========

**Shared database, model-loading, and vacua-vault infrastructure for
string-compactification workflows.**

StringForge provides the cross-package layer used to query Calabi-Yau
geometry databases, construct JAXVacua models from database rows, persist
vacuum solutions, and coordinate shared HuggingFace datasets.  The physics
engines live in sibling packages; this site documents the infrastructure
interfaces that connect them.

StringForge is intentionally solver-light.  It owns catalogue access, cache
management, model handoff, vault layout, validation, and provenance.  It does
not replace JAXVacua, and it is not a public release of planned packages such
as KahlerJAX or JAXiverse.

How to navigate
---------------

.. grid:: 1 1 2 2
   :gutter: 2

   .. grid-item-card:: New to StringForge
      :link: getting_started
      :link-type: doc

      Start with installation, the package boundary, and the shortest
      database-to-JAXVacua model-loading example.

   .. grid-item-card:: Running examples
      :link: tutorials
      :link-type: doc

      Use the tutorial catalogue for quickstart, database access, vault,
      cluster, and ecosystem workflow notebooks.

   .. grid-item-card:: Database and vault concepts
      :link: intro/database
      :link-type: doc

      Read about HuggingFace layouts, catalogue versus mirror conventions,
      lazy caching, offline mode, and the shared vacua vault.

   .. grid-item-card:: Cluster production runs
      :link: tutorials/database_and_infrastructure/vulcan_cluster_runs
      :link-type: doc

      Use Vulcan to stage vacuum batches on workers and batch-commit them
      to a shared HuggingFace dataset repo within the HF commit-rate cap.

   .. grid-item-card:: API lookup
      :link: api/index
      :link-type: doc

      Go to the curated API reference when you already know which database,
      vault, or model-loading class you need.

Recommended first path
----------------------

For a first pass through the documentation, read:

1. :doc:`Getting started <getting_started>` for installation and package
   orientation.
2. :doc:`Calabi-Yau Geometry Database <intro/database>` for data layout,
   cache behaviour, and convention boundaries.
3. :doc:`Tutorials <tutorials>` for executable notebooks.
4. :doc:`API reference <api/index>` once you need precise signatures.

Users working specifically with the curated KKLT index should read
:doc:`KKLT Database <intro/kklt_database>` after the general
TDF/CICY database material.

Reference lookup
----------------

* :ref:`genindex`
* :ref:`modindex`

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Start here

   getting_started
   tutorials
   api/index

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Background

   intro/database
   intro/kklt_database

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Ecosystem

   ecosystem/overview
   ecosystem/architecture

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Package overviews

   packages/jaxvacua
   packages/jaxpolylog
   packages/kahlerjax
   packages/jaxiverse
   packages/cytools
