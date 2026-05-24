StringForge
===========

**Shared database, model-loading, and vacua-vault infrastructure for the
StringForge ecosystem.**

StringForge provides the cross-package layer used to query Calabi-Yau
geometry databases, construct JAXVacua models from database rows, persist
vacuum solutions, and coordinate shared HuggingFace datasets.  The physics
engines live in the sibling packages; this site documents the umbrella
interfaces that connect them.

The documentation is organised around how users usually arrive here: first
install and orient yourself, then choose the database or vault workflow you
need, then use the API reference for exact entry points.

How to navigate
---------------

.. grid:: 1 1 2 2
   :gutter: 2

   .. grid-item-card:: New to StringForge
      :link: getting_started
      :link-type: doc

      Start with installation, precision requirements, and the shortest
      examples for the database and model-loading interfaces.

   .. grid-item-card:: Running examples
      :link: tutorials
      :link-type: doc

      Use the tutorial catalogue.  It groups notebooks by quickstart,
      database access, vacua storage, cluster workflows, and ecosystem
      examples.

   .. grid-item-card:: Database and vault concepts
      :link: intro/database
      :link-type: doc

      Read the database introduction for cache behaviour, HuggingFace
      layout, catalog conventions, offline mode, and vault-related
      environment variables.

   .. grid-item-card:: Looking for a function
      :link: api/index
      :link-type: doc

      Go to the API reference when you already know which module or class
      you need.  The API pages are curated around the public workflow
      entry points.

Recommended first path
----------------------

For a first pass through the documentation, read:

1. :doc:`Getting started <getting_started>` for installation and the package
   orientation.
2. :doc:`Calabi-Yau Geometry Database <intro/database>` for the data layout,
   cache model, and the boundary between catalog and mirror conventions.
3. :doc:`Tutorials <tutorials>` for executable notebooks.
4. :doc:`API reference <api/index>` once you need precise signatures.

Users working specifically with the curated KKLT-vacua subset should also
read :doc:`KKLT-Vacua Database <intro/kklt_vacua_database>`.

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
   intro/kklt_vacua_database

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
