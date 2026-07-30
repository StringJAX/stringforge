Tutorials
=========

This page collects the executable notebooks in one place.  Use it as the main
entry point once you want to run code rather than read background material.

Choosing a path
---------------

.. grid:: 1 1 2 2
   :gutter: 2

   .. grid-item-card:: First orientation
      :link: tutorials/quickstart
      :link-type: doc

      Query a catalogue, load a model as ``lcs_tree``, construct a JAXVacua
      ``FluxVacuaFinder``, and learn where StringForge stops.

   .. grid-item-card:: Database access
      :link: tutorials/database_and_infrastructure/database_interface
      :link-type: doc

      Query HuggingFace-hosted catalogues, load Calabi-Yau data, batch models,
      and manage cache/offline workflows.

   .. grid-item-card:: Vacua storage
      :link: tutorials/vault_workflow
      :link-type: doc

      Validate, designate, store, and prepare vacuum datasets for the shared
      ``vacua_vault``.

   .. grid-item-card:: Cluster workflows
      :link: tutorials/database_and_infrastructure/cluster_parallelisation
      :link-type: doc

      Older chunk-and-merge pattern; prefer Vulcan for new cluster runs.

   .. grid-item-card:: Production cluster runs
      :link: tutorials/database_and_infrastructure/vulcan_cluster_runs
      :link-type: doc

      Stage vacuum batches on workers, batch-commit on a head node within
      HuggingFace's commit-rate cap, query and stream the resulting dataset.

Tutorial catalogue
------------------

Quickstart and overview
~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Notebook
     - Use it for
   * - :doc:`Quickstart <tutorials/quickstart>`
     - The shortest StringForge-first database-to-JAXVacua workflow.
   * - :doc:`StringForge ecosystem pipeline <tutorials/ecosystem_pipeline>`
     - Public package choreography plus schematic planning notes for future packages.

Geometry input and databases
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Notebook
     - Use it for
   * - :doc:`CYTools interface <tutorials/cytools_interface>`
     - Understanding the boundary between CYTools objects, database rows, and JAXVacua model data.
   * - :doc:`Complete Intersection Calabi-Yau Threefolds <tutorials/cicy>`
     - Working with CICY catalogue data, with provenance caveats.
   * - :doc:`Database interface <tutorials/database_and_infrastructure/database_interface>`
     - Querying catalogues, loading individual models, batch loading, offline mode, and cache management.

Vacua vault and infrastructure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Notebook
     - Use it for
   * - :doc:`Vault workflow <tutorials/vault_workflow>`
     - Validating, designating, and preparing vacuum datasets for upload.
   * - :doc:`Storing vacuum objects <tutorials/storing_vacuum_objects>`
     - Writing ``jaxvacua.vacuum`` ``Vacuum`` / ``PFV`` / ``AFV`` objects directly (curated and Vulcan tiers) with exact, finder-free read-back.
   * - :doc:`Vacua storage <tutorials/database_and_infrastructure/vacua_storage>`
     - Local vacuum storage, querying, designation, retraction, and sharing.
   * - :doc:`Cluster parallelisation <tutorials/database_and_infrastructure/cluster_parallelisation>`
     - Exporting scan chunks, processing them on a cluster, and merging results.
   * - :doc:`Vulcan cluster runs <tutorials/database_and_infrastructure/vulcan_cluster_runs>`
     - Production vacuum forging: stage on workers, batch-commit on the head node, query and stream as an ML dataset.

Advanced curated subsets
~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Page
     - Use it for
   * - :doc:`KKLT Database <intro/kklt_database>`
     - Specialised conifold-class indexing and tags for KKLT-style searches.

.. toctree::
   :hidden:
   :maxdepth: 2

   tutorials/quickstart
   tutorials/cytools_interface
   tutorials/cicy
   tutorials/database_and_infrastructure/database_interface
   tutorials/vault_workflow
   tutorials/storing_vacuum_objects
   tutorials/database_and_infrastructure/vacua_storage
   tutorials/database_and_infrastructure/cluster_parallelisation
   tutorials/database_and_infrastructure/vulcan_cluster_runs
   tutorials/ecosystem_pipeline
