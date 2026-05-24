Tutorials
=========

This page collects the executable notebooks in one place.  Use it as the
main entry point once you want to run code rather than read background
material.

Choosing a path
---------------

.. grid:: 1 1 2 2
   :gutter: 2

   .. grid-item-card:: First orientation
      :link: tutorials/quickstart
      :link-type: doc

      Start here for the broadest overview of how StringForge relates to
      the sibling packages in the ecosystem.

   .. grid-item-card:: Database access
      :link: tutorials/database_and_infrastructure/database_interface
      :link-type: doc

      Query the HuggingFace-hosted catalogues, load Calabi-Yau data, and
      construct JAXVacua models through ``LCSDatabase``.

   .. grid-item-card:: Vacua storage
      :link: tutorials/vault_workflow
      :link-type: doc

      Learn how designated vacua are validated, stored locally, and prepared
      for the public vacua vault.

   .. grid-item-card:: Cluster workflows
      :link: tutorials/database_and_infrastructure/cluster_parallelisation
      :link-type: doc

      Use the infrastructure notebooks when a scan needs chunking, merging,
      and persistent run output.

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
     - A first orientation to the package and its relationship to JAXVacua.
   * - :doc:`StringForge ecosystem pipeline <tutorials/ecosystem_pipeline>`
     - End-to-end package choreography across the broader ecosystem.

Geometry input and databases
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Notebook
     - Use it for
   * - :doc:`CYTools interface <tutorials/cytools_interface>`
     - Loading Kreuzer-Skarke geometries through CYTools and translating
       them into model data.
   * - :doc:`Complete Intersection Calabi-Yau Threefolds <tutorials/cicy>`
     - Working with CICY data and CICY model loading.
   * - :doc:`Database interface <tutorials/database_and_infrastructure/database_interface>`
     - Querying catalogues, loading individual models, batch loading,
       offline mode, and cache management.

Vacua vault and infrastructure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Notebook
     - Use it for
   * - :doc:`Vault workflow <tutorials/vault_workflow>`
     - Validating, designating, and preparing vacuum datasets for upload.
   * - :doc:`Vacua storage <tutorials/database_and_infrastructure/vacua_storage>`
     - Local vacuum storage, querying, designation, retraction, and sharing.
   * - :doc:`Cluster parallelisation <tutorials/database_and_infrastructure/cluster_parallelisation>`
     - Exporting scan chunks, processing them on a cluster, and merging
       results back into the vault workflow.

.. toctree::
   :hidden:
   :maxdepth: 2

   tutorials/quickstart
   tutorials/cytools_interface
   tutorials/cicy
   tutorials/database_and_infrastructure/database_interface
   tutorials/vault_workflow
   tutorials/database_and_infrastructure/vacua_storage
   tutorials/database_and_infrastructure/cluster_parallelisation
   tutorials/ecosystem_pipeline
