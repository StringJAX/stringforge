StringForge
=========

**Differentiable tools for string compactifications with JAX.**

StringForge is a Python framework for the systematic construction and
analysis of string vacua. It provides a unified computational pipeline
from Calabi–Yau compactification data to four-dimensional effective
field theories, vacuum solutions, and physical observables — with
automatic differentiation, just-in-time compilation, and hardware
acceleration throughout.

The framework is organised as an ecosystem of interoperable packages,
each targeting a specific layer of the compactification problem.
This site documents the umbrella package and links out to the
per-package documentation via intersphinx.

.. toctree::
   :maxdepth: 2
   :caption: Getting started

   getting_started

.. toctree::
   :maxdepth: 1
   :caption: The ecosystem

   ecosystem/overview
   ecosystem/architecture
   ecosystem/migration_from_jaxvacua

.. toctree::
   :maxdepth: 1
   :caption: Background

   intro/database

.. toctree::
   :maxdepth: 1
   :caption: Tutorials

   tutorials/quickstart
   tutorials/cytools_interface
   tutorials/cicy
   tutorials/vault_workflow
   tutorials/database_and_infrastructure/database_interface
   tutorials/database_and_infrastructure/cluster_parallelisation
   tutorials/database_and_infrastructure/vacua_storage
   tutorials/ecosystem_pipeline

.. toctree::
   :maxdepth: 1
   :caption: Package overviews

   packages/jaxvacua
   packages/jaxpolylog
   packages/kahlerjax
   packages/jaxiverse
   packages/cytools

.. toctree::
   :maxdepth: 2
   :caption: API reference

   api/index
