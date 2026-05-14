stringjax.cy_io
==================

.. currentmodule:: stringjax.cy_io

.. automodule:: stringjax.cy_io


Database classes
-----------------------------------

Pure-I/O classes for reading the HuggingFace-hosted ``cy-database``.  These
classes have **no downstream-package dependencies** and serve as the shared
geometry-database layer for jaxvacua, kahlerjax, and other sibling packages.
For model construction (``lcs_tree`` / ``FluxVacuaFinder``) and vacua
persistence, see ``jaxvacua.lcs_database`` and ``jaxvacua.vacua_writer``.

.. autosummary::
    :toctree: _autosummary
    :template: custom-class-template.rst

    CYDatabase
    TDFDatabase
    CICYDatabase


Discovery
-----------------------------------

.. autosummary::
    :toctree: _autosummary

    CYDatabase.info
    CYDatabase.query
    CYDatabase.query_conifolds


Module-level convenience functions
-----------------------------------

.. autosummary::
    :toctree: _autosummary

    load_catalog
    query_models
