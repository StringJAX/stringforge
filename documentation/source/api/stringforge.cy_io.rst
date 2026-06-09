stringforge.cy_io
==================

.. currentmodule:: stringforge.cy_io

.. automodule:: stringforge.cy_io


Database classes
-----------------------------------

Pure-I/O classes for reading the HuggingFace-hosted ``cy-database``.  These
classes have **no downstream-package dependencies** and serve as the shared
geometry-database layer for JAXVacua and planned sibling packages.
For model construction (``lcs_tree`` / ``FluxVacuaFinder``) and vacua
persistence, see ``stringforge.lcs_database`` and ``stringforge.vacua_writer``.

.. autosummary::
    :toctree: ../_autosummary
    :template: custom-class-template.rst

    CYDatabase
    TDFDatabase
    CICYDatabase


Discovery
-----------------------------------

* :meth:`CYDatabase.info`
* :meth:`CYDatabase.query`
* :meth:`CYDatabase.query_conifolds`


Module-level convenience functions
-----------------------------------

* :func:`load_catalog`
* :func:`query_models`
