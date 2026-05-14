stringjax.vacua_writer
========================

.. currentmodule:: stringjax.vacua_writer

.. automodule:: stringjax.vacua_writer


VacuaWriter
-----------------------------------

Standalone class for writing, querying, retracting, and pushing flux-vacuum
solutions to the vault and to the community HuggingFace repository.  Wraps a
:class:`stringjax.cy_io.CYDatabase` (or any subclass) instance and forwards
attribute lookups to it via ``__getattr__``.

Users can either call the methods directly on an explicit ``VacuaWriter(db)``
instance, or use the thin delegation methods exposed on
:class:`stringjax.lcs_database.LCSDatabase` — both paths call the same code.

.. autosummary::
    :toctree: _autosummary
    :template: custom-class-template.rst

    VacuaWriter


Local / vault operations
-----------------------------------

.. autosummary::
    :toctree: _autosummary

    VacuaWriter.designate_vacua
    VacuaWriter.retract_designated
    VacuaWriter.purge_retracted
    VacuaWriter.query_designated
    VacuaWriter.load_designated
    VacuaWriter.designated_info
    VacuaWriter.load_local_vacua


HuggingFace Hub operations
-----------------------------------

.. autosummary::
    :toctree: _autosummary

    VacuaWriter.push_vacua_to_hub
    VacuaWriter.fetch_vacua_from_hub
    VacuaWriter.list_hub_vacua


Session-tier helpers
-----------------------------------

.. autosummary::
    :toctree: _autosummary

    VacuaWriter.vacua_writer
    VacuaWriter.query_vacua
    VacuaWriter.load_vacua
    VacuaWriter.solution_exists
    VacuaWriter.find_similar_vacua
    VacuaWriter.vacua_info
    VacuaWriter.delete_vacua
    VacuaWriter.validate_vacua
