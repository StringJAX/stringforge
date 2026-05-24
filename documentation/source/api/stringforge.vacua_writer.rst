stringforge.vacua_writer
========================

.. currentmodule:: stringforge.vacua_writer

.. automodule:: stringforge.vacua_writer


VacuaWriter
-----------------------------------

Standalone class for writing, querying, retracting, and pushing flux-vacuum
solutions to the vault and to the community HuggingFace repository.  Wraps a
:class:`stringforge.cy_io.CYDatabase` (or any subclass) instance and forwards
attribute lookups to it via ``__getattr__``.

Users can either call the methods directly on an explicit ``VacuaWriter(db)``
instance, or use the thin delegation methods exposed on
:class:`stringforge.lcs_database.LCSDatabase` — both paths call the same code.

.. raw:: html
   :file: ../_static/figures/f3_vault_workflow.html

.. autosummary::
    :toctree: ../_autosummary
    :template: custom-class-template.rst

    VacuaWriter


Local / vault operations
-----------------------------------

* :meth:`VacuaWriter.designate_vacua`
* :meth:`VacuaWriter.retract_designated`
* :meth:`VacuaWriter.purge_retracted`
* :meth:`VacuaWriter.query_designated`
* :meth:`VacuaWriter.load_designated`
* :meth:`VacuaWriter.designated_info`
* :meth:`VacuaWriter.load_local_vacua`


HuggingFace Hub operations
-----------------------------------

* :meth:`VacuaWriter.push_vacua_to_hub`
* :meth:`VacuaWriter.fetch_vacua_from_hub`
* :meth:`VacuaWriter.list_hub_vacua`


Session-tier helpers
-----------------------------------

* :meth:`VacuaWriter.vacua_writer`
* :meth:`VacuaWriter.query_vacua`
* :meth:`VacuaWriter.load_vacua`
* :meth:`VacuaWriter.solution_exists`
* :meth:`VacuaWriter.find_similar_vacua`
* :meth:`VacuaWriter.vacua_info`
* :meth:`VacuaWriter.delete_vacua`
* :meth:`VacuaWriter.validate_vacua`
