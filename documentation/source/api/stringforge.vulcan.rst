stringforge.vulcan
==================

.. currentmodule:: stringforge.vulcan

.. automodule:: stringforge.vulcan


Vulcan
-----------------------------------

Production vacuum-forging pipeline for StringForge.  ``Vulcan`` is the
cluster-side counterpart to :class:`stringforge.vacua_writer.VacuaWriter`:
where ``VacuaWriter`` *designates* curated vacua into the paper-aligned
``vacua_vault`` repo, ``Vulcan`` *forges* high-volume production runs into a
separate, production HuggingFace repo with a uniform parquet schema and
geometry-disjoint VulcanMLView splits.

Worker code never touches HuggingFace.  :meth:`Vulcan.write` lands validated
parquet shards under ``{staging_dir}/pending/`` together with a per-file
metadata sidecar.  A separate sync tier (:meth:`Vulcan.sync`, the
``vulcan sync`` CLI, or a cron job on a head node) batches pending shards
into one :func:`huggingface_hub.HfApi.create_commit` call per
``max_batch``-sized chunk (default 500 files per commit), respecting
HuggingFace's documented 100-commit-per-hour-per-repo cap
(see https://huggingface.co/docs/hub/repositories-recommendations) via the
rolling-window budget in :mod:`~stringforge.vulcan.ratelimit`.

.. autosummary::
    :toctree: ../_autosummary
    :template: custom-class-template.rst

    Vulcan
    VulcanReader
    VulcanMLView


Worker-side: stage vacua
-----------------------------------

* :meth:`Vulcan.write`
* :meth:`Vulcan.write_vacua` — stage :class:`jaxvacua.vacuum.Vacuum` objects (one shard per geometry)
* :func:`stringforge.vulcan.writer.vacua_to_vulcan_df` — build the shard DataFrame from vacua
* :meth:`Vulcan.render_run_id`
* :meth:`Vulcan.list_pending`
* :meth:`Vulcan.remaining_budget`


Sync tier: HuggingFace commits
-----------------------------------

* :meth:`Vulcan.sync`
* :func:`stringforge.vulcan.sync.sync_pending`
* :func:`stringforge.vulcan.sync.remaining_budget`


Read path: query and fetch
-----------------------------------

* :meth:`Vulcan.reader`
* :meth:`Vulcan.query`
* :meth:`Vulcan.fetch_run`
* :meth:`Vulcan.read_vacua` — reconstruct :class:`jaxvacua.vacuum.Vacuum` objects from a shard
* :meth:`VulcanReader.catalog`
* :meth:`VulcanReader.query`
* :meth:`VulcanReader.fetch_shard`


ML view: train / val / test splits
-----------------------------------

* :meth:`Vulcan.ml_view`
* :meth:`VulcanMLView.as_dataframe`
* :meth:`VulcanMLView.as_hf_dataset`
* :meth:`VulcanMLView.split_assignments`
* :class:`stringforge.vulcan.ml_view.FeatureSpec`
* :func:`stringforge.vulcan.ml_view.assign_split`


Submodules
-----------------------------------

.. autosummary::
    :toctree: ../_autosummary
    :template: custom-module-template.rst

    schema
    runid
    writer
    ratelimit
    hf_io
    sync
    reader
    ml_view


Environment variables
-----------------------------------

* ``STRINGFORGE_VULCAN_REPO`` — target HuggingFace dataset repo.  No
  package-level default; each :class:`Vulcan` must be told where to write.
* ``STRINGFORGE_VULCAN_STAGING_DIR`` — local staging root.  When unset (and
  no explicit ``staging_dir=`` is passed to :class:`Vulcan`),
  :func:`stringforge.vulcan.resolve_staging_dir` falls back to
  ``<repo_root>/vulcan_staging/`` inside a stringforge source checkout, or
  ``<stringforge.data_dir>/vulcan_staging/`` otherwise.
* ``STRINGFORGE_VULCAN_PROJECT`` — default ``{project}`` substitution for
  the run-id template.
* ``STRINGFORGE_VULCAN_BUDGET`` — commits-per-hour ceiling (default 90;
  the HuggingFace cap is 100).
* ``STRINGFORGE_VULCAN_TOKEN`` — HuggingFace write token.  Falls back to
  ``HF_TOKEN``.

Public setters in :mod:`stringforge`:
:func:`stringforge.set_vulcan_repo`,
:func:`stringforge.set_vulcan_staging_dir`,
:func:`stringforge.set_vulcan_project`,
:func:`stringforge.set_vulcan_budget`,
:func:`stringforge.set_vulcan_token`.
