# Copyright 2026 Andreas Schachner
#
# This file is part of StringForge.
#
# StringForge is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

r"""
Vulcan: production vacuum-forging pipeline for StringForge.

Vulcan is the high-volume, cluster-side, append-only writer that
forges vacuum-search output into a HuggingFace dataset repo.  It is
the production counterpart to
:class:`stringforge.vacua_writer.VacuaWriter` -- VacuaWriter
*designates* curated vacua into ``vacua_vault``, Vulcan *forges*
production runs into a separate, ML-friendly repo.

Architecture summary:

* Worker code calls :meth:`Vulcan.write` to stage validated parquet
  shards under ``{staging_dir}/pending/``.  No HuggingFace I/O.
* A separate sync tier (:meth:`Vulcan.sync`, the ``vulcan sync`` CLI,
  or a cron job on a head node) batches pending shards into one
  :func:`huggingface_hub.HfApi.create_commit` call per
  ``max_batch``-sized chunk (default 500 files per commit),
  respecting HF's 100-commit-per-hour cap via
  :mod:`~stringforge.vulcan.ratelimit`.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence, Union

import pandas as pd

from . import schema as _schema_module
from .certify import (
    CertifyResult,
    CertRow,
    StabilityRecord,
    build_jaxvacua_certifier,
    check_endpoint_stability,
    certify_shard,
)
from .community import (
    CommunityGateResult,
    dedup_key,
    review_submission,
)
from .ml_view import (
    DEFAULT_FRACTIONS,
    DEFAULT_SPLITS,
    FeatureSpec,
    VulcanMLView,
    assign_split,
)
from .ratelimit import DEFAULT_BUDGET, DEFAULT_WINDOW_S
from .reader import CATALOG_COLUMNS, DEFAULT_CACHE_SIZE, VulcanReader
from .runid import DEFAULT_TEMPLATE, render as render_run_id
from .snapshot import (
    MANIFEST_FILENAME,
    SnapshotManifest,
    build_manifest,
    freeze_snapshot,
)
from .sync import (
    DEFAULT_MAX_BATCH,
    DEFAULT_MAX_RETRIES,
    SyncReport,
    remaining_budget,
    sync_pending,
)
from .verifier import (
    CANONICAL_CHECK_NAMES,
    FULL_CASCADE_CHECKS,
    VerifierRegistry,
    VerifierSpec,
    verifier_id,
)
from .writer import (
    FAILED_DIRNAME,
    PENDING_DIRNAME,
    SYNCED_DIRNAME,
    StagedShard,
    list_pending,
    vacua_to_vulcan_df,
    write_shard,
)

__all__ = (
    "Vulcan",
    "VulcanReader",
    "VulcanMLView",
    "FeatureSpec",
    "StagedShard",
    "SyncReport",
    "VerifierSpec",
    "VerifierRegistry",
    "verifier_id",
    "StabilityRecord",
    "check_endpoint_stability",
    "SnapshotManifest",
    "build_manifest",
    "freeze_snapshot",
    "CertRow",
    "CertifyResult",
    "certify_shard",
    "build_jaxvacua_certifier",
    "CommunityGateResult",
    "dedup_key",
    "review_submission",
    "resolve_staging_dir",
    "vacua_to_vulcan_df",
)


def resolve_staging_dir(explicit: Optional[Union[str, Path]] = None) -> Path:
    r"""
    **Description:**
    Resolve the Vulcan staging directory used by workers and the sync
    tier.

    Resolution order -- uses whichever stage succeeds first:

    1. ``explicit`` argument when provided.
    2. ``STRINGFORGE_VULCAN_STAGING_DIR`` environment variable.
    3. ``<stringforge_repo_root>/vulcan_staging/`` when running from a
       stringforge source checkout.
    4. ``<stringforge.data_dir>/vulcan_staging/``.

    The directory is **not** created here; the writer / sync helpers
    materialise its ``pending/``, ``synced/`` and ``failed/`` subdirs
    on first use via
    :func:`~stringforge.vulcan.writer.ensure_staging_layout`.

    Args:
        explicit: An explicit path that overrides every other source.

    Returns:
        Path: Absolute path to the chosen staging directory.
    """
    if explicit is not None:
        return Path(explicit).resolve()

    env = os.environ.get("STRINGFORGE_VULCAN_STAGING_DIR")
    if env:
        return Path(env).resolve()

    # Try repo-root detection, falling through to the data-dir.
    try:
        from stringforge.cy_io import _find_stringforge_repo_root  # type: ignore
        root = _find_stringforge_repo_root()
    except Exception:
        root = None
    if root is not None:
        return (Path(root) / "vulcan_staging").resolve()

    try:
        from stringforge import data_dir as _data_dir
    except ImportError:
        _data_dir = os.environ.get(
            "STRINGFORGE_DATA_DIR",
            str(Path.home() / ".stringforge_cache"),
        )
    return (Path(_data_dir) / "vulcan_staging").resolve()


class Vulcan:
    r"""
    **Description:**
    Per-repo handle for the production vacuum-forging pipeline.

    A ``Vulcan`` instance is bound to a single HuggingFace repo and a
    single local staging directory.  Workers call :meth:`write` to
    stage validated shards; a head node (cron job, daemon, or
    ``python -m stringforge.vulcan sync``) calls :meth:`sync` to push
    them to the Hub.  Multiple ``Vulcan`` instances may co-exist in
    the same process to write to different repos -- useful for
    cross-project workflows where different paper drafts target
    different production repos.

    See :class:`stringforge.vacua_writer.VacuaWriter` for the curated
    counterpart that targets the paper-aligned ``vacua_vault`` repo.

    Args:
        repo: ``"user/repo"`` on HuggingFace.  Required; no global
            default.  Override per-call via :meth:`sync` with
            ``repo=...``.
        staging_dir: Local staging root.  Resolution order is
            documented in :func:`resolve_staging_dir`.
        run_id_template: Either a format string with ``{name}`` /
            ``{name?}`` placeholders, a callable returning the
            run_id, or ``None`` for the UUID fallback.  See
            :mod:`~stringforge.vulcan.runid`.
        project: Default ``{project}`` value for the run-id template.
        token: HuggingFace write token.  Falls back to
            ``STRINGFORGE_VULCAN_TOKEN`` then ``HF_TOKEN`` at sync
            time.
        budget: Commits-per-hour ceiling.  Defaults to 90 (10-commit
            margin below HuggingFace's 100/hour cap).
        window_s: Rolling-window length, in seconds, over which
            ``budget`` applies.  Defaults to 3600 s.
        extra_kwargs: Default substitutions for the run-id template
            applied to every :meth:`write` call.  Per-call
            ``extra_kwargs=`` arguments take precedence.

    Raises:
        ValueError: When ``repo`` is empty.
    """

    def __init__(
        self,
        *,
        repo: str,
        staging_dir: Optional[Union[str, Path]] = None,
        run_id_template: Optional[Union[str, Callable[..., str]]] = DEFAULT_TEMPLATE,
        project: str = "",
        token: Optional[str] = None,
        budget: int = DEFAULT_BUDGET,
        window_s: float = DEFAULT_WINDOW_S,
        extra_kwargs: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if not repo:
            raise ValueError("Vulcan requires a HuggingFace repo (e.g. 'user/repo').")
        self.repo = str(repo)
        self.staging_dir = resolve_staging_dir(staging_dir)
        self.run_id_template = run_id_template
        self.project = str(project)
        self._token = token
        self.budget = int(budget)
        self.window_s = float(window_s)
        self.extra_kwargs = dict(extra_kwargs) if extra_kwargs else {}

    # ── factory ────────────────────────────────────────────────────

    @classmethod
    def from_env(
        cls,
        *,
        repo: Optional[str] = None,
        run_id_template: Optional[Union[str, Callable[..., str]]] = DEFAULT_TEMPLATE,
        **kwargs: Any,
    ) -> "Vulcan":
        r"""
        **Description:**
        Construct a :class:`Vulcan` from ``STRINGFORGE_VULCAN_*``
        environment variables.

        Environment variables read:

        * ``STRINGFORGE_VULCAN_REPO`` -- target HF repo.
        * ``STRINGFORGE_VULCAN_PROJECT`` -- default ``{project}``
          substitution.
        * ``STRINGFORGE_VULCAN_BUDGET`` -- commits/hour ceiling.
        * ``STRINGFORGE_VULCAN_TOKEN`` -- HuggingFace write token.

        Explicit keyword arguments take precedence over the
        environment.

        Args:
            repo: Optional override for ``STRINGFORGE_VULCAN_REPO``.
            run_id_template: Override for the run-id template; see
                :class:`Vulcan`'s constructor.
            **kwargs: Forwarded to :class:`Vulcan`.

        Returns:
            Vulcan: A configured instance.

        Raises:
            ValueError: When ``repo`` is unset and
                ``STRINGFORGE_VULCAN_REPO`` is empty, or when
                ``STRINGFORGE_VULCAN_BUDGET`` is non-empty but not a
                valid integer.
        """
        repo = repo or os.environ.get("STRINGFORGE_VULCAN_REPO")
        if not repo:
            raise ValueError(
                "Vulcan.from_env: no repo specified.  Set "
                "STRINGFORGE_VULCAN_REPO or pass repo=... explicitly."
            )
        project_kw = kwargs.pop("project", None)
        project = project_kw if project_kw is not None else os.environ.get(
            "STRINGFORGE_VULCAN_PROJECT", "",
        )
        budget_env = (os.environ.get("STRINGFORGE_VULCAN_BUDGET") or "").strip()
        if budget_env:
            try:
                env_budget = int(budget_env)
            except ValueError:
                raise ValueError(
                    f"STRINGFORGE_VULCAN_BUDGET must be an integer; got "
                    f"{budget_env!r}"
                ) from None
        else:
            env_budget = DEFAULT_BUDGET
        budget_kw = kwargs.pop("budget", None)
        budget = budget_kw if budget_kw is not None else env_budget
        token_kw = kwargs.pop("token", None)
        token = token_kw if token_kw is not None else os.environ.get("STRINGFORGE_VULCAN_TOKEN")
        return cls(
            repo=repo,
            run_id_template=run_id_template,
            project=project,
            budget=budget,
            token=token,
            **kwargs,
        )

    def __repr__(self) -> str:
        return (
            f"Vulcan(repo={self.repo!r}, staging_dir={str(self.staging_dir)!r}, "
            f"project={self.project!r}, budget={self.budget})"
        )

    # ── worker-side ────────────────────────────────────────────────

    def render_run_id(self, **substitutions: Any) -> str:
        r"""
        **Description:**
        Render a ``run_id`` using the instance's template, the
        instance-wide ``extra_kwargs`` defaults, and the supplied
        per-call substitutions.

        Per-call substitutions take precedence over instance defaults.

        Args:
            **substitutions: Values to interpolate into the template.

        Returns:
            str: A slug-safe run identifier.
        """
        env: dict[str, Any] = dict(self.extra_kwargs)
        env.update(substitutions)
        return render_run_id(
            self.run_id_template,
            project=self.project,
            **env,
        )

    def write(
        self,
        df: pd.DataFrame,
        *,
        geometry: Mapping[str, Any],
        run_id: Optional[str] = None,
        tadpole_charge: Optional[int] = None,
        solver: Optional[Mapping[str, Any]] = None,
        provenance: Optional[Mapping[str, Any]] = None,
        extra_kwargs: Optional[Mapping[str, Any]] = None,
    ) -> StagedShard:
        r"""
        **Description:**
        Stage a vacuum batch as a parquet shard in ``pending/``.

        This method never contacts HuggingFace.  Workers can call
        it safely from any cluster node, including nodes with no
        outbound network.  Pending shards are picked up by
        :meth:`sync` (or the ``vulcan sync`` CLI) and uploaded
        in batches.

        Args:
            df: Per-vacuum DataFrame.  Must carry the vault-floor
                columns (``flux``, ``moduli_re``, ``moduli_im``,
                ``tau_re``, ``tau_im``).  The writer is authoritative
                on identity: ``run_id``, ``geometry_id`` and the
                geometry-key columns are unconditionally overwritten
                with the values resolved here.  ``tadpole_charge`` and
                ``created_at`` are filled in only when absent from
                ``df`` (the caller may pre-populate them).
            geometry: ``{h11, h12, ks_id, triang_id, conifold_id,
                cicy_id}`` mapping.  Missing keys are filled with the
                ``-1`` sentinel.  ``h11``, ``h12`` are interpreted in
                mirror (jaxvacua / lcs_tree) convention, matching
                :class:`~stringforge.vacua_writer.LCSDatabase`.
            run_id: Override the rendered run_id.  Defaults to
                :meth:`render_run_id` driven by the geometry-key
                substitutions ``{h11, h12, ks_id, triang_id,
                conifold_id, cicy_id}``, plus ``solver`` (when
                ``solver['name']`` is supplied), ``seed`` (when
                ``provenance['seed']`` is supplied), and any
                instance- or call-level ``extra_kwargs``.
            tadpole_charge: Production-tier required column.  May be
                supplied here or pre-populated on ``df``; filled in
                only when absent.
            solver: ``{name, config_hash, ...}`` -- embedded in the
                parquet kv-metadata.
            provenance: ``{git_sha, seed, wall_clock_s, ...}`` --
                embedded alongside ``solver``.
            extra_kwargs: Extra substitutions for the run-id template
                (e.g. a custom ``{variant}`` placeholder).

        Returns:
            StagedShard: Reference to the on-disk parquet + sidecar
            pair and the canonical HF-repo-relative path.
        """
        substitutions: dict[str, Any] = {
            "h11": geometry.get("h11"),
            "h12": geometry.get("h12"),
            "ks_id": geometry.get("ks_id"),
            "triang_id": geometry.get("triang_id"),
            "conifold_id": geometry.get("conifold_id"),
            "cicy_id": geometry.get("cicy_id"),
        }
        if solver is not None and "name" in solver:
            substitutions["solver"] = solver["name"]
        if provenance is not None and "seed" in provenance:
            substitutions["seed"] = provenance["seed"]
        if extra_kwargs:
            substitutions.update(extra_kwargs)

        rid = run_id if run_id is not None else self.render_run_id(**substitutions)
        return write_shard(
            df,
            staging_dir=self.staging_dir,
            run_id=rid,
            geometry=geometry,
            tadpole_charge=tadpole_charge,
            solver=solver,
            provenance=provenance,
        )

    # ── vacuum-object convenience (jaxvacua.vacuum objects) ─────────

    def write_vacua(
        self,
        vacua: Any,
        *,
        models: Any = None,
        finder: Any = None,
        geometry: Optional[Mapping[str, Any]] = None,
        store_full: bool = True,
        store_trajectory: bool = False,
        tadpole_charge: Optional[int] = None,
        run_id: Optional[str] = None,
        solver: Optional[Mapping[str, Any]] = None,
        provenance: Optional[Mapping[str, Any]] = None,
    ) -> Any:
        r"""
        **Description:**
        Stage ``jaxvacua.vacuum.Vacuum`` objects as Vulcan shards — the
        production-tier analogue of
        :meth:`stringforge.vacua_writer.LCSDatabase.write_vacua`.

        The vacua are grouped by geometry (one shard = one geometry, the Vulcan
        invariant) and each group is written via :meth:`write`.  Geometry is
        resolved from (in order) an explicit *geometry* mapping, each vacuum's
        stamped ``metadata["identity"]``, or *finder* / *models*.

        Args:
            vacua: A ``Vacuum`` / ``PFV`` / ``AFV`` or an iterable of them.
            models: Optional finder resolution — a single finder, a
                ``PromotionModels``, or a ``{model_name: finder|PromotionModels}``
                dict (used for ``tadpole_charge`` and the geometry fallback).
            finder: A single finder applied to every group (overrides *models*).
            geometry: Explicit ``{h11, h12, ks_id, ...}`` mapping (overrides the
                stamped identity).  Required when neither identity nor a finder
                is available.
            store_full (bool): Embed the full record in ``extra_data``.
            store_trajectory (bool): Keep the verbose trajectory in the record.
            tadpole_charge (int | None): Override the per-row tadpole charge.
            run_id, solver, provenance: Forwarded to :meth:`write`.

        Returns:
            StagedShard, or a ``list`` of them when the input spans multiple
            geometries.
        """
        from .._vacuum_adapter import is_vacuum
        from ..vacua_writer import _extract_model_identity, _resolve_finder
        from .schema import GEOMETRY_KEYS, GEOMETRY_KEY_SENTINEL
        from .writer import vacua_to_vulcan_df

        items = [vacua] if is_vacuum(vacua) else list(vacua)
        groups: dict[str, list] = {}
        for v in items:
            meta = getattr(v, "metadata", None) or {}
            ident = meta.get("identity") or {}
            name = meta.get("model_name") or ident.get("model_name") or "unknown"
            groups.setdefault(str(name), []).append(v)

        shards: list = []
        for name, group in groups.items():
            fdr = finder if finder is not None else _resolve_finder(models, name)
            if fdr is None and tadpole_charge is None:
                raise ValueError(
                    f"vulcan.write_vacua: geometry {name!r} — the production tier "
                    f"requires a D3 tadpole charge (a REQUIRED column). Pass "
                    f"models=/finder= (tadpole computed via finder.tadpole) or an "
                    f"explicit tadpole_charge=."
                )
            ident = (getattr(group[0], "metadata", None) or {}).get("identity") or {}
            if not ident and fdr is not None:
                try:
                    ident = _extract_model_identity(fdr)
                except Exception:
                    ident = {}
            base = {**ident, **(dict(geometry) if geometry else {})}
            if base.get("h11") is None or base.get("h12") is None:
                raise ValueError(
                    f"vulcan.write_vacua: geometry for group {name!r} is unknown "
                    f"— pass geometry=..., models=..., or promote with identity "
                    f"stamping."
                )
            geom = {k: int(base.get(k, GEOMETRY_KEY_SENTINEL)) for k in GEOMETRY_KEYS}
            df = vacua_to_vulcan_df(group, finder=fdr, store_full=store_full,
                                    store_trajectory=store_trajectory)
            shards.append(self.write(
                df, geometry=geom, run_id=run_id, tadpole_charge=tadpole_charge,
                solver=solver, provenance=provenance,
            ))
        return shards[0] if len(shards) == 1 else shards

    def read_vacua(self, source: Any, *, finder: Any = None) -> list:
        r"""
        **Description:**
        Reconstruct ``jaxvacua.vacuum.Vacuum`` objects from a Vulcan shard, a
        parquet path, or an already-read :class:`pandas.DataFrame` — the inverse
        of :meth:`write_vacua`.

        Rows carrying a stored record are rebuilt exactly (finder-free); rows
        written with ``store_full=False`` (no record) are skipped.

        Args:
            source: A :class:`~stringforge.vulcan.writer.StagedShard`, a parquet
                path, or a :class:`pandas.DataFrame`.
            finder: Optional finder (reserved for rebuilding record-less rows).

        Returns:
            list: Reconstructed ``Vacuum`` objects.
        """
        from .._vacuum_adapter import row_to_vacuum

        if isinstance(source, pd.DataFrame):
            df = source
        else:
            path = getattr(source, "parquet_path", source)
            df = pd.read_parquet(path)
        out: list = []
        for _, r in df.iterrows():
            v = row_to_vacuum(r.get("extra_data"), finder=finder)
            if v is not None:
                out.append(v)
        return out

    # ── observability ──────────────────────────────────────────────

    def list_pending(self) -> list[StagedShard]:
        r"""
        **Description:**
        Return shards currently awaiting sync from this Vulcan's
        staging directory.

        Returns:
            list[StagedShard]: One entry per pending shard.
        """
        return list_pending(self.staging_dir)

    def remaining_budget(self) -> int:
        r"""
        **Description:**
        Report how many commits are still available in the current
        rolling window for this Vulcan's staging directory.

        Returns:
            int: Non-negative slot count.
        """
        return remaining_budget(
            self.staging_dir, budget=self.budget, window_s=self.window_s,
        )

    # ── read path ──────────────────────────────────────────────────

    def reader(self, *, cache_size: int = DEFAULT_CACHE_SIZE) -> "VulcanReader":
        r"""
        **Description:**
        Return a :class:`VulcanReader` bound to this Vulcan's local
        ``synced/`` directory.

        Args:
            cache_size: LRU capacity passed through to the reader.

        Returns:
            VulcanReader: A local-source reader.
        """
        return VulcanReader.from_local(self.staging_dir, cache_size=cache_size)

    def query(self, **filters: Any) -> pd.DataFrame:
        r"""
        **Description:**
        Convenience wrapper around :meth:`VulcanReader.query` using
        this Vulcan's local ``synced/`` directory.

        Args:
            **filters: Forwarded to :meth:`VulcanReader.query`.

        Returns:
            pd.DataFrame: Matching rows.
        """
        return self.reader().query(**filters)

    def fetch_run(self, run_id: str) -> pd.DataFrame:
        r"""
        **Description:**
        Convenience wrapper around :meth:`VulcanReader.fetch_run`
        using this Vulcan's local ``synced/`` directory.

        Args:
            run_id: Identifier carried on each row of the target
                shards.

        Returns:
            pd.DataFrame: Concatenated rows.
        """
        return self.reader().fetch_run(run_id)

    def ml_view(
        self,
        *,
        fractions: Optional[Sequence[float]] = None,
        splits: Optional[Sequence[str]] = None,
        salt: str = "",
        feature_spec: Optional["FeatureSpec"] = None,
    ) -> "VulcanMLView":
        r"""
        **Description:**
        Return a :class:`VulcanMLView` backed by this Vulcan's local
        ``synced/`` directory.

        Args:
            fractions: Per-split fractions; defaults to
                :data:`~stringforge.vulcan.ml_view.DEFAULT_FRACTIONS`.
            splits: Split labels; defaults to
                :data:`~stringforge.vulcan.ml_view.DEFAULT_SPLITS`.
            salt: Optional salt for cross-validation folds.
            feature_spec: Default :class:`FeatureSpec`.

        Returns:
            VulcanMLView: A ready-to-use ML view.
        """
        return VulcanMLView(
            self.reader(),
            fractions=fractions or DEFAULT_FRACTIONS,
            splits=splits or DEFAULT_SPLITS,
            salt=salt,
            feature_spec=feature_spec,
        )

    def freeze_snapshot(
        self,
        *,
        tag: str,
        created_at: str,
        repo: Optional[str] = None,
        require_fully_certified: bool = True,
        dry_run: bool = False,
        **kwargs: Any,
    ) -> "SnapshotManifest":
        r"""
        **Description:**
        Freeze a citeable snapshot of this Vulcan's local ``synced/``
        view (convenience wrapper around
        :func:`stringforge.vulcan.snapshot.freeze_snapshot`).

        Args:
            tag: Immutable git tag to create (e.g. ``"v2026.06"``).
            created_at: UTC timestamp string (caller-supplied for
                determinism).
            repo: HuggingFace repo to tag; defaults to this Vulcan's
                ``repo``.  Ignored when ``dry_run``.
            require_fully_certified: Refuse to tag a not-fully-certified
                snapshot (default ``True``).
            dry_run: Build the manifest only; no network call, no tag.
            **kwargs: Forwarded to
                :func:`~stringforge.vulcan.snapshot.freeze_snapshot`
                (e.g. ``local_manifest_dir``, ``token``, ``message``).

        Returns:
            SnapshotManifest: The manifest.
        """
        token = kwargs.pop("token", None)
        if token is None:
            token = self._token
            if token is None:
                token = os.environ.get(
                    "STRINGFORGE_VULCAN_TOKEN", os.environ.get("HF_TOKEN"),
                )
        return freeze_snapshot(
            self.reader(),
            tag=tag,
            created_at=created_at,
            repo=repo or self.repo,
            token=token,
            require_fully_certified=require_fully_certified,
            dry_run=dry_run,
            **kwargs,
        )

    # ── sync tier ──────────────────────────────────────────────────

    def sync(
        self,
        *,
        repo: Optional[str] = None,
        token: Optional[str] = None,
        budget: Optional[int] = None,
        max_batch: int = DEFAULT_MAX_BATCH,
        max_retries: int = DEFAULT_MAX_RETRIES,
        create_pr: bool = False,
        revision: Optional[str] = None,
        commit_prefix: str = "vulcan: forge",
        dry_run: bool = False,
    ) -> SyncReport:
        r"""
        **Description:**
        Drain ``pending/`` into HuggingFace commits.

        Wraps :func:`~stringforge.vulcan.sync.sync_pending` with this
        instance's repo, staging directory, token, budget, and
        window.

        Args:
            repo: Override the instance's target repo.
            token: HuggingFace write token; falls back to
                ``STRINGFORGE_VULCAN_TOKEN`` then ``HF_TOKEN``.
            budget: Override the instance's commits-per-hour ceiling.
            max_batch: Maximum files per HuggingFace commit.
            max_retries: Per-batch retry budget on 429/503 responses.
            create_pr: Open PRs instead of pushing to the default
                branch.
            revision: HF revision name; defaults to the repo's default
                branch.
            commit_prefix: Commit-message prefix used for each batch.
            dry_run: Skip the network call (still reserves budget
                slots).

        Returns:
            SyncReport: Summary of the work done.
        """
        effective_token = token if token is not None else self._token
        if effective_token is None:
            effective_token = os.environ.get(
                "STRINGFORGE_VULCAN_TOKEN", os.environ.get("HF_TOKEN"),
            )
        return sync_pending(
            self.staging_dir,
            repo_id=repo or self.repo,
            token=effective_token,
            budget=budget if budget is not None else self.budget,
            window_s=self.window_s,
            max_batch=max_batch,
            max_retries=max_retries,
            create_pr=create_pr,
            revision=revision,
            commit_prefix=commit_prefix,
            dry_run=dry_run,
        )
