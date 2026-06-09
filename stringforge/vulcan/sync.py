# Copyright 2026 Andreas Schachner
#
# This file is part of StringForge.
#
# StringForge is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

r"""
Vulcan sync tier: batch staged shards into HuggingFace commits.

A single :func:`sync_pending` call moves the contents of
``{staging_dir}/pending/`` into ``synced/`` (or ``failed/``) by
issuing batched ``HfApi.create_commit`` operations.  At most
:data:`~stringforge.vulcan.ratelimit.DEFAULT_BUDGET` commits are
issued per rolling-window hour (subject to the persisted budget
state); each commit can carry up to ``max_batch`` files.

On a 429 from HuggingFace the offending batch is retried with
exponential backoff that honours the server's ``Retry-After``;
persistent failures land in ``failed/`` for manual inspection.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from .hf_io import CommitResult, create_batched_commit
from .ratelimit import (
    CommitBudget,
    DEFAULT_BUDGET,
    DEFAULT_WINDOW_S,
    sleep_with_backoff,
)
from .writer import (
    FAILED_DIRNAME,
    PENDING_DIRNAME,
    StagedShard,
    SYNCED_DIRNAME,
    list_pending,
    move_to,
)

#: Default maximum files per commit.  HuggingFace places no strict
#: cap, but very large commits slow down the Hub UI and the diff
#: viewer.  500 keeps shard-batches usable.
DEFAULT_MAX_BATCH: int = 500

#: Default maximum retries on a single batch hitting a rate-limit
#: response.  Three retries with exponential backoff covers most
#: transient outages; beyond that we relegate the batch to ``failed/``.
DEFAULT_MAX_RETRIES: int = 3

#: Filename of the persisted commit-window state under the staging
#: directory.  One file per Vulcan staging dir; the budget is
#: per-staging-dir, not per-repo, so a single sync process targeting
#: multiple repos through one staging area shares the budget across
#: repos.  (For multi-repo writes you'd use multiple staging dirs.)
BUDGET_STATE_FILENAME: str = "_commit_budget.json"


@dataclass
class SyncReport:
    r"""
    Summary of a single :func:`sync_pending` invocation.

    Attributes:
        n_committed: Total files that landed in successful commits.
        n_failed: Files moved into ``failed/`` after exhausting retries.
        n_remaining: Files still in ``pending/`` (budget-exhausted).
        n_commits: Number of HF commits issued.
        commit_oids: Per-commit OIDs (None entries for dry-run).
        errors: Strings describing per-batch failures, when any.
    """
    n_committed: int = 0
    n_failed: int = 0
    n_remaining: int = 0
    n_commits: int = 0
    commit_oids: List[Optional[str]] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


def _chunk(items: List[StagedShard], size: int) -> List[List[StagedShard]]:
    if size <= 0:
        raise ValueError("max_batch must be >= 1")
    return [items[i:i + size] for i in range(0, len(items), size)]


def _attempt_commit(
    *,
    repo_id: str,
    batch: List[StagedShard],
    commit_message: str,
    token: Optional[str],
    create_pr: bool,
    revision: Optional[str],
    dry_run: bool,
    max_retries: int,
) -> CommitResult:
    r"""
    Run :func:`create_batched_commit` with retry+backoff on rate-limit
    responses.  Other failures (auth, network, etc.) are returned
    after a single attempt.
    """
    # Only the parquet files travel to HuggingFace.  Sidecars are a
    # local tracking concern and the parquet's kv-metadata blob
    # already carries the same provenance; uploading sidecars too
    # would just inflate the per-commit file count without giving
    # consumers anything they can't get from the parquet itself.
    paths_and_locals = [(s.path_in_repo, s.parquet_path) for s in batch]

    last: Optional[CommitResult] = None
    for attempt in range(max_retries + 1):
        result = create_batched_commit(
            repo_id=repo_id,
            paths_and_locals=paths_and_locals,
            commit_message=commit_message,
            token=token,
            create_pr=create_pr,
            revision=revision,
            dry_run=dry_run,
        )
        last = result
        if result.ok:
            return result
        if result.retry_after_s is None and not getattr(result, "is_transient", False):
            # Not a rate-limit error and not a transient network failure -- don't retry.
            return result
        if attempt == max_retries:
            return result
        sleep_with_backoff(attempt, retry_after_s=result.retry_after_s)
    assert last is not None
    return last


def sync_pending(
    staging_dir: Path,
    *,
    repo_id: str,
    token: Optional[str] = None,
    budget: int = DEFAULT_BUDGET,
    window_s: float = DEFAULT_WINDOW_S,
    max_batch: int = DEFAULT_MAX_BATCH,
    max_retries: int = DEFAULT_MAX_RETRIES,
    repo_type: str = "dataset",
    create_pr: bool = False,
    revision: Optional[str] = None,
    dry_run: bool = False,
    commit_prefix: str = "vulcan: forge",
) -> SyncReport:
    r"""
    **Description:**
    Drain ``{staging_dir}/pending/`` into HuggingFace commits.

    Shards are processed in batches of at most ``max_batch`` files.
    Each batch consumes one slot from the rolling-window budget
    (advisory: see :class:`~stringforge.vulcan.ratelimit.CommitBudget`);
    once the budget is exhausted, remaining batches stay in
    ``pending/`` for a later invocation to pick up.  Successful
    batches move their shards to ``synced/``; persistent failures
    (after ``max_retries`` rate-limit retries with exponential
    backoff) move to ``failed/`` for manual inspection.

    Args:
        staging_dir: Vulcan staging root.
        repo_id: HuggingFace dataset repo target.
        token: HF write token; falls back to ``HF_TOKEN``.
        budget: Maximum commits permitted per rolling ``window_s``.
        window_s: Window length (seconds).  Defaults to 1 hour.
        max_batch: Maximum files per HuggingFace commit.
        max_retries: Per-batch retry budget on 429/503 responses.
        repo_type: HF repo type; default ``"dataset"``.
        create_pr: Open PRs instead of pushing to the default branch.
        revision: HF revision name; defaults to the repo's default
            branch.
        dry_run: Skip the network call.  Shards stay in ``pending/``;
            budget slots are still reserved.
        commit_prefix: Prefix for the auto-generated commit message.

    Returns:
        SyncReport: Summary of what landed, what remains, and what
        failed.
    """
    staging_dir = Path(staging_dir)
    pending = list_pending(staging_dir)
    report = SyncReport()
    if not pending:
        return report

    budget_state = CommitBudget(
        staging_dir / BUDGET_STATE_FILENAME,
        budget=budget,
        window_s=window_s,
    )

    batches = _chunk(pending, max_batch)
    for batch in batches:
        if not budget_state.reserve():
            report.n_remaining += len(batch)
            continue

        run_ids = sorted({s.run_id for s in batch})
        if len(run_ids) == 1:
            run_summary = run_ids[0]
        else:
            run_summary = f"{run_ids[0]} (+{len(run_ids) - 1} more)"
        commit_message = (
            f"{commit_prefix}: {len(batch)} shards, run={run_summary}"
        )

        result = _attempt_commit(
            repo_id=repo_id,
            batch=batch,
            commit_message=commit_message,
            token=token,
            create_pr=create_pr,
            revision=revision,
            dry_run=dry_run,
            max_retries=max_retries,
        )
        report.n_commits += 1
        report.commit_oids.append(result.commit_oid)

        if result.ok:
            if not dry_run:
                for shard in batch:
                    move_to(shard, staging_dir, SYNCED_DIRNAME)
            report.n_committed += len(batch)
        else:
            report.errors.append(result.error or "unknown commit failure")
            if not dry_run:
                for shard in batch:
                    move_to(shard, staging_dir, FAILED_DIRNAME)
            report.n_failed += len(batch)

    # Any batches we didn't try because we ran out of budget are still
    # in `pending/` -- ``list_pending`` will pick them up next time.
    return report


def remaining_budget(staging_dir: Path, *, budget: int = DEFAULT_BUDGET,
                     window_s: float = DEFAULT_WINDOW_S) -> int:
    r"""
    **Description:**
    Report how many commit slots are currently available for the
    given staging directory.

    Args:
        staging_dir: Vulcan staging root.
        budget: Ceiling to evaluate against (defaults to the module
            default).
        window_s: Window length in seconds.

    Returns:
        int: Non-negative slot count.
    """
    return CommitBudget(
        Path(staging_dir) / BUDGET_STATE_FILENAME,
        budget=budget,
        window_s=window_s,
    ).remaining()
