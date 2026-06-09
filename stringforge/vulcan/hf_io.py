# Copyright 2026 Andreas Schachner
#
# This file is part of StringForge.
#
# StringForge is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

r"""
HuggingFace Hub I/O wrappers for Vulcan.

Vulcan uses :class:`huggingface_hub.HfApi.create_commit` to upload
many staged shards in **one** commit -- avoiding the
one-commit-per-file pattern in
:func:`stringforge.vacua_writer.VacuaWriter.designate_vacua` that
would hit HF's 100-commit-per-hour cap almost immediately on a cluster
workload.

The HuggingFace SDK is an *optional* runtime dependency: this module
loads it lazily so the rest of Vulcan (worker writes, schema
validation, run-id rendering, queries against a local cache) keeps
working in environments without ``huggingface_hub`` installed.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence


@dataclass
class CommitResult:
    r"""
    Outcome of a single batched commit.

    Attributes:
        ok: ``True`` if the commit landed on HF.
        commit_oid: The commit OID returned by the Hub (``None`` on
            failure or dry-run).
        paths: HF-repo-relative paths that were carried in this commit.
        error: Stringified error when ``ok`` is False.
        retry_after_s: Minimum retry interval for rate-limit responses;
            ``0.0`` when the server didn't suggest one, ``None`` when
            the failure is not rate-limit-shaped.
        is_transient: ``True`` when the failure looks like a transient
            network error (connection reset, timeout) with no HTTP
            response attached.  The sync tier treats this as
            retry-eligible under its own back-off schedule.
    """
    ok: bool
    commit_oid: Optional[str]
    paths: List[str]
    error: Optional[str] = None
    retry_after_s: Optional[float] = None
    is_transient: bool = False


def _load_hf_api() -> Any:
    r"""
    Lazily import :class:`huggingface_hub.HfApi`.

    Raises a clear ``ImportError`` referencing the M2 dependency hint
    rather than the SDK's own message.
    """
    try:
        from huggingface_hub import HfApi  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required for Vulcan sync. "
            "Install it via `pip install 'stringforge[sync]'`."
        ) from exc
    return HfApi


def _commit_operations(paths_and_locals: Sequence[tuple[str, Path]]) -> list:
    r"""
    Build a list of ``CommitOperationAdd`` operations for
    ``HfApi.create_commit``.  ``paths_and_locals`` pairs the in-repo
    path with the on-disk source.
    """
    try:
        from huggingface_hub import CommitOperationAdd  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Vulcan sync requires `huggingface_hub.CommitOperationAdd`. "
            "Install or upgrade huggingface_hub."
        ) from exc
    ops = []
    for path_in_repo, local_path in paths_and_locals:
        ops.append(
            CommitOperationAdd(
                path_in_repo=path_in_repo,
                path_or_fileobj=str(local_path),
            )
        )
    return ops


def _extract_retry_after(exc: BaseException) -> Optional[float]:
    r"""
    **Description:**
    Best-effort extraction of a ``Retry-After`` value from an
    :class:`huggingface_hub.errors.HfHubHTTPError` or its underlying
    ``requests.Response``.

    Args:
        exc: The exception raised by the HF client.

    Returns:
        Optional[float]: Seconds to wait, or ``None`` if the response
        carried no header (or no response object).
    """
    response = getattr(exc, "response", None)
    if response is None:
        return None
    headers = getattr(response, "headers", None)
    if not headers:
        return None
    from .ratelimit import parse_retry_after
    return parse_retry_after(headers.get("Retry-After"))


def _is_rate_limit_error(exc: BaseException) -> bool:
    r"""
    Heuristic for "this looks like an HF rate-limit response".  Accepts
    HTTP 429 and 503; rejects everything else.  We intentionally do
    not catch generic network errors here -- those are handled by the
    sync tier's wider retry envelope.
    """
    response = getattr(exc, "response", None)
    if response is None:
        return False
    status = getattr(response, "status_code", None)
    return status in (429, 503)


def _is_transient_network_error(exc: BaseException) -> bool:
    r"""
    **Description:**
    Heuristic for "this looks like a transient network failure" --
    i.e.  a connection reset or timeout with no HTTP response object
    attached.  Such failures are retry-eligible under the sync tier's
    own back-off schedule.

    Args:
        exc: The exception raised by the HF client.

    Returns:
        bool: ``True`` if the exception is a connection / timeout
        class with no response object, ``False`` otherwise.
    """
    if getattr(exc, "response", None) is not None:
        return False
    try:
        import requests.exceptions as _rexc  # type: ignore
    except ImportError:
        _rexc = None  # type: ignore[assignment]
    if _rexc is not None and isinstance(
        exc, (_rexc.ConnectionError, _rexc.Timeout)
    ):
        return True
    return exc.__class__.__name__ in {"ConnectionError", "Timeout", "ReadTimeout"}


def create_batched_commit(
    *,
    repo_id: str,
    paths_and_locals: Sequence[tuple[str, Path]],
    commit_message: str,
    token: Optional[str] = None,
    repo_type: str = "dataset",
    create_pr: bool = False,
    revision: Optional[str] = None,
    dry_run: bool = False,
) -> CommitResult:
    r"""
    **Description:**
    Upload a batch of files to ``repo_id`` in a single commit.

    The call is atomic on the HuggingFace side: either every file
    lands or none does.  Failures (auth, network, rate-limit) are
    reported back as a :class:`CommitResult` with ``ok=False`` rather
    than being raised, so the sync tier can route them to ``failed/``
    or retry without losing the batch.

    On ``dry_run=True`` no network call is made and ``ok`` is reported
    as ``True`` with ``commit_oid=None``.

    Args:
        repo_id: ``"user/repo"`` on the Hub.
        paths_and_locals: Iterable of ``(path_in_repo, local_path)``
            pairs.  Each ``local_path`` must be readable.
        commit_message: Single commit message for the whole batch.
        token: HF write token.  Falls back to ``HF_TOKEN`` env var if
            unset.
        repo_type: ``"dataset"`` for Vulcan; the SDK default would be
            ``"model"``.
        create_pr: When ``True``, opens a PR instead of pushing to the
            default branch.
        revision: Optional ref name; defaults to the repo's default
            branch.
        dry_run: Skip the network call.  Used by the sync CLI in
            inspect mode.

    Returns:
        CommitResult: Summary of the outcome.  ``retry_after_s`` is
        populated only on rate-limit responses.
    """
    paths = [p for (p, _) in paths_and_locals]
    if not paths:
        return CommitResult(ok=True, commit_oid=None, paths=[])

    if dry_run:
        return CommitResult(ok=True, commit_oid=None, paths=paths)

    HfApi = _load_hf_api()
    api = HfApi(token=token or os.environ.get("HF_TOKEN"))
    operations = _commit_operations(paths_and_locals)

    try:
        commit_info = api.create_commit(
            repo_id=repo_id,
            repo_type=repo_type,
            operations=operations,
            commit_message=commit_message,
            create_pr=create_pr,
            revision=revision,
        )
    except Exception as exc:  # noqa: BLE001 -- we report this back, not raise
        if _is_rate_limit_error(exc):
            retry_after = _extract_retry_after(exc)
            if retry_after is None:
                retry_after = 0.0   # rate-limit-shaped, no Retry-After: fall back to exponential schedule
        else:
            retry_after = None
        is_transient = _is_transient_network_error(exc)
        return CommitResult(
            ok=False,
            commit_oid=None,
            paths=paths,
            error=f"{type(exc).__name__}: {exc}",
            retry_after_s=retry_after,
            is_transient=is_transient,
        )

    oid = (
        getattr(commit_info, "oid", None)
        or getattr(commit_info, "commit_oid", None)
        or getattr(commit_info, "commit_url", None)
    )
    return CommitResult(ok=True, commit_oid=oid, paths=paths)


def ensure_repo_exists(
    repo_id: str,
    *,
    token: Optional[str] = None,
    repo_type: str = "dataset",
    private: bool = False,
) -> None:
    r"""
    **Description:**
    Idempotent :func:`huggingface_hub.HfApi.create_repo` wrapper.

    Safe to invoke before every sync; the SDK no-ops when the repo
    already exists (using ``exist_ok=True`` where available, with a
    fallback to swallowing "already exists" errors on older SDKs).

    Args:
        repo_id: ``"user/repo"`` on the Hub.
        token: HF write token; falls back to ``HF_TOKEN``.
        repo_type: ``"dataset"`` for Vulcan.
        private: Mark the repo private at creation time.

    Returns:
        None
    """
    HfApi = _load_hf_api()
    api = HfApi(token=token or os.environ.get("HF_TOKEN"))
    try:
        api.create_repo(
            repo_id=repo_id,
            repo_type=repo_type,
            private=private,
            exist_ok=True,
        )
    except TypeError:
        # Older huggingface_hub: no exist_ok kwarg.  Swallow the
        # "already exists" path; re-raise anything else.
        try:
            api.create_repo(repo_id=repo_id, repo_type=repo_type, private=private)
        except Exception as exc:  # noqa: BLE001
            if "exists" not in str(exc).lower():
                raise
