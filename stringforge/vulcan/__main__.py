# Copyright 2026 Andreas Schachner
#
# This file is part of StringForge.
#
# StringForge is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

r"""
CLI for the Vulcan sync tier.

Examples:

    # Inspect what's pending without touching HF.
    python -m stringforge.vulcan status

    # Drain `pending/` to the repo (one cron-tick worth of work).
    python -m stringforge.vulcan sync --repo aschachner/vacua_forge

    # Same as above but issue PRs instead of direct commits.
    python -m stringforge.vulcan sync --repo aschachner/vacua_forge --create-pr

    # Estimate what would happen without contacting HF.
    python -m stringforge.vulcan sync --repo aschachner/vacua_forge --dry-run
"""
from __future__ import annotations

import argparse
import os
import sys

from . import Vulcan, resolve_staging_dir
from .ratelimit import DEFAULT_BUDGET, DEFAULT_WINDOW_S
from .sync import DEFAULT_MAX_BATCH, DEFAULT_MAX_RETRIES
from .writer import list_pending


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--staging-dir",
        default=None,
        help="Staging directory (defaults to STRINGFORGE_VULCAN_STAGING_DIR).",
    )


def _cmd_status(args: argparse.Namespace) -> int:
    staging = resolve_staging_dir(args.staging_dir)
    pending = list_pending(staging)
    print(f"staging_dir: {staging}")
    print(f"pending shards: {len(pending)}")
    if pending:
        for sh in pending[:10]:
            print(f"  {sh.run_id}  {sh.path_in_repo}  ({sh.n_rows} rows)")
        if len(pending) > 10:
            print(f"  ... and {len(pending) - 10} more")
    return 0


def _cmd_sync(args: argparse.Namespace) -> int:
    repo = args.repo or os.environ.get("STRINGFORGE_VULCAN_REPO")
    if not repo:
        print(
            "error: --repo is required (or set STRINGFORGE_VULCAN_REPO).",
            file=sys.stderr,
        )
        return 2

    forge = Vulcan(
        repo=repo,
        staging_dir=args.staging_dir,
        budget=args.budget,
        window_s=args.window_s,
    )
    report = forge.sync(
        budget=args.budget,
        max_batch=args.max_batch,
        max_retries=args.max_retries,
        create_pr=args.create_pr,
        dry_run=args.dry_run,
    )
    print(
        f"committed={report.n_committed}  "
        f"failed={report.n_failed}  "
        f"remaining={report.n_remaining}  "
        f"commits={report.n_commits}"
    )
    if report.errors:
        for err in report.errors:
            print(f"  ! {err}", file=sys.stderr)
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m stringforge.vulcan",
        description="Vulcan production vacuum-forging CLI.",
    )
    subs = parser.add_subparsers(dest="command", required=True)

    status = subs.add_parser("status", help="Inspect the staging directory.")
    _add_common(status)
    status.set_defaults(func=_cmd_status)

    sync = subs.add_parser(
        "sync",
        help="Batch staged shards into HuggingFace commits.",
    )
    _add_common(sync)
    sync.add_argument(
        "--repo",
        default=None,
        help="Target HF repo (or STRINGFORGE_VULCAN_REPO).",
    )
    sync.add_argument("--budget", type=int, default=DEFAULT_BUDGET,
                      help=f"Commits-per-hour ceiling (default: {DEFAULT_BUDGET}).")
    sync.add_argument("--window-s", type=float, default=DEFAULT_WINDOW_S,
                      help=f"Rolling-window seconds (default: {DEFAULT_WINDOW_S}).")
    sync.add_argument("--max-batch", type=int, default=DEFAULT_MAX_BATCH,
                      help=f"Files per commit (default: {DEFAULT_MAX_BATCH}).")
    sync.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES,
                      help=f"Per-batch retries on 429 (default: {DEFAULT_MAX_RETRIES}).")
    sync.add_argument("--create-pr", action="store_true",
                      help="Open PRs instead of pushing to main.")
    sync.add_argument("--dry-run", action="store_true",
                      help="Skip the HF call; reserve budget slots only.")
    sync.set_defaults(func=_cmd_sync)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
