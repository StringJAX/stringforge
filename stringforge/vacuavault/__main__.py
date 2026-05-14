r"""
CLI dispatcher for the ``stringforge.vacuavault`` tooling.

Usage:

    python -m stringforge.vacuavault validate [--base-branch main] [--repo-path .]
    python -m stringforge.vacuavault rebuild_catalog [--repo-path .] [--output catalog.parquet]
    python -m stringforge.vacuavault curate <community_file> [--repo-root .] [--new-label LABEL] [--dry-run]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .ci      import validate_pr_diff
from .catalog import rebuild_catalog
from .curate  import curate_submission


def _cmd_validate(args: argparse.Namespace) -> int:
    # Pure schema-only validation — this CLI imports nothing from
    # jaxvacua.  Physics-aware variants live in a separate
    # downstream CLI (e.g. ``jaxvacua-vault validate
    # --check-physics``), which can pass a populated
    # ``model_loader`` to :func:`validate_pr_diff` directly.
    return validate_pr_diff(
        repo_path=args.repo_path,
        base_branch=args.base_branch,
        report_path=args.report_path,
        catalog_preview_path=args.preview_path,
        model_loader=None,
        verbose=True,
    )


def _cmd_rebuild(args: argparse.Namespace) -> int:
    rebuild_catalog(
        repo_path=args.repo_path,
        output=args.output,
        verbose=True,
    )
    return 0


def _cmd_curate(args: argparse.Namespace) -> int:
    target = curate_submission(
        community_path=args.file,
        repo_root=args.repo_root,
        new_label=args.new_label,
        dry_run=args.dry_run,
        verbose=True,
    )
    print(f"curated → {target}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m stringforge.vacuavault",
        description="Tooling for the HuggingFace vacua_vault dataset repo.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # validate
    pv = sub.add_parser(
        "validate",
        help="Validate parquet files in a PR diff (CI entry point).",
    )
    pv.add_argument("--repo-path",    default=".",  help="Vault repo root (default: cwd).")
    pv.add_argument("--base-branch",  default="main",
                    help="PR base branch (default: main).")
    pv.add_argument("--report-path",  default="validation_report.json")
    pv.add_argument("--preview-path", default="catalog_preview.md")
    pv.set_defaults(func=_cmd_validate)

    # rebuild_catalog
    pr = sub.add_parser(
        "rebuild_catalog",
        help="Regenerate catalog.parquet from the repo contents.",
    )
    pr.add_argument("--repo-path", default=".")
    pr.add_argument("--output",    default=None,
                    help="Output path (default: <repo>/catalog.parquet).")
    pr.set_defaults(func=_cmd_rebuild)

    # curate
    pc = sub.add_parser(
        "curate",
        help="Promote a community/ submission up to the model directory.",
    )
    pc.add_argument("file", help="Path to the file under community/.")
    pc.add_argument("--repo-root", default=".")
    pc.add_argument("--new-label", default=None,
                    help="Override the target filename stem.")
    pc.add_argument("--dry-run", action="store_true")
    pc.set_defaults(func=_cmd_curate)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
