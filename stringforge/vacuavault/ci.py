r"""
Server-side PR validation for the ``vacua_vault`` repo.

Called by CI on every PR.  Parses the git diff for added/modified
parquet files, re-validates them (no trust in client-side validation),
and writes:

- ``validation_report.json`` — machine-readable per-file results.
- ``catalog_preview.md`` — human-readable summary for a PR comment.

Exit code is 0 on pass, 1 on fail (blocks merge).
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from .schema import validate_parquet_file, SCHEMA_VERSION


def _git(args: list[str], cwd: str | Path) -> str:
    res = subprocess.run(
        ["git", *args], cwd=str(cwd),
        capture_output=True, text=True, check=False,
    )
    return res.stdout.strip()


def _changed_parquet_paths(
    repo_path: str | Path, base_branch: str,
) -> List[Path]:
    r"""Return Paths of added/modified parquet files in the diff."""
    out = _git(
        ["diff", "--name-only", f"origin/{base_branch}...HEAD", "--",
         "*.parquet"],
        cwd=repo_path,
    )
    files: List[Path] = []
    for line in out.splitlines():
        p = Path(repo_path) / line.strip()
        if p.exists() and p.suffix == ".parquet":
            files.append(p)
    return files


def validate_pr_diff(
    repo_path: str | Path,
    base_branch: str = "main",
    report_path: str = "validation_report.json",
    catalog_preview_path: str = "catalog_preview.md",
    model_loader: Optional[Any] = None,
    verbose: bool = True,
) -> int:
    r"""
    **Description:**
    Validate every added/changed parquet in the PR diff.

    Args:
        repo_path: Path to the checked-out vault repo.
        base_branch: Base branch of the PR (usually ``main``).
        report_path: Where to write ``validation_report.json``.
        catalog_preview_path: Where to write the Markdown preview
            posted as a PR comment.
        model_loader: Optional callable ``(ks_id, triang_id, cicy_id)
            → model`` for F-term physics checks.  When omitted, only
            schema/structural checks are performed.
        verbose: Print progress.

    Returns:
        int: Exit code (0 = all green, 1 = at least one failure).
    """
    repo = Path(repo_path).resolve()
    changed = _changed_parquet_paths(repo, base_branch)

    if verbose:
        print(f"[ci] {len(changed)} changed parquet file(s) in PR")

    per_file: List[Dict[str, Any]] = []
    any_failed = False

    for p in changed:
        rel = p.relative_to(repo)
        model = None
        if model_loader is not None:
            # Infer identity from the path
            parts = rel.parts
            try:
                if parts[0] == "tdf":
                    ks_tri = parts[2]  # ks_X_tri_Y
                    import re
                    m = re.match(r"ks_(\d+)_tri_(\d+)", ks_tri)
                    if m:
                        model = model_loader(
                            ks_id=int(m.group(1)),
                            triang_id=int(m.group(2)),
                        )
                elif parts[0] == "cicy":
                    import re
                    m = re.match(r"cicy_(\d+)", parts[1])
                    if m:
                        model = model_loader(cicy_id=int(m.group(1)))
            except Exception as e:
                if verbose:
                    print(f"[ci] model_loader failed for {rel}: {e}")

        result = validate_parquet_file(
            p, model=model, expected_schema_version=SCHEMA_VERSION,
        )
        entry = {
            "file_path":      str(rel),
            "passed":         result["passed"],
            "n_rows":         result["n_rows"],
            "n_failed":       result["n_failed"],
            "errors":         result["errors"],
            "warnings":       result["warnings"],
            "metadata":       result["metadata"],
        }
        if not result["passed"]:
            any_failed = True
            entry["per_row_errors"] = [
                {"index": r["index"], "errors": r["errors"]}
                for r in result["per_row_report"] if not r["passed"]
            ]
        per_file.append(entry)
        if verbose:
            flag = "✓" if result["passed"] else "✗"
            print(f"[ci] {flag} {rel}  rows={result['n_rows']} "
                  f"failed={result['n_failed']}")

    # Write JSON report
    report = {
        "pr_base":          base_branch,
        "schema_version":   SCHEMA_VERSION,
        "n_files":          len(changed),
        "n_failed_files":   sum(1 for e in per_file if not e["passed"]),
        "per_file":         per_file,
    }
    rp = Path(report_path)
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(json.dumps(report, indent=2))

    # Write Markdown preview
    lines: List[str] = ["## vacua_vault CI preview", ""]
    if not changed:
        lines.append("No parquet file changes in this PR.")
    else:
        total_rows = sum(e["n_rows"] for e in per_file)
        n_fail     = sum(1 for e in per_file if not e["passed"])
        lines.append(
            f"- **{len(changed)}** file(s) changed, "
            f"**{total_rows:,}** total rows."
        )
        lines.append(
            f"- **{n_fail}** file(s) failed validation."
            if n_fail else "- All files passed validation."
        )
        lines.append("")
        lines.append("| File | Rows | Status |")
        lines.append("| --- | --- | --- |")
        for e in per_file:
            status = "✅ pass" if e["passed"] else "❌ fail"
            lines.append(f"| `{e['file_path']}` | {e['n_rows']} | {status} |")
        # Detailed errors
        for e in per_file:
            if not e["passed"]:
                lines.append("")
                lines.append(f"### Errors in `{e['file_path']}`")
                for err in e["errors"]:
                    lines.append(f"- {err}")
    cp = Path(catalog_preview_path)
    cp.parent.mkdir(parents=True, exist_ok=True)
    cp.write_text("\n".join(lines) + "\n")

    return 1 if any_failed else 0
