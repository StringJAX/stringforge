# Copyright 2026 Andreas Schachner
#
# This file is part of StringForge.
#
# StringForge is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
r"""
Cross-package compatibility test: a freshly written Vulcan shard
must pass :func:`stringforge.vacuavault.schema.validate_parquet_file`
without errors and without warnings about missing schema-version
metadata (finding #14).

This is a structural contract: the Vulcan production tier and the
curated VacuaVault tier share a schema-version (both at version 1 as
of 2026-05), so a shard produced by the writer must be readable by
the vault-floor validator -- a precondition for any future promotion
step.

The test never contacts HuggingFace; the writer runs locally and the
validator inspects an on-disk parquet file.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent))

from stringforge.vacuavault.schema import validate_parquet_file
from stringforge.vulcan import Vulcan


def _vacuum_df(n: int = 3) -> pd.DataFrame:
    return pd.DataFrame({
        "flux": [[1, 0, -2, 3, 0, 1]] * n,
        "moduli_re": [[0.0, 0.0]] * n,
        "moduli_im": [[2.5, 3.0]] * n,
        "tau_re": [0.0] * n,
        "tau_im": [4.0] * n,
    })


def test_vulcan_shard_passes_vault_floor_validation(tmp_path):
    r"""
    Write a single Vulcan shard, run it through the vault-floor
    validator with ``physics_checks="off"`` (so no jaxvacua-style
    physics validation is attempted), and confirm the validator
    reports success.

    The schema-version metadata key is unconditionally written by
    :mod:`stringforge.vulcan.writer`; the validator must therefore
    *not* emit the "no 'schema_version' metadata" warning that it
    appends when the key is absent.
    """
    forge = Vulcan(
        repo="user/repo",
        staging_dir=tmp_path,
        run_id_template="run-{seq}",
        project="t",
    )
    geom = {"h11": 3, "h12": 2, "ks_id": 1, "triang_id": 0}
    shard = forge.write(
        _vacuum_df(), geometry=geom, tadpole_charge=12, run_id="run-vlt-1",
    )

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        result = validate_parquet_file(
            shard.parquet_path,
            physics_checks="off",
        )

    assert result["passed"] is True, (
        f"vault-floor validator rejected a Vulcan shard: "
        f"errors={result['errors']!r}, warnings={result['warnings']!r}"
    )
    assert len(result["errors"]) == 0
    assert "schema_version" in (result["metadata"] or {})

    # The validator must not warn about missing schema-version
    # metadata; the Vulcan writer always writes it.
    missing_sv_msgs = [
        w for w in captured
        if "no 'schema_version' metadata" in str(w.message)
    ]
    assert missing_sv_msgs == [], (
        f"unexpected schema_version warnings emitted: "
        f"{[str(w.message) for w in missing_sv_msgs]}"
    )
    # The same message can also surface in the validator's own
    # ``warnings`` list -- check that channel too.
    assert not any(
        "no 'schema_version' metadata" in w for w in result["warnings"]
    ), f"validator reported missing schema_version: {result['warnings']!r}"
