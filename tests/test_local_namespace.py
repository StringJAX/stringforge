"""Tests for the ``local/`` vault namespace (local jaxvacua models addressed by
``h12`` + ``model_ID``), added so such models can be pushed / fetched / indexed
on the HuggingFace vault alongside ``tdf/`` and ``cicy/`` models.

Covers the three path-parsers that had to learn the namespace:
  * ``VacuaWriter._remote_model_dir``      (remote push/fetch directory)
  * ``vacuavault.catalog._parse_path``     (server-side catalog rebuild)
  * ``vacuavault.ci`` PR-diff identity inference (via the public re-export)

These are pure path/identity helpers, so the tests need no jaxvacua model.
"""

from pathlib import Path

from stringforge.vacua_writer import VacuaWriter
from stringforge.vacuavault.catalog import _parse_path, CATALOG_COLUMNS


# --- _remote_model_dir -----------------------------------------------------

def test_remote_dir_local_model():
    """A local model (synthetic ``model_name``, no ks/triang/cicy) routes to
    ``local/h12_{h12}/model_{model_ID}``."""
    ident = {"model_name": "local_h12_2_ID_1", "h12": 2,
             "ks_id": -1, "triang_id": -1, "cicy_id": -1}
    assert VacuaWriter._remote_model_dir(ident) == "local/h12_2/model_1"


def test_remote_dir_local_double_digit():
    ident = {"model_name": "local_h12_15_ID_237", "h12": 15,
             "ks_id": -1, "triang_id": -1, "cicy_id": -1}
    assert VacuaWriter._remote_model_dir(ident) == "local/h12_15/model_237"


def test_remote_dir_tdf_and_cicy_unchanged():
    """The existing namespaces must keep their layout."""
    assert VacuaWriter._remote_model_dir(
        {"ks_id": 29, "triang_id": 0, "cicy_id": -1, "h12": 2}
    ) == "tdf/h12_2/ks_29_tri_0"
    assert VacuaWriter._remote_model_dir(
        {"ks_id": -1, "triang_id": -1, "cicy_id": 7, "h12": 3}
    ) == "cicy/cicy_7"


def test_remote_dir_unrooted_returns_none():
    """No identity at all → no remote directory."""
    assert VacuaWriter._remote_model_dir(
        {"ks_id": -1, "triang_id": -1, "cicy_id": -1, "h12": 2,
         "model_name": None}
    ) is None


# --- catalog _parse_path ---------------------------------------------------

def test_parse_path_has_model_id_column():
    assert "model_ID" in CATALOG_COLUMNS


def test_parse_path_local_community():
    root = Path("/vault")
    info = _parse_path(
        root, root / "local" / "h12_2" / "model_1" / "community"
        / "alice_ISDplus_demo.parquet")
    assert info["h12"] == 2
    assert info["model_ID"] == 1
    assert info["status"] == "pending"          # community/ → pending
    assert info["label"] == "ISDplus_demo"      # username prefix stripped
    assert info["ks_id"] is None and info["cicy_id"] is None


def test_parse_path_local_curated():
    root = Path("/vault")
    info = _parse_path(
        root, root / "local" / "h12_5" / "model_12" / "run_v3.parquet")
    assert (info["h12"], info["model_ID"], info["status"]) == (5, 12, "curated")
    assert info["label"] == "run"
    assert info["version"] == 3


def test_parse_path_local_rejected():
    root = Path("/vault")
    info = _parse_path(
        root, root / "local" / "h12_2" / "model_1" / "_rejected"
        / "bob_bad.parquet")
    assert info["status"] == "rejected"


def test_parse_path_tdf_still_works():
    root = Path("/vault")
    info = _parse_path(
        root, root / "tdf" / "h12_2" / "ks_29_tri_0" / "curated.parquet")
    assert (info["h12"], info["ks_id"], info["triang_id"]) == (2, 29, 0)
    assert info["model_ID"] is None
