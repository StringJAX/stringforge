import pandas as pd

from stringforge.lcs_database import LCSDatabase


def _vacuum_dataframe():
    return pd.DataFrame(
        {
            "flux": [[1, 0, -2, 3, 0, 1]],
            "moduli_re": [[0.0, 0.0]],
            "moduli_im": [[2.5, 3.0]],
            "tau_re": [0.0],
            "tau_im": [4.0],
            "is_susy": [True],
        }
    )


def test_tdf_designated_paths_include_hodge_sector(tmp_path, monkeypatch):
    monkeypatch.setenv("STRINGFORGE_VAULT", str(tmp_path / "vault"))
    db = LCSDatabase(dataset="tdf", cache_dir=str(tmp_path / "cache"), offline=True)

    path_a = db._resolve_vacua_dir(h11=5, h12=2, ks_id=10, triang_id=0)
    path_b = db._resolve_vacua_dir(h11=6, h12=2, ks_id=10, triang_id=0)

    assert path_a != path_b
    assert "h11_5" in str(path_a)
    assert "h11_6" in str(path_b)
    assert "ks_10" in str(path_a)
    assert "tri_0" in str(path_a)


def test_duplicate_flux_designation_is_scoped_to_model_identity(tmp_path, monkeypatch):
    monkeypatch.setenv("STRINGFORGE_VAULT", str(tmp_path / "vault"))
    db = LCSDatabase(dataset="tdf", cache_dir=str(tmp_path / "cache"), offline=True)

    first_ids = db.designate_vacua(
        _vacuum_dataframe(),
        label="model_a",
        committed_by="A. Schachner",
        h11=5,
        h12=2,
        ks_id=10,
        triang_id=0,
        validate=True,
    )
    second_ids = db.designate_vacua(
        _vacuum_dataframe(),
        label="model_b",
        committed_by="A. Schachner",
        h11=6,
        h12=2,
        ks_id=11,
        triang_id=0,
        validate=True,
    )

    assert first_ids == [0]
    assert second_ids == [1]
