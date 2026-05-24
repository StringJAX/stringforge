import pandas as pd

from stringforge.lcs_database import LCSDatabase


def test_kklt_dataset_dispatch_accepts_positional_dataset(tmp_path):
    db = LCSDatabase("kklt_vacua", cache_dir=str(tmp_path), offline=True)

    assert type(db).__name__ == "KKLTDatabase"
    assert db.dataset == "kklt_vacua"
    assert hasattr(db, "_tdf_seed")


def test_query_exposes_hodge_numbers_in_mirror_convention(tmp_path, monkeypatch):
    db = LCSDatabase(dataset="tdf", cache_dir=str(tmp_path), offline=True)
    db._catalog = pd.DataFrame(
        [
            {
                "h11": 2,
                "h12": 272,
                "chi": -540,
                "ks_id": 10,
                "triang_id": 0,
                "has_gv": True,
                "n_conifolds": 3,
            },
            {
                "h11": 3,
                "h12": 111,
                "chi": -216,
                "ks_id": 11,
                "triang_id": 0,
                "has_gv": False,
                "n_conifolds": 0,
            },
        ]
    )
    monkeypatch.setattr(db, "_ensure_catalog", lambda: None)

    result = db.query(h11=272, h12=2, has_conifolds=True)

    assert list(result[["h11", "h12", "ks_id", "triang_id"]].iloc[0]) == [
        272,
        2,
        10,
        0,
    ]
    assert "chi" not in result.columns
