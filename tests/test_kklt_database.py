import pandas as pd

from stringforge.kklt_database import KKLTDatabase


def test_run_log_attempt_index_counts_attempts_not_rows(tmp_path):
    db = KKLTDatabase(cache_dir=str(tmp_path), offline=True)

    run_id = db.start_run(
        "class",
        ks_id=384564,
        coni_class_id=0,
        task="scan",
        host="test-host",
    )
    db.finish_run(run_id, "done", n_solutions=1)
    db.start_run(
        "class",
        ks_id=384564,
        coni_class_id=0,
        task="scan",
        host="test-host",
    )

    runs = db.list_runs(ks_id=384564, coni_class_id=0, task="scan")

    assert runs["attempt_index"].tolist() == [1, 1, 2]
    assert runs["status"].tolist() == ["running", "done", "running"]


def test_stage_fragment_roundtrip_stamps_identity_and_extra_columns(tmp_path):
    db = KKLTDatabase(cache_dir=str(tmp_path), offline=True)

    path = db.write_pfvs(
        pd.DataFrame({"flux": [[1, 0, -1, 2]], "score": [0.25]}),
        kind="harvested",
        h11=272,
        h12=2,
        ks_id=384564,
        coni_class_id=0,
        coni_id=75,
        run_id="run-a",
        extra_columns={"solver": "isd", "window": [1, 2, 3]},
    )
    loaded = db.load_pfvs(
        kind="harvested",
        h11=272,
        h12=2,
        ks_id=384564,
        coni_class_id=0,
        coni_id=75,
        run_id="run-a",
    )

    assert path.exists()
    assert len(loaded) == 1
    assert loaded.loc[0, "h11"] == 272
    assert loaded.loc[0, "h12"] == 2
    assert loaded.loc[0, "ks_id"] == 384564
    assert loaded.loc[0, "coni_class_id"] == 0
    assert loaded.loc[0, "coni_id"] == 75
    assert loaded.loc[0, "run_id"] == "run-a"
    assert loaded.loc[0, "solver"] == "isd"
    assert list(loaded.loc[0, "window"]) == [1, 2, 3]
