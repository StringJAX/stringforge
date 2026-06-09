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


def test_query_helpers_filter_scalar_tag_metadata(tmp_path):
    db = KKLTDatabase(cache_dir=str(tmp_path), offline=True)
    db._catalog = pd.DataFrame(
        [
            {
                "ks_id": 1,
                "h11": 2,
                "h12": 10,
                "chi": -16,
                "Q": 14,
                "n_rigids_dual": 3,
                "n_coni_classes": 1,
                "tags": "kklt_candidate,q_ge_100",
            },
            {
                "ks_id": 2,
                "h11": 3,
                "h12": 9,
                "chi": -12,
                "Q": 14,
                "n_rigids_dual": 4,
                "n_coni_classes": 1,
                "tags": "kklt_candidate,review_later",
            },
        ]
    )
    db._class_catalog = pd.DataFrame(
        [
            {
                "ks_id": 1,
                "coni_class_id": 0,
                "h11": 2,
                "h12": 10,
                "one_face_divisor": "[1]",
                "n_conifolds_in_class": 2,
                "tags": "kklt_conifold_class,one_face_divisor_available",
            }
        ]
    )
    db._conifold_catalog = pd.DataFrame(
        [
            {
                "ks_id": 1,
                "coni_class_id": 0,
                "coni_id": 4,
                "h11": 2,
                "h12": 10,
                "triang_id": 0,
                "tdf_conifold_id": 7,
                "tdf_status": "ok",
                "tags": "kklt_conifold,tdf_link_ok,has_kklt_gv",
            }
        ]
    )

    polys = db.query_polytopes(tags_include="q_ge_100")
    classes = db.query_classes(tags_include="one_face_divisor_available")
    conifolds = db.query_conifolds(
        tags_include=["tdf_link_ok", "has_kklt_gv"],
        tags_exclude="orphaned",
    )

    assert polys["ks_id"].tolist() == [1]
    assert classes["coni_class_id"].tolist() == [0]
    assert conifolds["coni_id"].tolist() == [4]
