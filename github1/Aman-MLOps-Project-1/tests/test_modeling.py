from src.house_price.modeling import build_candidates, train_and_evaluate


def test_candidate_models_exist():
    assert set(build_candidates()) == {"baseline", "linear_regression", "random_forest"}


def test_training_writes_model_and_metrics(tmp_path, monkeypatch):
    import src.house_price.modeling as modeling

    monkeypatch.setattr(modeling, "ARTIFACT_DIR", tmp_path)
    monkeypatch.setattr(modeling, "MODEL_PATH", tmp_path / "model.joblib")
    monkeypatch.setattr(modeling, "METRICS_PATH", tmp_path / "metrics.json")
    report = train_and_evaluate(n_samples=120, random_state=5)
    assert report["selected_model"] in report["models"]
    assert modeling.MODEL_PATH.exists()
    assert modeling.METRICS_PATH.exists()
    assert all("mae" in scores and "rmse" in scores and "r2" in scores
               for scores in report["models"].values())
