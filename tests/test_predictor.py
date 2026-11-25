"""Automated tests for the Random Forest Disease Prediction System."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from app import create_app  # noqa: E402
from disease_predictor.catalog import ALL_SYMPTOMS, DISEASE_PROFILES  # noqa: E402
from disease_predictor.data_loader import generate_dataset, load_dataset  # noqa: E402
from disease_predictor.model import train_and_evaluate  # noqa: E402
from disease_predictor.predictor import DiseasePredictor  # noqa: E402


def test_catalog_symptoms_are_consistent():
    known = set(ALL_SYMPTOMS)
    assert len(known) == len(ALL_SYMPTOMS), "Duplicate symptoms found in ALL_SYMPTOMS"
    for disease, profile in DISEASE_PROFILES.items():
        for sym in profile["core"] + profile["secondary"]:
            assert sym in known, f"Unknown symptom '{sym}' in profile for '{disease}'"


def test_train_and_predict_end_to_end(tmp_path: Path):
    dataset_path = tmp_path / "disease_symptoms.csv"
    metadata_path = tmp_path / "disease_metadata.csv"
    model_path = tmp_path / "rf_model.joblib"
    metrics_path = tmp_path / "metrics.json"

    df, meta = generate_dataset(
        output_csv=dataset_path,
        metadata_csv=metadata_path,
        samples_per_disease=60,
        seed=42,
    )
    assert len(df) == len(DISEASE_PROFILES) * 60
    assert len(meta) == len(DISEASE_PROFILES)

    X, y = load_dataset(dataset_path)
    assert X.shape[1] == len(ALL_SYMPTOMS)
    assert y.nunique() == len(DISEASE_PROFILES)

    metrics = train_and_evaluate(
        dataset_path=dataset_path,
        model_path=model_path,
        metrics_path=metrics_path,
        n_estimators=120,
        cv_folds=3,
        random_state=42,
    )
    assert model_path.exists()
    assert metrics_path.exists()
    assert metrics["evaluation"]["test_accuracy"] >= 0.90
    assert metrics["evaluation"]["cv_mean_accuracy"] >= 0.90

    predictor = DiseasePredictor(model_path=model_path, auto_train=False)

    migraine_res = predictor.predict(
        ["throbbing_headache", "sensitivity_to_light", "nausea", "blurred_vision"]
    )
    assert migraine_res["top_prediction"]["disease"] == "Migraine"
    assert len(migraine_res["predictions"]) == 5

    uti_res = predictor.predict(
        ["painful_urination", "frequent_urination", "cloudy_urine"]
    )
    assert uti_res["top_prediction"]["disease"] == "Urinary Tract Infection (UTI)"

    unknown_res = predictor.predict(["high_fever", "chills", "muscle_aches", "fver"])
    assert len(unknown_res["unknown_symptoms"]) == 1
    assert "fever" in unknown_res["unknown_symptoms"][0]["suggestions"]

    with pytest.raises(ValueError):
        predictor.predict(["totally_unknown_xyz"])


def test_flask_api(tmp_path: Path):
    dataset_path = tmp_path / "api_symptoms.csv"
    model_path = tmp_path / "api_rf.joblib"
    metrics_path = tmp_path / "api_metrics.json"

    train_and_evaluate(
        dataset_path=dataset_path,
        model_path=model_path,
        metrics_path=metrics_path,
        n_estimators=100,
        cv_folds=3,
        regenerate_data=True,
    )

    flask_app = create_app(model_path=model_path)
    client = flask_app.test_client()

    assert client.get("/").status_code == 200
    assert client.get("/api/health").get_json()["status"] == "ok"

    sym_resp = client.get("/api/symptoms")
    assert sym_resp.status_code == 200
    assert sym_resp.get_json()["total_diseases"] == len(DISEASE_PROFILES)

    metrics_resp = client.get("/api/metrics")
    assert metrics_resp.status_code == 200
    assert metrics_resp.get_json()["algorithm"] == "RandomForestClassifier"

    pred_resp = client.post(
        "/api/predict",
        json={"symptoms": ["wheezing", "shortness_of_breath", "chest_tightness", "dry_cough"]},
    )
    assert pred_resp.status_code == 200
    assert pred_resp.get_json()["top_prediction"]["disease"] == "Bronchial Asthma"

    bad_resp = client.post("/api/predict", json={"symptoms": []})
    assert bad_resp.status_code == 400
