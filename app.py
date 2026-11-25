"""Flask web application and JSON API for the Random Forest Disease Prediction System."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from flask import Flask, jsonify, render_template, request

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from disease_predictor.model import DEFAULT_MODEL_PATH  # noqa: E402
from disease_predictor.predictor import DiseasePredictor  # noqa: E402

SAMPLE_PRESETS = [
    {
        "name": "Flu-like Syndrome",
        "symptoms": ["high_fever", "chills", "muscle_aches", "fatigue", "cough"],
    },
    {
        "name": "Respiratory Wheeze",
        "symptoms": ["wheezing", "shortness_of_breath", "chest_tightness", "dry_cough"],
    },
    {
        "name": "Migraine Episode",
        "symptoms": ["throbbing_headache", "sensitivity_to_light", "nausea", "blurred_vision"],
    },
    {
        "name": "GI Distress",
        "symptoms": ["diarrhea", "nausea", "vomiting", "abdominal_pain", "dehydration"],
    },
    {
        "name": "Metabolic Signs",
        "symptoms": ["excessive_thirst", "frequent_urination", "fatigue", "blurred_vision"],
    },
]


def create_app(model_path: Path | str = DEFAULT_MODEL_PATH) -> Flask:
    """Create and configure the Flask application."""
    app = Flask(
        __name__,
        template_folder=str(PROJECT_ROOT / "templates"),
        static_folder=str(PROJECT_ROOT / "static"),
    )

    predictor = DiseasePredictor(model_path=model_path, auto_train=True)
    app.config["PREDICTOR"] = predictor

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/api/health")
    def health():
        return jsonify({"status": "ok", "algorithm": "RandomForestClassifier"})

    @app.get("/api/symptoms")
    def get_symptoms():
        return jsonify(
            {
                "categories": predictor.get_symptom_catalog(),
                "total_symptoms": len(predictor.feature_names),
                "total_diseases": len(predictor.classes),
                "presets": SAMPLE_PRESETS,
            }
        )

    @app.get("/api/metrics")
    def get_metrics():
        return jsonify(predictor.metrics)

    @app.post("/api/predict")
    def predict():
        payload = request.get_json(silent=True) or {}
        symptoms = payload.get("symptoms", [])
        top_k = payload.get("top_k", 5)

        if not isinstance(symptoms, list):
            return jsonify({"error": "'symptoms' must be a list of symptom strings."}), 400

        try:
            top_k_int = max(1, min(int(top_k), len(predictor.classes)))
        except (TypeError, ValueError):
            top_k_int = 5

        try:
            result = predictor.predict(symptoms, top_k=top_k_int)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

        return jsonify(result)

    return app


app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    host = "0.0.0.0" if "PORT" in os.environ else "127.0.0.1"
    app.run(host=host, port=port, debug=False)
