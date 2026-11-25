"""Inference engine for predicting diseases from patient symptoms."""

from __future__ import annotations

import difflib
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .catalog import DISEASE_PROFILES, SYMPTOM_CATEGORIES, format_symptom_label
from .model import DEFAULT_MODEL_PATH, load_model_bundle, train_and_evaluate

MEDICAL_DISCLAIMER = (
    "This AI/ML prediction is for educational and informational decision-support only. "
    "It is NOT a medical diagnosis. Always consult a qualified healthcare professional "
    "for clinical evaluation, testing, and treatment."
)


class DiseasePredictor:
    """Wrapper around the trained Random Forest model for structured disease inference."""

    def __init__(self, model_path: Path | str = DEFAULT_MODEL_PATH, auto_train: bool = True):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            if not auto_train:
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            train_and_evaluate(model_path=self.model_path)

        bundle = load_model_bundle(self.model_path)
        self.model = bundle["model"]
        self.feature_names: list[str] = bundle["feature_names"]
        self.classes: list[str] = bundle["classes"]
        self.metrics: dict[str, Any] = bundle.get("metrics", {})
        self._feature_index = {name: idx for idx, name in enumerate(self.feature_names)}

    def get_symptom_catalog(self) -> list[dict[str, Any]]:
        """Return symptoms grouped by clinical category for UI/API consumption."""
        known_set = set(self.feature_names)
        categorized: list[dict[str, Any]] = []
        used: set[str] = set()

        for category, symptoms in SYMPTOM_CATEGORIES.items():
            items = [
                {"key": s, "label": format_symptom_label(s)}
                for s in symptoms
                if s in known_set
            ]
            if items:
                categorized.append({"category": category, "symptoms": items})
                used.update(s["key"] for s in items)

        remaining = [
            {"key": s, "label": format_symptom_label(s)}
            for s in self.feature_names
            if s not in used
        ]
        if remaining:
            categorized.append({"category": "Other Symptoms", "symptoms": remaining})

        return categorized

    def normalize_symptoms(
        self, symptoms: Iterable[str]
    ) -> tuple[list[str], list[dict[str, Any]]]:
        """Validate and normalize input symptom names against trained model features."""
        valid: list[str] = []
        unknown: list[dict[str, Any]] = []
        seen: set[str] = set()

        for raw in symptoms:
            cleaned = str(raw).strip().lower().replace(" ", "_").replace("-", "_")
            if not cleaned:
                continue
            if cleaned in self._feature_index:
                if cleaned not in seen:
                    valid.append(cleaned)
                    seen.add(cleaned)
            else:
                suggestions = difflib.get_close_matches(
                    cleaned, self.feature_names, n=3, cutoff=0.55
                )
                unknown.append({"input": raw, "suggestions": suggestions})

        return valid, unknown

    def predict(
        self, symptoms: Iterable[str], top_k: int = 5
    ) -> dict[str, Any]:
        """Predict top-K likely diseases given a list of symptom keys or labels."""
        valid_symptoms, unknown_symptoms = self.normalize_symptoms(symptoms)
        if not valid_symptoms:
            raise ValueError(
                "Please provide at least one recognized symptom to generate a prediction."
            )

        vector = np.zeros((1, len(self.feature_names)), dtype=int)
        for sym in valid_symptoms:
            vector[0, self._feature_index[sym]] = 1

        feature_df = pd.DataFrame(vector, columns=self.feature_names)
        probabilities = self.model.predict_proba(feature_df)[0]

        ranked_indices = np.argsort(probabilities)[::-1][: max(1, top_k)]
        valid_set = set(valid_symptoms)
        importances = self.model.feature_importances_

        predictions: list[dict[str, Any]] = []
        for idx in ranked_indices:
            disease_name = str(self.classes[idx])
            prob = float(probabilities[idx])
            profile = DISEASE_PROFILES.get(disease_name, {})
            core = profile.get("core", [])
            secondary = profile.get("secondary", [])

            matched_core = [s for s in core if s in valid_set]
            matched_secondary = [s for s in secondary if s in valid_set]

            predictions.append(
                {
                    "disease": disease_name,
                    "probability": round(prob, 4),
                    "confidence_pct": round(prob * 100, 2),
                    "confidence_band": self._confidence_band(prob),
                    "description": profile.get(
                        "description",
                        "No clinical summary available for this condition.",
                    ),
                    "precautions": profile.get("precautions", []),
                    "specialist": profile.get("specialist", "General Physician"),
                    "urgency": profile.get("urgency", "Moderate"),
                    "matched_core_symptoms": [
                        format_symptom_label(s) for s in matched_core
                    ],
                    "matched_secondary_symptoms": [
                        format_symptom_label(s) for s in matched_secondary
                    ],
                    "core_coverage": (
                        round(len(matched_core) / len(core), 2) if core else None
                    ),
                }
            )

        symptom_contributions = sorted(
            [
                {
                    "symptom": sym,
                    "label": format_symptom_label(sym),
                    "global_importance": round(float(importances[self._feature_index[sym]]), 4),
                }
                for sym in valid_symptoms
            ],
            key=lambda item: item["global_importance"],
            reverse=True,
        )

        return {
            "top_prediction": predictions[0],
            "predictions": predictions,
            "input_symptoms": [
                {"key": s, "label": format_symptom_label(s)} for s in valid_symptoms
            ],
            "unknown_symptoms": unknown_symptoms,
            "symptom_contributions": symptom_contributions,
            "model_info": {
                "algorithm": self.metrics.get("algorithm", "RandomForestClassifier"),
                "n_estimators": self.metrics.get("hyperparameters", {}).get("n_estimators"),
                "test_accuracy": self.metrics.get("evaluation", {}).get("test_accuracy"),
                "cv_mean_accuracy": self.metrics.get("evaluation", {}).get("cv_mean_accuracy"),
            },
            "disclaimer": MEDICAL_DISCLAIMER,
        }

    @staticmethod
    def _confidence_band(probability: float) -> str:
        if probability >= 0.65:
            return "High"
        if probability >= 0.30:
            return "Moderate"
        return "Low"
