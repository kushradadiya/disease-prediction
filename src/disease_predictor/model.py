"""Random Forest training, cross-validation, evaluation, and model persistence."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from .catalog import format_symptom_label
from .data_loader import DEFAULT_DATASET_PATH, generate_dataset, load_dataset

DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[2] / "models"
DEFAULT_MODEL_PATH = DEFAULT_MODEL_DIR / "random_forest_model.joblib"
DEFAULT_METRICS_PATH = DEFAULT_MODEL_DIR / "metrics.json"


def train_and_evaluate(
    dataset_path: Path | str = DEFAULT_DATASET_PATH,
    model_path: Path | str = DEFAULT_MODEL_PATH,
    metrics_path: Path | str = DEFAULT_METRICS_PATH,
    n_estimators: int = 300,
    max_depth: int | None = None,
    min_samples_leaf: int = 1,
    test_size: float = 0.2,
    cv_folds: int = 5,
    random_state: int = 42,
    regenerate_data: bool = False,
) -> dict[str, Any]:
    """Train a Random Forest disease classifier, evaluate it, and save artifacts."""
    dataset_file = Path(dataset_path)
    if regenerate_data or not dataset_file.exists():
        generate_dataset(output_csv=dataset_file, seed=random_state)

    X, y = load_dataset(dataset_file)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight="balanced",
        oob_score=True,
        n_jobs=-1,
        random_state=random_state,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    test_accuracy = float(accuracy_score(y_test, y_pred))
    test_precision = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
    test_recall = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
    test_f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))

    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    cv_scores = cross_val_score(
        RandomForestClassifier(
            n_estimators=min(n_estimators, 200),
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            class_weight="balanced",
            n_jobs=-1,
            random_state=random_state,
        ),
        X,
        y,
        cv=skf,
        scoring="accuracy",
    )

    # Refit final production model on full dataset for maximum predictive coverage.
    production_clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight="balanced",
        oob_score=True,
        n_jobs=-1,
        random_state=random_state,
    )
    production_clf.fit(X, y)

    importances = production_clf.feature_importances_
    top_indices = np.argsort(importances)[::-1][:15]
    top_features = [
        {
            "symptom": X.columns[idx],
            "label": format_symptom_label(X.columns[idx]),
            "importance": round(float(importances[idx]), 4),
        }
        for idx in top_indices
    ]

    per_class = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    metrics: dict[str, Any] = {
        "algorithm": "RandomForestClassifier",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "path": str(dataset_file),
            "total_samples": int(len(X)),
            "train_samples": int(len(X_train)),
            "test_samples": int(len(X_test)),
            "num_features": int(X.shape[1]),
            "num_diseases": int(y.nunique()),
        },
        "hyperparameters": {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf,
            "class_weight": "balanced",
            "oob_score": True,
            "random_state": random_state,
        },
        "evaluation": {
            "test_accuracy": round(test_accuracy, 4),
            "test_precision_weighted": round(test_precision, 4),
            "test_recall_weighted": round(test_recall, 4),
            "test_f1_weighted": round(test_f1, 4),
            "holdout_oob_score": round(float(clf.oob_score_), 4),
            "production_oob_score": round(float(production_clf.oob_score_), 4),
            "cv_folds": cv_folds,
            "cv_mean_accuracy": round(float(np.mean(cv_scores)), 4),
            "cv_std_accuracy": round(float(np.std(cv_scores)), 4),
        },
        "top_feature_importances": top_features,
        "per_class_f1": {
            label: round(float(vals["f1-score"]), 4)
            for label, vals in per_class.items()
            if isinstance(vals, dict) and label not in ("macro avg", "weighted avg")
        },
    }

    model_file = Path(model_path)
    metrics_file = Path(metrics_path)
    model_file.parent.mkdir(parents=True, exist_ok=True)
    metrics_file.parent.mkdir(parents=True, exist_ok=True)

    bundle = {
        "model": production_clf,
        "feature_names": list(X.columns),
        "classes": list(production_clf.classes_),
        "metrics": metrics,
    }
    joblib.dump(bundle, model_file)

    with metrics_file.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def load_model_bundle(model_path: Path | str = DEFAULT_MODEL_PATH) -> dict[str, Any]:
    """Load the saved Random Forest model bundle from disk."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Trained model not found at {path}. Run training first."
        )
    return joblib.load(path)
