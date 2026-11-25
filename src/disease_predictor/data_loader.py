"""Dataset generation and loading utilities for the Disease Prediction System."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from .catalog import ALL_SYMPTOMS, DISEASE_PROFILES

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data"
DEFAULT_DATASET_PATH = DEFAULT_DATA_DIR / "disease_symptoms.csv"
DEFAULT_METADATA_PATH = DEFAULT_DATA_DIR / "disease_metadata.csv"


def generate_dataset(
    output_csv: Path | str = DEFAULT_DATASET_PATH,
    metadata_csv: Path | str = DEFAULT_METADATA_PATH,
    samples_per_disease: int = 160,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate a realistic synthetic symptom-disease dataset with controlled variability.

    Each generated case samples core symptoms with high probability (0.82-0.96),
    secondary symptoms with moderate probability (0.30-0.65), and low background
    noise (0.012) to reflect real-world clinical presentation variability.
    """
    rng = np.random.default_rng(seed)
    symptom_index = {symptom: idx for idx, symptom in enumerate(ALL_SYMPTOMS)}

    rows: list[dict[str, int | str]] = []
    for disease, profile in DISEASE_PROFILES.items():
        core = [s for s in profile["core"] if s in symptom_index]
        secondary = [s for s in profile["secondary"] if s in symptom_index]

        for _ in range(samples_per_disease):
            vec = (rng.random(len(ALL_SYMPTOMS)) < 0.012).astype(int)

            for s in core:
                if rng.random() < rng.uniform(0.82, 0.96):
                    vec[symptom_index[s]] = 1

            for s in secondary:
                if rng.random() < rng.uniform(0.30, 0.65):
                    vec[symptom_index[s]] = 1

            # Ensure at least two hallmark core symptoms are present per patient sample.
            present_core = [s for s in core if vec[symptom_index[s]] == 1]
            if len(present_core) < min(2, len(core)):
                chosen = rng.choice(core, size=min(2, len(core)), replace=False)
                for s in chosen:
                    vec[symptom_index[s]] = 1

            row = {symptom: int(vec[i]) for i, symptom in enumerate(ALL_SYMPTOMS)}
            row["disease"] = disease
            rows.append(row)

    dataset_df = pd.DataFrame(rows)
    dataset_df = dataset_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    metadata_rows = [
        {
            "disease": disease,
            "description": profile["description"],
            "precautions": json.dumps(profile["precautions"]),
            "specialist": profile["specialist"],
            "urgency": profile["urgency"],
            "core_symptoms": json.dumps(profile["core"]),
            "secondary_symptoms": json.dumps(profile["secondary"]),
        }
        for disease, profile in DISEASE_PROFILES.items()
    ]
    metadata_df = pd.DataFrame(metadata_rows)

    output_path = Path(output_csv)
    metadata_path = Path(metadata_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    dataset_df.to_csv(output_path, index=False)
    metadata_df.to_csv(metadata_path, index=False)

    return dataset_df, metadata_df


def load_dataset(
    dataset_path: Path | str = DEFAULT_DATASET_PATH,
    target_column: str | None = None,
    generate_if_missing: bool = True,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Load feature matrix X and target series y from a CSV dataset."""
    path = Path(dataset_path)
    if not path.exists():
        if not generate_if_missing:
            raise FileNotFoundError(f"Dataset not found at {path}")
        generate_dataset(output_csv=path)

    df = pd.read_csv(path)
    if target_column is None:
        for candidate in ("disease", "prognosis", "target", "label"):
            if candidate in df.columns:
                target_column = candidate
                break
        if target_column is None:
            target_column = df.columns[-1]

    feature_cols = [col for col in df.columns if col != target_column]
    # Drop unnamed index columns if present in external datasets (e.g., Kaggle CSVs).
    feature_cols = [col for col in feature_cols if not col.startswith("Unnamed:")]

    X = df[feature_cols].fillna(0).astype(int)
    y = df[target_column].astype(str).str.strip()
    return X, y
