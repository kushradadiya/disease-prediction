# MediForest AI — Disease Prediction System (Random Forest)

An end-to-end AI/ML **Disease Prediction System** powered by scikit-learn's **Random Forest Classifier** (`RandomForestClassifier`), complete with dataset generation/loading, cross-validated training, a CLI tool, an interactive web interface, and automated tests.

> **Medical Disclaimer:** This project is for educational and informational decision-support purposes only. It is **not** a medical diagnosis tool. Always consult a qualified healthcare professional.

## Features

- **Random Forest Classifier (`300` trees):** Stratified train/test evaluation, 5-fold stratified cross-validation, out-of-bag (OOB) score estimation, weighted precision/recall/F1, and Gini feature importance rankings.
- **30 Diseases & 79 Categorized Symptoms:** Covers Respiratory, Systemic, Gastrointestinal, Neurological, Musculoskeletal, Dermatological, Cardiovascular/Urinary, and Ophthalmic symptom groups.
- **Clinical Guidance Metadata:** Returns Top-K differential probabilities, matched core & secondary symptoms, urgency level, recommended specialist, and practical precautions.
- **Interactive Web App (`app.py`):** Searchable symptom picker, quick clinical presets, probability distribution bars, and model metrics dashboard.
- **CLI (`cli.py`):** Train on built-in or custom CSV datasets, run single predictions, or launch an interactive terminal session.

## Project Structure

```text
d:\health\
├── app.py                              # Flask web server & REST API
├── cli.py                              # CLI for training and prediction
├── requirements.txt
├── src/disease_predictor/
│   ├── catalog.py                      # Symptom categories & disease profiles
│   ├── data_loader.py                  # Dataset generator & CSV loader
│   ├── model.py                        # Random Forest training & evaluation
│   └── predictor.py                    # Top-K inference & symptom matching
├── templates/index.html                # Web UI
├── static/style.css & static/app.js    # Web UI styling & client logic
├── data/                               # Generated CSV dataset & metadata
├── models/                             # Saved joblib model & metrics.json
└── tests/test_predictor.py             # Pytest suite
```

## Quick Start

### 1. Activate Virtual Environment

```powershell
d:\health\.venv\Scripts\Activate.ps1
```

### 2. Train the Random Forest Model

```powershell
python cli.py train
```

To train using your own CSV dataset (binary `0/1` symptom columns + `disease` or `prognosis` target column):

```powershell
python cli.py train --dataset path\to\your_dataset.csv
```

### 3. Run Predictions via CLI

```powershell
python cli.py predict --symptoms "high_fever,chills,muscle_aches,fatigue,cough"
```

Or launch interactive mode:

```powershell
python cli.py interactive
```

### 4. Launch the Web Application

```powershell
python app.py
```

Open `http://127.0.0.1:5000` in your browser.

### 5. Run Tests

```powershell
pytest -v
```
