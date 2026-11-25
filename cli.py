"""Command-line interface for training and querying the Random Forest Disease Predictor."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from disease_predictor.model import DEFAULT_MODEL_PATH, train_and_evaluate  # noqa: E402
from disease_predictor.predictor import DiseasePredictor  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="AI/ML Disease Prediction System using Random Forest"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_cmd = subparsers.add_parser("train", help="Generate/load dataset and train Random Forest")
    train_cmd.add_argument("--dataset", type=str, default=None, help="Optional path to custom CSV dataset")
    train_cmd.add_argument("--n-estimators", type=int, default=300, help="Number of trees in Random Forest")
    train_cmd.add_argument("--regenerate", action="store_true", help="Regenerate synthetic dataset before training")

    predict_cmd = subparsers.add_parser("predict", help="Predict disease from comma-separated symptoms")
    predict_cmd.add_argument(
        "--symptoms",
        type=str,
        required=True,
        help="Comma-separated symptoms, e.g. 'high_fever,chills,muscle_aches,cough'",
    )
    predict_cmd.add_argument("--top-k", type=int, default=5, help="Number of top predictions to display")
    predict_cmd.add_argument("--json", action="store_true", help="Print raw JSON output")

    subparsers.add_parser("interactive", help="Interactive terminal symptom checker")
    return parser


def run_interactive(predictor: DiseasePredictor) -> None:
    print("\n=== Random Forest Disease Prediction System (Interactive Mode) ===")
    print("Type 'list' to view available symptoms, or enter comma-separated symptoms.")
    print("Type 'quit' to exit.\n")

    while True:
        user_input = input("Symptoms > ").strip()
        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit", "q"}:
            print("Goodbye.")
            break
        if user_input.lower() == "list":
            for group in predictor.get_symptom_catalog():
                keys = ", ".join(item["key"] for item in group["symptoms"])
                print(f"\n[{group['category']}]\n  {keys}")
            print()
            continue

        symptoms = [item.strip() for item in user_input.split(",") if item.strip()]
        try:
            result = predictor.predict(symptoms, top_k=5)
        except ValueError as exc:
            print(f"Error: {exc}\n")
            continue

        top = result["top_prediction"]
        print(f"\nTop Prediction: {top['disease']} ({top['confidence_pct']}% | Urgency: {top['urgency']})")
        print(f"Specialist:     {top['specialist']}")
        print(f"Description:    {top['description']}")
        print("Differential Probabilities:")
        for pred in result["predictions"]:
            print(f"  - {pred['disease']}: {pred['confidence_pct']}%")
        if result["unknown_symptoms"]:
            print(f"Unrecognized inputs ignored: {result['unknown_symptoms']}")
        print(f"Disclaimer: {result['disclaimer']}\n")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "train":
        kwargs = {
            "n_estimators": args.n_estimators,
            "regenerate_data": args.regenerate,
        }
        if args.dataset:
            kwargs["dataset_path"] = Path(args.dataset)
        metrics = train_and_evaluate(**kwargs)
        print(json.dumps(metrics, indent=2))
        return

    predictor = DiseasePredictor(model_path=DEFAULT_MODEL_PATH, auto_train=True)

    if args.command == "predict":
        symptoms = [s.strip() for s in args.symptoms.split(",") if s.strip()]
        result = predictor.predict(symptoms, top_k=args.top_k)
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            top = result["top_prediction"]
            print(f"\nPredicted Disease : {top['disease']} ({top['confidence_pct']}%)")
            print(f"Confidence Band   : {top['confidence_band']}")
            print(f"Urgency           : {top['urgency']}")
            print(f"Recommended Care  : {top['specialist']}")
            print(f"Description       : {top['description']}")
            print("\nPrecautions:")
            for step in top["precautions"]:
                print(f"  * {step}")
            print("\nTop Differential Diagnoses:")
            for item in result["predictions"]:
                print(f"  - {item['disease']}: {item['confidence_pct']}%")
            print(f"\nNote: {result['disclaimer']}")
        return

    if args.command == "interactive":
        run_interactive(predictor)


if __name__ == "__main__":
    main()
