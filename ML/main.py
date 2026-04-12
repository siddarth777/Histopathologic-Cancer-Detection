
import argparse
import csv
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

SCRIPTS = {
    "logistic_regression": PROJECT_ROOT / "models" / "logistic_regression.py",
    "random_forest": PROJECT_ROOT / "models" / "random_forest.py",
    "naive_bayes": PROJECT_ROOT / "models" / "naive_bayes.py",
    "xgboost": PROJECT_ROOT / "models" / "xgboost_model.py",
    "catboost": PROJECT_ROOT / "models" / "catboost_model.py",
    "svm": PROJECT_ROOT / "models" / "svm.py",
    "ensemble": PROJECT_ROOT / "run_ensemble_models.py",
}


def run_python_script(script_path: Path, args: list[str]) -> None:
    command = [sys.executable, str(script_path), *args]
    print(f"\n[RUN] {' '.join(command)}")
    subprocess.run(command, check=True, cwd=PROJECT_ROOT)


def get_best_model_from_results(results_csv: Path) -> tuple[str, str, float]:
    if not results_csv.exists():
        raise FileNotFoundError(f"Results file not found: {results_csv}")

    with open(results_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"model", "features", "roc_auc"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required columns in {results_csv}: {sorted(missing)}")

        best_row = None
        best_auc = float("-inf")
        for row in reader:
            auc = float(row["roc_auc"])
            if auc > best_auc:
                best_auc = auc
                best_row = row

    if best_row is None:
        raise ValueError(f"No rows found in results CSV: {results_csv}")

    return str(best_row["model"]), str(best_row["features"]), float(best_row["roc_auc"])


def normalize_model_name(model_name: str) -> str:
    if model_name.startswith("ensemble_"):
        return "ensemble"
    return model_name


def build_common_args(parsed: argparse.Namespace, result_file: str, include_trials: bool) -> list[str]:
    args = [
        "--train_path", parsed.train_path,
        "--test_path", parsed.test_path,
        "--selected_features_path", parsed.selected_features_path,
        "--result_file", result_file,
    ]
    if include_trials:
        args.extend(["--n_trials", str(parsed.n_trials)])
    return args


def run_all(parsed: argparse.Namespace) -> None:
    model_order = [
        "logistic_regression",
        "random_forest",
        "naive_bayes",
        "xgboost",
        "catboost",
        "svm",
    ]

    for model_name in model_order:
        script = SCRIPTS[model_name]
        result_file = f"results_{model_name}.txt"
        include_trials = model_name != "naive_bayes"
        args = build_common_args(parsed, result_file, include_trials)
        run_python_script(script, args)

    if parsed.include_ensemble:
        ensemble_args = [
            "--train_path", parsed.train_path,
            "--test_path", parsed.test_path,
            "--selected_features_path", parsed.selected_features_path,
            "--result_file", parsed.ensemble_result_file,
            "--n_trials_model", str(parsed.n_trials_model),
            "--n_trials_weights", str(parsed.n_trials_weights),
        ]
        run_python_script(SCRIPTS["ensemble"], ensemble_args)


def run_best(parsed: argparse.Namespace) -> None:
    best_model_raw, best_features, best_auc = get_best_model_from_results(Path(parsed.results_csv))
    best_model = normalize_model_name(best_model_raw)

    print(
        f"[BEST] model={best_model_raw} | normalized={best_model} | "
        f"features={best_features} | roc_auc={best_auc:.6f}"
    )

    if best_model not in SCRIPTS:
        raise ValueError(f"Unsupported best model from results CSV: {best_model_raw}")

    if best_model == "ensemble":
        ensemble_args = [
            "--train_path", parsed.train_path,
            "--test_path", parsed.test_path,
            "--selected_features_path", parsed.selected_features_path,
            "--result_file", parsed.ensemble_result_file,
            "--n_trials_model", str(parsed.n_trials_model),
            "--n_trials_weights", str(parsed.n_trials_weights),
        ]
        run_python_script(SCRIPTS["ensemble"], ensemble_args)
        return

    result_file = f"results_{best_model}.txt"
    include_trials = best_model != "naive_bayes"
    args = build_common_args(parsed, result_file, include_trials)
    run_python_script(SCRIPTS[best_model], args)


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment runner")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run-all", action="store_true", help="Run all experiments")
    mode.add_argument(
        "--run-best",
        action="store_true",
        help="Run only the best model based on highest roc_auc in --results-csv",
    )

    parser.add_argument("--train_path", required=True, help="Path to training CSV")
    parser.add_argument("--test_path", required=True, help="Path to test CSV")
    parser.add_argument("--selected_features_path", required=True, help="Path to selected-features CSV")

    parser.add_argument(
        "--results-csv",
        default="results/results_full96.csv",
        help="CSV used to pick the best model when --run-best is set",
    )
    parser.add_argument("--n-trials", type=int, default=10, help="Optuna trials for single-model scripts")
    parser.add_argument("--n-trials-model", type=int, default=10, help="Model-tuning trials for ensemble script")
    parser.add_argument("--n-trials-weights", type=int, default=10, help="Weight-tuning trials for ensemble script")
    parser.add_argument(
        "--include-ensemble",
        action="store_true",
        help="Also run run_ensemble_models.py when using --run-all",
    )
    parser.add_argument(
        "--ensemble-result-file",
        default="results_ensemble.txt",
        help="Output text file for ensemble results",
    )

    args = parser.parse_args()

    if args.run_all:
        run_all(args)
    else:
        run_best(args)

    print("\n[DONE] experiment run complete")


if __name__ == "__main__":
    main()
