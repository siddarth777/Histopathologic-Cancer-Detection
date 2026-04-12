import argparse
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


def _run_command(command: list[str]) -> None:
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def _run_dl_lda(data_dir: str, out_dir: str) -> None:
    """Run the known-good LDA entry command."""
    _run_command([
        sys.executable,
        '-m', 'DL_exp.src_lda',
        '--task', 'all',
        '--data-dir', data_dir,
        '--out-dir', out_dir,
    ])


def _run_dl_optuna(
    data_dir: str,
    out_dir: str,
    model: str,
    n_trials: int,
    n_jobs: int,
    epochs: int,
    timeout_minutes: float | None,
) -> None:
    """Run the known-good Optuna entry command."""
    command = [
        sys.executable,
        '-m', 'DL_exp.src_optuna',
        '--model', model,
        '--data-dir', data_dir,
        '--out-dir', out_dir,
        '--n-trials', str(n_trials),
        '--n-jobs', str(n_jobs),
        '--epochs', str(epochs),
    ]
    if timeout_minutes is not None:
        command.extend(['--timeout-minutes', str(timeout_minutes)])
    _run_command(command)


def _run_ml(
    run_best: bool,
    train_path: str,
    test_path: str,
    selected_features_path: str,
    results_csv: str,
    n_trials: int,
    include_ensemble: bool,
    n_trials_model: int,
    n_trials_weights: int,
    ensemble_result_file: str,
) -> None:
    """Run the ML pipeline with absolute path resolution."""
    train_path = str((PROJECT_ROOT / train_path).resolve())
    test_path = str((PROJECT_ROOT / test_path).resolve())
    selected_features_path = str((PROJECT_ROOT / selected_features_path).resolve())
    results_csv = str((PROJECT_ROOT / results_csv).resolve())
    ensemble_result_file = str((PROJECT_ROOT / ensemble_result_file).resolve())

    command = [
        sys.executable,
        '-m', 'ML.main',
        '--run-best' if run_best else '--run-all',
        '--train_path', train_path,
        '--test_path', test_path,
        '--selected_features_path', selected_features_path,
        '--n-trials', str(n_trials),
        '--n-trials-model', str(n_trials_model),
        '--n-trials-weights', str(n_trials_weights),
        '--ensemble-result-file', ensemble_result_file,
    ]

    if run_best:
        command.extend(['--results-csv', results_csv])
    if include_ensemble:
        command.append('--include-ensemble')

    # Debug prints (optional but useful)
    print("\n[ML PATH DEBUG]")
    print("Train:", train_path)
    print("Test:", test_path)
    print("Features:", selected_features_path)
    print()

    _run_command(command)


def main() -> None:
    parser = argparse.ArgumentParser(description='Pipeline dispatcher for EDA, DL, and ML runs')
    parser.add_argument('--dl-optuna', action='store_true', help='Run the DL Optuna tuning pipeline')
    parser.add_argument('--dl-lda', action='store_true', help='Run the DL LDA pipeline')
    parser.add_argument('--ml-run-all', action='store_true', help='Run all ML experiments')
    parser.add_argument('--ml-run-best', action='store_true', help='Run best ML experiment from a results CSV')
    parser.add_argument('--eda', action='store_true', help='Run the EDA pipeline')

    parser.add_argument('--data-dir', default='data', help='Dataset directory passed to DL pipelines')
    parser.add_argument('--out-dir', default='outputs', help='Output directory passed to DL pipelines')
    parser.add_argument('--model', default='all', help='Model name for Optuna runs, or all')
    parser.add_argument('--n-trials', type=int, default=20, help='Number of Optuna trials per model')
    parser.add_argument('--n-jobs', type=int, default=2, help='Number of Optuna jobs')
    parser.add_argument('--timeout-minutes', type=float, default=None, help='Optuna timeout in minutes')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs for Optuna trials')
    parser.add_argument('--eda-batch-size', type=int, default=1024, help='Batch size for the EDA pipeline')

    parser.add_argument('--ml-train-path', default='ML/datasets/full96_train.csv', help='Training CSV path for ML pipeline')
    parser.add_argument('--ml-test-path', default='ML/datasets/full96_test.csv', help='Testing CSV path for ML pipeline')
    parser.add_argument('--ml-selected-features-path', default='EDA/outputs/reports/selected_features_Full96.csv', help='Selected-features CSV path for ML pipeline')
    parser.add_argument('--ml-results-csv', default='ML/results/results_full96.csv', help='Results CSV used by --ml-run-best')

    parser.add_argument('--ml-n-trials', type=int, default=10, help='Optuna trials for single-model ML scripts')
    parser.add_argument('--ml-include-ensemble', action='store_true', help='Also run ensemble script in --ml-run-all mode')
    parser.add_argument('--ml-n-trials-model', type=int, default=10, help='Model tuning trials for ML ensemble')
    parser.add_argument('--ml-n-trials-weights', type=int, default=10, help='Weight tuning trials for ML ensemble')
    parser.add_argument('--ml-ensemble-result-file', default='results_ensemble.txt', help='Output file for ML ensemble metrics')

    args = parser.parse_args()

    if args.ml_run_all and args.ml_run_best:
        parser.error('Use only one of --ml-run-all or --ml-run-best')

    selected = [args.eda, args.dl_lda, args.dl_optuna, args.ml_run_all, args.ml_run_best]
    if not any(selected):
        parser.error('Specify at least one pipeline: --eda, --dl-lda, --dl-optuna, --ml-run-all, or --ml-run-best')

    if args.eda:
        os.environ['EDA_DATA_DIR'] = str((PROJECT_ROOT / args.data_dir).resolve())
        os.environ['EDA_OUTPUT_DIR'] = str((PROJECT_ROOT / args.out_dir).resolve())
        from EDA.src.eda import run_eda
        run_eda(batch_size=args.eda_batch_size)

    if args.dl_lda:
        _run_dl_lda(args.data_dir, args.out_dir)

    if args.dl_optuna:
        _run_dl_optuna(
            data_dir=args.data_dir,
            out_dir=args.out_dir,
            model=args.model,
            n_trials=args.n_trials,
            n_jobs=args.n_jobs,
            epochs=args.epochs,
            timeout_minutes=args.timeout_minutes,
        )

    if args.ml_run_all or args.ml_run_best:
        _run_ml(
            run_best=args.ml_run_best,
            train_path=args.ml_train_path,
            test_path=args.ml_test_path,
            selected_features_path=args.ml_selected_features_path,
            results_csv=args.ml_results_csv,
            n_trials=args.ml_n_trials,
            include_ensemble=args.ml_include_ensemble,
            n_trials_model=args.ml_n_trials_model,
            n_trials_weights=args.ml_n_trials_weights,
            ensemble_result_file=args.ml_ensemble_result_file,
        )


if __name__ == '__main__':
    main()
