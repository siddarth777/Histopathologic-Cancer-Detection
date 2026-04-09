from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from EDA.eda.src.eda import run_eda


PROJECT_ROOT = Path(__file__).resolve().parent


def _run_command(command: list[str]) -> None:
	subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def main() -> None:
	parser = argparse.ArgumentParser(description='Pipeline dispatcher for EDA, LDA, and Optuna runs')
	parser.add_argument('--dl-optuna', action='store_true', help='Run the DL Optuna tuning pipeline')
	parser.add_argument('--dl-lda', action='store_true', help='Run the DL LDA pipeline')
	parser.add_argument('--eda', action='store_true', help='Run the EDA pipeline')
	parser.add_argument('--data-dir', default='data', help='Dataset directory passed to DL pipelines')
	parser.add_argument('--out-dir', default='outputs', help='Output directory passed to DL pipelines')
	parser.add_argument('--model', default='all', help='Model name for Optuna runs, or all')
	parser.add_argument('--n-trials', type=int, default=20, help='Number of Optuna trials per model')
	parser.add_argument('--n-jobs', type=int, default=2, help='Number of Optuna jobs')
	parser.add_argument('--timeout-minutes', type=float, default=None, help='Optuna timeout in minutes')
	parser.add_argument('--epochs', type=int, default=5, help='Number of epochs for Optuna trials')
	parser.add_argument('--eda-batch-size', type=int, default=1024, help='Batch size for the EDA pipeline')
	args = parser.parse_args()

	selected = [args.eda, args.dl_lda, args.dl_optuna]
	if not any(selected):
		parser.error('Specify at least one of --eda, --dl-lda, or --dl-optuna')

	if args.eda:
		run_eda(batch_size=args.eda_batch_size)

	if args.dl_lda:
		_run_command([
			sys.executable,
			'-m', 'DL_exp.src_lda',
			'--task', 'all',
			'--data-dir', args.data_dir,
			'--out-dir', args.out_dir,
		])

	if args.dl_optuna:
		command = [
			sys.executable,
			'-m', 'DL_exp.src_optuna',
			'--model', args.model,
			'--data-dir', args.data_dir,
			'--out-dir', args.out_dir,
			'--n-trials', str(args.n_trials),
			'--n-jobs', str(args.n_jobs),
			'--epochs', str(args.epochs),
		]
		if args.timeout_minutes is not None:
			command.extend(['--timeout-minutes', str(args.timeout_minutes)])
		_run_command(command)


if __name__ == '__main__':
	main()
