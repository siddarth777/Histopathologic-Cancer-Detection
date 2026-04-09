from __future__ import annotations

import argparse
import copy
import os

from .analysis import export_study_artifacts
from .config import CFG
from .objective import objective_factory
from .utils import load_train_dataframe, model_out_dir


def _build_sampler(optuna, cfg: dict):
    kind = cfg.get('sampler_kind', 'tpe')
    if kind == 'random':
        return optuna.samplers.RandomSampler(seed=cfg['seed'])
    return optuna.samplers.TPESampler(seed=cfg['seed'])


def _build_pruner(optuna, cfg: dict):
    kind = cfg.get('pruner_kind', 'median')
    if kind == 'none':
        return optuna.pruners.NopPruner()
    if kind == 'percentile':
        return optuna.pruners.PercentilePruner(percentile=50.0, n_startup_trials=5, n_warmup_steps=1)
    return optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)


def run_model_study(model_name: str, cfg: dict, train_df, val_df):
    try:
        import optuna
    except Exception as exc:
        raise RuntimeError('optuna is required. Install with: pip install optuna') from exc

    objective = objective_factory(model_name, train_df, val_df, cfg)
    sampler = _build_sampler(optuna, cfg)
    pruner = _build_pruner(optuna, cfg)

    study = optuna.create_study(
        direction=cfg['study_direction'],
        sampler=sampler,
        pruner=pruner,
        study_name=f'{model_name}_optuna',
    )

    timeout_seconds = None
    if cfg.get('timeout_minutes') is not None:
        timeout_seconds = int(cfg['timeout_minutes'] * 60)

    study.optimize(
        objective,
        n_trials=int(cfg['n_trials']),
        n_jobs=int(cfg.get('n_jobs', 1)),
        timeout=timeout_seconds,
        gc_after_trial=True,
        show_progress_bar=True,
    )

    artifacts = export_study_artifacts(study, model_name, cfg['out_dir'])
    complete_trials = [t for t in study.trials if t.state.name == 'COMPLETE' and t.value is not None]
    if complete_trials:
        best_trial = max(complete_trials, key=lambda t: float(t.value))
        print(f"[OPTUNA] {model_name}: best_auc={float(best_trial.value):.5f} best_params={best_trial.params}")
    else:
        print(f"[OPTUNA] {model_name}: no completed trials (all pruned/failed)")
    print(f"[OPTUNA] artifacts: {artifacts}")


def main():
    parser = argparse.ArgumentParser(description='Optuna tuner for all DL models in src')
    parser.add_argument('--model', default='all', help='One of cnn, alexnet, resnet50, vgg16, or all')
    parser.add_argument('--data-dir', default=CFG['data_dir'])
    parser.add_argument('--out-dir', default=CFG['out_dir'])
    parser.add_argument('--n-trials', type=int, default=CFG['n_trials'])
    parser.add_argument('--n-jobs', type=int, default=CFG['n_jobs'])
    parser.add_argument('--timeout-minutes', type=float, default=CFG['timeout_minutes'])
    parser.add_argument('--epochs', type=int, default=CFG['epochs'])
    args = parser.parse_args()

    cfg = copy.deepcopy(CFG)
    cfg['data_dir'] = args.data_dir
    cfg['out_dir'] = args.out_dir
    cfg['n_trials'] = args.n_trials
    cfg['n_jobs'] = args.n_jobs
    cfg['timeout_minutes'] = args.timeout_minutes
    cfg['epochs'] = args.epochs

    os.makedirs(cfg['out_dir'], exist_ok=True)

    train_df, val_df = load_train_dataframe(cfg['data_dir'], cfg['seed'], cfg['val_split'])

    if args.model == 'all':
        model_names = list(cfg['models'])
    else:
        if args.model not in cfg['models']:
            raise ValueError(f'Unknown model: {args.model}. Choose from {cfg["models"]} or all')
        model_names = [args.model]

    if cfg['n_jobs'] > 1:
        print('[WARN] n_jobs > 1 can increase GPU memory pressure. Reduce n_jobs if OOM occurs.')

    for model_name in model_names:
        model_out_dir(cfg['out_dir'], model_name)
        run_model_study(model_name, cfg, train_df, val_df)


if __name__ == '__main__':
    main()
