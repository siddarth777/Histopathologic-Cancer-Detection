from __future__ import annotations

from src.config import CFG as BASE_CFG

CFG = {
    **BASE_CFG,
    'study_direction': 'maximize',
    'optuna_metric': 'auc',
    'n_trials': 20,
    'n_jobs': 2,
    'timeout_minutes': None,
    'sampler_kind': 'tpe',
    'pruner_kind': 'median',
    'search_space': {
        'lr': (1e-5, 1e-2),
        'weight_decay': (1e-6, 1e-2),
        'optimizer': ['adamw', 'sgd'],
        'sgd_momentum': (0.8, 0.99),
    },
}
