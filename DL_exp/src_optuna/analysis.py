from __future__ import annotations

import os

from .utils import write_json


def export_study_artifacts(study, model_name: str, out_dir: str):
    model_dir = os.path.join(out_dir, 'optuna', model_name)
    os.makedirs(model_dir, exist_ok=True)

    trials_csv = os.path.join(model_dir, 'trials.csv')
    study.trials_dataframe().to_csv(trials_csv, index=False)

    best_json = os.path.join(model_dir, 'best_params.json')
    complete_trials = [t for t in study.trials if t.state.name == 'COMPLETE' and t.value is not None]
    if complete_trials:
        best_trial = max(complete_trials, key=lambda t: float(t.value))
        best_payload = {
            'model_name': model_name,
            'best_trial': int(best_trial.number),
            'best_value': float(best_trial.value),
            'direction': str(study.direction),
            'params': dict(best_trial.params),
            'checkpoint_path': best_trial.user_attrs.get('checkpoint_path'),
            'trial_dir': best_trial.user_attrs.get('trial_dir'),
        }
    else:
        best_payload = {
            'model_name': model_name,
            'best_trial': None,
            'best_value': None,
            'direction': str(study.direction),
            'params': {},
            'checkpoint_path': None,
            'trial_dir': None,
            'note': 'No complete trials available.',
        }

    write_json(best_json, best_payload)

    topk_csv = os.path.join(model_dir, 'top_trials.csv')
    complete_trials.sort(key=lambda t: float(t.value), reverse=True)
    rows = []
    for t in complete_trials[:5]:
        rows.append(
            {
                'trial': int(t.number),
                'value': float(t.value),
                'params': dict(t.params),
                'checkpoint_path': t.user_attrs.get('checkpoint_path'),
            }
        )

    import pandas as pd

    pd.DataFrame(rows).to_csv(topk_csv, index=False)

    return {
        'trials_csv': trials_csv,
        'best_json': best_json,
        'topk_csv': topk_csv,
    }
