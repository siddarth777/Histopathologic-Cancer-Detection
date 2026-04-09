from __future__ import annotations

from .training import train_trial
from .utils import set_trial_seed


def suggest_hparams(trial, cfg: dict) -> dict:
    lr_lo, lr_hi = cfg['search_space']['lr']
    wd_lo, wd_hi = cfg['search_space']['weight_decay']
    optimizers = cfg['search_space']['optimizer']

    hparams = {
        'lr': trial.suggest_float('lr', lr_lo, lr_hi, log=True),
        'weight_decay': trial.suggest_float('weight_decay', wd_lo, wd_hi, log=True),
        'optimizer': trial.suggest_categorical('optimizer', optimizers),
        'sgd_momentum': 0.9,
    }
    if hparams['optimizer'] == 'sgd':
        m_lo, m_hi = cfg['search_space']['sgd_momentum']
        hparams['sgd_momentum'] = trial.suggest_float('sgd_momentum', m_lo, m_hi)
    return hparams


def objective_factory(model_name: str, train_df, val_df, cfg: dict):
    def objective(trial):
        set_trial_seed(cfg['seed'], trial.number)
        hparams = suggest_hparams(trial, cfg)
        result = train_trial(model_name=model_name, train_df=train_df, val_df=val_df, cfg=cfg, hparams=hparams, trial=trial)
        trial.set_user_attr('checkpoint_path', result['checkpoint_path'])
        trial.set_user_attr('trial_dir', result['trial_dir'])
        return result['best_val_auc']

    return objective
