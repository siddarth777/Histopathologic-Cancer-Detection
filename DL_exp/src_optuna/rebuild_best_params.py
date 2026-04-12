
import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch
from sklearn.metrics import confusion_matrix, precision_score, recall_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from ..src_lda.data import CancerDataset
    from .models import get_model
    from .config import CFG
    from .utils import load_train_dataframe
except ImportError:
    from DL_exp.src_lda.data import CancerDataset
    from DL_exp.src_optuna.models import get_model
    from DL_exp.src_optuna.config import CFG
    from DL_exp.src_optuna.utils import load_train_dataframe


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _try_number(value: str) -> Any:
    if value is None:
        return value
    text = str(value)
    try:
        if any(ch in text for ch in ['.', 'e', 'E']):
            return float(text)
        return int(text)
    except ValueError:
        return value


def _relative(path: Path) -> str:
    try:
        return os.path.relpath(path, Path.cwd())
    except ValueError:
        return str(path)


def _extract_trial_number(path: Path) -> int | None:
    parent = path.parent.name
    if not parent.startswith('trial_'):
        return None
    suffix = parent.replace('trial_', '', 1)
    return int(suffix) if suffix.isdigit() else None


def _resolve_checkpoint_path(model_dir: Path, payload: dict[str, Any]) -> Path | None:
    checkpoint_path = payload.get('checkpoint_path')
    if checkpoint_path:
        candidate = Path(checkpoint_path)
        if not candidate.is_absolute():
            candidate = Path.cwd() / candidate
        if candidate.exists():
            return candidate

    trial_dir = payload.get('trial_dir')
    if trial_dir:
        trial_path = Path(trial_dir)
        if not trial_path.is_absolute():
            trial_path = Path.cwd() / trial_path
        if trial_path.exists():
            trial_number = payload.get('best_trial')
            model_name = str(payload.get('model_name', model_dir.name))
            if trial_number is not None:
                candidate = trial_path / f'{model_name}_best.pth'
                if candidate.exists():
                    return candidate

    trial_number = payload.get('best_trial')
    if trial_number is not None:
        candidate = model_dir / f'trial_{int(trial_number):04d}' / f"{payload.get('model_name', model_dir.name)}_best.pth"
        if candidate.exists():
            return candidate

    return None


def _evaluate_checkpoint(ckpt_path: Path, payload: dict[str, Any]) -> dict[str, Any] | None:
    try:
        try:
            checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=True)
        except TypeError:
            checkpoint = torch.load(ckpt_path, map_location='cpu')
    except Exception:
        return None

    model_name = str(checkpoint.get('model_name', payload.get('model_name', ckpt_path.parent.parent.name)))
    if model_name not in CFG['models']:
        return None

    train_df, val_df = load_train_dataframe(CFG['data_dir'], CFG['seed'], CFG['val_split'])
    img_root = os.path.join(CFG['data_dir'], 'train')
    val_ds = CancerDataset(val_df, img_root, 'val')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=CFG['batch_size'] * 2,
        shuffle=False,
        num_workers=CFG['num_workers'],
        pin_memory=CFG['pin_memory'],
        drop_last=False,
    )

    model = get_model(model_name).to(device)
    state_dict = checkpoint.get('state_dict')
    if state_dict is None:
        return None
    model.load_state_dict(state_dict)
    model.eval()

    labels = []
    probs = []
    with torch.no_grad():
        for imgs, lbls, _ in loader:
            imgs = imgs.to(device, non_blocking=True)
            logits = model(imgs)
            batch_probs = torch.sigmoid(logits).squeeze(1).detach().cpu().tolist()
            probs.extend(batch_probs)
            labels.extend(lbls.detach().cpu().tolist())

    preds = [1 if p >= 0.5 else 0 for p in probs]
    precision = float(precision_score(labels, preds, zero_division=0)) if len(labels) else 0.0
    recall = float(recall_score(labels, preds, zero_division=0)) if len(labels) else 0.0
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel().tolist()

    return {
        'val_precision': precision,
        'val_recall': recall,
        'confusion_matrix_sample_counts': {
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
            'tp': int(tp),
            'support_negative': int(tn + fp),
            'support_positive': int(fn + tp),
            'total': int(tn + fp + fn + tp),
        },
    }


def _best_from_checkpoints(model_dir: Path) -> dict[str, Any] | None:
    ckpt_paths = sorted(model_dir.glob('trial_*/*_best.pth'))
    best_payload = None

    for ckpt_path in ckpt_paths:
        try:
            payload = torch.load(ckpt_path, map_location='cpu', weights_only=True)
        except TypeError:
            payload = torch.load(ckpt_path, map_location='cpu')
        except Exception:
            continue

        val_auc = _safe_float(payload.get('val_auc'))
        if val_auc is None:
            continue

        trial_number = _extract_trial_number(ckpt_path)
        candidate = {
            'model_name': str(payload.get('model_name', model_dir.name)),
            'best_trial': trial_number,
            'best_value': val_auc,
            'direction': 'StudyDirection.MAXIMIZE',
            'params': payload.get('hparams', {}),
            'checkpoint_path': _relative(ckpt_path),
            'trial_dir': _relative(ckpt_path.parent),
            'best_run_stats': {
                'epoch': int(payload.get('epoch', -1)) if payload.get('epoch') is not None else None,
                'val_auc': val_auc,
                'val_acc': _safe_float(payload.get('val_acc')),
                'val_f1': _safe_float(payload.get('val_f1')),
                'val_precision': None,
                'val_recall': None,
                'confusion_matrix_sample_counts': None,
            },
        }

        evaluated = _evaluate_checkpoint(ckpt_path, candidate)
        if evaluated is not None:
            candidate['best_run_stats'].update(evaluated)

        if best_payload is None or candidate['best_value'] > best_payload['best_value']:
            best_payload = candidate

    return best_payload


def _best_from_trials_csv(model_dir: Path) -> dict[str, Any] | None:
    trials_csv = model_dir / 'trials.csv'
    if not trials_csv.exists():
        return None

    best_row = None
    best_value = None
    with open(trials_csv, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            state = (row.get('state') or '').upper()
            value = _safe_float(row.get('value'))
            if state != 'COMPLETE' or value is None:
                continue
            if best_value is None or value > best_value:
                best_value = value
                best_row = row

    if best_row is None or best_value is None:
        return None

    params = {}
    for key, value in best_row.items():
        if not key.startswith('params_'):
            continue
        name = key.replace('params_', '', 1)
        if value is None or value == '':
            continue
        params[name] = _try_number(value)

    trial_number = _safe_float(best_row.get('number'))
    trial_index = int(trial_number) if trial_number is not None else None
    trial_dir = model_dir / f'trial_{trial_index:04d}' if trial_index is not None else None
    checkpoint_path = None
    if trial_dir is not None:
        candidate = trial_dir / f'{model_dir.name}_best.pth'
        if candidate.exists():
            checkpoint_path = candidate

    payload = {
        'model_name': model_dir.name,
        'best_trial': trial_index,
        'best_value': best_value,
        'direction': 'StudyDirection.MAXIMIZE',
        'params': params,
        'checkpoint_path': _relative(checkpoint_path) if checkpoint_path is not None else None,
        'trial_dir': _relative(trial_dir) if trial_dir is not None else None,
        'best_run_stats': {
            'epoch': None,
            'val_auc': best_value,
            'val_acc': None,
            'val_f1': None,
            'val_precision': None,
            'val_recall': None,
            'confusion_matrix_sample_counts': None,
        },
        'note': 'Recovered from trials.csv because no usable checkpoint was found.',
    }

    if checkpoint_path is not None:
        evaluated = _evaluate_checkpoint(checkpoint_path, payload)
        if evaluated is not None:
            payload['best_run_stats'].update(evaluated)

    return payload


def recover_best_params(optuna_dir: Path, force: bool = False) -> list[dict[str, Any]]:
    summaries = []

    for model_dir in sorted(p for p in optuna_dir.iterdir() if p.is_dir()):
        out_path = model_dir / 'best_params.json'
        if out_path.exists() and not force:
            summaries.append({'model': model_dir.name, 'status': 'skipped-existing', 'path': _relative(out_path)})
            continue

        payload = _best_from_checkpoints(model_dir)
        if payload is None:
            payload = _best_from_trials_csv(model_dir)

        if payload is None:
            summaries.append({'model': model_dir.name, 'status': 'no-data'})
            continue

        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2, sort_keys=True)

        summaries.append({
            'model': model_dir.name,
            'status': 'written',
            'path': _relative(out_path),
            'best_trial': payload.get('best_trial'),
            'best_value': payload.get('best_value'),
        })

    return summaries


def main():
    parser = argparse.ArgumentParser(
        description='Rebuild best_params.json from Optuna output folders and include best-run stats.'
    )
    parser.add_argument('--optuna-dir', default='outputs/optuna', help='Path to outputs/optuna root folder')
    parser.add_argument('--force', action='store_true', help='Overwrite existing best_params.json files')
    args = parser.parse_args()

    optuna_dir = Path(args.optuna_dir)
    if not optuna_dir.exists() or not optuna_dir.is_dir():
        raise FileNotFoundError(f'optuna directory not found: {optuna_dir}')

    summary = recover_best_params(optuna_dir, force=args.force)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
