from __future__ import annotations

import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, roc_auc_score
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from ..src_lda.data import CancerDataset
from ..src_lda.logging_utils import Logger

from .models import get_model

from .utils import cleanup_cuda, get_device, trial_dir


def _reshape_binary_labels(labels: torch.Tensor) -> torch.Tensor:
    labels = labels.float()
    if labels.ndim == 1:
        return labels.unsqueeze(1)
    return labels.reshape(labels.size(0), -1)


def _compute_binary_metrics(labels, probs) -> dict:
    labels_arr = np.asarray(labels, dtype=np.float32)
    probs_arr = np.asarray(probs, dtype=np.float32)
    finite_mask = np.isfinite(labels_arr) & np.isfinite(probs_arr)
    labels_arr = labels_arr[finite_mask]
    probs_arr = probs_arr[finite_mask]

    if labels_arr.size == 0:
        return {'acc': 0.0, 'auc': 0.5, 'f1': 0.0}

    preds = (probs_arr >= 0.5).astype(np.int32)
    acc = float((preds == labels_arr).mean()) if len(labels_arr) else 0.0
    auc = float(roc_auc_score(labels_arr, probs_arr)) if len(np.unique(labels_arr)) > 1 else 0.5
    f1 = float(f1_score(labels_arr, preds, zero_division=0)) if len(labels_arr) else 0.0
    return {'acc': acc, 'auc': auc, 'f1': f1}


def _build_loaders(train_df, val_df, cfg):
    img_root = os.path.join(cfg['data_dir'], 'train')
    train_ds = CancerDataset(train_df, img_root, 'train')
    val_ds = CancerDataset(val_df, img_root, 'val')

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg['batch_size'],
        shuffle=True,
        num_workers=cfg['num_workers'],
        pin_memory=cfg['pin_memory'],
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg['batch_size'] * 2,
        shuffle=False,
        num_workers=cfg['num_workers'],
        pin_memory=cfg['pin_memory'],
        drop_last=False,
    )
    return train_loader, val_loader


def run_epoch(model, loader, criterion, optimizer, scaler, device, phase, epoch, amp_enabled):
    model.train() if phase == 'train' else model.eval()
    total_loss, all_labels, all_probs = 0.0, [], []
    t0 = time.time()

    ctx = torch.enable_grad() if phase == 'train' else torch.no_grad()
    with ctx:
        for imgs, labels, _ in loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = _reshape_binary_labels(labels).to(device, non_blocking=True)

            with autocast(enabled=amp_enabled):
                logits = model(imgs)
                loss = criterion(logits, labels)

            if not torch.isfinite(loss):
                continue

            if phase == 'train':
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()

            total_loss += loss.item() * imgs.size(0)
            probs = torch.sigmoid(logits).squeeze(1)
            all_probs.extend(probs.detach().cpu().numpy().tolist())
            all_labels.extend(labels.squeeze(1).detach().cpu().numpy().tolist())

    n = max(1, len(all_labels))
    metrics = _compute_binary_metrics(all_labels, all_probs)
    metrics['loss'] = total_loss / n
    metrics['elapsed_s'] = time.time() - t0
    metrics['labels'] = all_labels
    metrics['probs'] = all_probs
    return metrics


def build_optimizer(model, hparams: dict):
    kind = hparams['optimizer']
    if kind == 'adamw':
        return optim.AdamW(model.parameters(), lr=hparams['lr'], weight_decay=hparams['weight_decay'])
    if kind == 'sgd':
        return optim.SGD(
            model.parameters(),
            lr=hparams['lr'],
            momentum=hparams['sgd_momentum'],
            weight_decay=hparams['weight_decay'],
            nesterov=True,
        )
    raise ValueError(f'Unknown optimizer: {kind}')


def train_trial(model_name: str, train_df, val_df, cfg: dict, hparams: dict, trial=None):
    device = get_device()
    train_loader, val_loader = _build_loaders(train_df, val_df, cfg)

    model = get_model(model_name).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = build_optimizer(model, hparams)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['epochs'], eta_min=1e-6)
    scaler = GradScaler(enabled=cfg['amp'] and device.type == 'cuda')

    trial_number = int(trial.number) if trial is not None else -1
    tdir = trial_dir(cfg['out_dir'], model_name, trial_number)
    logger = Logger(os.path.join(tdir, 'log.csv'))

    best_auc = -1.0
    best_metrics = None
    ckpt_path = os.path.join(tdir, f'{model_name}_best.pth')

    try:
        for epoch in range(1, cfg['epochs'] + 1):
            tr = run_epoch(model, train_loader, criterion, optimizer, scaler, device, 'train', epoch, cfg['amp'] and device.type == 'cuda')
            vl = run_epoch(model, val_loader, criterion, None, scaler, device, 'val', epoch, cfg['amp'] and device.type == 'cuda')
            scheduler.step()

            logger.log({'epoch': epoch, 'phase': 'train', 'loss': tr['loss'], 'acc': tr['acc'], 'auc': tr['auc'], 'f1': tr['f1'], 'lr': scheduler.get_last_lr()[0], 'elapsed_s': tr['elapsed_s']})
            logger.log({'epoch': epoch, 'phase': 'val', 'loss': vl['loss'], 'acc': vl['acc'], 'auc': vl['auc'], 'f1': vl['f1'], 'lr': scheduler.get_last_lr()[0], 'elapsed_s': vl['elapsed_s']})

            if trial is not None:
                trial.report(vl['auc'], step=epoch)
                if trial.should_prune():
                    import optuna
                    raise optuna.TrialPruned()

            if vl['auc'] > best_auc:
                best_auc = vl['auc']
                best_metrics = vl
                torch.save(
                    {
                        'epoch': epoch,
                        'model_name': model_name,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'val_auc': best_auc,
                        'val_acc': vl['acc'],
                        'val_f1': vl['f1'],
                        'hparams': hparams,
                    },
                    ckpt_path,
                )
    finally:
        del model, optimizer, scheduler, scaler
        cleanup_cuda()

    return {
        'best_val_auc': float(best_auc),
        'best_metrics': best_metrics,
        'checkpoint_path': ckpt_path,
        'trial_dir': tdir,
    }
