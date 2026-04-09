from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from sklearn.metrics import f1_score, roc_auc_score
from tqdm import tqdm

from .config import CFG


def binary_metrics(labels, probs):
    labels = np.asarray(labels, dtype=np.float32)
    probs = np.asarray(probs, dtype=np.float32)
    preds = (probs >= 0.5).astype(np.int32)
    acc = float((preds == labels).mean()) if len(labels) else 0.0
    auc = float(roc_auc_score(labels, probs)) if len(np.unique(labels)) > 1 else 0.5
    f1 = float(f1_score(labels, preds, zero_division=0)) if len(labels) else 0.0
    return {'acc': acc, 'auc': auc, 'f1': f1}


def _reshape_binary_labels(labels: torch.Tensor) -> torch.Tensor:
    labels = labels.float()
    if labels.ndim == 1:
        return labels.unsqueeze(1)
    return labels.reshape(labels.size(0), -1)


def run_image_epoch(rank, model, loader, criterion, optimizer, scaler, phase, epoch):
    model.train() if phase == 'train' else model.eval()
    total_loss, all_labels, all_probs = 0.0, [], []
    t0 = time.time()

    pbar = tqdm(loader, desc=f'{phase.upper()} Epoch {epoch}', dynamic_ncols=True, leave=False) if rank == 0 else loader
    ctx = torch.enable_grad() if phase == 'train' else torch.no_grad()
    with ctx:
        for imgs, labels, _ in pbar:
            imgs = imgs.to(rank, non_blocking=True)
            labels = _reshape_binary_labels(labels).to(rank, non_blocking=True)

            with autocast(enabled=CFG['amp']):
                logits = model(imgs)
                loss = criterion(logits, labels)

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

            if rank == 0:
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    elapsed = time.time() - t0
    n = max(1, len(all_labels))
    return {
        'loss': total_loss / n,
        'elapsed_s': elapsed,
        'labels': all_labels,
        'probs': all_probs,
    }


def run_feature_epoch(rank, model, loader, criterion, optimizer, scaler, phase, epoch):
    model.train() if phase == 'train' else model.eval()
    total_loss, all_labels, all_probs = 0.0, [], []
    t0 = time.time()

    pbar = tqdm(loader, desc=f'{phase.upper()} Epoch {epoch}', dynamic_ncols=True, leave=False) if rank == 0 else loader
    ctx = torch.enable_grad() if phase == 'train' else torch.no_grad()
    with ctx:
        for feats, labels in pbar:
            feats = feats.to(rank, non_blocking=True)
            labels = _reshape_binary_labels(labels).to(rank, non_blocking=True)

            with autocast(enabled=CFG['amp']):
                logits = model(feats)
                loss = criterion(logits, labels)

            if phase == 'train':
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()

            total_loss += loss.item() * feats.size(0)
            probs = torch.sigmoid(logits).squeeze(1)
            all_probs.extend(probs.detach().cpu().numpy().tolist())
            all_labels.extend(labels.squeeze(1).detach().cpu().numpy().tolist())

    elapsed = time.time() - t0
    n = max(1, len(all_labels))
    return {
        'loss': total_loss / n,
        'elapsed_s': elapsed,
        'labels': all_labels,
        'probs': all_probs,
    }
