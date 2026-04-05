import time

import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from sklearn.metrics import f1_score, roc_auc_score
from tqdm import tqdm

from .config import CFG


def run_epoch(rank, model, loader, criterion, optimizer, scaler, phase, epoch):
    model.train() if phase == 'train' else model.eval()
    total_loss, all_labels, all_probs = 0.0, [], []
    t0 = time.time()

    pbar = tqdm(loader,
                desc=f"{phase.upper()} Epoch {epoch}",
                dynamic_ncols=True,
                leave=False) if rank == 0 else loader

    ctx = torch.enable_grad() if phase == 'train' else torch.no_grad()
    with ctx:
        for imgs, labels, _ in pbar:
            imgs = imgs.to(rank, non_blocking=True)
            labels = labels.float().unsqueeze(1).to(rank, non_blocking=True)

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
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})

    n = len(all_labels)
    avg_loss = total_loss / n
    preds = [1 if p >= 0.5 else 0 for p in all_probs]
    acc = sum(p == l for p, l in zip(preds, all_labels)) / n
    auc = roc_auc_score(all_labels, all_probs)
    f1 = f1_score(all_labels, preds, zero_division=0)
    elapsed = time.time() - t0

    return {
        'loss': avg_loss,
        'acc': acc,
        'auc': auc,
        'f1': f1,
        'elapsed_s': elapsed,
        'labels': all_labels,
        'probs': all_probs,
    }
