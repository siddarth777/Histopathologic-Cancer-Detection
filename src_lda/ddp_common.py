from __future__ import annotations

import os

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from src.logging_utils import Logger
from src.plotting import plot_confusion_and_roc, plot_training_curves
from src.utils import seed_everything

from .config import CFG
from .metrics import compute_binary_metrics
from .trainers import binary_metrics


def init_ddp(rank: int, world_size: int, master_port: str = '12355'):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = master_port
    dist.init_process_group('nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    seed_everything(CFG['seed'] + rank)


def finalize_ddp():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def aggregate_epoch_result(rank: int, result: dict, world_size: int):
    gathered = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, (result['labels'], result['probs'], result['loss'], result['elapsed_s']))
    if rank != 0:
        return None

    labels, probs = [], []
    loss_vals, elapsed_vals = [], []
    for item_labels, item_probs, loss, elapsed in gathered:
        labels.extend(item_labels)
        probs.extend(item_probs)
        loss_vals.append(loss)
        elapsed_vals.append(elapsed)

    metrics = compute_binary_metrics(labels, probs)
    metrics['loss'] = float(sum(loss_vals) / len(loss_vals)) if loss_vals else 0.0
    metrics['elapsed_s'] = float(sum(elapsed_vals) / len(elapsed_vals)) if elapsed_vals else 0.0
    metrics['labels'] = labels
    metrics['probs'] = probs
    return metrics


def log_and_maybe_checkpoint(rank: int, logger: Logger | None, model, optimizer, out_dir: str,
                             model_name: str, epoch: int, phase: str, metrics: dict, lr: float,
                             best_auc: float, best_val_results: dict | None):
    if rank == 0 and logger is not None:
        logger.log({
            'epoch': epoch,
            'phase': phase,
            'loss': metrics['loss'],
            'acc': metrics['acc'],
            'auc': metrics['auc'],
            'f1': metrics['f1'],
            'lr': lr,
            'elapsed_s': metrics['elapsed_s'],
        })

    if phase == 'val' and metrics['auc'] > best_auc and rank == 0:
        ckpt_path = os.path.join(out_dir, f'{model_name}_best.pth')
        torch.save({
            'epoch': epoch,
            'model_name': model_name,
            'state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'val_auc': metrics['auc'],
            'val_acc': metrics['acc'],
            'val_f1': metrics['f1'],
        }, ckpt_path)
        print(f'  [CKPT] New best AUC={metrics["auc"]:.4f} → {ckpt_path}')
        best_auc = metrics['auc']
        best_val_results = metrics

    return best_auc, best_val_results


def finish_run(out_dir: str, model_name: str, logger: Logger | None, best_val_results: dict | None, best_auc: float):
    if logger is not None:
        plot_training_curves(os.path.join(out_dir, f'{model_name}_log.csv'), model_name, out_dir)
        if best_val_results is not None:
            plot_confusion_and_roc(best_val_results['labels'], best_val_results['probs'], model_name, out_dir)
        print(f'\n  [DONE] {model_name.upper()}  best_val_auc={best_auc:.4f}')

