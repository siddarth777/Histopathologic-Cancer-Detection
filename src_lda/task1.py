from __future__ import annotations

import argparse
import os

import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp

from src.logging_utils import Logger

from .config import CFG
from .data import build_image_datasets
from .features import BaseCNNBackbone
from .heads import CNNMLPNet
from .trainers import run_image_epoch
from .ddp_common import aggregate_epoch_result, finalize_ddp, finish_run, init_ddp


def ddp_worker(rank: int, world_size: int, train_df: pd.DataFrame, val_df: pd.DataFrame, out_dir: str, model_name: str = 'cnn_mlp'):
    init_ddp(rank, world_size)

    is_main = rank == 0
    if is_main:
        print('\n' + '=' * 70)
        print(f'  TRAINING: {model_name.upper()}   [DDP]')
        print('=' * 70)

    train_ds, val_ds = build_image_datasets(train_df, val_df)
    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(train_ds, batch_size=CFG['batch_size'], sampler=train_sampler,
                              num_workers=CFG['num_workers'], pin_memory=CFG['pin_memory'], drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=CFG['batch_size'] * 2, sampler=val_sampler,
                            num_workers=CFG['num_workers'], pin_memory=CFG['pin_memory'], drop_last=False)

    backbone = BaseCNNBackbone()
    model = CNNMLPNet(backbone, feature_dim=BaseCNNBackbone.output_dim, hidden_dim=CFG['mlp_hidden_dim'], dropout=CFG['mlp_dropout']).to(rank)
    model = DDP(model, device_ids=[rank], find_unused_parameters=False)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=CFG['lr'], weight_decay=CFG['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG['epochs'], eta_min=1e-6)
    scaler = GradScaler(enabled=CFG['amp'])

    logger = Logger(os.path.join(out_dir, f'{model_name}_log.csv')) if is_main else None
    best_auc = 0.0
    best_val_results = None

    for epoch in range(1, CFG['epochs'] + 1):
        if is_main:
            print(f'\n  ── Epoch {epoch:02d}/{CFG["epochs"]}  lr={scheduler.get_last_lr()[0]:.2e} ──')

        train_sampler.set_epoch(epoch)
        tr_local = run_image_epoch(rank, model, train_loader, criterion, optimizer, scaler, 'train', epoch)
        vl_local = run_image_epoch(rank, model, val_loader, criterion, None, scaler, 'val', epoch)
        scheduler.step()

        tr_metrics = aggregate_epoch_result(rank, tr_local, world_size)
        vl_metrics = aggregate_epoch_result(rank, vl_local, world_size)

        if rank == 0:
            logger.log({'epoch': epoch, 'phase': 'train', 'loss': tr_metrics['loss'], 'acc': tr_metrics['acc'], 'auc': tr_metrics['auc'], 'f1': tr_metrics['f1'], 'lr': scheduler.get_last_lr()[0], 'elapsed_s': tr_metrics['elapsed_s']})
            logger.log({'epoch': epoch, 'phase': 'val', 'loss': vl_metrics['loss'], 'acc': vl_metrics['acc'], 'auc': vl_metrics['auc'], 'f1': vl_metrics['f1'], 'lr': scheduler.get_last_lr()[0], 'elapsed_s': vl_metrics['elapsed_s']})
            if vl_metrics['auc'] > best_auc:
                best_auc = vl_metrics['auc']
                ckpt_path = os.path.join(out_dir, f'{model_name}_best.pth')
                torch.save({
                    'epoch': epoch,
                    'model_name': model_name,
                    'state_dict': model.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'val_auc': best_auc,
                    'val_acc': vl_metrics['acc'],
                    'val_f1': vl_metrics['f1'],
                }, ckpt_path)
                print(f'  [CKPT] New best AUC={best_auc:.4f} → {ckpt_path}')
                best_val_results = vl_metrics

    if is_main:
        finish_run(out_dir, model_name, logger, best_val_results, best_auc)

    finalize_ddp()


def main():
    parser = argparse.ArgumentParser(description='Task 1: CNN feature vector + MLP head (DDP)')
    parser.add_argument('--world-size', type=int, default=2)
    parser.add_argument('--data-dir', default=CFG['data_dir'])
    parser.add_argument('--out-dir', default=CFG['out_dir'])
    args = parser.parse_args()

    from .utils import load_train_dataframe

    train_df, val_df = load_train_dataframe(args.data_dir, CFG['seed'], CFG['val_split'])
    os.makedirs(args.out_dir, exist_ok=True)
    mp.spawn(ddp_worker, args=(args.world_size, train_df, val_df, args.out_dir, 'cnn_mlp'), nprocs=args.world_size, join=True)


if __name__ == '__main__':
    main()
