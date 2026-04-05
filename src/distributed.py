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

from .config import CFG
from .data import CancerDataset
from .logging_utils import Logger
from .models import get_model
from .plotting import plot_confusion_and_roc, plot_training_curves
from .training import run_epoch
from .utils import seed_everything


def ddp_worker(rank: int, world_size: int, model_name: str,
               train_df: pd.DataFrame, val_df: pd.DataFrame, out_dir: str):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group('nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    seed_everything(CFG['seed'] + rank)

    is_main = (rank == 0)
    if is_main:
        print(f'\n' + '=' * 70)
        print(f'  TRAINING: {model_name.upper()}   [2x T4 DDP]')
        print('=' * 70)

    train_ds = CancerDataset(train_df, os.path.join(CFG['data_dir'], 'train'), 'train')
    val_ds = CancerDataset(val_df, os.path.join(CFG['data_dir'], 'train'), 'val')

    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=CFG['batch_size'],
        sampler=train_sampler,
        num_workers=CFG['num_workers'],
        pin_memory=CFG['pin_memory'],
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=CFG['batch_size'] * 2,
        sampler=val_sampler,
        num_workers=CFG['num_workers'],
        pin_memory=CFG['pin_memory'],
    )

    model = get_model(model_name).to(rank)
    model = DDP(model, device_ids=[rank], find_unused_parameters=False)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=CFG['lr'], weight_decay=CFG['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG['epochs'], eta_min=1e-6)
    scaler = GradScaler(enabled=CFG['amp'])

    if is_main:
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'  Parameters: {n_params:,}  ({n_params / 1e6:.2f}M)')
        logger = Logger(os.path.join(out_dir, f'{model_name}_log.csv'))

    best_auc = 0.0
    best_val_results = None

    for epoch in range(1, CFG['epochs'] + 1):
        train_sampler.set_epoch(epoch)

        if is_main:
            print(f'\n  ── Epoch {epoch:02d}/{CFG["epochs"]}  lr={scheduler.get_last_lr()[0]:.2e} ──')

        tr = run_epoch(rank, model, train_loader, criterion, optimizer, scaler, 'train', epoch)
        vl = run_epoch(rank, model, val_loader, criterion, None, scaler, 'val', epoch)

        scheduler.step()

        auc_tensor = torch.tensor(vl['auc']).to(rank)
        dist.all_reduce(auc_tensor, op=dist.ReduceOp.AVG)
        synced_auc = auc_tensor.item()

        if is_main:
            current_lr = scheduler.get_last_lr()[0]
            logger.log({'epoch': epoch, 'phase': 'train',
                        'loss': tr['loss'], 'acc': tr['acc'],
                        'auc': tr['auc'], 'f1': tr['f1'],
                        'lr': current_lr, 'elapsed_s': tr['elapsed_s']})
            logger.log({'epoch': epoch, 'phase': 'val',
                        'loss': vl['loss'], 'acc': vl['acc'],
                        'auc': vl['auc'], 'f1': vl['f1'],
                        'lr': current_lr, 'elapsed_s': vl['elapsed_s']})

            if synced_auc > best_auc:
                best_auc = synced_auc
                ckpt_path = os.path.join(out_dir, f'{model_name}_best.pth')
                torch.save({
                    'epoch': epoch,
                    'model_name': model_name,
                    'state_dict': model.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'val_auc': best_auc,
                    'val_acc': vl['acc'],
                    'val_f1': vl['f1'],
                }, ckpt_path)
                print(f'  [CKPT] New best AUC={best_auc:.4f} → {ckpt_path}')
                best_val_results = vl

    if is_main:
        plot_training_curves(os.path.join(out_dir, f'{model_name}_log.csv'), model_name, out_dir)
        if best_val_results is not None:
            plot_confusion_and_roc(
                best_val_results['labels'], best_val_results['probs'],
                model_name, out_dir,
            )
        print(f'\n  [DONE] {model_name.upper()}  best_val_auc={best_auc:.4f}')

    dist.destroy_process_group()
