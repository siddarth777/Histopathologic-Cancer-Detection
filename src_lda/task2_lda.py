from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader

from src.logging_utils import Logger
from src.plotting import plot_training_curves, plot_confusion_and_roc
from src.utils import seed_everything

from .config import CFG
from .data import build_image_datasets, make_feature_loader
from .features import build_backbone
from .heads import MLPHead
from .metrics import compute_binary_metrics
from .trainers import run_feature_epoch
from .transforms import LDAProjector


def _sample_features(loader: DataLoader, backbone: torch.nn.Module, device: torch.device, max_samples: int):
    """Sample features from backbone for LDA fitting."""
    backbone.eval()
    feature_batches, label_batches = [], []
    seen = 0
    with torch.no_grad():
        for imgs, lbls, _ in loader:
            if seen >= max_samples:
                break
            imgs = imgs.to(device, non_blocking=True)
            feats = backbone(imgs).detach().cpu()
            lbls = lbls.detach().cpu()
            take = min(max_samples - seen, feats.size(0))
            if take <= 0:
                break
            feature_batches.append(feats[:take])
            label_batches.append(lbls[:take])
            seen += take
    if feature_batches:
        return torch.cat(feature_batches, dim=0), torch.cat(label_batches, dim=0)
    return torch.empty(0), torch.empty(0)


def _collect_projected_features(loader: DataLoader, backbone: torch.nn.Module, projector: LDAProjector, device: torch.device):
    """Project all features through LDA projector."""
    backbone.eval()
    projected_batches, label_batches = [], []
    with torch.no_grad():
        for imgs, lbls, _ in loader:
            imgs = imgs.to(device, non_blocking=True)
            feats = backbone(imgs).detach().cpu().numpy()
            projected = projector.transform(feats)
            projected_batches.append(torch.tensor(projected, dtype=torch.float32))
            label_batches.append(lbls.detach().cpu().float())
    if projected_batches:
        return torch.cat(projected_batches, dim=0), torch.cat(label_batches, dim=0)
    return torch.empty(0), torch.empty(0)


def train_with_lda(model_name: str, train_df: pd.DataFrame, val_df: pd.DataFrame, out_dir: str, projector_kind: str = 'lda'):
    """Train model with LDA/RegLDA projection on single GPU."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed_everything(CFG['seed'])

    print('\n' + '=' * 70)
    print(f'  TRAINING: {model_name.upper()} + {projector_kind.upper()}   [Single GPU]')
    print('=' * 70)

    # Build dataloaders and backbone
    train_ds, val_ds = build_image_datasets(train_df, val_df)
    train_loader = DataLoader(train_ds, batch_size=CFG['batch_size'], shuffle=False,
                              num_workers=CFG['num_workers'], pin_memory=CFG['pin_memory'], drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=CFG['batch_size'], shuffle=False,
                            num_workers=CFG['num_workers'], pin_memory=CFG['pin_memory'], drop_last=False)

    backbone = build_backbone(model_name).to(device)
    
    # Sample and fit LDA
    max_global = int(CFG.get('projector_max_samples', 32768))
    sampled_train_feats, sampled_train_labels = _sample_features(train_loader, backbone, device, max_global)

    feature_chunks = [sampled_train_feats.numpy()] if sampled_train_feats.numel() > 0 else []
    label_chunks = [sampled_train_labels.numpy()] if sampled_train_labels.numel() > 0 else []
    
    if feature_chunks and label_chunks:
        train_features = np.concatenate(feature_chunks, axis=0)
        train_labels = np.concatenate(label_chunks, axis=0)
    else:
        train_features = np.empty((0, 1), dtype=np.float32)
        train_labels = np.empty((0,), dtype=np.float32)

    projector = LDAProjector(kind=projector_kind)
    projector.fit(train_features, train_labels)

    # Project all features
    train_projected, train_labels_full = _collect_projected_features(train_loader, backbone, projector, device)
    val_projected, val_labels_full = _collect_projected_features(val_loader, backbone, projector, device)
    
    train_loader = make_feature_loader(train_projected, train_labels_full.float(), CFG['batch_size'], shuffle=True)
    val_loader = make_feature_loader(val_projected, val_labels_full.float(), CFG['batch_size'], shuffle=False)

    # Train MLP on projected features
    model = MLPHead(train_projected.shape[1], hidden_dim=CFG['mlp_hidden_dim'], dropout=CFG['mlp_dropout']).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=CFG['lr'], weight_decay=CFG['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG['epochs'], eta_min=1e-6)
    scaler = GradScaler(enabled=CFG['amp'])

    logger = Logger(os.path.join(out_dir, f'{model_name}_{projector_kind}_log.csv'))
    best_auc = 0.0
    best_val_results = None

    for epoch in range(1, CFG['epochs'] + 1):
        print(f'\n  ── Epoch {epoch:02d}/{CFG["epochs"]}  lr={scheduler.get_last_lr()[0]:.2e} ──')

        tr_metrics = run_feature_epoch(model, train_loader, criterion, optimizer, scaler, 'train', epoch, device)
        vl_metrics = run_feature_epoch(model, val_loader, criterion, None, scaler, 'val', epoch, device)
        scheduler.step()

        logger.log({'epoch': epoch, 'phase': 'train', 'loss': tr_metrics['loss'], 'acc': tr_metrics['acc'], 'auc': tr_metrics['auc'], 'f1': tr_metrics['f1'], 'lr': scheduler.get_last_lr()[0], 'elapsed_s': tr_metrics['elapsed_s']})
        logger.log({'epoch': epoch, 'phase': 'val', 'loss': vl_metrics['loss'], 'acc': vl_metrics['acc'], 'auc': vl_metrics['auc'], 'f1': vl_metrics['f1'], 'lr': scheduler.get_last_lr()[0], 'elapsed_s': vl_metrics['elapsed_s']})

        if vl_metrics['auc'] > best_auc:
            best_auc = vl_metrics['auc']
            ckpt_path = os.path.join(out_dir, f'{model_name}_{projector_kind}_best.pth')
            torch.save({
                'epoch': epoch,
                'model_name': model_name,
                'projector_kind': projector_kind,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'val_auc': best_auc,
                'val_acc': vl_metrics['acc'],
                'val_f1': vl_metrics['f1'],
            }, ckpt_path)
            print(f'  [CKPT] New best AUC={best_auc:.4f} → {ckpt_path}')
            best_val_results = vl_metrics

    plot_training_curves(os.path.join(out_dir, f'{model_name}_{projector_kind}_log.csv'), f'{model_name}_{projector_kind}', out_dir)
    if best_val_results is not None:
        plot_confusion_and_roc(best_val_results['labels'], best_val_results['probs'], f'{model_name}_{projector_kind}', out_dir)
    print(f'\n  [DONE] {model_name.upper()} + {projector_kind.upper()}  best_val_auc={best_auc:.4f}')


def main():
    parser = argparse.ArgumentParser(description='Task 2: LDA on backbone features')
    parser.add_argument('--data-dir', default=CFG['data_dir'])
    parser.add_argument('--out-dir', default=CFG['out_dir'])
    parser.add_argument('--model', default=None, help='One of cnn, alexnet, resnet50, vgg16. Defaults to all.')
    args = parser.parse_args()

    from .utils import load_train_dataframe

    train_df, val_df = load_train_dataframe(args.data_dir, CFG['seed'], CFG['val_split'])
    model_names = [args.model] if args.model else CFG['models']
    os.makedirs(args.out_dir, exist_ok=True)
    for model_name in model_names:
        train_with_lda(model_name, train_df, val_df, args.out_dir, 'lda')


if __name__ == '__main__':
    main()
    init_ddp(rank, world_size)

    is_main = rank == 0
    if is_main:
        print('\n' + '=' * 70)
        print(f'  TRAINING: {model_name.upper()} + {projector_kind.upper()}   [DDP]')
        print('=' * 70)

    train_ds, val_ds = build_image_datasets(train_df, val_df)
    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=False)
    val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(train_ds, batch_size=CFG['batch_size'], sampler=train_sampler,
                              num_workers=CFG['num_workers'], pin_memory=CFG['pin_memory'], drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=CFG['batch_size'], sampler=val_sampler,
                            num_workers=CFG['num_workers'], pin_memory=CFG['pin_memory'], drop_last=False)

    backbone = build_backbone(model_name).to(rank)
    max_global = int(CFG.get('projector_max_samples', 32768))
    max_per_rank = max(1, int(math.ceil(max_global / world_size)))
    sampled_train_feats, sampled_train_labels = _sample_local_features(rank, train_loader, backbone, max_per_rank)

    gathered_train = [None for _ in range(world_size)]
    dist.all_gather_object(gathered_train, (sampled_train_feats.numpy(), sampled_train_labels.numpy()))

    if is_main:
        feature_chunks = [item[0] for item in gathered_train if len(item[0])]
        label_chunks = [item[1] for item in gathered_train if len(item[1])]
        if feature_chunks and label_chunks:
            train_features = np.concatenate(feature_chunks, axis=0)
            train_labels = np.concatenate(label_chunks, axis=0)
        else:
            train_features = np.empty((0, 1), dtype=np.float32)
            train_labels = np.empty((0,), dtype=np.float32)
        projector = LDAProjector(kind=projector_kind)
        projector.fit(train_features, train_labels)
        payload = projector.dumps()
    else:
        payload = None

    broadcast_payload = [payload]
    dist.broadcast_object_list(broadcast_payload, src=0)
    projector = LDAProjector.loads(broadcast_payload[0])

    train_projected, train_labels_local = _collect_projected_features(rank, train_loader, backbone, projector)
    val_projected, val_labels_local = _collect_projected_features(rank, val_loader, backbone, projector)
    train_loader = make_feature_loader(train_projected, train_labels_local.float(), CFG['batch_size'], shuffle=True)
    val_loader = make_feature_loader(val_projected, val_labels_local.float(), CFG['batch_size'], shuffle=False)

    model = MLPHead(train_projected.shape[1], hidden_dim=CFG['mlp_hidden_dim'], dropout=CFG['mlp_dropout']).to(rank)
    model = DDP(model, device_ids=[rank], find_unused_parameters=False)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=CFG['lr'], weight_decay=CFG['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG['epochs'], eta_min=1e-6)
    scaler = GradScaler(enabled=CFG['amp'])

    logger = Logger(os.path.join(out_dir, f'{model_name}_{projector_kind}_log.csv')) if is_main else None
    best_auc = 0.0
    best_val_results = None

    for epoch in range(1, CFG['epochs'] + 1):
        if is_main:
            print(f'\n  ── Epoch {epoch:02d}/{CFG["epochs"]}  lr={scheduler.get_last_lr()[0]:.2e} ──')

        train_sampler.set_epoch(epoch)
        tr_local = run_feature_epoch(rank, model, train_loader, criterion, optimizer, scaler, 'train', epoch)
        vl_local = run_feature_epoch(rank, model, val_loader, criterion, None, scaler, 'val', epoch)
        scheduler.step()

        tr_metrics = aggregate_epoch_result(rank, tr_local, world_size)
        vl_metrics = aggregate_epoch_result(rank, vl_local, world_size)

        if rank == 0:
            logger.log({'epoch': epoch, 'phase': 'train', 'loss': tr_metrics['loss'], 'acc': tr_metrics['acc'], 'auc': tr_metrics['auc'], 'f1': tr_metrics['f1'], 'lr': scheduler.get_last_lr()[0], 'elapsed_s': tr_metrics['elapsed_s']})
            logger.log({'epoch': epoch, 'phase': 'val', 'loss': vl_metrics['loss'], 'acc': vl_metrics['acc'], 'auc': vl_metrics['auc'], 'f1': vl_metrics['f1'], 'lr': scheduler.get_last_lr()[0], 'elapsed_s': vl_metrics['elapsed_s']})
            if vl_metrics['auc'] > best_auc:
                best_auc = vl_metrics['auc']
                ckpt_path = os.path.join(out_dir, f'{model_name}_{projector_kind}_best.pth')
                torch.save({
                    'epoch': epoch,
                    'model_name': model_name,
                    'projector_kind': projector_kind,
                    'state_dict': model.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'val_auc': best_auc,
                    'val_acc': vl_metrics['acc'],
                    'val_f1': vl_metrics['f1'],
                }, ckpt_path)
                print(f'  [CKPT] New best AUC={best_auc:.4f} → {ckpt_path}')
                best_val_results = vl_metrics

    if is_main:
        finish_run(out_dir, f'{model_name}_{projector_kind}', logger, best_val_results, best_auc)

    finalize_ddp()


def main():
    parser = argparse.ArgumentParser(description='Task 2: LDA on backbone features (DDP)')
    parser.add_argument('--world-size', type=int, default=2)
    parser.add_argument('--data-dir', default=CFG['data_dir'])
    parser.add_argument('--out-dir', default=CFG['out_dir'])
    parser.add_argument('--model', default=None, help='One of cnn, alexnet, resnet50, vgg16. Defaults to all.')
    args = parser.parse_args()

    from .utils import load_train_dataframe

    train_df, val_df = load_train_dataframe(args.data_dir, CFG['seed'], CFG['val_split'])
    model_names = [args.model] if args.model else CFG['models']
    os.makedirs(args.out_dir, exist_ok=True)
    for model_name in model_names:
        mp.spawn(ddp_worker, args=(args.world_size, model_name, train_df, val_df, args.out_dir, 'lda'), nprocs=args.world_size, join=True)


if __name__ == '__main__':
    main()
