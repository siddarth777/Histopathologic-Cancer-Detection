from __future__ import annotations

import argparse

import pandas as pd
import torch.multiprocessing as mp

from .config import CFG

from .task2_lda import ddp_worker


def ddp_worker_reglda(rank: int, world_size: int, model_name: str, train_df: pd.DataFrame, val_df: pd.DataFrame, out_dir: str):
    return ddp_worker(rank, world_size, model_name, train_df, val_df, out_dir, projector_kind='reglda')


def main():
    parser = argparse.ArgumentParser(description='Task 3: Regularized LDA on backbone features (DDP)')
    parser.add_argument('--world-size', type=int, default=2)
    parser.add_argument('--data-dir', default=CFG['data_dir'])
    parser.add_argument('--out-dir', default=CFG['out_dir'])
    parser.add_argument('--model', default=None, help='One of cnn, alexnet, resnet50, vgg16. Defaults to all.')
    args = parser.parse_args()

    from .utils import load_train_dataframe

    train_df, val_df = load_train_dataframe(args.data_dir, CFG['seed'], CFG['val_split'])
    model_names = [args.model] if args.model else CFG['models']
    import os
    os.makedirs(args.out_dir, exist_ok=True)
    for model_name in model_names:
        mp.spawn(ddp_worker, args=(args.world_size, model_name, train_df, val_df, args.out_dir, 'reglda'), nprocs=args.world_size, join=True)


if __name__ == '__main__':
    main()
