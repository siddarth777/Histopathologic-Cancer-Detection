from __future__ import annotations

import argparse
import os

import torch.multiprocessing as mp

from .config import CFG
from .task1 import ddp_worker as task1_worker
from .task2_lda import ddp_worker as task2_worker
from .task3_reglda import ddp_worker_reglda as task3_worker
from .utils import load_train_dataframe


def spawn(worker, world_size: int, args: tuple):
    mp.spawn(worker, args=(world_size, *args), nprocs=world_size, join=True)


def main():
    parser = argparse.ArgumentParser(description='Distributed LDA experiment runner')
    parser.add_argument('--task', choices=['task1', 'task2', 'task3', 'all'], default='all')
    parser.add_argument('--world-size', type=int, default=2)
    parser.add_argument('--data-dir', default=CFG['data_dir'])
    parser.add_argument('--out-dir', default=CFG['out_dir'])
    args = parser.parse_args()

    train_df, val_df = load_train_dataframe(args.data_dir, CFG['seed'], CFG['val_split'])
    os.makedirs(args.out_dir, exist_ok=True)

    if args.task in ('task1', 'all'):
        spawn(task1_worker, args.world_size, (train_df, val_df, args.out_dir, 'cnn_mlp'))
    if args.task in ('task2', 'all'):
        for model_name in CFG['models']:
            spawn(task2_worker, args.world_size, (model_name, train_df, val_df, args.out_dir, 'lda'))
    if args.task in ('task3', 'all'):
        for model_name in CFG['models']:
            spawn(task3_worker, args.world_size, (model_name, train_df, val_df, args.out_dir))


if __name__ == '__main__':
    main()
