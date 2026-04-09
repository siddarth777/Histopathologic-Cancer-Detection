from __future__ import annotations

import argparse
import os

from .config import CFG
from .task2_lda import train_with_lda
from .utils import load_train_dataframe


def main():
    parser = argparse.ArgumentParser(description='LDA experiment runner')
    parser.add_argument('--task', choices=['task2', 'task3', 'all'], default='all')
    parser.add_argument('--data-dir', default=CFG['data_dir'])
    parser.add_argument('--out-dir', default=CFG['out_dir'])
    args = parser.parse_args()

    train_df, val_df = load_train_dataframe(args.data_dir, CFG['seed'], CFG['val_split'])
    os.makedirs(args.out_dir, exist_ok=True)
    
    if args.task in ('task2', 'all'):
        for model_name in CFG['models']:
            train_with_lda(model_name, train_df, val_df, args.out_dir, 'lda')
    
    if args.task in ('task3', 'all'):
        for model_name in CFG['models']:
            train_with_lda(model_name, train_df, val_df, args.out_dir, 'reglda')


if __name__ == '__main__':
    main()
