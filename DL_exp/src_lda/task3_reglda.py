
import argparse
import os

import pandas as pd

from .config import CFG
from .task2_lda import train_with_lda


def main():
    parser = argparse.ArgumentParser(description='Task 3: Regularized LDA on backbone features')
    parser.add_argument('--data-dir', default=CFG['data_dir'])
    parser.add_argument('--out-dir', default=CFG['out_dir'])
    parser.add_argument('--model', default=None, help='One of cnn, alexnet, resnet50, vgg16. Defaults to all.')
    args = parser.parse_args()

    from .utils import load_train_dataframe

    train_df, val_df = load_train_dataframe(args.data_dir, CFG['seed'], CFG['val_split'])
    model_names = [args.model] if args.model else CFG['models']
    os.makedirs(args.out_dir, exist_ok=True)
    for model_name in model_names:
        train_with_lda(model_name, train_df, val_df, args.out_dir, 'reglda')


if __name__ == '__main__':
    main()
