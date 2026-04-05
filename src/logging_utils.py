import csv


class Logger:
    FIELDS = ['epoch', 'phase', 'loss', 'acc', 'auc', 'f1', 'lr', 'elapsed_s']

    def __init__(self, path: str):
        self.path = path
        with open(path, 'w', newline='') as f:
            csv.DictWriter(f, fieldnames=self.FIELDS).writeheader()

    def log(self, row: dict):
        with open(self.path, 'a', newline='') as f:
            csv.DictWriter(f, fieldnames=self.FIELDS).writerow(row)
        tag = f"[{row['phase'].upper():5}]"
        print(
            f"  {tag} epoch={row['epoch']:02d}  loss={row['loss']:.4f}  "
            f"acc={row['acc']:.4f}  auc={row['auc']:.4f}  "
            f"f1={row['f1']:.4f}  lr={row['lr']:.2e}  time={row['elapsed_s']:.1f}s",
            flush=True,
        )
