
import csv
import os
from typing import Any


class Logger:
    """Simple CSV logger with stable header ordering from the first record."""

    def __init__(self, path: str):
        self.path = path
        self.fieldnames: list[str] | None = None
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)

    def log(self, row: dict[str, Any]) -> None:
        if self.fieldnames is None:
            self.fieldnames = list(row.keys())

        with open(self.path, 'a', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fieldnames)
            if handle.tell() == 0:
                writer.writeheader()
            writer.writerow({key: row.get(key, '') for key in self.fieldnames})
