
import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image

from src.config import CFG
from src.models import get_model


MEAN = [0.7009, 0.5384, 0.6916]
STD = [0.2125, 0.2432, 0.1939]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_path(path_str: str, base_dir: Path | None = None) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    base = base_dir if base_dir is not None else _repo_root()
    return (base / path).resolve()


def _load_checkpoint_path(checkpoint_arg: str | None, best_params_arg: str | None) -> Path:
    root = _repo_root()
    if checkpoint_arg:
        return _resolve_path(checkpoint_arg, root)

    best_params_path = _resolve_path(best_params_arg or 'outputs/optuna/cnn/best_params.json', root)
    with best_params_path.open('r', encoding='utf-8') as handle:
        metadata = json.load(handle)

    checkpoint_path = metadata.get('checkpoint_path')
    if not checkpoint_path:
        raise ValueError(f'checkpoint_path missing from {best_params_path}')
    return _resolve_path(checkpoint_path, root)


def _load_labels(data_dir: Path) -> pd.DataFrame:
    csv_path = data_dir / 'train_labels.csv'
    if not csv_path.exists():
        raise FileNotFoundError(f'Missing labels CSV: {csv_path}')
    return pd.read_csv(csv_path)


def _pick_sample(df: pd.DataFrame, label: int, image_id: str | None) -> pd.Series:
    subset = df[df['label'] == label]
    if subset.empty:
        raise ValueError(f'No samples found for label={label}')

    if image_id is not None:
        match = subset[subset['id'] == image_id]
        if match.empty:
            raise ValueError(f'Image id {image_id} was not found for label={label}')
        return match.iloc[0]

    return subset.sample(1, random_state=CFG['seed']).iloc[0]


def _build_input_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image.convert('RGB'), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    mean = torch.tensor(MEAN).view(3, 1, 1)
    std = torch.tensor(STD).view(3, 1, 1)
    return (tensor - mean) / std


def _get_module_by_path(model: nn.Module, path: str) -> nn.Module:
    current: nn.Module | nn.Sequential | nn.ModuleList | torch.Tensor = model
    for part in path.split('.'):
        if part.isdigit():
            current = current[int(part)]
        else:
            current = getattr(current, part)
    if not isinstance(current, nn.Module):
        raise TypeError(f'Target layer {path} did not resolve to a module')
    return current


def _compute_gradcam(model: nn.Module, target_layer: nn.Module, input_tensor: torch.Tensor) -> tuple[np.ndarray, float]:
    activation_store: dict[str, torch.Tensor] = {}
    gradient_store: dict[str, torch.Tensor] = {}

    def forward_hook(_module, _inputs, output):
        activation_store['value'] = output.detach()

    def backward_hook(_module, _grad_input, grad_output):
        gradient_store['value'] = grad_output[0].detach()

    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)

    try:
        model.zero_grad(set_to_none=True)
        logits = model(input_tensor)
        score = logits.reshape(-1)[0]
        prob = torch.sigmoid(score).item()
        score.backward()

        activations = activation_store['value']
        gradients = gradient_store['value']
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * activations).sum(dim=1, keepdim=True))
        cam = torch.nn.functional.interpolate(cam, size=input_tensor.shape[-2:], mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam -= cam.min()
        cam_max = cam.max()
        if cam_max > 0:
            cam /= cam_max
        return cam.astype(np.float32), float(prob)
    finally:
        forward_handle.remove()
        backward_handle.remove()


def _save_gradcam_figure(original: Image.Image, cam: np.ndarray, output_path: Path, title: str):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle(title, fontsize=13, fontweight='bold')

    axes[0].imshow(original)
    axes[0].set_title('Original')
    axes[0].axis('off')

    axes[1].imshow(cam, cmap='jet')
    axes[1].set_title('Grad-CAM')
    axes[1].axis('off')

    axes[2].imshow(original)
    axes[2].imshow(cam, cmap='jet', alpha=0.45)
    axes[2].set_title('Overlay')
    axes[2].axis('off')

    plt.tight_layout()
    fig.savefig(output_path, format='jpeg', dpi=150, bbox_inches='tight')
    plt.close(fig)


def _load_model(checkpoint_path: Path, device: torch.device) -> nn.Module:
    model = get_model('cnn').to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get('state_dict', checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _iter_targets(df: pd.DataFrame, malignant_id: str | None, non_malignant_id: str | None) -> Iterable[tuple[str, pd.Series]]:
    yield 'malignant', _pick_sample(df, 1, malignant_id)
    yield 'non_malignant', _pick_sample(df, 0, non_malignant_id)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Generate Grad-CAM JPEGs for one malignant and one non-malignant CNN sample.')
    parser.add_argument('--data-dir', default='data', help='Project data directory containing train_labels.csv and train/')
    parser.add_argument('--output-dir', default='outputs/gradcam', help='Directory where JPEG outputs will be saved')
    parser.add_argument('--checkpoint', default=None, help='Optional path to a CNN checkpoint (.pth)')
    parser.add_argument('--best-params', default='outputs/optuna/cnn/best_params.json', help='Optuna best_params JSON used to resolve the checkpoint when --checkpoint is omitted')
    parser.add_argument('--malignant-id', default=None, help='Optional malignant image id to use instead of auto-selecting one')
    parser.add_argument('--non-malignant-id', default=None, help='Optional non-malignant image id to use instead of auto-selecting one')
    parser.add_argument('--target-layer', default='features.3.3', help='CNN layer used for Grad-CAM hooks')
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = _repo_root()
    data_dir = _resolve_path(args.data_dir, root)
    images_dir = data_dir / 'train'
    output_dir = _resolve_path(args.output_dir, root)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = _load_checkpoint_path(args.checkpoint, args.best_params)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f'Checkpoint not found: {checkpoint_path}')

    df = _load_labels(data_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = _load_model(checkpoint_path, device)
    target_layer = _get_module_by_path(model, args.target_layer)

    for class_name, row in _iter_targets(df, args.malignant_id, args.non_malignant_id):
        image_id = row['id']
        label = int(row['label'])
        image_path = images_dir / f'{image_id}.tif'
        if not image_path.exists():
            raise FileNotFoundError(f'Missing image: {image_path}')

        original = Image.open(image_path).convert('RGB')
        input_tensor = _build_input_tensor(original).unsqueeze(0).to(device)
        cam, prob = _compute_gradcam(model, target_layer, input_tensor)

        label_name = 'malignant' if label == 1 else 'non_malignant'
        output_path = output_dir / f'{class_name}_{label_name}_{image_id}_gradcam.jpeg'
        title = f'{class_name.replace("_", " ").title()} | label={label_name} | predicted p(malignant)={prob:.4f}'
        _save_gradcam_figure(original, cam, output_path, title)
        print(f'Saved {output_path}')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())