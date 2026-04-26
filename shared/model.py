"""TinyCNN model + transfer-learning helper.

Kept architecturally identical to the notebook's original definition so that
the saved baseline checkpoint loads cleanly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn


class TinyCNN(nn.Module):
    """Small CNN for 28x28 grayscale character classification."""

    def __init__(self, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def _get_conv_modules(model: TinyCNN):
    """Return the two Conv2d modules inside the sequential."""
    return [m for m in model.net if isinstance(m, nn.Conv2d)]


def _get_classifier_head(model: TinyCNN) -> nn.Linear:
    """Return the final Linear layer (the classifier head)."""
    last_linear = None
    for m in model.net:
        if isinstance(m, nn.Linear):
            last_linear = m
    assert last_linear is not None, "TinyCNN should always contain Linear layers"
    return last_linear


def build_transfer_model(
    num_classes: int,
    checkpoint_path: Union[str, Path],
    *,
    freeze_conv: bool = True,
    reset_head: bool = True,
    device: Optional[torch.device] = None,
) -> TinyCNN:
    """Load the baseline EMNIST checkpoint and prepare it for fine-tuning.

    Steps:
      1. Build a fresh TinyCNN and load the baseline weights into it.
      2. Optionally freeze the conv layers so only the dense layers train.
      3. Optionally re-initialize the final classifier head, which is useful
         if the downstream task uses a different ``num_classes`` or if you
         want the head to adapt aggressively to a new writer's handwriting.

    Parameters
    ----------
    num_classes:
        Number of output classes for the new task. Must match the baseline
        when ``reset_head=False``.
    checkpoint_path:
        Path to the baseline ``state_dict`` produced by the EMNIST training
        loop in the notebook.
    freeze_conv:
        If True, ``requires_grad`` is set to False on the two Conv2d layers.
    reset_head:
        If True, the final Linear classifier is replaced with a fresh
        ``Linear(128, num_classes)``. The head's parameters always remain
        trainable.
    device:
        Optional device to move the model onto after loading.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Baseline checkpoint not found: {checkpoint_path}")

    state = torch.load(checkpoint_path, map_location="cpu")

    head_weight_key = "net.9.weight"
    if head_weight_key in state:
        baseline_num_classes = state[head_weight_key].shape[0]
    else:
        baseline_num_classes = num_classes

    model = TinyCNN(baseline_num_classes)
    model.load_state_dict(state)

    if reset_head or baseline_num_classes != num_classes:
        new_head = nn.Linear(128, num_classes)
        nn.init.kaiming_uniform_(new_head.weight, a=5 ** 0.5)
        nn.init.zeros_(new_head.bias)
        model.net[-1] = new_head

    if freeze_conv:
        for conv in _get_conv_modules(model):
            for p in conv.parameters():
                p.requires_grad = False

    if device is not None:
        model = model.to(device)

    return model


def trainable_parameter_summary(model: nn.Module) -> dict:
    """Small helper for printing what's frozen vs trainable in the notebook."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_params": int(total),
        "trainable_params": int(trainable),
        "frozen_params": int(total - trainable),
    }
