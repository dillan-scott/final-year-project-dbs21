from __future__ import annotations

from typing import Mapping

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score
from tqdm.auto import tqdm


M_HIDDEN_DEFAULT  = (256, 128, 64)
M_EPOCHS_DEFAULT  = 30
M_LR_DEFAULT      = 1e-3
M_BATCH_DEFAULT   = 4096


class BaseLearnerMLP(nn.Module):
    """Binary MLP classifier: BENIGN (0) vs ATTACK (1)."""

    def __init__(self, input_dim: int, hidden_dims: tuple = M_HIDDEN_DEFAULT):
        super().__init__()
        layers, prev = [], input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_base_learner(
    model: BaseLearnerMLP,
    X: np.ndarray,
    y: np.ndarray,
    epochs: int = M_EPOCHS_DEFAULT,
    lr: float = M_LR_DEFAULT,
    batch_size: int = M_BATCH_DEFAULT,
    class_weight: np.ndarray | None = None,
    device: torch.device | str | None = None,
    loss_log_every: int = 20,
) -> None:
    """Train M from scratch (or continue training) on (X, y) with cross-entropy."""
    dev = torch.device(device) if device is not None else next(model.parameters()).device
    model.train()
    X_t = torch.from_numpy(X).to(dev)
    y_t = torch.from_numpy(y).long().to(dev)
    opt = optim.Adam(model.parameters(), lr=lr)
    w = (torch.tensor(class_weight, dtype=torch.float32, device=dev)
         if class_weight is not None else None)
    ce = nn.CrossEntropyLoss(weight=w)
    for e in tqdm(range(epochs), leave=False):
        perm = torch.randperm(len(X_t), device=dev)
        epoch_loss, n_batches = 0.0, 0
        for i in range(0, len(X_t), batch_size):
            b = perm[i:i + batch_size]
            opt.zero_grad()
            loss = ce(model(X_t[b]), y_t[b])
            loss.backward()
            opt.step()
            epoch_loss += float(loss.detach())
            n_batches += 1
        if loss_log_every and ((e + 1) % loss_log_every == 0 or (e + 1) == epochs):
            print(f"    [M {e + 1}/{epochs}] loss={epoch_loss / max(n_batches, 1):.4f}")
    model.eval()


@torch.no_grad()
def eval_base_learner(
    model: BaseLearnerMLP,
    X_eval: torch.Tensor,
    y_eval_bin: np.ndarray,
    novel_mask: np.ndarray,
    per_class_masks: Mapping[str, np.ndarray] | None = None,
    device: torch.device | str | None = None,
) -> dict:
    """
    Evaluate M on a fixed balanced eval set.

    Returns:
        dict with keys 'accuracy', 'f1', 'fnr_novel', plus 'fnr_<name>' for each
        per-class mask provided
    """
    dev = torch.device(device) if device is not None else next(model.parameters()).device
    model.eval()
    preds = model(X_eval.to(dev)).argmax(1).cpu().numpy()

    def _fnr(mask: np.ndarray) -> float:
        return float(1 - accuracy_score(y_eval_bin[mask], preds[mask])) if mask.any() else float('nan')

    out = {
        'accuracy':  float(accuracy_score(y_eval_bin, preds)),
        'f1':        float(f1_score(y_eval_bin, preds, zero_division=0)),
        'fnr_novel': _fnr(novel_mask),
    }
    if per_class_masks:
        for name, mask in per_class_masks.items():
            out[f'fnr_{name}'] = _fnr(mask)
    return out


@torch.no_grad()
def eval_on_stream_batch(
    model: BaseLearnerMLP,
    X_b: np.ndarray,
    y_b_bin: np.ndarray,
    device: torch.device | str | None = None,
) -> dict:
    """
    Evaluate M on the current stream batch's own samples.

    Different from eval_base_learner (which uses the fixed pre-built benchmark);
    this captures M's instantaneous performance on the traffic actually flowing
    through the IDS at this batch. Under attack, y_b_bin contains the labels
    as observed by the system (including any attacker forged labels).

    Returns:
        dict with keys 'accuracy_stream', 'f1_stream'
    """
    dev = torch.device(device) if device is not None else next(model.parameters()).device
    model.eval()
    preds = model(torch.from_numpy(X_b).to(dev)).argmax(1).cpu().numpy()
    return {
        'accuracy_stream': float(accuracy_score(y_b_bin, preds)),
        'f1_stream':       float(f1_score(y_b_bin, preds, zero_division=0)),
    }


class GPULoader:
    """Minimal GPU-resident data loader for the detector training functions."""

    def __init__(self, X: np.ndarray, y: np.ndarray, batch_size: int, device: torch.device | str):
        self.X = torch.from_numpy(X).to(device)
        self.y = torch.from_numpy(y).to(device)
        self.bs = batch_size
        self.n = len(X)

    def __len__(self) -> int:
        return (self.n + self.bs - 1) // self.bs

    def __iter__(self):
        perm = torch.randperm(self.n, device=self.X.device)
        for i in range(0, self.n, self.bs):
            b = perm[i:i + self.bs]
            yield self.X[b], self.y[b]
