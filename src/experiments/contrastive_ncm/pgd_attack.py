from __future__ import annotations

import numpy as np
import torch


def drift_score(autoencoder, ncm, X: torch.Tensor) -> torch.Tensor:
    """
    Minimum hybrid NCM distance to the known prototypes (DD's drift score).

    Differentiable w.r.t. ``X``; a sample is flagged drifted when this exceeds the
    detector's ``drift_threshold``.
    """
    h = autoencoder.encode(X)
    D = ncm._compute_distance(h)  # (N, C)
    return D.min(dim=1).values  # (N,)


def flag_rate(autoencoder, ncm, drift_threshold: float, X, device=None) -> float:
    """
    Fraction of ``X`` the detector flags as drifted (score > threshold).
    """
    device = device or next(autoencoder.parameters()).device
    Xt = torch.as_tensor(np.asarray(X), dtype=torch.float32, device=device)
    autoencoder.eval()
    with torch.no_grad():
        s = drift_score(autoencoder, ncm, Xt)
    return float((s > drift_threshold).float().mean().item())


def pgd_drift_attack(
    autoencoder,
    ncm,
    X,
    *,
    eps: float,
    steps: int = 50,
    alpha: float | None = None,
    norm: str = "linf",
    x_min=None,
    x_max=None,
    device=None,
):
    """
    PGD that MAXIMISES the drift score, pushing samples past the threshold.
    """
    was_numpy = isinstance(X, np.ndarray)
    device = device or next(autoencoder.parameters()).device
    X0 = torch.as_tensor(np.asarray(X), dtype=torch.float32, device=device)
    if alpha is None:
        alpha = 2.5 * eps / max(steps, 1)
    lo = (
        None
        if x_min is None
        else torch.as_tensor(np.asarray(x_min), dtype=torch.float32, device=device)
    )
    hi = (
        None
        if x_max is None
        else torch.as_tensor(np.asarray(x_max), dtype=torch.float32, device=device)
    )

    autoencoder.eval()
    delta = torch.zeros_like(X0)
    for _ in range(steps):
        delta.requires_grad_(True)
        score = drift_score(autoencoder, ncm, X0 + delta).sum()
        (grad,) = torch.autograd.grad(score, delta)
        with torch.no_grad():
            if norm == "linf":
                delta = (delta + alpha * grad.sign()).clamp_(-eps, eps)
            elif norm == "l2":
                g = grad / grad.flatten(1).norm(dim=1).clamp_min(1e-12).view(-1, 1)
                delta = delta + alpha * g
                scale = (eps / delta.flatten(1).norm(dim=1).clamp_min(1e-12)).clamp_max(1.0)
                delta = delta * scale.view(-1, 1)
            else:
                raise ValueError(f"unknown norm {norm!r}")
            if lo is not None or hi is not None:
                X_adv = X0 + delta
                if lo is not None:
                    X_adv = torch.maximum(X_adv, lo)
                if hi is not None:
                    X_adv = torch.minimum(X_adv, hi)
                delta = X_adv - X0
        delta = delta.detach()
    X_adv = (X0 + delta).detach()
    return X_adv.cpu().numpy() if was_numpy else X_adv


def pgd_directional_attack(
    autoencoder,
    X,
    direction,
    *,
    eps: float,
    sign: float = 1.0,
    steps: int = 50,
    alpha: float | None = None,
    norm: str = "linf",
    x_min=None,
    x_max=None,
    device=None,
):
    """
    PGD that pushes embeddings maximally along ``sign * direction`` in latent space.
    """
    was_numpy = isinstance(X, np.ndarray)
    device = device or next(autoencoder.parameters()).device
    X0 = torch.as_tensor(np.asarray(X), dtype=torch.float32, device=device)
    u = torch.as_tensor(np.asarray(direction), dtype=torch.float32, device=device)
    u = u / u.norm().clamp_min(1e-12)
    if alpha is None:
        alpha = 2.5 * eps / max(steps, 1)
    lo = (
        None
        if x_min is None
        else torch.as_tensor(np.asarray(x_min), dtype=torch.float32, device=device)
    )
    hi = (
        None
        if x_max is None
        else torch.as_tensor(np.asarray(x_max), dtype=torch.float32, device=device)
    )

    autoencoder.eval()
    delta = torch.zeros_like(X0)
    for _ in range(steps):
        delta.requires_grad_(True)
        obj = sign * (autoencoder.encode(X0 + delta) @ u).sum()  # push each embedding along sign*u
        (grad,) = torch.autograd.grad(obj, delta)
        with torch.no_grad():
            if norm == "linf":
                delta = (delta + alpha * grad.sign()).clamp_(-eps, eps)
            elif norm == "l2":
                g = grad / grad.flatten(1).norm(dim=1).clamp_min(1e-12).view(-1, 1)
                delta = delta + alpha * g
                scale = (eps / delta.flatten(1).norm(dim=1).clamp_min(1e-12)).clamp_max(1.0)
                delta = delta * scale.view(-1, 1)
            else:
                raise ValueError(f"unknown norm {norm!r}")
            if lo is not None or hi is not None:
                X_adv = X0 + delta
                if lo is not None:
                    X_adv = torch.maximum(X_adv, lo)
                if hi is not None:
                    X_adv = torch.minimum(X_adv, hi)
                delta = X_adv - X0
        delta = delta.detach()
    X_adv = (X0 + delta).detach()
    return X_adv.cpu().numpy() if was_numpy else X_adv


def pgd_meanshift_attack(
    autoencoder,
    X,
    *,
    eps: float,
    steps: int = 50,
    alpha: float | None = None,
    norm: str = "linf",
    x_min=None,
    x_max=None,
    device=None,
):
    """
    Two-pool attack that DIRECTLY maximises the latent batch-mean separation.
    """
    was_numpy = isinstance(X, np.ndarray)
    device = device or next(autoencoder.parameters()).device
    X0 = torch.as_tensor(np.asarray(X), dtype=torch.float32, device=device)
    if alpha is None:
        alpha = 2.5 * eps / max(steps, 1)
    lo = (
        None
        if x_min is None
        else torch.as_tensor(np.asarray(x_min), dtype=torch.float32, device=device)
    )
    hi = (
        None
        if x_max is None
        else torch.as_tensor(np.asarray(x_max), dtype=torch.float32, device=device)
    )

    def _project(delta):
        if norm == "linf":
            delta = delta.clamp(-eps, eps)
        elif norm == "l2":
            scale = (eps / delta.flatten(1).norm(dim=1).clamp_min(1e-12)).clamp_max(1.0)
            delta = delta * scale.view(-1, 1)
        else:
            raise ValueError(f"unknown norm {norm!r}")
        if lo is not None or hi is not None:
            x_adv = X0 + delta
            if lo is not None:
                x_adv = torch.maximum(x_adv, lo)
            if hi is not None:
                x_adv = torch.minimum(x_adv, hi)
            delta = x_adv - X0
        return delta

    autoencoder.eval()
    rng = np.random.default_rng(0)
    d0 = rng.standard_normal(X0.shape[1]).astype(np.float32)
    d0 = d0 / (np.abs(d0).max() + 1e-12) * (0.5 * eps)
    d0 = torch.as_tensor(d0, dtype=torch.float32, device=device)
    dA = _project(d0.unsqueeze(0).expand_as(X0).clone())
    dB = _project((-d0).unsqueeze(0).expand_as(X0).clone())
    for _ in range(steps):
        dA.requires_grad_(True)
        dB.requires_grad_(True)
        mA = autoencoder.encode(X0 + dA).mean(dim=0)
        mB = autoencoder.encode(X0 + dB).mean(dim=0)
        obj = (mA - mB).norm()
        gA, gB = torch.autograd.grad(obj, [dA, dB])
        with torch.no_grad():
            if norm == "linf":
                dA = _project(dA + alpha * gA.sign())
                dB = _project(dB + alpha * gB.sign())
            else:
                gA = gA / gA.flatten(1).norm(dim=1).clamp_min(1e-12).view(-1, 1)
                gB = gB / gB.flatten(1).norm(dim=1).clamp_min(1e-12).view(-1, 1)
                dA = _project(dA + alpha * gA)
                dB = _project(dB + alpha * gB)
        dA = dA.detach()
        dB = dB.detach()
    A = (X0 + dA).detach()
    B = (X0 + dB).detach()
    if was_numpy:
        return A.cpu().numpy(), B.cpu().numpy()
    return A, B
