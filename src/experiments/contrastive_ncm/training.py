from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from src.detectors.contrastive_ncm import ContrastiveNCMDetector

from .base_learner import (
    BaseLearnerMLP,
    GPULoader,
    train_base_learner,
)


DD_EPOCHS_DEFAULT       = 300
DD_LR_DEFAULT           = 1e-4
DD_BATCH_DEFAULT        = 4096
DD_LATENT_DEFAULT       = 32
DD_HIDDEN_DEFAULT       = 64
DD_TEMP_DEFAULT         = 0.1
DD_CAL_QUANTILE_DEFAULT = 0.90
DD_T_QUANTILE_DEFAULT   = 0.995

M_EPOCHS_INIT_DEFAULT   = 100
M_LR_DEFAULT            = 1e-3
M_BATCH_DEFAULT         = 4096

CONCEPT_BATCH_DEFAULT   = 256


@dataclass
class InitialTrainingResult:
    """Outputs of initial_training"""
    detector: ContrastiveNCMDetector
    base_learner: BaseLearnerMLP
    X_exemplars:   np.ndarray
    y_exemplars:   np.ndarray
    y_exemplars_mc: np.ndarray
    drift_threshold: float
    concept_threshold: float
    m_init_state: dict
    dd_init_ae: dict
    dd_init_ncm: object
    X_cal:        np.ndarray
    y_cal_mc:     np.ndarray


def calibrate_drift_threshold(
    detector: ContrastiveNCMDetector,
    X_cal: np.ndarray,
    quantile: float = DD_CAL_QUANTILE_DEFAULT,
) -> float:
    """Set detector.drift_threshold at the given percentile of known-class NCM distances"""
    with torch.no_grad():
        _, _, cal_dists = detector.detect(torch.from_numpy(X_cal))
    thr = float(torch.quantile(cal_dists, quantile).item())
    detector.drift_threshold = thr
    return thr


def calibrate_concept_threshold(
    detector: ContrastiveNCMDetector,
    X_cal: np.ndarray,
    device: torch.device | str,
    concept_batch: int = CONCEPT_BATCH_DEFAULT,
    quantile: float = DD_T_QUANTILE_DEFAULT,
    n_runs: int = 200,
    batches_per_run: int = 20,
    seed: int = 0,
) -> float:
    """Set detector.concept_threshold at the given percentile of batch-delta norms
    observed on known-class data (random sub-batches of size CONCEPT_BATCH)"""
    rng = np.random.default_rng(seed)
    delta_norms: list[float] = []
    for _ in range(n_runs):
        prev_mean = None
        delta_acc = None
        for _ in range(batches_per_run):
            idx = rng.choice(len(X_cal), concept_batch, replace=True)
            with torch.no_grad():
                h = detector.encode(torch.from_numpy(X_cal[idx]))
            d = detector.ncm._compute_distance(h.to(device)).min(dim=1).values.cpu()
            mask = d > detector.drift_threshold
            if not mask.any():
                continue
            bm = h[mask].cpu().mean(dim=0)
            if prev_mean is not None:
                delta = bm - prev_mean
                delta_acc = delta if delta_acc is None else delta_acc + delta
                delta_norms.append(delta_acc.norm().item())
            prev_mean = bm
    T = float(np.percentile(delta_norms, quantile * 100)) if delta_norms else 16.5
    detector.concept_threshold = T
    return T


def build_known_exemplars(
    detector: ContrastiveNCMDetector,
    X_tr: np.ndarray,
    y_tr_mc: np.ndarray,
    y_tr_bin: np.ndarray,
    n_known: int,
    E: int = 500,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """iCaRL-style herding: pick E exemplars per known class closest to its NCM prototype"""
    exemplar_X: list[np.ndarray] = []
    exemplar_y_bin: list[np.ndarray] = []
    exemplar_y_mc: list[np.ndarray] = []
    for k in range(n_known):
        mask_k = (y_tr_mc == k)
        X_k = X_tr[mask_k]
        with torch.no_grad():
            h_k = detector.encode(torch.from_numpy(X_k)).cpu()
        proto_k = detector.ncm.prototypes[k].cpu()
        top_idx = (h_k - proto_k).norm(dim=1).argsort()[:E]
        idx_np = top_idx.numpy()
        exemplar_X.append(X_k[idx_np])
        exemplar_y_bin.append(y_tr_bin[mask_k][idx_np])
        exemplar_y_mc.append(y_tr_mc[mask_k][idx_np])
    return (
        np.vstack(exemplar_X),
        np.concatenate(exemplar_y_bin),
        np.concatenate(exemplar_y_mc),
    )


def initial_training(
    X_tr: np.ndarray,
    y_tr_mc: np.ndarray,
    y_tr_bin: np.ndarray,
    n_known: int,
    input_dim: int,
    device: torch.device | str,
    *,
    dd_epochs: int = DD_EPOCHS_DEFAULT,
    dd_lr: float = DD_LR_DEFAULT,
    dd_batch: int = DD_BATCH_DEFAULT,
    dd_latent: int = DD_LATENT_DEFAULT,
    dd_hidden: int = DD_HIDDEN_DEFAULT,
    dd_temp: float = DD_TEMP_DEFAULT,
    dd_cal_quantile: float = DD_CAL_QUANTILE_DEFAULT,
    dd_t_quantile: float = DD_T_QUANTILE_DEFAULT,
    concept_batch: int = CONCEPT_BATCH_DEFAULT,
    m_hidden: tuple = (256, 128, 64),
    m_epochs_init: int = M_EPOCHS_INIT_DEFAULT,
    m_lr: float = M_LR_DEFAULT,
    m_batch: int = M_BATCH_DEFAULT,
    use_class_weights_on_m: bool = True,
    random_seed: int = 42,
) -> InitialTrainingResult:
    """End-to-end initial training pipeline

    1. Holds out 10% of X_tr for calibration
    2. Trains DD on the remaining 90%;
    3. Calibrates drift_threshold and concept_threshold
    4. Trains M with inverse-frequency class weights (if use_class_weights_on_m=True)
    5. Builds the known-class exemplar set with both binary and multi-class labels
    """
    X_dd, X_cal, y_dd, y_cal_mc = train_test_split(
        X_tr, y_tr_mc, test_size=0.10, random_state=random_seed, stratify=y_tr_mc,
    )
    dd_loader = GPULoader(X_dd, y_dd, dd_batch, device)
    detector = ContrastiveNCMDetector(
        input_dim=input_dim, hidden_dim=dd_hidden, latent_dim=dd_latent,
        temperature=dd_temp, concept_threshold=3.5, device=device,
    )
    detector.fit(dd_loader, epochs=dd_epochs, lr=dd_lr, num_classes=n_known)

    drift_threshold = calibrate_drift_threshold(detector, X_cal, dd_cal_quantile)
    concept_threshold = calibrate_concept_threshold(
        detector, X_cal, device, concept_batch, dd_t_quantile,
    )

    # M with optional inverse-frequency class weighting
    base_learner = BaseLearnerMLP(input_dim, m_hidden).to(device)
    class_weight = None
    if use_class_weights_on_m:
        n_cls = np.bincount(y_tr_bin)
        class_weight = (n_cls.max() / n_cls).astype(np.float32)
    train_base_learner(
        base_learner, X_tr, y_tr_bin,
        epochs=m_epochs_init, lr=m_lr, batch_size=m_batch,
        class_weight=class_weight, device=device,
    )

    X_ex, y_ex_bin, y_ex_mc = build_known_exemplars(
        detector, X_tr, y_tr_mc, y_tr_bin, n_known,
    )

    return InitialTrainingResult(
        detector=detector,
        base_learner=base_learner,
        X_exemplars=X_ex,
        y_exemplars=y_ex_bin,
        y_exemplars_mc=y_ex_mc,
        drift_threshold=drift_threshold,
        concept_threshold=concept_threshold,
        m_init_state=copy.deepcopy(base_learner.state_dict()),
        dd_init_ae=copy.deepcopy(detector.autoencoder.state_dict()),
        dd_init_ncm=copy.deepcopy(detector.ncm),
        X_cal=X_cal,
        y_cal_mc=y_cal_mc.astype(np.int64),
    )
