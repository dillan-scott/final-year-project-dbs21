from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Callable, Mapping

import numpy as np
import torch

from .base_learner import (
    BaseLearnerMLP,
    GPULoader,
    eval_base_learner,
    eval_on_stream_batch,
    train_base_learner,
)
from .training import calibrate_concept_threshold, build_known_exemplars

_STREAM_VERBOSE = True


def set_stream_verbose(enabled: bool = True) -> None:
    global _STREAM_VERBOSE
    _STREAM_VERBOSE = enabled


def _vlog(*args, **kwargs) -> None:
    if _STREAM_VERBOSE:
        print(*args, **kwargs)


@dataclass
class BatchResult:
    """Snapshot of one prequential batch's processing outcome, including M/DD
    metrics and system state."""

    batch_idx: int
    day_idx: int
    drift_frac: float
    concepts_found: int
    retrain_fired: bool
    n_poisoned: int
    buf_size: int
    accuracy: float
    f1: float
    fnr_novel: float
    extras: dict = field(default_factory=dict)

    @classmethod
    def from_metrics(
        cls,
        batch_idx: int,
        day_idx: int,
        drift_frac: float,
        concepts_found: int,
        retrain_fired: bool,
        n_poisoned: int,
        buf_size: int,
        metrics: dict,
        extras_extra: dict | None = None,
    ) -> "BatchResult":
        """Build a BatchResult from an eval_base_learner dict + system extras."""
        explicit = {"accuracy", "f1", "fnr_novel"}
        extras = {k: v for k, v in metrics.items() if k not in explicit}
        if extras_extra:
            extras.update(extras_extra)
        return cls(
            batch_idx=batch_idx,
            day_idx=day_idx,
            drift_frac=drift_frac,
            concepts_found=concepts_found,
            retrain_fired=retrain_fired,
            n_poisoned=n_poisoned,
            buf_size=buf_size,
            accuracy=metrics["accuracy"],
            f1=metrics["f1"],
            fnr_novel=metrics["fnr_novel"],
            extras=extras,
        )


class ConceptBuffer:
    """Accumulates DD-flagged drifted instances between concept-discovery firings"""

    def __init__(self):
        self.X_buf: list[np.ndarray] = []
        self.y_buf: list[np.ndarray] = []

    def push(self, X_d: np.ndarray, y_d: np.ndarray) -> None:
        if len(X_d):
            self.X_buf.append(X_d)
            self.y_buf.append(y_d)

    @property
    def size(self) -> int:
        return sum(len(x) for x in self.X_buf)

    def get(self, X_ex: np.ndarray, y_ex: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.vstack(self.X_buf + [X_ex]),
            np.concatenate(self.y_buf + [y_ex]),
        )

    def reset(self) -> None:
        self.X_buf = []
        self.y_buf = []


class StaticDStaticM:
    """M trained once, never retrains. Provides the reference FNR for attack impact"""

    def __init__(
        self,
        base_learner: BaseLearnerMLP,
        X_eval: torch.Tensor,
        y_eval_bin: np.ndarray,
        novel_mask: np.ndarray,
        per_class_masks: Mapping[str, np.ndarray] | None = None,
    ):
        self.base_learner = base_learner
        self.X_eval = X_eval
        self.y_eval_bin = y_eval_bin
        self.novel_mask = novel_mask
        self.per_class_masks = per_class_masks or {}

    def process_batch(
        self,
        batch_idx: int,
        day_b: np.ndarray,
        X_b: np.ndarray,
        y_b: np.ndarray,
        n_poisoned: int = 0,
    ) -> BatchResult:
        m = eval_base_learner(
            self.base_learner, self.X_eval, self.y_eval_bin, self.novel_mask, self.per_class_masks
        )
        stream_m = eval_on_stream_batch(self.base_learner, X_b, y_b)
        return BatchResult.from_metrics(
            batch_idx=batch_idx,
            day_idx=int(round(float(day_b.mean()))),
            drift_frac=float("nan"),
            concepts_found=0,
            retrain_fired=False,
            n_poisoned=n_poisoned,
            buf_size=0,
            metrics=m,
            extras_extra=stream_m,
        )


class StaticDAdaptiveM:
    """DD encoder is held static (only NCM prototypes grow via concept discovery).
    M retrains from scratch on (concept buffer + known exemplars + accumulated novel
    exemplars) when concept discovery fires.
    """

    # iCaRL-style anti-forgetting budget for newly-discovered concepts
    NOVEL_EXEMPLARS_PER_CONCEPT = 500

    # Deviation from paper: gate concept-discovery firing on buffer composition.
    #
    # Two-part check:
    #   - MIN_ATTACK_FOR_FIRE: absolute floor on attack-labelled drifted samples
    #   - MIN_ATTACK_FRAC_FOR_FIRE: composition floor — the buffer's attack fraction
    #     must exceed this to fire
    # Set BOTH to 0 to recover paper-literal behaviour.
    MIN_ATTACK_FOR_FIRE = 500
    MIN_ATTACK_FRAC_FOR_FIRE = 0.3

    def __init__(
        self,
        detector,
        base_learner: BaseLearnerMLP,
        X_exemplars: np.ndarray,
        y_exemplars: np.ndarray,
        m_init_state: dict,
        X_eval: torch.Tensor,
        y_eval_bin: np.ndarray,
        novel_mask: np.ndarray,
        per_class_masks: Mapping[str, np.ndarray] | None = None,
        concept_batch: int = 256,
    ):
        self.detector = detector
        self.base_learner = base_learner
        self.X_ex = X_exemplars
        self.y_ex = y_exemplars
        self.m_init_state = m_init_state
        self.X_eval = X_eval
        self.y_eval_bin = y_eval_bin
        self.novel_mask = novel_mask
        self.per_class_masks = per_class_masks or {}
        self.concept_batch = concept_batch
        self.n_proto_at_last_retrain = detector.ncm.num_classes
        self.novel_X: np.ndarray | None = None
        self.novel_y: np.ndarray | None = None
        self.retrain_count = 0

    def _rebuild_base_learner(
        self, X_ret_bin: np.ndarray, y_ret_bin: np.ndarray, epochs: int = 100
    ) -> None:
        """Rebuild M from scratch on the retrain set, mirroring the initial recipe."""
        for module in self.base_learner.modules():
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()
        n_cls = np.bincount(y_ret_bin.astype(np.int64), minlength=2)
        class_weight = (n_cls.max() / np.clip(n_cls, 1, None)).astype(np.float32)
        train_base_learner(
            self.base_learner, X_ret_bin, y_ret_bin, epochs=epochs, class_weight=class_weight
        )

    def process_batch(
        self,
        batch_idx: int,
        day_b: np.ndarray,
        X_b: np.ndarray,
        y_b: np.ndarray,
        concept_buf: ConceptBuffer,
        n_poisoned: int = 0,
    ) -> BatchResult:
        X_t = torch.from_numpy(X_b)

        m = eval_base_learner(
            self.base_learner, self.X_eval, self.y_eval_bin, self.novel_mask, self.per_class_masks
        )

        _, is_drifted, _ = self.detector.detect(X_t)
        drift_frac = float(is_drifted.float().mean())

        concept_buf.push(X_b[is_drifted.numpy()], y_b[is_drifted.numpy()])

        n_attack_in_buf = sum(int((y == 1).sum()) for y in concept_buf.y_buf)
        n_buf = concept_buf.size
        attack_frac = (n_attack_in_buf / n_buf) if n_buf > 0 else 0.0
        allow_fire = (
            n_attack_in_buf >= self.MIN_ATTACK_FOR_FIRE
            and attack_frac >= self.MIN_ATTACK_FRAC_FOR_FIRE
        )

        concepts_found = 0
        for s in range(0, len(X_t), self.concept_batch):
            sub = X_t[s : s + self.concept_batch]
            _, is_dr_sub, _ = self.detector.detect(sub)
            if is_dr_sub.any():
                z_sub = self.detector.encode(sub[is_dr_sub])
                if self.detector.update_with_batch_drifted(z_sub, allow_fire=allow_fire):
                    concepts_found += 1

        # Gate-state trace: which of the three firing conditions is binding this
        # batch. delta_norm is the telescoping displacement accumulator (reset to
        # None on a fire, so reads 0 on firing batches); compared against concept_T.
        _da = self.detector._delta_accumulated
        gate_trace = {
            "n_attack_in_buf": int(n_attack_in_buf),
            "attack_frac": float(attack_frac),
            "allow_fire": bool(allow_fire),
            "delta_norm": float(torch.norm(_da)) if _da is not None else 0.0,
            "concept_T": float(self.detector.concept_threshold),
        }

        retrain_fired = False
        if concepts_found > 0:
            X_buf = np.vstack(concept_buf.X_buf)
            y_buf = np.concatenate(concept_buf.y_buf)
            n_attack = int((y_buf == 1).sum())
            n_benign = int((y_buf == 0).sum())
            _vlog(
                f"  [FIRE {type(self).__name__}] "
                f"batch={batch_idx} day={int(round(float(day_b.mean())))} "
                f"buf={len(y_buf)} attack={n_attack} benign={n_benign} "
                f"concepts={concepts_found}"
            )
            with torch.no_grad():
                h_buf = self.detector.encode(torch.from_numpy(X_buf)).cpu()
            new_proto_idx = range(self.n_proto_at_last_retrain, self.detector.ncm.num_classes)
            harvested_X, harvested_y = [], []
            for pi in new_proto_idx:
                proto = self.detector.ncm.prototypes[pi].cpu()
                dists = (h_buf - proto).norm(dim=1)
                E = min(self.NOVEL_EXEMPLARS_PER_CONCEPT, len(X_buf))
                top_idx = dists.argsort()[:E].numpy()
                harvested_X.append(X_buf[top_idx])
                harvested_y.append(y_buf[top_idx])
            if harvested_X:
                new_ex_X = np.vstack(harvested_X)
                new_ex_y = np.concatenate(harvested_y)
                self.novel_X = (
                    new_ex_X if self.novel_X is None else np.vstack([self.novel_X, new_ex_X])
                )
                self.novel_y = (
                    new_ex_y if self.novel_y is None else np.concatenate([self.novel_y, new_ex_y])
                )
            self.n_proto_at_last_retrain = self.detector.ncm.num_classes
            parts_X = [X_buf, self.X_ex]
            parts_y = [y_buf, self.y_ex]
            if self.novel_X is not None:
                parts_X.append(self.novel_X)
                parts_y.append(self.novel_y)
            X_ret = np.vstack(parts_X)
            y_ret = np.concatenate(parts_y)
            self._rebuild_base_learner(X_ret, y_ret)
            concept_buf.reset()
            self.retrain_count += 1
            retrain_fired = True
        stream_m = eval_on_stream_batch(self.base_learner, X_b, y_b)
        return BatchResult.from_metrics(
            batch_idx=batch_idx,
            day_idx=int(round(float(day_b.mean()))),
            drift_frac=drift_frac,
            concepts_found=concepts_found,
            retrain_fired=retrain_fired,
            n_poisoned=n_poisoned,
            buf_size=concept_buf.size,
            metrics=m,
            extras_extra={**stream_m, **gate_trace},
        )


class AdaptiveDAdaptiveM(StaticDAdaptiveM):
    """Both DD's encoder and M retrain when concept discovery fires.

    Extends StaticDDAdaptiveM with one extra step: after M's retrain, DD's
    encoder also retrains using the same training set as M but with multi-class
    labels (known exemplars keep their original class index; drifted-buffer
    samples are assigned the new prototype class created by concept discovery)
    """

    # Anchoring knobs for the DD retrain set, to prevent catastrophic forgetting of
    # known-class geometry
    # Capping the buffer contribution and tiling the known exemplars restores a
    # roughly balanced per-class contribution to the loss
    DD_RETRAIN_BUF_MAX = 5_000
    DD_RETRAIN_EX_REPEATS = 2

    def __init__(
        self,
        detector,
        base_learner: BaseLearnerMLP,
        X_exemplars: np.ndarray,
        y_exemplars: np.ndarray,
        y_exemplars_mc: np.ndarray,
        m_init_state: dict,
        X_eval: torch.Tensor,
        y_eval_bin: np.ndarray,
        novel_mask: np.ndarray,
        X_dd_eval_pool: np.ndarray,
        per_class_masks: Mapping[str, np.ndarray] | None = None,
        concept_batch: int = 256,
        dd_retrain_epochs: int = 300,
        dd_retrain_lr: float = 1e-4,
        dd_retrain_batch: int = 4096,
        X_cal: np.ndarray | None = None,
        y_cal_mc: np.ndarray | None = None,
    ):
        super().__init__(
            detector=detector,
            base_learner=base_learner,
            X_exemplars=X_exemplars,
            y_exemplars=y_exemplars,
            m_init_state=m_init_state,
            X_eval=X_eval,
            y_eval_bin=y_eval_bin,
            novel_mask=novel_mask,
            per_class_masks=per_class_masks,
            concept_batch=concept_batch,
        )
        self.y_ex_mc = y_exemplars_mc
        self.X_dd_eval_pool = X_dd_eval_pool
        self.dd_retrain_epochs = dd_retrain_epochs
        self.dd_retrain_lr = dd_retrain_lr
        self.dd_retrain_batch = dd_retrain_batch
        self.n_dd_retrains = 0

        self.X_cal = X_cal
        self.y_cal_mc = y_cal_mc

        self.novel_y_mc: np.ndarray | None = None

    @torch.no_grad()
    def _eval_dd_drift_rate(self) -> float:
        _, is_drifted, _ = self.detector.detect(torch.from_numpy(self.X_dd_eval_pool))
        return float(is_drifted.float().mean())

    def process_batch(
        self,
        batch_idx: int,
        day_b: np.ndarray,
        X_b: np.ndarray,
        y_b: np.ndarray,
        concept_buf: ConceptBuffer,
        n_poisoned: int = 0,
    ) -> BatchResult:
        X_t = torch.from_numpy(X_b)

        m = eval_base_learner(
            self.base_learner, self.X_eval, self.y_eval_bin, self.novel_mask, self.per_class_masks
        )

        _, is_drifted, _ = self.detector.detect(X_t)
        drift_frac = float(is_drifted.float().mean())
        concept_buf.push(X_b[is_drifted.numpy()], y_b[is_drifted.numpy()])

        n_attack_in_buf = sum(int((y == 1).sum()) for y in concept_buf.y_buf)
        n_buf = concept_buf.size
        attack_frac = (n_attack_in_buf / n_buf) if n_buf > 0 else 0.0
        allow_fire = (
            n_attack_in_buf >= self.MIN_ATTACK_FOR_FIRE
            and attack_frac >= self.MIN_ATTACK_FRAC_FOR_FIRE
        )

        concepts_found = 0
        for s in range(0, len(X_t), self.concept_batch):
            sub = X_t[s : s + self.concept_batch]
            _, is_dr_sub, _ = self.detector.detect(sub)
            if is_dr_sub.any():
                z_sub = self.detector.encode(sub[is_dr_sub])
                if self.detector.update_with_batch_drifted(z_sub, allow_fire=allow_fire):
                    concepts_found += 1

        # Gate-state trace: which of the three firing conditions is binding this
        # batch. delta_norm is the telescoping displacement accumulator (reset to
        # None on a fire, so reads 0 on firing batches); compared against concept_T.
        _da = self.detector._delta_accumulated
        gate_trace = {
            "n_attack_in_buf": int(n_attack_in_buf),
            "attack_frac": float(attack_frac),
            "allow_fire": bool(allow_fire),
            "delta_norm": float(torch.norm(_da)) if _da is not None else 0.0,
            "concept_T": float(self.detector.concept_threshold),
        }

        retrain_fired = False
        if concepts_found > 0:
            X_buf = np.vstack(concept_buf.X_buf)
            y_buf = np.concatenate(concept_buf.y_buf)
            n_attack = int((y_buf == 1).sum())
            n_benign = int((y_buf == 0).sum())
            _vlog(
                f"  [FIRE {type(self).__name__}] "
                f"batch={batch_idx} day={int(round(float(day_b.mean())))} "
                f"buf={len(y_buf)} attack={n_attack} benign={n_benign} "
                f"concepts={concepts_found}"
            )
            with torch.no_grad():
                h_buf = self.detector.encode(torch.from_numpy(X_buf)).cpu()
            new_proto_idx = range(self.n_proto_at_last_retrain, self.detector.ncm.num_classes)
            harvested_X, harvested_y_bin, harvested_y_mc = [], [], []
            for pi in new_proto_idx:
                proto = self.detector.ncm.prototypes[pi].cpu()
                dists = (h_buf - proto).norm(dim=1)
                E = min(self.NOVEL_EXEMPLARS_PER_CONCEPT, len(X_buf))
                top_idx = dists.argsort()[:E].numpy()
                harvested_X.append(X_buf[top_idx])
                harvested_y_bin.append(y_buf[top_idx])

                harvested_y_mc.append(np.full(len(top_idx), pi, dtype=np.int64))
            if harvested_X:
                new_ex_X = np.vstack(harvested_X)
                new_ex_y_bin = np.concatenate(harvested_y_bin)
                new_ex_y_mc = np.concatenate(harvested_y_mc)
                self.novel_X = (
                    new_ex_X if self.novel_X is None else np.vstack([self.novel_X, new_ex_X])
                )
                self.novel_y = (
                    new_ex_y_bin
                    if self.novel_y is None
                    else np.concatenate([self.novel_y, new_ex_y_bin])
                )
                self.novel_y_mc = (
                    new_ex_y_mc
                    if self.novel_y_mc is None
                    else np.concatenate([self.novel_y_mc, new_ex_y_mc])
                )
            n_new_protos = self.detector.ncm.num_classes - self.n_proto_at_last_retrain
            self.n_proto_at_last_retrain = self.detector.ncm.num_classes

            parts_X_bin = [X_buf, self.X_ex]
            parts_y_bin = [y_buf, self.y_ex]
            if self.novel_X is not None:
                parts_X_bin.append(self.novel_X)
                parts_y_bin.append(self.novel_y)
            X_ret_bin = np.vstack(parts_X_bin)
            y_ret_bin = np.concatenate(parts_y_bin)
            self._rebuild_base_learner(X_ret_bin, y_ret_bin)

            latest_new_proto_idx = self.detector.ncm.num_classes - 1
            if len(X_buf) > self.DD_RETRAIN_BUF_MAX:
                sub_idx = np.random.default_rng(self.n_dd_retrains).choice(
                    len(X_buf), self.DD_RETRAIN_BUF_MAX, replace=False
                )
                X_buf_dd = X_buf[sub_idx]
            else:
                X_buf_dd = X_buf
            y_buf_mc_dd = np.full(len(X_buf_dd), latest_new_proto_idx, dtype=np.int64)
            X_ex_tile = np.tile(self.X_ex, (self.DD_RETRAIN_EX_REPEATS, 1))
            y_ex_mc_tile = np.tile(self.y_ex_mc, self.DD_RETRAIN_EX_REPEATS)
            parts_X_mc = [X_buf_dd, X_ex_tile]
            parts_y_mc = [y_buf_mc_dd, y_ex_mc_tile]
            if self.novel_X is not None:
                parts_X_mc.append(self.novel_X)
                parts_y_mc.append(self.novel_y_mc)
            X_ret_mc = np.vstack(parts_X_mc)
            y_ret_mc = np.concatenate(parts_y_mc)
            _vlog(
                f"  [DD retrain set] buf={len(X_buf_dd):,} (capped from {len(X_buf):,})  "
                f"ex={len(X_ex_tile):,} ({self.DD_RETRAIN_EX_REPEATS}x)  "
                f"novel={len(self.novel_X) if self.novel_X is not None else 0}"
            )
            dd_loader = GPULoader(
                X_ret_mc,
                y_ret_mc,
                self.dd_retrain_batch,
                device=next(self.base_learner.parameters()).device,
            )
            self.detector.retrain(
                dd_loader,
                epochs=self.dd_retrain_epochs,
                lr=self.dd_retrain_lr,
            )
            self.n_dd_retrains += 1

            # Recalibrate the concept-discovery threshold T on the new encoder
            T_old = self.detector.concept_threshold
            if self.X_cal is not None:
                cal_src = self.X_cal
                cal_label = "X_cal"
            else:
                cal_src = self.X_ex
                cal_label = "X_ex (fallback)"
            T_raw = calibrate_concept_threshold(
                self.detector,
                cal_src,
                device=next(self.base_learner.parameters()).device,
            )
            T_floor = 2.0 * self.detector.drift_threshold
            if T_raw < T_floor:
                self.detector.concept_threshold = T_floor
            _vlog(
                f"  [T recalibrated on {cal_label}] {T_old:.3f} -> "
                f"{self.detector.concept_threshold:.3f} (raw {T_raw:.3f}, "
                f"floor {T_floor:.3f})"
            )

            concept_buf.reset()
            self.retrain_count += 1
            retrain_fired = True

        dd_drift_rate_pool = self._eval_dd_drift_rate()
        stream_m = eval_on_stream_batch(self.base_learner, X_b, y_b)
        extras_extra = {
            "dd_drift_rate_pool": dd_drift_rate_pool,
            "n_dd_retrains": self.n_dd_retrains,
            "n_prototypes": int(self.detector.ncm.num_classes),
            **gate_trace,
            **stream_m,
        }
        return BatchResult.from_metrics(
            batch_idx=batch_idx,
            day_idx=int(round(float(day_b.mean()))),
            drift_frac=drift_frac,
            concepts_found=concepts_found,
            retrain_fired=retrain_fired,
            n_poisoned=n_poisoned,
            buf_size=concept_buf.size,
            metrics=m,
            extras_extra=extras_extra,
        )


def run_stream(
    system,
    X_stream: np.ndarray,
    y_stream_bin: np.ndarray,
    d_stream: np.ndarray,
    novel_pool: np.ndarray,
    *,
    poison_frac: float = 0.0,
    schedule_fn: Callable[[int], float] | None = None,
    rng_seed: int = 1041,
    batch_size: int = 1000,
    min_inject_day: int = 1,
    inject_label: int = 0,
) -> list[BatchResult]:
    """
    Prequential streaming loop with optional adversarial injection.

    When schedule_fn is None (default), the per-batch poison fraction is the
    constant poison_frac

    When schedule_fn is provided, the per-batch poison fraction is
    schedule_fn(batch_idx)

    min_inject_day gates injection to batches whose day index reaches it; the
    default of 1 skips the all-benign Monday warm-up. Set to 0 to permit
    injection from the very start of the stream (used by the manufactured-novelty
    force-false experiment, which injects in a pre-drift window).

    inject_label is the label attached to every injected (novel-class) flow.
    The default 0 (BENIGN) is the label-flipping payload shared by the suppress
    and oscillating force-false attacks. Set to 1 (ATTACK) to inject under the
    true coarse label, which leaves the buffer's attack fraction high enough to
    pass the concept-discovery gate.
    """
    from tqdm.auto import tqdm

    rng_atk = np.random.default_rng(rng_seed)
    is_coupled = isinstance(system, StaticDAdaptiveM)
    concept_buf = ConceptBuffer() if is_coupled else None
    results: list[BatchResult] = []

    for b_start in tqdm(
        range(0, len(X_stream), batch_size), leave=False, desc=f"p={poison_frac:.0%}"
    ):
        X_b = X_stream[b_start : b_start + batch_size].copy()
        y_b = y_stream_bin[b_start : b_start + batch_size].copy()
        d_b = d_stream[b_start : b_start + batch_size]
        batch_idx = b_start // batch_size

        if schedule_fn is not None:
            this_p = schedule_fn(batch_idx)
        else:
            this_p = poison_frac

        n_poison = 0
        if this_p > 0 and (d_b >= min_inject_day).any():
            n_poison = int(len(X_b) * this_p)
            if n_poison > 0:
                adv_idx = rng_atk.choice(len(novel_pool), n_poison, replace=True)
                X_adv = novel_pool[adv_idx]
                y_adv = np.full(n_poison, inject_label, dtype=np.int64)
                X_b = np.vstack([X_b, X_adv])
                y_b = np.concatenate([y_b, y_adv])
                perm = rng_atk.permutation(len(X_b))
                X_b, y_b = X_b[perm], y_b[perm]

        if is_coupled:
            r = system.process_batch(batch_idx, d_b, X_b, y_b, concept_buf, n_poison)
        else:
            r = system.process_batch(batch_idx, d_b, X_b, y_b, n_poison)
        results.append(r)
    return results


def fresh_system(
    system_type: str,
    *,
    detector_init,
    m_init_state: dict,
    X_exemplars: np.ndarray,
    y_exemplars: np.ndarray,
    X_eval: torch.Tensor,
    y_eval_bin: np.ndarray,
    novel_mask: np.ndarray,
    per_class_masks: Mapping[str, np.ndarray] | None = None,
    input_dim: int,
    m_hidden: tuple = (256, 128, 64),
    device: torch.device | str = "cuda",
    concept_batch: int = 256,
):
    """Factory for a fresh system instance reset to its initial state."""
    det = detector_init()
    ml = BaseLearnerMLP(input_dim, m_hidden).to(device)
    ml.load_state_dict(copy.deepcopy(m_init_state))

    if system_type == "static":
        return StaticDStaticM(ml, X_eval, y_eval_bin, novel_mask, per_class_masks)
    if system_type == "static_dd_adaptive_m" or system_type == "coupled":
        return StaticDAdaptiveM(
            detector=det,
            base_learner=ml,
            X_exemplars=X_exemplars,
            y_exemplars=y_exemplars,
            m_init_state=m_init_state,
            X_eval=X_eval,
            y_eval_bin=y_eval_bin,
            novel_mask=novel_mask,
            per_class_masks=per_class_masks,
            concept_batch=concept_batch,
        )
    raise ValueError(f"Unknown system_type: {system_type!r}")
