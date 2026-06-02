from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Protocol

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from alibi_detect.ad import AdversarialAE  # type: ignore
from src.detectors.contrastive_ncm.detector import ContrastiveNCMDetector


# ---------------------------------------------------------------------------
# Common protocol
# ---------------------------------------------------------------------------

class DriftDetector(Protocol):
    """Structural interface that all detector adapters must satisfy."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the detector on reference (clean) data."""
        ...

    def detect(self, X: np.ndarray, y: np.ndarray) -> bool:
        """Return True if drift / anomaly is detected in this batch."""
        ...

    def adapt(self, X: np.ndarray, y: np.ndarray) -> None:
        """Update the detector after confirmed drift."""
        ...


# ---------------------------------------------------------------------------
# Adapters
# ---------------------------------------------------------------------------

class ContrastiveNCMAdapter:
    """Translates ContrastiveNCMDetector into the DriftDetector protocol."""

    def __init__(
        self,
        detector: ContrastiveNCMDetector,
        batch_size: int = 64,
        fit_epochs: int = 300,
        fit_lr: float = 1e-4,
    ) -> None:
        self._detector = detector
        self._batch_size = batch_size
        self._fit_epochs = fit_epochs
        self._fit_lr = fit_lr
        # Drifted embeddings cached from detect(), consumed by adapt().
        self._pending_drift_embeddings: torch.Tensor | None = None

    @property
    def detector(self) -> ContrastiveNCMDetector:
        return self._detector

    def _make_loader(self, X: np.ndarray, y: np.ndarray) -> DataLoader:
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.long)
        return DataLoader(TensorDataset(X_t, y_t), batch_size=self._batch_size, shuffle=True)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        loader = self._make_loader(X, y)
        num_classes = int(np.max(y)) + 1
        self._detector.fit(loader, epochs=self._fit_epochs, lr=self._fit_lr, num_classes=num_classes)

    def detect(self, X: np.ndarray, y: np.ndarray) -> bool:
        """
        Identifies drifted samples via NCM distance (Section 4.1).

        Caches the latent embeddings of drifted samples so adapt() can feed
        them into the concept-discovery buffer (Eq. 6) without a second encode
        pass.  No state mutation happens here.
        """
        x_t = torch.tensor(X, dtype=torch.float32)
        _, is_drifted, _ = self._detector.detect(x_t)

        if is_drifted.any():
            h = self._detector.encode(x_t)
            self._pending_drift_embeddings = h[is_drifted]
        else:
            self._pending_drift_embeddings = None

        return bool(is_drifted.any())

    def adapt(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Feeds drifted embeddings into the recursive concept-discovery buffer
        (Eq. 6, Section 4.2).  Per the paper's Fig. 4 flow, the encoder is
        only retrained when update_with_batch_drifted() returns True — i.e.
        when the accumulated delta norm crosses threshold T and a new
        prototype is created.  Drift alone does not trigger a retrain.
        """
        if self._pending_drift_embeddings is None:
            return

        new_concept_found = self._detector.update_with_batch_drifted(
            self._pending_drift_embeddings,
        )

        self._pending_drift_embeddings = None

        if new_concept_found:
            loader = self._make_loader(X, y)
            self._detector.retrain(loader, epochs=self._fit_epochs, lr=self._fit_lr)


class AdversarialAEAdapter:
    """Translates alibi-detect AdversarialAE into the DriftDetector protocol."""

    def __init__(
        self,
        detector: AdversarialAE,
        fit_epochs: int = 50,
        fit_batch_size: int = 64,
    ) -> None:
        self._detector = detector
        self._fit_epochs = fit_epochs
        self._fit_batch_size = fit_batch_size

    @property
    def detector(self) -> AdversarialAE:
        return self._detector

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        # AdversarialAE is unsupervised; labels are not used.
        self._detector.fit(
            X.astype(np.float32),
            epochs=self._fit_epochs,
            batch_size=self._fit_batch_size,
            verbose=False,
        )

    def detect(self, X: np.ndarray, y: np.ndarray) -> bool:
        result = self._detector.predict(X.astype(np.float32), return_instance_score=True)
        return bool(result["data"]["is_adversarial"].any())

    def adapt(self, X: np.ndarray, y: np.ndarray) -> None:
        self._detector.fit(
            X.astype(np.float32),
            epochs=self._fit_epochs,
            batch_size=self._fit_batch_size,
            verbose=False,
        )


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class BatchResult:
    batch_idx: int
    n_samples: int
    n_correct: int
    drift_detected: bool
    was_poisoned: bool


@dataclass
class PipelineMetrics:
    total_samples: int = 0
    total_correct: int = 0
    drift_events: int = 0
    batches_processed: int = 0
    batch_results: list[BatchResult] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        return self.total_correct / self.total_samples if self.total_samples else 0.0


# ---------------------------------------------------------------------------
# Type alias for the poisoning hook
# ---------------------------------------------------------------------------

PoisonFn = Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class StreamingPipeline:
    """
    Processes a data stream through a classifier monitored by a drift detector.

    The data stream yields (X_batch, y_batch) pairs.  Before any batch reaches
    the classifier or detector, it passes through `poison_fn` — a callable with
    signature (X, y) -> (X, y).  When `poison_fn` is None the pipeline runs on
    clean data.  Call `set_poison_fn()` to inject or remove an attack at any
    point without restarting the pipeline.
    """

    def __init__(
        self,
        data_stream: Iterable[tuple[np.ndarray, np.ndarray]],
        classifier,
        detector: ContrastiveNCMAdapter | AdversarialAEAdapter,
        poison_fn: PoisonFn | None = None,
    ) -> None:
        self.data_stream = data_stream
        self.classifier = classifier
        self.detector = detector
        self.poison_fn = poison_fn
        self._metrics = PipelineMetrics()

    def set_poison_fn(self, fn: PoisonFn | None) -> None:
        """Attach or detach a poisoning function without restarting the pipeline."""
        self.poison_fn = fn

    def fit(self, X_ref: np.ndarray, y_ref: np.ndarray) -> None:
        """Train the classifier and detector on clean reference data."""
        self.classifier.fit(X_ref, y_ref)
        self.detector.fit(X_ref, y_ref)

    def run(self) -> PipelineMetrics:
        """
        Consume the data stream one batch at a time.

        For each batch:
          1. Apply poison_fn (no-op when None) — attack injection point.
          2. Predict with the classifier.
          3. Check for drift with the detector.
          4. If drift is detected, call detector.adapt() on the batch.
        """
        self._metrics = PipelineMetrics()

        for batch_idx, (X_batch, y_batch) in enumerate(self.data_stream):
            was_poisoned = self.poison_fn is not None

            # --- poisoning injection point -----------------------------------
            y_clean = y_batch.copy()  # Keep a clean copy for metrics, if needed.
            if self.poison_fn is not None:
                X_batch, y_batch = self.poison_fn(X_batch, y_batch)
            # -----------------------------------------------------------------

            predictions = self.classifier.predict(X_batch)
            n_correct = int(np.sum(predictions == y_clean))

            drift_detected = self.detector.detect(X_batch, y_batch)
            if drift_detected:
                self._metrics.drift_events += 1
                self.detector.adapt(X_batch, y_batch)

            result = BatchResult(
                batch_idx=batch_idx,
                n_samples=len(y_batch),
                n_correct=n_correct,
                drift_detected=drift_detected,
                was_poisoned=was_poisoned,
            )
            self._metrics.batch_results.append(result)
            self._metrics.total_samples += result.n_samples
            self._metrics.total_correct += result.n_correct
            self._metrics.batches_processed += 1

        return self._metrics

    @property
    def metrics(self) -> PipelineMetrics:
        return self._metrics
