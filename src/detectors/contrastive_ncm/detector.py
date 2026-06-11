import torch
from torch.utils.data import DataLoader

from .autoencoder import DriftAutoencoder, train_contrastive_autoencoder
from .ncm_classifier import NCMClassifier


class ContrastiveNCMDetector:
    """
    Drift detector from Kuppa & Le-Khac (2022): "Learn to adapt: Robust drift detection
    in security domain."

    Pipeline:
      1. A contrastive autoencoder maps inputs to a latent space where same-class samples
         cluster tightly (MSE + supervised contrastive loss).
      2. An NCM classifier assigns incoming samples to the nearest class prototype using a
         combined Euclidean + cosine (Riemannian proxy) distance (Eq. 3).
      3. A sample is flagged as *drifted* when its minimum NCM distance exceeds
         drift_threshold (needs calibration per dataset).
      4. Drifted embeddings are accumulated via the recursive update in Eq. 6. When the
         L2 norm of the accumulated delta vector crosses concept_threshold T (paper
         default: 3.5), a new class prototype is created from the mean of buffered
         embeddings.
    """

    def __init__(
        self,
        input_dim: int = 72,
        hidden_dim: int = 64,
        latent_dim: int = 32,
        lambda_1: float = 0.1,
        temperature: float = 0.1,
        drift_threshold: float = 1.0,
        concept_threshold: float = 3.5,
        device: str | torch.device = "cpu",
    ):
        self.device = torch.device(device)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.autoencoder = DriftAutoencoder(input_dim, hidden_dim, latent_dim).to(self.device)
        self.ncm = NCMClassifier(lambda_1)
        self.drift_threshold = drift_threshold
        self.concept_threshold = concept_threshold
        self.temperature = temperature
        self.num_classes: int = 0

        self._drift_buffer: list[torch.Tensor] = []
        self._prev_drift_z: torch.Tensor | None = None
        self._delta_accumulated: torch.Tensor | None = None

    def fit(
        self,
        data_loader: DataLoader,
        epochs: int = 300,
        lr: float = 0.0001,
        num_classes: int | None = None,
        tqdm_position: int = 0,
        desc_prefix: str = "",
    ) -> None:
        """Train the autoencoder, then fit NCM prototypes on the resulting embeddings."""
        train_contrastive_autoencoder(
            self.autoencoder,
            data_loader,
            epochs=epochs,
            lr=lr,
            temperature=self.temperature,
            num_classes=num_classes,
            tqdm_position=tqdm_position,
            desc_prefix=desc_prefix,
        )

        all_h, all_labels = [], []

        for x, labels in data_loader:
            h = self.encode(x)
            all_h.append(h)
            all_labels.append(labels)

        all_h_tensor = torch.cat(all_h, dim=0)
        all_labels_tensor = torch.cat(all_labels, dim=0)

        if num_classes is None:
            num_classes = int(all_labels_tensor.max().item()) + 1

        self.num_classes = num_classes
        self.ncm.fit(all_h_tensor, all_labels_tensor, num_classes)

    def detect(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Classify incoming samples and flag drifted ones.

        Args:
            x: Input tensor of shape (N, input_dim).

        Returns:
            predictions:   (N,) predicted class indices (closest prototype).
            is_drifted:    (N,) bool mask — True when min distance > drift_threshold.
            min_distances: (N,) minimum distance to the nearest class prototype.
        """
        h = self.encode(x)

        distances = self.ncm._compute_distance(h)
        min_distances, predictions = distances.min(dim=1)
        is_drifted = min_distances > self.drift_threshold

        return predictions.cpu(), is_drifted.cpu(), min_distances.cpu()

    def update_with_batch_drifted(
        self,
        z_batch: torch.Tensor,
        allow_fire: bool = True,
    ) -> bool:
        """
        Batch-level concept-discovery matching the paper's time-window semantics
        (Eq. 6, Kuppa & Le-Khac 2022).

        Args:
            z_batch: (M, latent_dim) tensor of all drifted embeddings in one batch.
            allow_fire: if False, the accumulator is still updated but the firing
                trigger is suppressed (used by the coupled-system wrapper to gate
                concept-discovery on buffer composition — see streaming.py).

        Returns:
            True if a new class prototype was registered, False otherwise.
        """
        if z_batch.shape[0] == 0:
            return False

        z_batch = z_batch.detach()

        for z in z_batch:
            self._drift_buffer.append(z)

        batch_mean = z_batch.mean(dim=0)

        if self._prev_drift_z is not None:
            delta = batch_mean - self._prev_drift_z
            if self._delta_accumulated is None:
                self._delta_accumulated = delta.clone()
            else:
                self._delta_accumulated = self._delta_accumulated + delta

            if allow_fire and torch.norm(self._delta_accumulated) > self.concept_threshold:
                new_prototype = torch.stack(self._drift_buffer).mean(dim=0)
                self.ncm.add_prototype(new_prototype)
                self.num_classes += 1
                self._reset_drift_buffer()
                return True

        self._prev_drift_z = batch_mean
        return False

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return latent embeddings for input x without gradient tracking."""
        self.autoencoder.eval()
        with torch.no_grad():
            h = self.autoencoder.encode(x.to(self.device))
        return h

    def retrain(
        self,
        data_loader: DataLoader,
        epochs: int = 300,
        lr: float = 0.0001,
        from_scratch: bool = True,
    ) -> None:
        """Retrain the autoencoder on updated data and refresh all NCM prototypes."""
        if from_scratch:
            self._reset_autoencoder()
        self.fit(data_loader, epochs=epochs, lr=lr, num_classes=self.num_classes)

    def _reset_autoencoder(self) -> None:
        """Re-initialise all autoencoder weights to a fresh random state."""
        for module in self.autoencoder.modules():
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()

    def _reset_drift_buffer(self) -> None:
        self._drift_buffer = []
        self._prev_drift_z = None
        self._delta_accumulated = None
