import torch
import torch.nn.functional as F


class NCMClassifier:
    """
    Nearest Class Mean (NCM) classifier that combines Euclidean distance and a proxy for
    Riemannian divergence.
    """

    def __init__(self, lambda_1: float = 0.1):
        self.lambda_1: float = lambda_1
        self.prototypes: torch.Tensor | None = None
        self.num_classes: int | None = None

    def fit(self, h: torch.Tensor, labels: torch.Tensor, num_classes: int):
        self.num_classes = num_classes
        latent_dim = h.shape[1]

        self.prototypes = torch.zeros((num_classes, latent_dim), device=h.device)

        for c in range(num_classes):
            class_mask = labels == c
            class_embeddings = h[class_mask]

            if class_embeddings.shape[0] > 0:
                self.prototypes[c] = class_embeddings.mean(dim=0)
            else:
                print(f"Warning: No samples found for class {c}.")

    def _compute_distance(self, h_j: torch.Tensor) -> torch.Tensor:
        if self.prototypes is None:
            raise ValueError("Prototypes not computed. Call fit() first.")

        prototypes = self.prototypes.to(h_j.device)

        # Euclidean Distance
        d_E = torch.cdist(h_j, prototypes, p=2.0)

        # Riemannian divergence: geodesic (arc-length) distance on the unit hypersphere.
        # d_R(h, p) = arccos(<h/‖h‖, p/‖p‖>)  — proper metric on S^{n-1}.
        # Clamp before acos to guard against numerical values slightly outside [-1, 1].
        h_j_norm = F.normalize(h_j, p=2, dim=1)
        proto_norm = F.normalize(prototypes, p=2, dim=1)
        cosine_sim = torch.matmul(h_j_norm, proto_norm.T).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        d_R = torch.acos(cosine_sim)

        D = d_R + (self.lambda_1 * d_E)

        return D

    def predict(self, h_j):
        distances = self._compute_distance(h_j)
        predictions = torch.argmin(distances, dim=1)

        return predictions

    def add_prototype(self, prototype: torch.Tensor) -> None:
        if self.prototypes is None or self.num_classes is None:
            raise ValueError("Classifier has not been fit yet. Call fit() first.")

        new_proto = prototype.unsqueeze(0).to(self.prototypes.device)
        self.prototypes = torch.cat([self.prototypes, new_proto], dim=0)
        self.num_classes += 1
