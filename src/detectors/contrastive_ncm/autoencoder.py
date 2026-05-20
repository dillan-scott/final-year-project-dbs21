import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class DriftAutoencoder(nn.Module):
    """
    Complete Autoencoder as described in the paper.
    """

    def __init__(self, input_dim: int = 72, hidden_dim: int = 64, latent_dim: int = 32):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encode(x)
        x_recon = self.decode(h)
        return h, x_recon


class PrototypeContrastiveLoss(nn.Module):
    """
    Prototype-based contrastive loss (Eq. 2 of Kuppa & Le-Khac 2022).

    For each sample h_i of class k, maximises cosine similarity to the batch
    prototype v_k while minimising similarity to all other class prototypes v_j.
    Batch prototypes are the per-class means of the current mini-batch embeddings.

    num_classes must be passed explicitly so that classes absent from a batch
    still occupy a slot in the softmax denominator (as zero prototypes), keeping
    the denominator size constant across batches. Without this, batches that
    happen to miss the highest-indexed class silently shrink the denominator.
    """

    def __init__(self, temperature: float = 0.1, num_classes: int | None = None):
        super().__init__()
        self.temperature = temperature
        self.num_classes = num_classes

    def forward(self, h: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        num_classes = self.num_classes if self.num_classes is not None else int(labels.max().item()) + 1

        # Compute per-class prototype means from the current batch (v_k in the paper)
        prototypes = torch.zeros(num_classes, h.shape[1], device=h.device)
        for c in range(num_classes):
            mask = labels == c
            if mask.any():
                prototypes[c] = h[mask].mean(dim=0)

        h_norm = F.normalize(h, p=2, dim=1)
        proto_norm = F.normalize(prototypes, p=2, dim=1)

        # s(h_i, v_j) = cosine_sim / tau  →  shape (N, num_classes)
        sim = torch.matmul(h_norm, proto_norm.T) / self.temperature

        # -sum_i log [ exp(s(h_i, v_k)) / sum_j exp(s(h_i, v_j)) ]
        return F.cross_entropy(sim, labels)


def train_contrastive_autoencoder(
    model: DriftAutoencoder,
    data_loader: torch.utils.data.DataLoader,
    epochs: int = 300,
    lr: float = 0.0001,
    temperature: float = 0.1,
    num_classes: int | None = None,
) -> float:

    device = next(model.parameters()).device
    optimizer = optim.Adam(model.parameters(), lr=lr)
    mse_loss_fn = nn.MSELoss()
    contrastive_loss_fn = PrototypeContrastiveLoss(temperature=temperature, num_classes=num_classes)

    model.train()
    total_loss = 0.0
    for epoch in range(epochs):
        epoch_loss = 0.0

        for x, labels in data_loader:
            x, labels = x.to(device), labels.to(device)
            optimizer.zero_grad()

            h, x_recon = model(x)

            mse_loss = mse_loss_fn(x_recon, x)
            contrastive_loss = contrastive_loss_fn(h, labels)

            loss = mse_loss + contrastive_loss

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            print(f"Batch Loss: {loss.item():.4f}\r", end="")

        total_loss += epoch_loss / len(data_loader)

        if (epoch + 1) % 1 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss / len(data_loader):.4f}")

    avg_loss = total_loss / epochs
    print(f"Average Loss over {epochs} epochs: {avg_loss:.4f}")

    return avg_loss


def train_plain_autoencoder(
    model: DriftAutoencoder,
    data_loader: torch.utils.data.DataLoader,
    epochs: int = 300,
    lr: float = 0.0001,
) -> float:
    """Train the autoencoder with MSE reconstruction loss only (no contrastive term)."""
    device = next(model.parameters()).device
    optimizer = optim.Adam(model.parameters(), lr=lr)
    mse_loss_fn = nn.MSELoss()

    model.train()
    total_loss = 0.0
    for epoch in range(epochs):
        epoch_loss = 0.0

        for x, _ in data_loader:
            x = x.to(device)
            optimizer.zero_grad()

            _, x_recon = model(x)

            loss = mse_loss_fn(x_recon, x)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            print(f"Batch Loss: {loss.item():.4f}\r", end="")

        total_loss += epoch_loss / len(data_loader)

        if (epoch + 1) % 1 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss / len(data_loader):.4f}")

    avg_loss = total_loss / epochs
    print(f"Average Loss over {epochs} epochs: {avg_loss:.4f}")
    
    return avg_loss
