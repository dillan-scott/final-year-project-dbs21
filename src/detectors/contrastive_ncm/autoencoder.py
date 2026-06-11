import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm.auto import tqdm


class DriftAutoencoder(nn.Module):
    """Complete Autoencoder as described in the paper."""

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
    """Prototype-based contrastive loss (Eq. 2 of Kuppa & Le-Khac 2022)."""

    def __init__(self, temperature: float = 0.1, num_classes: int | None = None):
        super().__init__()
        self.temperature = temperature
        self.num_classes = num_classes

    def forward(self, h: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        num_classes = (
            self.num_classes if self.num_classes is not None else int(labels.max().item()) + 1
        )

        one_hot = torch.zeros(h.shape[0], num_classes, device=h.device)
        one_hot.scatter_(1, labels.unsqueeze(1), 1.0)
        counts = one_hot.sum(0).clamp(min=1).unsqueeze(1)
        prototypes = (one_hot.T @ h) / counts

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
    tqdm_position: int = 0,
    desc_prefix: str = "",
    disable_tqdm: bool = False,
    loss_log_every: int = 20,
) -> float:
    """Train the autoencoder with contrastive loss."""
    device = next(model.parameters()).device
    optimizer = optim.Adam(model.parameters(), lr=lr)
    mse_loss_fn = nn.MSELoss()
    contrastive_loss_fn = PrototypeContrastiveLoss(temperature=temperature, num_classes=num_classes)

    model.train()
    total_loss = 0.0
    pbar = tqdm(
        range(epochs),
        desc=f"{desc_prefix}Contrastive",
        position=tqdm_position,
        leave=True,
        disable=disable_tqdm,
    )
    for e in pbar:
        epoch_loss = torch.tensor(0.0, device=device)

        for x, labels in data_loader:
            x, labels = x.to(device), labels.to(device)
            optimizer.zero_grad()

            h, x_recon = model(x)

            mse_loss = mse_loss_fn(x_recon, x)
            contrastive_loss = contrastive_loss_fn(h, labels)

            loss = mse_loss + contrastive_loss

            loss.backward()
            optimizer.step()

            epoch_loss += loss.detach()

        epoch_loss_val = epoch_loss.item() / len(data_loader)
        total_loss += epoch_loss_val
        pbar.set_postfix(loss=f"{epoch_loss_val:.4f}")
        if loss_log_every and ((e + 1) % loss_log_every == 0 or (e + 1) == epochs):
            print(f"    [{desc_prefix}Contrastive {e + 1}/{epochs}] loss={epoch_loss_val:.4f}")

    return total_loss / epochs


def train_plain_autoencoder(
    model: DriftAutoencoder,
    data_loader: torch.utils.data.DataLoader,
    epochs: int = 300,
    lr: float = 0.0001,
    tqdm_position: int = 1,
    desc_prefix: str = "",
    disable_tqdm: bool = False,
    loss_log_every: int = 20,
) -> float:
    """Train the autoencoder with MSE reconstruction loss only (no contrastive term)."""
    device = next(model.parameters()).device
    optimizer = optim.Adam(model.parameters(), lr=lr)
    mse_loss_fn = nn.MSELoss()

    model.train()
    total_loss = 0.0
    pbar = tqdm(
        range(epochs),
        desc=f"{desc_prefix}Plain AE",
        position=tqdm_position,
        leave=True,
        disable=disable_tqdm,
    )
    for e in pbar:
        epoch_loss = torch.tensor(0.0, device=device)

        for x, _ in data_loader:
            x = x.to(device)
            optimizer.zero_grad()

            _, x_recon = model(x)

            loss = mse_loss_fn(x_recon, x)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.detach()

        epoch_loss_val = epoch_loss.item() / len(data_loader)
        total_loss += epoch_loss_val
        pbar.set_postfix(loss=f"{epoch_loss_val:.4f}")
        if loss_log_every and ((e + 1) % loss_log_every == 0 or (e + 1) == epochs):
            print(f"    [{desc_prefix}Plain AE {e + 1}/{epochs}] loss={epoch_loss_val:.4f}")

    return total_loss / epochs
