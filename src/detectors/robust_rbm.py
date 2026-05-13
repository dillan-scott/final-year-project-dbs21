import math
from typing import Tuple

import torch
from torch import nn


class RobustRBM(nn.Module):
    """
    Implements a Robust Restricted Boltzmann Machine (RRBM) in PyTorch.
    """

    def __init__(
        self,
        n_visible: int,
        n_hidden: int,
        n_classes: int,
        k: int = 1,
        learning_rate: float = 0.005,
        delta: float = 0.05,
        ema_decay: float = 0.99,
        random_state: int = 42,
    ) -> None:
        """
        Initializes the Robust RBM with given dimensions and random seed for reproducibility.

        Args:
        n_visible (int): Number of visible neurons, V
        n_hidden (int): Number of hidden neurons, H
        n_classes (int): Number of class neurons, Z
        """
        super().__init__()
        self.V = n_visible
        self.H = n_hidden
        self.Z = n_classes
        self.k = k
        self.lr = learning_rate
        self.delta = delta
        self.ema_decay = ema_decay

        torch.manual_seed(random_state)

        # W: weights between visible (v) and hidden (h) layers
        self.W = nn.Parameter(torch.randn(self.V, self.H) * 0.01)
        # U: weights between hidden (h) and class (z) layers
        self.U = nn.Parameter(torch.randn(self.H, self.Z) * 0.01)

        # Bias for visible layer v
        self.a = nn.Parameter(torch.zeros(self.V))
        # Bias for hidden layer h
        self.b = nn.Parameter(torch.zeros(self.H))
        # Bias for class layer z
        self.c = nn.Parameter(torch.zeros(self.Z))

        self.register_buffer("noise_mean", torch.zeros(self.V))
        self.register_buffer("noise_var", torch.ones(self.V))

    def sample_hidden(self, v: torch.Tensor, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculates the activation probability of the hidden neurons and samples their states.
        Computes P(h_j | v, z) using Eq. 10 from the paper.

        Args:
            v (torch.Tensor): Tensor of visible layer inputs. Shape: (batch_size, V)
            z (torch.Tensor): Tensor of class layer inputs. Shape: (batch_size, Z)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - prob: Tensor of hidden activation probabilities. Shape: (batch_size, H)
                - sample: Binary tensor of sampled hidden states. Shape: (batch_size, H)
        """

        prob = torch.sigmoid(v @ self.W + z @ self.U.T + self.b)
        return prob, torch.bernoulli(prob)

    def sample_visible(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculates the activation probability of the visible neurons and samples their states.
        Computes P(v_i | h) using Eq. 11 from the paper.

        Args:
            h (torch.Tensor): Tensor of hidden layer states. Shape: (batch_size, H)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - prob: Tensor of visible activation probabilities. Shape: (batch_size, V)
                - sample: Binary tensor of sampled visible states. Shape: (batch_size, V)
        """
        prob = torch.sigmoid(h @ self.W.T + self.a)
        return prob, torch.bernoulli(prob)

    def sample_class(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculates the probability of predicting each class and samples class assignments.
        Computes P(z = 1_k | h) using Eq. 12 from the paper.

        Args:
            h (torch.Tensor): Tensor of hidden layer states. Shape: (batch_size, H)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - prob: Tensor of class probabilities. Shape: (batch_size, Z)
                - sample: Tensor of sampled class indices. Shape: (batch_size,)
        """
        logits = -(h @ self.U + self.c)
        prob = torch.softmax(logits, dim=-1)
        return prob, torch.multinomial(prob, 1).squeeze()

    def forward(self, v: torch.Tensor, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Performs a forward pass through the RRBM, returning the sampled hidden states,
        reconstructed visible probabilities, and reconstructed class probabilities.

        Args:
            v (torch.Tensor): Tensor of visible layer inputs. Shape: (batch_size, V)
            z (torch.Tensor): Tensor of class layer inputs. Shape: (batch_size, Z)

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
                - h_sample: Sampled hidden states. Shape: (batch_size, H)
                - v_recon_prob: Reconstructed visible probabilities. Shape: (batch_size, V)
                - z_recon_prob: Reconstructed class probabilities. Shape: (batch_size, Z)
        """
        h_prob, h_sample = self.sample_hidden(v, z)
        v_recon_prob, _ = self.sample_visible(h_sample)
        z_recon_prob, _ = self.sample_class(h_sample)

        return h_sample, v_recon_prob, z_recon_prob

    def compute_truncation_factor(self, v: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Computes robust truncation factors for each visible neuron (input feature).

        Uses 0-1 loss based on the predicted class vs. true class to estimate the per-feature loss
        distribution.
        This implementation uses a pseudo-Huber M-estimator, solved by iteratively re-weighted
        averaging. The returned value is a multiplicative truncation coefficient in $[0, 1]$ per
        visible feature, suitable for robust gradient descent updates.

        Args:
            v (torch.Tensor): Tensor of visible layer inputs. Shape: (batch_size, V)
            z (torch.Tensor): Tensor of class layer inputs. Shape: (batch_size, Z)

        Returns:
            torch.Tensor: Tensor of truncation factors per visible feature. Shape: (V,)
        """
        batch_size = v.size(0)

        if batch_size == 0:
            return torch.ones(self.V, device=v.device, dtype=v.dtype)

        with torch.no_grad():
            h_prob, _ = self.sample_hidden(v, z)
            z_prob, _ = self.sample_class(h_prob)

            z_pred = torch.argmax(z_prob, dim=1)
            z_true = torch.argmax(z, dim=1)

            instance_loss = (z_pred != z_true).float()
            v_norm = v / (v.abs().max(dim=0, keepdim=True)[0] + 1e-8)
            losses = instance_loss.unsqueeze(1) * v_norm

            gamma_i = losses.mean(dim=0)
            diff = losses - gamma_i.unsqueeze(0)

            sigma_hat = diff.std(dim=0, unbiased=False).clamp_min(1e-6)

            c = 0.5
            for _ in range(10):
                u = diff / sigma_hat.unsqueeze(0)

                weights = 1.0 / (1.0 + u**2)
                sigma_sq_new = (weights * (diff**2)).mean(dim=0) / c
                sigma_new = torch.sqrt(sigma_sq_new).clamp_min(1e-6)

                if torch.max(torch.abs(sigma_new - sigma_hat)).item() < 1e-6:
                    sigma_hat = sigma_new
                    break
                sigma_hat = sigma_new

            scale = sigma_hat * math.sqrt(batch_size / math.log(2.0 / self.delta))
            scale = scale.clamp_min(1e-6)

            theta_hat = losses.mean(dim=0)

            for _ in range(10):
                u = (losses - theta_hat.unsqueeze(0)) / scale.unsqueeze(0)
                weights = 1.0 / torch.sqrt(1.0 + u**2)
                theta_new = (weights * losses).sum(dim=0) / (weights.sum(dim=0) + 1e-8)
                if torch.max(torch.abs(theta_new - theta_hat)).item() < 1e-6:
                    theta_hat = theta_new
                    break
                theta_hat = theta_new

        return theta_hat.clamp(0.0, 1.0)

    def compute_energy_gating(self, v: torch.Tensor) -> torch.Tensor:
        """
        Computes the binary gating mask 'g' based on the Gaussian noise model.
        Updates the running mean and variance of the clean distribution.

        Args:
            v (torch.Tensor): Visible layer inputs. Shape: (batch_size, V)

        Returns:
            torch.Tensor: Binary gating mask g. Shape: (batch_size, V)
        """
        if self.training:
            batch_mean = v.mean(dim=0)
            batch_var = v.var(dim=0, unbiased=False)

            self.noise_mean: torch.Tensor = ( # pytlint: disable=attribute-defined-outside-init
                self.ema_decay * self.noise_mean + (1.0 - self.ema_decay) * batch_mean
            )
            self.noise_var: torch.Tensor = ( # pylint: disable=attribute-defined-outside-init
                self.ema_decay * self.noise_var + (1.0 - self.ema_decay) * batch_var
            )

        std = torch.sqrt(self.noise_var).clamp_min(1e-6)

        g = (torch.abs(v - self.noise_mean.unsqueeze(0)) <= 3.0 * std.unsqueeze(0)).float()

        return g

    def cd_k_step(self, v: torch.Tensor, z: torch.Tensor) -> float:
        """
        Performs one step of Contrastive Divergence (CD-k) to update model parameters.

        This method executes the positive phase (data-driven), follows with k steps
        of Gibbs sampling (reconstruction phase), and updates the weights (W, U)
        and biases (a, b, c) using the difference between the data and model expectations.

        Args:
            v (torch.Tensor): Tensor of visible layer inputs. Shape: (batch_size, V)
            z (torch.Tensor): Tensor of class layer inputs. Shape: (batch_size, Z)

        Returns:
            float: The reconstruction loss (MSE) after the CD-k step, useful for monitoring training
            progress.
        """
        batch_size = v.size(0)

        g = self.compute_energy_gating(v)
        v_gated = v * g
        v_k = v_gated.clone()

        # Positive phase: Sample hidden states given the data
        h_prob, _ = self.sample_hidden(v_gated, z)

        pos_assoc_W = v_gated.T @ h_prob / batch_size
        pos_assoc_U = h_prob.T @ z / batch_size
        pos_assoc_a = torch.mean(v_gated, dim=0)
        pos_assoc_b = torch.mean(h_prob, dim=0)
        pos_assoc_c = torch.mean(z, dim=0)

        # Gibbs sampling for k steps
        z_k = z
        for _ in range(self.k):
            _, h_k = self.sample_hidden(v_k, z_k)
            _, v_k = self.sample_visible(h_k)

            g_k = self.compute_energy_gating(v_k)
            v_k = v_k * g_k

            _, z_k = self.sample_class(h_k)
            # pylint: disable=not-callable
            z_k = nn.functional.one_hot(z_k, num_classes=self.Z).float()

        # Negative phase: Sample hidden states given the reconstructions
        h_prob_neg, _ = self.sample_hidden(v_k, z_k)

        neg_assoc_W = v_k.T @ h_prob_neg / batch_size
        neg_assoc_U = h_prob_neg.T @ z_k / batch_size
        neg_assoc_a = torch.mean(v_k, dim=0)
        neg_assoc_b = torch.mean(h_prob_neg, dim=0)
        neg_assoc_c = torch.mean(z_k, dim=0)

        theta_hat = self.compute_truncation_factor(v, z)

        grad_W = (pos_assoc_W * theta_hat.unsqueeze(1)) - neg_assoc_W
        grad_a = (pos_assoc_a * theta_hat) - neg_assoc_a
        grad_U = (pos_assoc_U * theta_hat.mean()) - neg_assoc_U
        grad_b = (pos_assoc_b * theta_hat.mean()) - neg_assoc_b
        grad_c = (pos_assoc_c * theta_hat.mean()) - neg_assoc_c

        recon_loss = torch.sqrt(
            torch.sum((v - v_k) ** 2, dim=1) + torch.sum((z - z_k) ** 2, dim=1)
        ).mean()

        # Parameter updates
        with torch.no_grad():
            self.W.add_(self.lr * grad_W)
            self.U.add_(self.lr * grad_U)
            self.a.add_(self.lr * grad_a)
            self.b.add_(self.lr * grad_b)
            self.c.add_(self.lr * grad_c)

        return recon_loss.item()

    def reconstruction_error(self, v: torch.Tensor, z: torch.Tensor) -> float:
        """
        Calculates the mean reconstruction error for a mini-batch of instances.

        It samples the hidden states conditionally, computes the reconstructed probabilities,
        and calculates the average Euclidean distance between the original and reconstructed inputs.

        Args:
            v (torch.Tensor): Tensor of visible layer inputs. Shape: (batch_size, V)
            z (torch.Tensor): Tensor of class layer inputs. Shape: (batch_size, Z)

        Returns:
            float: The mean reconstruction error for the mini-batch.
        """
        _, h_sample = self.sample_hidden(v, z)
        v_prob, _ = self.sample_visible(h_sample)
        z_prob, _ = self.sample_class(h_sample)

        error = torch.sqrt(
            torch.sum((v - v_prob) ** 2, dim=1) + torch.sum((z - z_prob) ** 2, dim=1)
        )

        return error.mean().item()
