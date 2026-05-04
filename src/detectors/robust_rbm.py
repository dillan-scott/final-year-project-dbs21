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
        Note: sign of softmax input is flipped.

        Args:
            h (torch.Tensor): Tensor of hidden layer states. Shape: (batch_size, H)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - prob: Tensor of class probabilities. Shape: (batch_size, Z)
                - sample: Tensor of sampled class indices. Shape: (batch_size,)
        """
        prob = torch.softmax(h @ self.U + self.c, dim=-1)
        return prob, torch.multinomial(prob, 1).squeeze()

    def compute_truncation_factor(self, v: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Computes robust truncation factors for each visible neuron (input feature).

        Uses 0-1 loss: $L_i(y_a; z) = \\mathbb{1}[|v_i - \\hat{v}_i| > 0.5]$.
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
            v_prob, _ = self.sample_visible(h_prob)

            losses = torch.abs(v - v_prob)

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

            empirical_loss = losses.mean(dim=0).clamp_min(1e-8)
            truncation = (theta_hat / empirical_loss).clamp(min=0.0, max=1.0)

        return truncation

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

        # Positive phase: Sample hidden states given the data
        h_prob, _ = self.sample_hidden(v, z)

        pos_assoc_W = v.T @ h_prob / batch_size
        pos_assoc_U = h_prob.T @ z / batch_size
        pos_assoc_a = torch.mean(v, dim=0)
        pos_assoc_b = torch.mean(h_prob, dim=0)
        pos_assoc_c = torch.mean(z, dim=0)

        # Gibbs sampling for k steps
        v_k, z_k = v, z
        for _ in range(self.k):
            _, h_k = self.sample_hidden(v_k, z_k)
            _, v_k = self.sample_visible(h_k)
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

        truncation_factor = self.compute_truncation_factor(v, z)
        global_truncation = torch.mean(truncation_factor)

        grad_W = (pos_assoc_W * truncation_factor.unsqueeze(1)) - neg_assoc_W
        grad_a = (pos_assoc_a * truncation_factor) - neg_assoc_a

        grad_U = (pos_assoc_U * global_truncation) - neg_assoc_U
        grad_b = (pos_assoc_b * global_truncation) - neg_assoc_b
        grad_c = (pos_assoc_c * global_truncation) - neg_assoc_c

        recon_loss = torch.mean((v - v_k) ** 2) + torch.mean((z - z_k) ** 2)

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
