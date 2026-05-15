import pytest
import torch

from src.detectors.RRBM_DD.robust_rbm import RobustRBM


def test_energy_gating_selectivity():
    """
    Objective:
        Prove that the gating mechanism successfully identifies
        and masks out statistical outliers while preserving clean data.
    """
    rbm = RobustRBM(n_visible=5, n_hidden=3, n_classes=2)
    rbm.train()  # Enable running stats updates

    clean_batch = torch.randn(100, 5) * 0.2 + 0.4

    rbm.compute_energy_gating(clean_batch)

    test_batch = clean_batch[:10].clone()
    # Target instance index 4, feature index 3 with an extreme anomaly
    test_batch[4, 3] = 50.0

    rbm.eval()  # Freeze stats for test evaluation
    gating_mask = rbm.compute_energy_gating(test_batch)

    assert gating_mask.shape == test_batch.shape, "Gating mask shape mismatch."
    assert torch.all(
        (gating_mask == 0.0) | (gating_mask == 1.0)
    ), "Gating mask must be strictly binary."
    assert torch.all(gating_mask[0] == 1.0), "Clean instance was incorrectly gated."
    assert gating_mask[4, 3].item() == 0.0, "Adversarial outlier was not successfully gated."
    assert (
        gating_mask[4, 0].item() == 1.0
    ), "Clean feature on anomalous instance was incorrectly gated."


def test_truncation_factor_numerical_stability():
    """
    Objective:
        Verify that the iterative M-estimator for computing the truncation factor is
        numerically stable and produces valid outputs across a range of input conditions.
    """
    rbm = RobustRBM(n_visible=4, n_hidden=2, n_classes=3)
    rbm.eval()

    batch_size = 10
    v = torch.rand(batch_size, 4)
    z = torch.zeros(batch_size, 3)
    z[:, 0] = 1.0  # Valid one-hot

    theta = rbm.compute_truncation_factor(v, z)

    assert theta.shape == (4,), "Truncation factor must return a 1D tensor of size V."
    assert torch.all(theta >= 0.0) and torch.all(theta <= 1.0), "Theta must be bounded in [0, 1]."
    assert not torch.isnan(theta).any(), "Iterative solver must not produce NaNs."

    # Edge Case 1: Pure Zero Input Batch
    zero_v = torch.zeros(batch_size, 4)
    zero_z = torch.zeros(batch_size, 3)
    zero_z[:, 0] = 1.0  # Valid one-hot

    theta_zeros = rbm.compute_truncation_factor(zero_v, zero_z)
    assert not torch.isnan(theta_zeros).any(), "Truncation factor produced NaN on zero inputs."
    assert torch.all(theta_zeros >= 0.0) and torch.all(
        theta_zeros <= 1.0
    ), "Output out of theoretical bounds [0, 1]."

    # Edge Case 2: Extreme Magnitude Input Batch
    extreme_v = torch.ones(batch_size, 4) * 1e10
    theta_extreme = rbm.compute_truncation_factor(extreme_v, zero_z)
    assert not torch.isnan(theta_extreme).any(), "Truncation factor produced NaN on extreme inputs."
    assert torch.all(theta_zeros >= 0.0) and torch.all(
        theta_zeros <= 1.0
    ), "Output out of theoretical bounds [0, 1]."


def test_cd_k_step_convergence():
    """
    Objective:
        Demonstrate that performing multiple CD-k updates on a static input distribution
        leads to a reduction in reconstruction error, indicating that the RBM is learning
        to model the underlying data structure effectively.
    """
    rbm = RobustRBM(n_visible=6, n_hidden=6, n_classes=2)
    rbm.train()

    # Fixed static input representing a highly structured data pattern
    v_static = torch.tensor([[1.0, 1.0, 0.0, 0.0, 1.0, 0.0]]).repeat(32, 1)
    z_static = torch.tensor([[1.0, 0.0]]).repeat(32, 1)

    with torch.no_grad():
        initial_error = rbm.reconstruction_error(v_static, z_static)

    # Over-fit the RBM to this single pattern
    for _ in range(100):
        rbm.cd_k_step(v_static, z_static)

    with torch.no_grad():
        final_error = rbm.reconstruction_error(v_static, z_static)

    assert (
        final_error < initial_error
    ), "CD-k updates must reduce reconstruction error on static distributions."


if __name__ == "__main__":
    pytest.main([__file__])
