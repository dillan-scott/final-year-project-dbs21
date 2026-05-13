import numpy as np
from statsmodels.tsa.stattools import grangercausalitytests, InfeasibleTestError  # type: ignore
import torch

from src.detectors.robust_rbm import RobustRBM
from src.detectors.trend_tracker import TrendTracker


class RRBM_DD:
    """
    Robust RBM Drift Detector (RRBM-DD) implementation based on the paper:
    "Adversarial concept drift detection under poisoning attacks for robust data stream mining"
    """

    def __init__(self, rbm_model: RobustRBM, window_size: int = 1000):
        self.rbm = rbm_model
        self.trend_tracker = TrendTracker(window_size)
        self.trend_history: list[float] = []

    def process_minibatch(self, v: torch.Tensor, z: torch.Tensor) -> bool:
        """
        Processes a new mini-batch, updates the RBM, and checks for concept drift.
        Returns True if a drift is detected, False otherwise.
        """
        # 1. Calculate reconstruction error BEFORE updating weights
        # Eq 27: Average reconstruction error over the recent mini-batch
        with torch.no_grad():
            batch_error = self.rbm.reconstruction_error(v, z)

        # 2. Update the RBM using CD-k
        self.rbm.cd_k_step(v, z)

        # 3. Calculate the new trend Q_r(t)
        current_trend = self.trend_tracker.update(batch_error)
        self.trend_history.append(current_trend)

        # 4. Check for Drift using Granger Causality on First Differences
        if current_trend <= 1e-5:
            return False

        is_drift = self._check_granger_causality()

        return is_drift

    def _check_granger_causality(self, test_window: int = 50, alpha: float = 0.05) -> bool:
        """
        Performs the Granger causality test on the first differences of the reconstruction error
        trends, as specified in the paper.

        Args:
            test_window (int): The number of recent trend points to use for the statistical test.
            alpha (float): The significance level for the hypothesis test.

        Returns:
            bool: True if concept drift is detected, False otherwise.
        """
        if len(self.trend_history) < test_window + 1:
            return False

        recent_trends = np.array(self.trend_history[-(test_window + 1) :])

        diffs = np.diff(recent_trends)

        if np.var(diffs) < 1e-8:
            return False

        y_t1 = diffs[1:]
        x_t = diffs[:-1]

        data = np.column_stack((y_t1, x_t))

        try:
            # maxlag=1 because we are looking at t and t+1
            gc_res = grangercausalitytests(data, maxlag=1)
            p_value = gc_res[1][0]["ssr_ftest"][1]

            # If p_value > alpha, we reject Granger causality -> Drift detected!
            if p_value > alpha:
                return True

        except (InfeasibleTestError, ValueError):
            # Handled in cases where variance is 0 (constant trend)
            pass

        return False
