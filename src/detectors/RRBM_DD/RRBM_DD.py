import numpy as np
from statsmodels.tsa.stattools import grangercausalitytests, InfeasibleTestError  # type: ignore
import torch

from src.detectors.RRBM_DD.robust_rbm import RobustRBM
from src.detectors.RRBM_DD.trend_tracker import TrendTracker


class RRBM_DD:
    """
    Robust RBM Drift Detector (RRBM-DD) implementation based on the paper:
    "Adversarial concept drift detection under poisoning attacks for robust data stream mining"
    """

    def __init__(self, rbm_model: RobustRBM):
        self.rbm = rbm_model
        self.trend_tracker = TrendTracker()
        self.trend_history: list[float] = []
        self.last_batch_error: float | None = None
        self.last_trend: float | None = None
        self.last_p_value: float | None = None
        self.last_gc_sample_size: int | None = None
        self.debug: str | None = None

    def process_minibatch(self, v: torch.Tensor, z: torch.Tensor) -> bool:
        """
        Processes a new mini-batch, updates the RBM, and checks for concept drift.
        Returns True if a drift is detected, False otherwise.
        """
        # 1. Calculate reconstruction error BEFORE updating weights
        # Eq 27: Average reconstruction error over the recent mini-batch
        with torch.no_grad():
            batch_error = self.rbm.reconstruction_error(v, z)
        self.last_batch_error = batch_error

        # 2. Update the RBM using CD-k
        self.rbm.cd_k_step(v, z)

        # 3. Calculate the new trend Q_r(t)
        current_trend = self.trend_tracker.update(batch_error)
        self.trend_history.append(current_trend)
        self.last_trend = current_trend

        # 4. Check for Drift using Granger Causality on First Difference
        is_drift = self._check_granger_causality()

        return is_drift

    def _check_granger_causality(self, test_window: int = 20, alpha: float = 0.05) -> bool:
        """
        Performs the Granger causality test on the first differences of the reconstruction error
        trends, as specified in the paper.

        Args:
            test_window (int): The number \of recent trend points to use for the statistical test.
            alpha (float): The significance level for the hypothesis test.

        Returns:
            bool: True if concept drift is detected, False otherwise.
        """
        self.debug = None
        if len(self.trend_history) < test_window + 1:
            self.last_p_value = None
            self.last_gc_sample_size = None
            self.debug = "Not enough history"
            return False

        recent_trends = np.array(self.trend_history[-(test_window + 1) :])

        diffs = np.diff(recent_trends)
        self.last_gc_sample_size = diffs.size

        if np.var(diffs) < 1e-8:
            self.last_p_value = None
            self.debug = "var too small"
            return False

        y_t1 = diffs[1:]
        x_t = diffs[:-1]

        data = np.column_stack((y_t1, x_t))

        try:
            # maxlag=1 because we are looking at t and t+1
            gc_res = grangercausalitytests(data, maxlag=1, verbose=False)
            p_value = gc_res[1][0]["ssr_ftest"][1]
            self.last_p_value = p_value

            # If p_value > alpha, we reject Granger causality -> Drift detected!
            if p_value > alpha:
                return True

        except (InfeasibleTestError, ValueError):
            # Handled in cases where variance is 0 (constant trend)
            self.debug = "var is 0"
            self.last_p_value = None
            pass

        return False
