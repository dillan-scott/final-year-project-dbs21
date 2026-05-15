import numpy as np
from river.drift import ADWIN  # type: ignore


class TrendTracker:
    """
    Implements the trend tracking mechanism using an ADWIN window, as described in the RRBM-DD
    paper. Maintains a sliding window of reconstruction errors and computes the trend Q_r(t) based
    on the synced window of errors.
    """

    def __init__(self, delta: float = 0.002):
        self.adwin = ADWIN(delta=delta)
        self.error_history: list[float] = []

    def update(self, current_error: float) -> float:
        """
        Updates the ADWIN window with the new reconstruction error and computes the current trend.

        Args:
            current_error (float): The average reconstruction error for the current mini-batch.
        Returns:
            float: The current trend Q_r(t) based on the synced window of errors.
        """
        self.adwin.update(current_error)
        self.error_history.append(current_error)

        if self.adwin.drift_detected:
            new_width = int(self.adwin.width)
            self.error_history = self.error_history[-new_width:]

        n_t = len(self.error_history)
        if n_t < 2:
            return 0.0

        time_steps = np.arange(1, n_t + 1)
        recent_errors = np.array(self.error_history)

        TR_t = np.sum(time_steps * recent_errors)
        T_t = np.sum(time_steps)
        R_t = np.sum(recent_errors)
        T2_t = np.sum(time_steps**2)

        numerator = (n_t * TR_t) - (T_t * R_t)
        denominator = (n_t * T2_t) - (T_t**2)

        if denominator == 0:
            return 0.0

        return numerator / denominator
