import pytest
import numpy as np

from src.detectors.RRBM_DD.trend_tracker import TrendTracker


def test_trend_tracker_insufficient_data():
    """
    Objective:
        Test that the tracker safely returns 0.0 when there are fewer than 2 data points,
        preventing division by zero errors during linear regression.
    """
    tracker = TrendTracker()

    trend = tracker.update(0.5)
    assert trend == 0.0, "Tracker must return 0.0 with only 1 data point."

    trend = tracker.update(0.6)
    assert trend != 0.0, "Tracker should compute a trend once 2 points are available."


def test_trend_tracker_perfect_slopes():
    """
    Objective:
        Test exact numerical verification for known linear slopes: positive, negative, and flat.
        For time steps t = [1, 2, 3, 4, 5]:
        - y = 0.1*t yields slope +0.1
        - y = -0.05*t yields slope -0.05
        - y = constant yields slope 0.0
    """
    # 1. Positive Slope (+0.1)
    tracker_pos = TrendTracker()
    for err in [0.1, 0.2, 0.3, 0.4, 0.5]:
        trend = tracker_pos.update(err)
    assert pytest.approx(trend, rel=1e-5) == 0.1

    # 2. Negative Slope (-0.05)
    tracker_neg = TrendTracker()
    for err in [1.0, 0.95, 0.90, 0.85, 0.80]:
        trend = tracker_neg.update(err)
    assert pytest.approx(trend, rel=1e-5) == -0.05

    # 3. Flat Slope (Zero Variance)
    tracker_flat = TrendTracker()
    for err in [0.5, 0.5, 0.5, 0.5, 0.5]:
        trend = tracker_flat.update(err)
    assert pytest.approx(trend, abs=1e-8) == 0.0


def test_trend_tracker_scale_invariance():
    """
    Objective:
        Verify that the linear regression slope scales perfectly linearly with the magnitude
        of the input data (scale invariance check).
    """
    base_errors = np.array([1.0, 2.0, 3.0, 4.0, 5.0])  # Base slope = +1.0
    scale_factor = 1e-4  # Tiny scale testing limits

    tracker = TrendTracker()
    for err in base_errors * scale_factor:
        trend = tracker.update(float(err))

    assert pytest.approx(trend, rel=1e-5) == (1.0 * scale_factor)


def test_trend_tracker_adwin_resizing():
    """
    Objective:
        Test that an abrupt, massive shift in the error distribution forces the internal
        ADWIN detector to trigger a drift signal, successfully truncating the older history buffer.
    """
    # Initialize ADWIN with a highly sensitive delta to guarantee triggering
    tracker = TrendTracker(delta=0.05)

    # Phase 1: Feed a long, stable concept (error ~ 0.1)
    for _ in range(100):
        tracker.update(0.1)

    buffer_length_before = len(tracker.error_history)
    assert buffer_length_before == 100, "History should grow linearly during stable periods."

    # Phase 2: Suddenly introduce a massive structural shift (error ~ 5.0)
    # Feeding enough points to ensure ADWIN detects the statistical difference between windows
    drift_triggered = False
    for _ in range(40):
        tracker.update(5.0)
        if tracker.adwin.drift_detected:
            drift_triggered = True
            break

    assert drift_triggered is True, "ADWIN failed to detect a massive, clear distribution shift."

    buffer_length_after = len(tracker.error_history)
    assert buffer_length_after < buffer_length_before, (
        f"Error history buffer was not truncated upon drift detection. "
        f"Length remained {buffer_length_after}."
    )


if __name__ == "__main__":
    pytest.main([__file__])
