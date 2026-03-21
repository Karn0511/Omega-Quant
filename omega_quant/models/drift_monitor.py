"""Phase 5: Model Drift Detection."""
import logging
from collections import deque
import numpy as np  # type: ignore

LOGGER = logging.getLogger(__name__)

class DriftMonitor:
    """Tracks historical vs recent accuracy to force active retraining on decay."""

    def __init__(self, history_window: int = 1000, recent_window: int = 50, decay_threshold: float = 0.15):
        """Auto-docstring."""
        self.history_window = history_window
        self.recent_window = recent_window
        self.decay_threshold = decay_threshold

        self.historical_outcomes = deque(maxlen=self.history_window)
        self.recent_outcomes = deque(maxlen=self.recent_window)

    def log_outcome(self, is_success: bool):
        """Auto-docstring."""
        val = 1.0 if is_success else 0.0
        self.historical_outcomes.append(val)
        self.recent_outcomes.append(val)

    def check_for_drift(self) -> bool:
        """Returns True if model performance has collapsed > threshold."""
        if len(self.historical_outcomes) < 100 or len(self.recent_outcomes) < self.recent_window:
            return False

        historical_acc = sum(self.historical_outcomes) / len(self.historical_outcomes)
        recent_acc = sum(self.recent_outcomes) / len(self.recent_outcomes)

        # Phase 5: Compare recent vs historical
        if (historical_acc - recent_acc) > self.decay_threshold:
            LOGGER.error("Model Drift Detected! Historical: %.2f vs Recent: %.2f.", historical_acc, recent_acc)
            return True
        return False
