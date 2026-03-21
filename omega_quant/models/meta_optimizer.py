"""Phase 5: Meta-Learning Layer for Dynamic Ensemble Weights."""
import logging
from typing import List, Tuple

# pylint: disable=import-error
import numpy as np  # type: ignore
# pylint: enable=import-error

LOGGER = logging.getLogger(__name__)

class MetaOptimizer:
    """Class configuration auto-docstring."""
    def __init__(self, learning_rate: float = 0.05):
        """Auto-docstring."""
        # Initial weights: QCNN, LSTM, XGBoost, Transformer
        self.weights = np.array([0.30, 0.25, 0.25, 0.20])
        self.learning_rate = learning_rate
        self.rolling_performance = []

    def update_weights(self, targets: List[int], predictions_list: List[np.ndarray]):
        """
        Dynamically adjusts ensemble weights based on rolling accuracy.
        predictions_list: List of [P_qcnn, P_lstm, P_xgb, P_trans] probabilities.
        """
        for target, preds in zip(targets, predictions_list):
            model_accuracies = []
            for p_model in preds:
                # How much probability mass did the model assign to the correct target?
                prob_correct = p_model[target]
                model_accuracies.append(prob_correct)

            self.rolling_performance.append(model_accuracies)
            if len(self.rolling_performance) > 100:
                self.rolling_performance.pop(0)

        # Calculate mean performance of each model over rolling window
        perf_matrix = np.array(self.rolling_performance)
        mean_perf = np.mean(perf_matrix, axis=0)

        # Softmax adjustment to weights
        exp_perf = np.exp(mean_perf / 0.1) # temperature trick
        new_weights = exp_perf / np.sum(exp_perf)

        # Smooth update
        self.weights = (1 - self.learning_rate) * self.weights + self.learning_rate * new_weights
        
        # Phase 5: Ensemble Diversification - Hard cap max reliance
        self.weights = np.clip(self.weights, 0.10, 0.35)
        self.weights /= np.sum(self.weights) # Renormalize to 1.0
        LOGGER.info("Meta-Optimizer updated Ensemble Weights: QCNN=%.3f, LSTM=%.3f, XGB=%.3f, Trans=%.3f",
                    self.weights[0], self.weights[1], self.weights[2], self.weights[3])

    def get_weights(self) -> Tuple[float, float, float, float]:
        """Auto-docstring."""
        return tuple(self.weights)
