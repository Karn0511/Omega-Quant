import logging
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

LOGGER = logging.getLogger(__name__)

class SignalCalibrationEngine:
    """
    Optimizes OMEGA-QUANT for live cloud trading calibration.
    Handles confidence distribution, threshold adaptation, and shadow trading.
    """
    def __init__(self, config: Dict):
        self.config = config
        self.log_dir = Path(config["data"].get("parquet_dir", "omega_quant/data"))
        self.calibration_path = self.log_dir / "calibration_state.json"
        self.load_state()

    def load_state(self):
        if self.calibration_path.exists():
            try:
                with open(self.calibration_path, "r") as f:
                    self.state = json.load(f)
            except Exception:
                self.state = self._default_state()
        else:
            self.state = self._default_state()

    def _default_state(self) -> Dict:
        return {
            "confidence_history": [],
            "shadow_trades": [],
            "adapted_threshold": 0.6,
            "win_rate_map": {},
            "temperature_suggested": 1.0,
            "drift_detected": False,
            "last_updated": datetime.now().isoformat()
        }

    def save_state(self):
        self.state["last_updated"] = datetime.now().isoformat()
        with open(self.calibration_path, "w") as f:
            json.dump(self.state, f, indent=4)

    def process_signal(self, symbol: str, predicted_class: str, confidence: float, features: Dict):
        """Phase 1: Track confidence distribution."""
        self.state["confidence_history"].append({
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "confidence": confidence,
            "class": predicted_class
        })
        
        # Limit history size to 1000 entries
        if len(self.state["confidence_history"]) > 1000:
            self.state["confidence_history"] = self.state["confidence_history"][-1000:]

        self._analyze_distribution()
        self._check_calibration(confidence)
        self._simulate_shadow_trade(symbol, predicted_class, confidence, features)

    def _analyze_distribution(self):
        """PHASE 1: Confidence Distribution Analysis."""
        history = [h["confidence"] for h in self.state["confidence_history"]]
        if not history:
            return

        # Histogram logic
        bands = [0, 0.2, 0.4, 0.6, 0.7, 0.8, 1.0]
        hist, _ = np.histogram(history, bins=bands)
        percentages = (hist / len(history)) * 100

        LOGGER.info("CALIBRATION [Phase 1]: Confidence Distribution: 0.4-0.6: %.1f%% | 0.6-0.7: %.1f%% | 0.7-0.8: %.1f%%",
                    percentages[2], percentages[3], percentages[4])

        if percentages[4] < 1.0:
            LOGGER.warning("CALIBRATION: CRITICAL - <1%% of signals reach Optimal Band (0.7-0.8). Model may be too compressed.")

    def _check_calibration(self, current_confidence: float):
        """PHASE 2 & 3: Signal Threshold Adaptation & Logit Check."""
        history = [h["confidence"] for h in self.state["confidence_history"]]
        avg_conf = np.mean(history) if history else 0
        
        # If no signals > 0.6 for extended time
        elite_signals = [h for h in history if h > 0.6]
        if len(history) > 50 and len(elite_signals) == 0:
            LOGGER.info("PHASE 2: Threshold Adaptation Active. Testing Lower Band (0.5-0.6).")
            self.state["adapted_threshold"] = 0.55
            self.state["temperature_suggested"] = 0.85 # Suggest sharpening via temp scaling
        else:
            self.state["adapted_threshold"] = 0.6

    def _simulate_shadow_trade(self, symbol: str, action: str, confidence: float, features: Dict):
        """PHASE 5: Shadow Trading Mode."""
        if action == "HOLD":
            return

        # Simulate for 0.4-0.7 range
        if 0.4 <= confidence <= 0.7:
            self.state["shadow_trades"].append({
                "timestamp": datetime.now().isoformat(),
                "symbol": symbol,
                "action": action,
                "confidence": confidence,
                "entry_price": features.get("close"),
                "status": "pending"
            })
            LOGGER.info("SHADOW MODE [Phase 5]: Simulating %s for %s at %.2f (Conf: %.2f)", 
                        action, symbol, features.get("close", 0), confidence)

    def get_calibration_report(self) -> str:
        """Returns a full summary for the supervisor pass."""
        history = [h["confidence"] for h in self.state["confidence_history"]]
        if not history:
            return "No calibration data available."

        avg = np.mean(history)
        peak = np.max(history)
        
        return f"CALIBRATION REPORT: Avg Conf: {avg:.3f} | Peak: {peak:.3f} | Adapted Threshold: {self.state['adapted_threshold']} | Temp Correction: {self.state['temperature_suggested']}"
