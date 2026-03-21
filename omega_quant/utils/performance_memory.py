"""Phase 1 & 9: Signal Quality Analysis and Performance Memory System."""
import logging
import json
from pathlib import Path
from datetime import datetime, timezone
# pylint: disable=import-error
import pandas as pd  # type: ignore
import numpy as np  # type: ignore
# pylint: enable=import-error
from typing import Dict, Any, List

LOGGER = logging.getLogger(__name__)

class PerformanceMemory:
    """Class configuration auto-docstring."""
    def __init__(self, storage_path: str = "omega_quant/data/performance_memory.json"):
        """Auto-docstring."""
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.memory: List[Dict] = self._load_memory()

    def _load_memory(self) -> list:
        """Auto-docstring."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except json.JSONDecodeError:
                return []
        return []

    def log_prediction(self, symbol: str, predicted_class: str, confidence: float, features: Dict[str, Any]):
        """Store prediction context to evaluate outcome later."""
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol": symbol,
            "predicted_class": predicted_class,
            "confidence": confidence,
            "features": features,
            "actual_outcome": None,
            "profit": 0.0
        }
        self.memory.append(record)
        self._save_memory()

    def log_outcome(self, symbol: str, profit: float, success: bool):
        """Update the latest unresolved prediction with real-world execution profit."""
        for record in reversed(self.memory):
            if record["symbol"] == symbol and record["actual_outcome"] is None:
                record["actual_outcome"] = "SUCCESS" if success else "FAILURE"
                record["profit"] = profit
                break
        self._save_memory()

    def _save_memory(self):
        """Institutional-grade serialization for complex data types."""
        def json_encoder(obj):
            if isinstance(obj, (pd.Timestamp, datetime)):
                return obj.isoformat()
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return str(obj)

        with open(self.storage_path, "w", encoding="utf-8") as f:
            json.dump(self.memory, f, indent=2, default=json_encoder)

    def analyze_signal_quality(self) -> Dict[str, Any]:
        """Phase 1: Compute Win rate, Precision/Recall, False signal rate, Profit."""
        df = pd.DataFrame(self.memory)
        if df.empty or "actual_outcome" not in df.columns:
            return {}

        completed = df[df["actual_outcome"].notna()].copy()
        if completed.empty:
            return {}

        completed["is_win"] = completed["actual_outcome"] == "SUCCESS"

        # 2. Compute Precision / Recall per class & False Signal Rate
        metrics = {}
        for cls in ["BUY", "SELL"]:
            subset = completed[completed["predicted_class"] == cls]
            if not subset.empty:
                true_pos = len(subset[subset["is_win"] == True])
                false_pos = len(subset[subset["is_win"] == False])
                precision = true_pos / len(subset) if len(subset) > 0 else 0
                metrics[f"{cls}_precision"] = precision
                metrics[f"{cls}_false_signal_rate"] = false_pos / len(subset) if len(subset) > 0 else 0

        metrics["win_rate"] = len(completed[completed["is_win"] == True]) / len(completed)
        metrics["avg_profit"] = completed["profit"].mean()

        LOGGER.info("Signal Quality: Win Rate=%.2f%%, Avg Profit=%.4f", metrics["win_rate"] * 100, metrics["avg_profit"])
        return metrics

    def train_meta_model(self):
        """Phase 9: Train meta-model 'Which situations lead to profit?'."""
        df = pd.DataFrame(self.memory)
        if df.empty or "actual_outcome" not in df.columns:
            LOGGER.warning("Not enough memory to train meta-model.")
            return None

        completed = df[df["actual_outcome"].notna()].copy()
        if len(completed) < 50:
            LOGGER.warning("Meta-Model requires at least 50 completed trades.")
            return None

        # Parse features out of the dictionary
        features_list = []
        for _, row in completed.iterrows():
            f_dict = row["features"].copy()
            f_dict["confidence"] = row["confidence"]
            features_list.append(f_dict)

        x = pd.DataFrame(features_list)
        y = (completed["actual_outcome"] == "SUCCESS").astype(int)

        # pylint: disable=import-error,import-outside-toplevel
        import xgboost as xgb
        # pylint: enable=import-error,import-outside-toplevel
        meta_model = xgb.XGBClassifier(n_estimators=50, max_depth=3)

        # Select numeric columns
        X_num = x.select_dtypes(include=['number']).fillna(0)
        meta_model.fit(X_num, y)

        LOGGER.info("Meta-Model officially trained on %d historical trades.", len(completed))

        # Find which situations lead to profit (feature importances in meta-model)
        importances = meta_model.feature_importances_
        sit_dict = sorted(zip(X_num.columns, importances), key=lambda x: x[1], reverse=True)
        LOGGER.info("Keys to Profitability (Meta-Model): %s", sit_dict[:3])

        # Save meta-model locally
        meta_path = self.storage_path.parent / "meta_model.json"
        meta_model.save_model(meta_path)
        return meta_model
