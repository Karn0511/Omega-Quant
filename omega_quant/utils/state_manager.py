"""Phase 7: Failsafe System & State Consistency."""
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

LOGGER = logging.getLogger(__name__)

class StateManager:
    """Safely persists active positions and equity on disk to resume after crashes."""

    def __init__(self, filename: str = "omega_quant/data/live_state.json"):
        """Auto-docstring."""
        self.filepath = Path(filename)
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self.state = self.load_state()

    def load_state(self) -> Dict[str, Any]:
        """Auto-docstring."""
        if self.filepath.exists():
            try:
                with open(self.filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                LOGGER.info("Failsafe: Successfully recovered active system state.")
                return data
            except json.JSONDecodeError:
                LOGGER.error("Failsafe: State file corrupt. Initializing fresh state.")
        return {"positions": {}, "equity": 10000.0, "last_trade_id": None}

    def save_state(self, positions: Dict[str, Any], equity: float, last_trade_id: str):
        """Atomically dump current exposure vectors to disk."""
        self.state["positions"] = positions
        self.state["equity"] = equity
        self.state["last_trade_id"] = last_trade_id

        # Write to temporary file then rename for atomic consistency
        temp_path = self.filepath.with_suffix(".tmp")
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(self.state, f, indent=2)
        temp_path.replace(self.filepath)

    def prevent_duplicate(self, trade_id: str) -> bool:
        """Auto-docstring."""
        if self.state.get("last_trade_id") == trade_id:
            LOGGER.warning("Failsafe: Duplicate trade attempt detected (%s). Blocking execution.", trade_id)
            return True
        return False
