"""Phase 1, 2, 3, 6, 7: Long-Term Alpha Preservation Engine."""
import json
import logging
from pathlib import Path
from typing import Dict, Any

# pylint: disable=import-error
import numpy as np  # type: ignore
# pylint: enable=import-error

LOGGER = logging.getLogger(__name__)

class AlphaPreservationEngine:
    """Manages alpha decay, cool-downs, transition shock, and global health."""
    
    def __init__(self, memory_path: str = "omega_quant/data/performance_memory.json"):
        """Auto-docstring."""
        self.memory_path = Path(memory_path)
        self.strategy_cooldowns: Dict[str, int] = {}
        self.regime_half_lives: Dict[str, int] = {}
        self.last_regime = "Sideways"
        self.regime_transition_cooldown = 0
        
    def _load_history(self) -> list:
        if not self.memory_path.exists():
            return []
        try:
            with open(self.memory_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    def evaluate_alpha_health(self) -> Dict[str, Any]:
        """Runs the 5 core preservation checks on the Live Database."""
        history = self._load_history()
        completed = [t for t in history if "actual_outcome" in t and t["actual_outcome"] is not None]
        
        state = {
            "health_score": 100.0,
            "decay_detected": False,
            "cooldowns": {},
            "transitioning": False
        }
        
        if len(completed) < 10:
            return state
            
        profits = np.array([float(t.get("profit", 0)) for t in completed])
        wins = np.array([1 if t["actual_outcome"] == "SUCCESS" else 0 for t in completed])
        regimes = [t.get("regime", "Unknown") for t in completed]
        
        # Phase 1: Alpha Decay Detection (20 vs 50 trade windows)
        w20_profits = profits[-20:]
        w50_profits = profits[-50:] if len(profits) >= 50 else profits
        
        pf_20 = sum([p for p in w20_profits if p > 0]) / (abs(sum([p for p in w20_profits if p < 0])) + 1e-9)
        pf_50 = sum([p for p in w50_profits if p > 0]) / (abs(sum([p for p in w50_profits if p < 0])) + 1e-9)
        
        win_20 = float(sum(wins[-20:]) / len(wins[-20:]))
        win_50 = float(sum(wins[-50:]) / len(wins[-50:])) if len(wins) >= 50 else win_20
        
        decay_rate = 0.0
        if pf_20 < (pf_50 * 0.8) and win_20 < (win_50 * 0.9):
            state["decay_detected"] = True
            LOGGER.warning("[ALPHA DECAY EVENT] Edge degrading (PF 50: %.2f -> %.2f)",
                           pf_50, pf_20)
            decay_rate = (pf_50 - pf_20) / pf_50

        # Phase 2 & 3: Strategy Cool-down and Edge Half-Life
        for r in set(regimes):
            r_profits = np.array([float(t.get("profit", 0)) for t in completed if t.get("regime") == r])
            if len(r_profits) < 5: 
                continue
            # Phase 1 & 3: Disable/Re-enable logic
            r_pf = sum([p for p in r_profits[-20:] if p > 0]) / (abs(sum([p for p in r_profits[-20:] if p < 0])) + 1e-9)
            if r_pf < 1.0:
                self.strategy_cooldowns[r] = float('inf') # Hard disable
                LOGGER.info("[STRATEGY DISABLED] Regime %s deep freeze (PF %.2f < 1.0).", r, r_pf)
            else:
                if r in self.strategy_cooldowns:
                    if r_pf > 1.2:
                        del self.strategy_cooldowns[r]
                        LOGGER.info("[STRATEGY AWAKENED] Regime %s stabilized (PF %.2f). Resuming.", r, r_pf)
                        
            # Record Half-life approximation
            if r_pf >= 1.0:
                self.regime_half_lives[r] = self.regime_half_lives.get(r, 0) + 1
            else:
                LOGGER.debug("Half-life met for %s: ~%d trades", r, self.regime_half_lives.get(r, 0))

        state["cooldowns"] = self.strategy_cooldowns

        # Phase 6: Regime Transition Detection
        current_regime = regimes[-1] if regimes else "Sideways"
        if current_regime != self.last_regime:
            LOGGER.warning("[REGIME TRANSITION] Detect shift from %s -> %s", self.last_regime, current_regime)
            self.regime_transition_cooldown = 5 # Suspend heavy sizing for 5 confirmation prints
            self.last_regime = current_regime
            
        if self.regime_transition_cooldown > 0:
            state["transitioning"] = True
            self.regime_transition_cooldown -= 1
            
        # Phase 7: Alpha Health Score
        expectancy = float(np.mean(profits))
        drawdown_penalty = 0.05 # Mock drawdown simulation
        stability = 1.0 - decay_rate
        
        health = (pf_20 * 20) + (expectancy * 100) - (drawdown_penalty * 100) + (stability * 30)
        state["health_score"] = max(0.0, min(100.0, float(health)))
        
        return state
