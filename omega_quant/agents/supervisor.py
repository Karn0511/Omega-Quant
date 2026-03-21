"""OMEGA-QUANT Supervisor Agent: Risk Audit and Override Authority."""
import logging
import json
import time
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List

LOGGER = logging.getLogger(__name__)

class SupervisorAgent:
    """Institutional guardian auditing every system cycle for anomalies and risk violations."""
    
    def __init__(self, config: Dict):
        """Auto-docstring."""
        self.config = config
        self.override_active = False
        self.safe_mode_reason = None
        self.forensic_log_path = Path("omega_quant/logs/supervisor_forensics.jsonl")
        self.last_audit_time = time.time()
        
        # Internal state tracking for anomalies
        self.trade_counts_window: List[float] = []
        self.last_pnl_swings: List[float] = []

    def audit_system_cycle(self, state: Dict[str, Any]):
        """Phase 1: Full System Audit (Every Cycle)."""
        # Critical inconsistencies check
        if state.get("probs_max", 0) > 1.0 or state.get("probs_max", 1) < 0:
            self._trigger_override("INVALID_MODEL_CONFIDENCE", "Model probability outside [0,1] range.")
        
        if state.get("equity", 0) <= 0:
            self._trigger_override("BANKRUPTCY_PROTECTION", "Equity dropped to zero/negative.")

        if state.get("risk_level", 0) > self.config["risk"]["max_position_pct"]:
            self._trigger_override("RISK_CEILING_VIOLATION", f"Position pct {state['risk_level']} exceeds limit.")

    def validate_trade_proposal(self, proposal: Dict[str, Any]) -> bool:
        """Phase 2: Trade Validation Before Execution."""
        # 1. Band check
        band = proposal.get("optimal_band", (0.0, 1.0))
        conf = proposal.get("confidence", 0.0)
        if not (band[0] <= conf <= band[1]):
            LOGGER.warning("SUPERVISOR: Trade Blocked. Confidence %.2f outside optimal band %s.", conf, band)
            return False

        # 2. Strategy PF check
        if proposal.get("strategy_pf", 1.0) < 1.0:
            LOGGER.warning("SUPERVISOR: Trade Blocked. Strategy PF < 1.0.")
            return False

        # 3. Cooldown check
        if proposal.get("in_cooldown", False):
            LOGGER.warning("SUPERVISOR: Trade Blocked. Alpha Preservation Cooldown active.")
            return False

        # 4. Liquidity check
        if proposal.get("liq_void", 0) > 0.4:
            LOGGER.warning("SUPERVISOR: Trade Blocked. Extreme Liquidity Void detected.")
            return False

        # 5. Overfit check
        if proposal.get("hard_guard", False):
            LOGGER.warning("SUPERVISOR: Trade Blocked. Hard Generalization Guard active.")
            return False

        # 6. Trade Size check (Phase 6)
        if proposal.get("capital_ratio", 0) > self.config["risk"]["max_position_pct"]:
            LOGGER.warning("SUPERVISOR: Trade Blocked. Size %.4f exceeds Risk Ceiling.", proposal.get("capital_ratio"))
            return False

        return True

    def post_trade_forensics(self, trade_data: Dict[str, Any]):
        """Phase 3: Post-Trade Forensic Analysis."""
        slippage = trade_data.get("slippage_pct", 0.0)
        outcome = trade_data.get("pnl_pct", 0.0)
        
        classification = "Acceptable"
        if slippage > 0.01:
            classification = "Wasteful"
        if outcome < -0.10:
            classification = "Dangerous"
        if slippage < 0.001 and outcome > 0.02:
            classification = "Efficient"

        forensic_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "classification": classification,
            "slippage": slippage,
            "pnl": outcome,
            "was_necessary": trade_data.get("was_necessary", True),
            "signal_accuracy": outcome > 0
        }
        
        self.forensic_log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.forensic_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(forensic_entry) + "\n")
            
        LOGGER.info("SUPERVISOR: Post-Trade Forensics complete. Class: %s", classification)

    def detect_alpha_drift(self, history: List[Dict]):
        """Phase 5: Alpha Drift Detection (20 vs 50 trades)."""
        if len(history) < 50:
            return False
            
        def calc_pf(trades):
            wins = [t.get("profit", 0) for t in trades if t.get("profit", 0) > 0]
            loss = [abs(t.get("profit", 0)) for t in trades if t.get("profit", 0) < 0]
            if not loss: return 2.0
            return sum(wins) / (sum(loss) + 1e-9)

        pf_20 = calc_pf(history[-20:])
        pf_50 = calc_pf(history[-50:])
        
        if pf_20 < pf_50 * 0.7:
            LOGGER.critical("SUPERVISOR: ALPHA DRIFT DETECTED. PF 20: %.2f vs PF 50: %.2f", pf_20, pf_50)
            return True
        return False

    def detect_anomalies(self, trade_count: int, last_pnl: float):
        """Phase 8: Anomaly Detection."""
        self.trade_counts_window.append(float(trade_count))
        if len(self.trade_counts_window) > 10:
            self.trade_counts_window.pop(0)
        
        # High Frequency Anomaly
        if len(self.trade_counts_window) == 10:
            avg_trades = sum(self.trade_counts_window) / 10
            if trade_count > avg_trades * 3:
                self._trigger_override("TRADE_FREQUENCY_ANOMALY", f"Trade spike: {trade_count} > {avg_trades*3}")

        # PnL Swing Anomaly
        if abs(last_pnl) > 0.15:
            self._trigger_override("VOLATILITY_ANOMALY", f"Abnormal PnL swing: {last_pnl*100:.2f}%")

    def execution_quality_check(self, slippage: float, latency: float):
        """Phase 7: Execution Quality Control."""
        if slippage > 0.005: # 0.5%
            LOGGER.warning("SUPERVISOR: Low Execution Quality: Slippage %.4f, Latency %.2fms", slippage, latency)
            return 0.8 # Scaler
        return 1.0

    def _trigger_override(self, reason_code: str, message: str):
        """Phase 10: Hard Override Authority."""
        self.override_active = True
        self.safe_mode_reason = reason_code
        LOGGER.critical("!!! SUPERVISOR OVERRIDE ACTUATED: %s - %s !!!", reason_code, message)

    def generate_status_report(self, alpha_state: Dict) -> str:
        """Phase 9: Real-Time Reporting."""
        status = "HEALTHY" if not self.override_active else f"HALTED ({self.safe_mode_reason})"
        report = "\n[OMEGA-QUANT SUPERVISOR REPORT]\n"
        report += f"System Status: {status}\n"
        report += f"Alpha Health Score: {alpha_state.get('health_score', 'N/A')}\n"
        report += f"Risk Level: {alpha_state.get('risk_level', 'N/A')}\n"
        return report
