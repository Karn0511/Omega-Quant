"""OMEGA-QUANT Live Intelligence Engine: Continuous Feedback and Rejection Analysis."""
import logging
import json
import time
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any

LOGGER = logging.getLogger(__name__)

class LiveIntelligenceEngine:
    """Institutional-level intelligence system for real-time audit and performance optimization."""
    
    def __init__(self, base_path: str = "omega_quant/data/"):
        """Auto-docstring."""
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.trades_log_path = self.base_path / "live_trades.jsonl"
        self.rejections_log_path = self.base_path / "signal_rejections.jsonl"
        self.intelligence_state_path = self.base_path / "live_intelligence_state.json"
        
        self.active_tracking: Dict[str, Any] = {} # For tracking rejected signals
        self.state = self._load_state()

    def _load_state(self) -> Dict:
        """Auto-docstring."""
        if self.intelligence_state_path.exists():
            with open(self.intelligence_state_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {
            "rejection_efficiency": 0.0,
            "correct_rejections": 0,
            "incorrect_rejections": 0,
            "strategy_stats": {},
            "execution_drag_total": 0.0,
            "last_report_date": None
        }

    def _save_state(self):
        """Auto-docstring."""
        with open(self.intelligence_state_path, "w", encoding="utf-8") as f:
            json.dump(self.state, f, indent=2)

    def log_executed_trade(self, trade_data: Dict[str, Any]):
        """Phase 1: Detailed Live Trade Logger."""
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "asset": trade_data.get("asset"),
            "strategy": trade_data.get("strategy"),
            "regime": trade_data.get("regime"),
            "confidence": float(trade_data.get("confidence", 0)),
            "direction": trade_data.get("direction"),
            "outcome": trade_data.get("outcome"),
            "pnl": float(trade_data.get("pnl", 0)),
            "slippage_pct": float(trade_data.get("slippage_pct", 0)),
            "latency_ms": float(trade_data.get("latency_ms", 0)),
            "expected_pnl": float(trade_data.get("expected_pnl", 0)) # Phase 3
        }
        
        with open(self.trades_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
            
        # Phase 3 & 4 Update
        exec_drag = record["expected_pnl"] - record["pnl"]
        self.state["execution_drag_total"] += exec_drag
        
        strat = record["strategy"]
        if strat not in self.state["strategy_stats"]:
            self.state["strategy_stats"][strat] = {"wins": 0, "losses": 0, "total_pnl": 0.0, "trades": 0}
            
        ss = self.state["strategy_stats"][strat]
        ss["trades"] += 1
        ss["total_pnl"] += record["pnl"]
        if record["pnl"] > 0: ss["wins"] += 1
        else: ss["losses"] += 1
        
        self._save_state()

    def log_signal_rejection(self, rejection_data: Dict[str, Any]):
        """Phase 2: Signal Rejection Analysis with Forward Tracking."""
        entry_id = f"{rejection_data['asset']}_{int(time.time())}"
        record = {
            "id": entry_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "asset": rejection_data.get("asset"),
            "reason": rejection_data.get("reason"),
            "confidence": float(rejection_data.get("confidence", 0)),
            "entry_price": float(rejection_data.get("price", 0)),
            "direction": rejection_data.get("direction"),
            "missed_outcome": None
        }
        
        # Add to active tracking for Phase 2/6 simulation
        self.active_tracking[entry_id] = record
        
        with open(self.rejections_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def track_rejection_outcomes(self, current_prices: Dict[str, float]):
        """Evaluate missed targets for Phase 6 efficiency score."""
        to_remove = []
        for rid, record in self.active_tracking.items():
            asset = record["asset"]
            if asset not in current_prices: continue
            
            cur_price = current_prices[asset]
            entry_price = record["entry_price"]
            
            # Simulated 1% Take Profit or 0.5% Stop Loss (Simplified for intelligence feedback)
            dist = (cur_price - entry_price) / entry_price
            if record["direction"] == "SELL": dist = -dist
            
            # Simple 10-minute timeout or target hit
            if abs(dist) > 0.01 or (time.time() - float(datetime.fromisoformat(record["timestamp"]).timestamp())) > 600:
                record["missed_outcome"] = dist
                to_remove.append(rid)
                
                # Phase 6: Update Efficiency Score
                if dist < 0:
                    self.state["correct_rejections"] += 1
                else:
                    self.state["incorrect_rejections"] += 1
                
                total_rejections = self.state["correct_rejections"] + self.state["incorrect_rejections"]
                if total_rejections > 0:
                    self.state["rejection_efficiency"] = self.state["correct_rejections"] / total_rejections
                
        for rid in to_remove:
            del self.active_tracking[rid]
        
        if to_remove:
            self._save_state()

    def generate_daily_live_report(self, alpha_health: float):
        """Phase 7: Daily Live Report Generator."""
        now = datetime.now()
        date_str = now.strftime("%Y%m%DD")
        filename = self.base_path / f"live_report_{date_str}.txt"
        
        best_strat = "None"
        worst_strat = "None"
        if self.state["strategy_stats"]:
            sorted_strats = sorted(self.state["strategy_stats"].items(), key=lambda x: x[1]["total_pnl"], reverse=True)
            best_strat = sorted_strats[0][0]
            worst_strat = sorted_strats[-1][0]

        report = f"OMEGA-QUANT LIVE INTELLIGENCE REPORT - {now.isoformat()}\n"
        report += "="*50 + "\n"
        report += f"Total Strategy Trades: {sum(s['trades'] for s in self.state['strategy_stats'].values())}\n"
        report += f"Total Intelligence PnL: ${sum(s['total_pnl'] for s in self.state['strategy_stats'].values()):.2f}\n"
        report += f"Best Performer: {best_strat}\n"
        report += f"Worst Performer: {worst_strat}\n"
        report += f"Rejection Efficiency Score: {self.state['rejection_efficiency']*100:.2f}%\n"
        report += f"Execution Drag (Expected vs Actual): ${self.state['execution_drag_total']:.2f}\n"
        report += f"Alpha Health Score (Live): {alpha_health:.4f}\n"
        report += "="*50 + "\n"
        
        with open(filename, "w", encoding="utf-8") as f:
            f.write(report)
        
        LOGGER.info("!!! PHASE 7: Daily Intelligence Report Generated: %s !!!", filename)

    def check_safe_scaling(self, drawdown: float, slippage: float) -> bool:
        """Phase 8: Safe Scaling Signal Logic."""
        # Baseline check
        if self.state["rejection_efficiency"] < 0.60: return False
        if drawdown > 0.08: return False
        if slippage > 0.005: return False
        
        # Check Strategy Live PF
        for stats in self.state["strategy_stats"].values():
            if stats["trades"] > 5:
                wins = stats["wins"]
                losses = stats["losses"]
                pf = (wins / losses) if losses > 0 else 2.0
                if pf < 1.3: return False
        
        return True
