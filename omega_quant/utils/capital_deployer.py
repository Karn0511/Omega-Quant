"""Capital Deployment Intelligence."""
import logging

LOGGER = logging.getLogger(__name__)

class CapitalDeployer:
    """Manages Phase 1 (Tiers), Phase 3 (Volatility Scaling), Phase 6 (Risk of Ruin), and Phase 7 (Strategy Scaling)."""
    
    def __init__(self):
        """Auto-docstring."""
        self.current_tier = 1
        self.capital_tiers = {1: 0.01, 2: 0.05, 3: 0.10, 4: 0.25}
        self.rolling_win_rate = 0.5
        self.rolling_pf = 1.0
        
        # Meta-layer strategy allocators (Phase 7)
        self.strategy_performance = {"TREND_FOLLOWING": 1.0, "MEAN_REVERSION": 1.0, "BREAKOUT": 1.0}

    def update_tier_metrics(self, win_rate: float, profit_factor: float, max_drawdown: float):
        """Phase 1: Gradual Capital Ramp - Upscale on stability, downscale on decay."""
        self.rolling_win_rate = win_rate
        self.rolling_pf = profit_factor
        
        # Promotion Logic
        if profit_factor > 1.5 and max_drawdown < 0.05 and win_rate > 0.55:
            if self.current_tier < 4:
                self.current_tier += 1
                LOGGER.info("CAPITAL DEPLOYER: Performance stable. PROMOTED to Tier %d (Alloc=%.0f%%)", 
                            self.current_tier, self.capital_tiers[self.current_tier] * 100)
        # Demotion Logic
        elif profit_factor < 1.0 or max_drawdown > 0.08 or win_rate < 0.45:
            if self.current_tier > 1:
                self.current_tier -= 1
                LOGGER.warning("CAPITAL DEPLOYER: Performance decayed. DEMOTED to Tier %d (Alloc=%.0f%%)", 
                               self.current_tier, self.capital_tiers[self.current_tier] * 100)

    def volatility_multiplier(self, atr: float, price: float) -> float:
        """Phase 3: Volatility-Based Scaling (High vol = lower size)."""
        volatility_pct = atr / price if price > 0 else 0
        
        # Baseline volatility assumed at 2%
        if volatility_pct > 0.04:
            return 0.5  # Extreme market chopped: Halve the size
        elif volatility_pct > 0.02:
            return 0.8
        elif volatility_pct < 0.01:
            return 1.25 # Dead market: Safely push more capital into standard deviations
        return 1.0
        
    def assess_risk_of_ruin(self, total_trades: int) -> float:
        """Phase 6: Risk of Ruin Probability Model."""
        if total_trades < 10:
            return 0.0
        # standard risk of ruin approximation based on flat bet sizing
        ror = ((1 - self.rolling_win_rate) / (1 + self.rolling_win_rate + 1e-9)) ** total_trades
        
        if ror > 0.01:
            LOGGER.critical("CAPITAL DEPLOYER: Risk of Ruin > 1%% (%.4f). Enforcing extreme capital suppression.", ror)
            return 0.25 # Quarter the portfolio
        return 1.0

    def update_strategy_performance(self, regime: str, is_win: bool):
        """Phase 7: Meta-layer track best performing strategies."""
        decay = 0.95
        reward = 1.05 if is_win else 0.90
        self.strategy_performance[regime] = (self.strategy_performance[regime] * decay) + (reward * (1 - decay))

    def get_deployed_capital_ratio(self, atr: float, price: float, total_trades: int, regime: str, alpha_health: dict = None) -> float:
        """Combines all Phase variables into one explicit sizing ratio."""
        import json
        from pathlib import Path
        decomp_path = Path("omega_quant/data/alpha_decomposition.json")
        decomp_multiplier = 1.0
        hard_guard_active = False
        
        if decomp_path.exists():
            try:
                with open(decomp_path, "r", encoding="utf-8") as f:
                    ds = json.load(f)
                    stats = ds.get("regime_stats", {})
                    # Phase 2: Identify structural leaders for redirection
                    best_regime = None
                    best_pf = 0.0
                    for reg, mets in stats.items():
                        if mets.get("profit_factor", 0) > best_pf:
                            best_pf = mets.get("profit_factor", 0)
                            best_regime = reg

                    if regime in stats:
                        if stats[regime].get("expectancy", 0.0) <= 0.0 or stats[regime].get("profit_factor", 1.0) < 1.0:
                            LOGGER.warning("CAPITAL DEPLOYER: Alpha Engine DISABLED capital for leaking regime [%s]", regime)
                            return 0.0
                        elif regime == best_regime and best_pf >= 1.2:
                            LOGGER.info("CAPITAL DEPLOYER: Redistributing maximum capital to Alpha Leader [%s]", regime)
                            decomp_multiplier = 2.0 # Heavy capital redirection
                            
                    if ds.get("hard_generalization_guard", False):
                        hard_guard_active = True

            except (FileNotFoundError, json.JSONDecodeError):
                pass
                
        # Phase 3, 6, 7: Long-Term Preservation Overrides
        if alpha_health:
            cooldowns = alpha_health.get("cooldowns", {})
            if regime in cooldowns:
                LOGGER.warning("PHASE 3: Strategy Cooldown blocking allocations to %s.", regime)
                return 0.0
                
            if alpha_health.get("transitioning", False):
                LOGGER.warning("PHASE 6: Regime Transition detected. Cutting size by 70%% pending stability confirmation.")
                decomp_multiplier *= 0.3
                
            score = alpha_health.get("health_score", 100)
            if score < 40:
                decomp_multiplier *= 0.1 # Phase 7 Drop size to 10%
            elif score > 80:
                decomp_multiplier *= 1.25

        if hard_guard_active:
            LOGGER.critical("CAPITAL DEPLOYER: HARD GENERALIZATION GUARD ACTIVE. Overfitting detected. Forcing Tier 1 (1%) and disabling scaling.")
            self.current_tier = 1
            return 0.01

        base_alloc = self.capital_tiers.get(self.current_tier, 0.01)
        vol_scaler = self.volatility_multiplier(atr, price)
        ror_scaler = self.assess_risk_of_ruin(total_trades)
        
        # Strategy specific override
        strat_weight = max(0.5, min(1.5, self.strategy_performance.get(regime, 1.0)))
        
        final_ratio = base_alloc * vol_scaler * ror_scaler * strat_weight * decomp_multiplier
        return min(final_ratio, 0.95) # Never exceed 95% total
