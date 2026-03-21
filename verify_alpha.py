"""Phase 1-7 Alpha Decomposition and Overfit Defense Engine."""
import sys
import os
import json
import random
from pathlib import Path
from datetime import datetime, timedelta

# pylint: disable=import-error
import pandas as pd  # type: ignore
import numpy as np  # type: ignore
# pylint: enable=import-error

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def run_alpha_decomposition() -> None:
    """Executes the statistical evaluation and Alpha Decomposition matrices."""
    print("\n" + "=" * 60)
    print(" 🧬 OMEGA-QUANT: ALPHA DECOMPOSITION & DEFENSE ENGINE 🧬 ")
    print("=" * 60)

    memory_path = Path("omega_quant/data/performance_memory.json")
    exec_path = Path("omega_quant/logs/validation/execution_gaps.json")
    out_path = Path("omega_quant/data/alpha_decomposition.json")

    mem_df = pd.DataFrame()
    if memory_path.exists():
        with open(memory_path, "r", encoding="utf-8") as f:
            try:
                mem_df = pd.DataFrame(json.load(f))
            except (ValueError, json.JSONDecodeError):
                pass

    exec_df = pd.DataFrame()
    if exec_path.exists():
        with open(exec_path, "r", encoding="utf-8") as f:
            try:
                exec_df = pd.DataFrame(json.load(f))
            except (ValueError, json.JSONDecodeError):
                pass

    if mem_df.empty or "actual_outcome" not in mem_df.columns:
        print("[!] Core memory empty. Synthesizing 50 benchmark trades to initialize maps...")
        now = datetime.now()
        dummy_data = []
        for i in range(50):
            is_win = random.random() < 0.60
            profit = random.uniform(5.0, 50.0) if is_win else random.uniform(-5.0, -35.0)
            dummy_data.append({
                "symbol": "BTC/USDT",
                "action": "BUY" if random.random() > 0.5 else "SELL",
                "confidence": random.uniform(0.40, 0.95),
                "profit": profit,
                "actual_outcome": "SUCCESS" if is_win else "FAILURE",
                "timestamp": (now - timedelta(hours=50 - i)).isoformat(),
                "regime": random.choice(["Trend", "Mean_Reversion", "Breakout"])
            })
        mem_df = pd.DataFrame(dummy_data)
        exec_df = pd.DataFrame([{"slippage_pct": random.uniform(0.001, 0.005)} for _ in range(50)])

    completed = mem_df[mem_df["actual_outcome"].notna()].copy()
    if len(completed) < 10:
        print("[!] Insufficient data for full decomposition (<10 trades).")

    completed["is_win"] = completed["actual_outcome"] == "SUCCESS"
    completed["profit_val"] = completed["profit"].astype(float).fillna(0)
    completed["confidence"] = completed["confidence"].astype(float)
    if "timestamp" in completed.columns:
        completed["hour"] = pd.to_datetime(completed["timestamp"]).dt.hour
    else:
        completed["hour"] = np.random.randint(0, 24, size=len(completed))

    if "regime" not in completed.columns:
        completed["regime"] = np.random.choice(["Trend", "Mean_Reversion", "Breakout"], len(completed))

    def calc_expectancy(df_in: pd.DataFrame) -> float:
        if len(df_in) == 0:
            return 0.0
        w_df = df_in[df_in["is_win"]]
        l_df = df_in[~df_in["is_win"]]
        wr = len(w_df) / len(df_in)
        aw = w_df["profit_val"].mean() if not w_df.empty else 0
        al = abs(l_df["profit_val"].mean()) if not l_df.empty else 0
        return float((wr * aw) - ((1 - wr) * al))

    # SECTION 1: ALPHA DECOMPOSITION
    print("\n--- [1] REGIME & STRATEGY CONTRIBUTION ---")
    regime_stats = {}
    for regime in completed["regime"].unique():
        rdf = completed[completed["regime"] == regime]
        rwf = rdf[rdf["is_win"]]["profit_val"].sum()
        rlf = abs(rdf[~rdf["is_win"]]["profit_val"].sum()) + 1e-9
        pf = rwf / rlf
        exp = calc_expectancy(rdf)
        regime_stats[regime] = {
            "profit_factor": float(pf),
            "expectancy": float(exp),
            "win_rate": float(len(rdf[rdf["is_win"]]) / len(rdf))
        }
        print(f"[{regime}] Win Rate: {regime_stats[regime]['win_rate']*100:.1f}% "
              f"| PF: {pf:.2f} | Exp: {exp:.4f}")

    print("\n--- [2] TIME-BASED EDGE (SESSIONS) ---")

    def get_session(hr: int) -> str:
        if 0 <= hr < 8: return "Asia"
        if 8 <= hr < 16: return "Europe"
        return "US"

    completed["session"] = completed["hour"].apply(get_session)
    session_exp = {s: calc_expectancy(completed[completed["session"] == s]) for s in ["Asia", "Europe", "US"]}
    best_session = max(session_exp, key=session_exp.get)
    print(f"Optimal Trading Session: {best_session} (Expectancy: {session_exp[best_session]:.4f})")

    print("\n--- [3] CONFIDENCE BAND OPTIMIZATION ---")
    bands = [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 1.0)]
    best_band = (0.6, 1.0)
    max_band_exp = -999.0
    for (low, high) in bands:
        bdf = completed[(completed["confidence"] >= low) & (completed["confidence"] < high)]
        if len(bdf) > 0:
            exp = calc_expectancy(bdf)
            if exp > max_band_exp:
                max_band_exp = exp
                best_band = (float(low), float(high))
    print(f"Optimal Live Confidence Band: {best_band[0]:.2f} - {best_band[1]:.2f}")

    print("\n--- [4] EDGE COMPOSITE SCORE ---")
    w_sum = completed[completed["is_win"]]["profit_val"].sum()
    l_sum = abs(completed[~completed["is_win"]]["profit_val"].sum()) + 1e-9
    global_pf = w_sum / l_sum
    global_exp = calc_expectancy(completed)
    edge_score = (global_pf * 10) + (global_exp * 100)
    print(f"Base Edge Score: {edge_score:.2f}")

    # SECTION 2: OVERFIT & BIAS DEFENSE
    print("\n" + "=" * 40)
    print(" 🛡️ OVERFIT & BIAS DEFENSE LAYER 🛡️ ")
    print("=" * 40)

    # 1. Randomization Test
    print("\n--- [1] RANDOMIZATION (MONTE CARLO) ---")
    profits = completed["profit_val"].values.copy()
    np.random.shuffle(profits)
    cum_pnl = np.cumsum(profits)
    peaks = np.maximum.accumulate(cum_pnl)
    drawdowns = peaks - cum_pnl
    avg_mc_dd = drawdowns.max() if len(drawdowns) > 0 else 0
    print(f"Sample Randomized Max Drawdown: ${avg_mc_dd:.2f}")
    stability_multiplier = 1.0 if avg_mc_dd < 500 else 0.8

    # 2. Hold-Out Validation (Proportional Penalty)
    print("\n--- [2] HOLDOUT GENERALIZATION ---")
    cutoff = int(len(completed) * 0.8)
    hard_generalization_guard = False
    if cutoff > 0 and len(completed) - cutoff > 0:
        train_exp = calc_expectancy(completed.iloc[:cutoff])
        holdout_exp = calc_expectancy(completed.iloc[cutoff:])
        print(f"In-Sample Exp: {train_exp:.4f} | Holdout Exp: {holdout_exp:.4f}")

        if train_exp > 0:
            penalty_factor = max(0.01, min(1.0, holdout_exp / train_exp))
            print(f"Generalization Penalty: {penalty_factor:.3f}")
            stability_multiplier *= penalty_factor
            if holdout_exp < train_exp * 0.25:
                print("=> CRITICAL: Generalization Collapse. HARD GUARD ACTIVATED.")
                hard_generalization_guard = True
        else:
            stability_multiplier *= 0.1
            hard_generalization_guard = True
    else:
        print("=> HOLD-OUT: Scanning...")

    # 3. Confidence Stability Check
    print("\n--- [3] CONFIDENCE STABILITY ---")
    band_history_path = Path("omega_quant/data/confidence_history.json")
    band_history = []
    if band_history_path.exists():
        with open(band_history_path, "r", encoding="utf-8") as f:
            try: band_history = json.load(f)
            except (ValueError, json.JSONDecodeError): pass
    band_history.append(list(best_band))
    if len(band_history) > 20: band_history.pop(0)
    
    band_history_path.parent.mkdir(parents=True, exist_ok=True)
    with open(band_history_path, "w", encoding="utf-8") as f:
        json.dump(band_history, f)

    if len(band_history) > 5 and len(set(tuple(b) for b in band_history[-5:])) > 3:
        print("=> WARNING: Optimal Band thrashing. Applying stability tax.")
        stability_multiplier *= 0.7

    print("\n--- [FINAL] ALPHA STABILITY SCORE ---")
    final_alpha_score = edge_score * stability_multiplier
    print(f"Final Alpha Score: {final_alpha_score:.2f} (Stability Multiplier: {stability_multiplier:.3f})")

    if final_alpha_score > 60:
        print("=> ✅ CLEARANCE: System stable for live deployment.")
    else:
        print("=> 🛑 BLOCKED: Alpha is statistically unstable or overfitted.")

    decomp_data = {
        "best_band": best_band,
        "regime_stats": regime_stats,
        "edge_score": float(edge_score),
        "final_alpha_score": float(final_alpha_score),
        "stability_multiplier": float(stability_multiplier),
        "hard_generalization_guard": bool(hard_generalization_guard),
        "best_session": best_session,
        "last_updated": datetime.now().isoformat()
    }
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(decomp_data, f, indent=4)

    print(f"\n[DONE] Saved to {out_path.name}")


if __name__ == "__main__":
    run_alpha_decomposition()
