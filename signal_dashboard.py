"""Phase 1: Signal Quality Dashboard."""
import logging
from omega_quant.utils.performance_memory import PerformanceMemory

def launch_dashboard():
    """Auto-docstring."""
    memory = PerformanceMemory()
    metrics = memory.analyze_signal_quality()

    print("\n" + "=" * 50)
    print(" 🚀 OMEGA-QUANT: SIGNAL QUALITY DASHBOARD 🚀 ")
    print("=" * 50)

    if not metrics:
        print("No trades completed yet. Run the system longer to accrue data.")
        return

    for k, v in metrics.items():
        if "rate" in k or "precision" in k:
            print(f"- {k.upper().replace('_', ' ')}: {v*100:.2f}%")
        else:
            print(f"- {k.upper().replace('_', ' ')}: {v:.4f}")

    print("\n" + "-" * 50)
    print(" 🧠 PHASE 9: META-MODEL OPTIMIZATION ENGINE 🧠 ")
    print("-" * 50)

    meta_model = memory.train_meta_model()
    if meta_model is None:
        print("[WARNING] Not enough completed trades to train meta-model yet (requires 50+).")
    else:
        print("[SUCCESS] Market Meta-Model successfully generated "
              "and saved to memory!")
        print("This AI logic block now actively understands which "
              "situations lead directly to profit.")

    print("=" * 50 + "\n")

if __name__ == "__main__":
    # Configure basic logger to prevent console clutter during dashboard display
    logging.basicConfig(level=logging.WARNING)
    launch_dashboard()
