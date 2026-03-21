"""Setup and execution script for OMEGA-QUANT."""
import os
import shutil
import subprocess
import sys

def main():
    """Auto-docstring."""
    root_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(root_dir)

    print("="*60)
    print("OMEGA-QUANT Automated Setup & Pipeline Runner")
    print("="*60)

    # 1. Restructure Workspace
    omega_dir = os.path.join(root_dir, "omega_quant")
    os.makedirs(omega_dir, exist_ok=True)

    modules_to_move = [
        "agents", "backtesting", "config", "data", "execution",
        "logs", "models", "strategies", "utils", "main.py",
        "__init__.py", "requirements.txt"
    ]

    print("\n[1/4] Restructuring workspace into `omega_quant` module...")
    moved_any = False
    for item in modules_to_move:
        src = os.path.join(root_dir, item)
        dst = os.path.join(omega_dir, item)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.move(src, dst)
            print(f"  Moved {item} -> omega_quant/{item}")
            moved_any = True

    if not moved_any:
        print("  Workspace already structured correctly.")

    # 2. Virtual Environment
    print("\n[2/4] Initializing Virtual Environment...")
    venv_dir = os.path.join(root_dir, ".venv")

    # Determine python and pip executables
    if os.name == "nt":
        venv_python = os.path.join(venv_dir, "Scripts", "python.exe")
        venv_pip = os.path.join(venv_dir, "Scripts", "pip.exe")
    else:
        venv_python = os.path.join(venv_dir, "bin", "python")
        venv_pip = os.path.join(venv_dir, "bin", "pip")

    if not os.path.exists(venv_dir) or not os.path.exists(venv_pip):
        print("  Creating or re-initializing virtual environment at .venv/")
        if os.path.exists(venv_dir):
            shutil.rmtree(venv_dir)
        subprocess.run([sys.executable, "-m", "venv", ".venv"], check=True)
    else:
        print("  Virtual environment already exists and is configured.")

    # 3. Install dependencies
    print("\n[3/4] Installing / Verifying Dependencies...")
    req_file = os.path.join(omega_dir, "requirements.txt")
    subprocess.run([venv_pip, "install", "-r", req_file], check=True)

    # 4. Run Initial Pipeline
    print("\n[4/4] Executing OMEGA-QUANT Pipeline...")

    print("\n--- Step 4.1: Download Data ---")
    subprocess.run([
        venv_python, "-m", "omega_quant.main", "download", 
        "--symbol", "BTC/USDT", "--timeframe", "1m"
    ], check=True)

    print("\n--- Step 4.2: Train Model ---")
    subprocess.run([venv_python, "-m", "omega_quant.main", "train"], check=True)

    print("\n--- Step 4.3: Backtest Strategy ---")
    subprocess.run([venv_python, "-m", "omega_quant.main", "backtest"], check=True)

    print("\n" + "="*60)
    print("SUCCESS: Setup and Pipeline complete!")
    print("System is fully runnable end-to-end without crashes.")
    print("="*60)

if __name__ == "__main__":
    main()
