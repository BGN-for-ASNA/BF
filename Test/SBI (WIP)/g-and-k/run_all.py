import argparse
import os
import subprocess
import sys

parser = argparse.ArgumentParser(description="Run all g-and-k parameter-recovery scripts")
parser.add_argument("--start", type=int, default=1,
                    help="Start from script number N (1-indexed, default 1)")
args = parser.parse_args()

python_exe = sys.executable

scripts = [
    "1.ABC.py",
    "2.ABC random forest.py",
    "3.NPE.py",
    "4.Bayesian NPE.py",
]

script_dir = os.path.dirname(os.path.abspath(__file__))
plots_dir = os.path.join(script_dir, "plots")
os.makedirs(plots_dir, exist_ok=True)

scripts = scripts[args.start - 1:]
print(f"Starting execution of {len(scripts)} scripts (from script {args.start})...")
print(f"  Plots → {plots_dir}")

env = os.environ.copy()
env["BF_NSIM"] = os.environ.get("BF_NSIM", "20")
env["BF_PLOTS_DIR"] = "plots"          # read by Utils.py

log_file = os.path.join(script_dir, "log.txt")

with open(log_file, "w") as f_log:
    f_log.write(f"Execution Log - {len(scripts)} scripts, BF_NSIM={env['BF_NSIM']}\n")
    f_log.write("=" * 50 + "\n")

    for script in scripts:
        print(f"\n{'=' * 50}")
        print(f"Running {script}...")
        print(f"{'=' * 50}")
        try:
            result = subprocess.run([python_exe, os.path.join(script_dir, script)], env=env,
                                    capture_output=True, text=True, check=True, cwd=script_dir)
            print(f"Successfully completed {script}")
            f_log.write(f"[SUCCESS] {script}\n")
            # keep the recovery metrics table in the log
            metrics = [l for l in result.stdout.splitlines()
                       if "RMSE" in l or l.strip().split(" ")[0] in ("A", "B", "g", "k")]
            f_log.write("\n".join(metrics) + "\n")
            f_log.flush()
        except subprocess.CalledProcessError as e:
            print(f"Error running {script}:")
            print(e.stderr)
            f_log.write(f"[FAILURE] {script}\n")
            f_log.write(f"Error:\n{e.stderr}\n")
            f_log.write("-" * 30 + "\n")
            f_log.flush()
        except Exception as e:
            print(f"Unexpected error: {e}")
            f_log.write(f"[ERROR] {script}: {e}\n")
            f_log.flush()

print("\nAll scripts finished.")
print(f"Check {log_file} for details.")
