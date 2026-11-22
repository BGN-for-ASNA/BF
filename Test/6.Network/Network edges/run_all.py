import argparse
import os
import subprocess
import sys

parser = argparse.ArgumentParser(description="Run all Network edges R scripts sequentially")
parser.add_argument("--start", type=int, default=1,
                    help="Start from script number N (1-indexed, default 1)")
args = parser.parse_args()

scripts = [
    "run_binary_directed_no_zi.R",
    "run_binary_directed_zi.R",
    "run_binary_undirected_no_zi.R",
    "run_binary_undirected_zi.R",
    "run_count_directed_no_zi.R",
    "run_count_directed_zi.R",
    "run_count_undirected_no_zi.R",
    "run_count_undirected_zi.R"
]

script_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(script_dir, "run_all_log.txt")

scripts = scripts[args.start - 1:]
print(f"Starting execution of {len(scripts)} scripts (from script {args.start})...")

env = os.environ.copy()

with open(log_file, "w") as f_log:
    f_log.write(f"Execution Log - {len(scripts)} scripts\n")
    f_log.write("="*50 + "\n")

    for i, script in enumerate(scripts, start=args.start):
        print(f"\n{'='*50}")
        print(f"Running [{i}/{len(scripts) + args.start - 1}] {script}...")
        print(f"{'='*50}")
        
        try:
            # Run the script using Rscript
            script_path = os.path.join(script_dir, "R_scripts", script)
            result = subprocess.run(["Rscript", script_path], env=env, capture_output=True, text=True, check=True, cwd=script_dir)
            print(f"Successfully completed {script}")
            f_log.write(f"[SUCCESS] {script}\n")
            f_log.flush()
        except subprocess.CalledProcessError as e:
            print(f"Error running {script}:")
            print(e.stdout)
            print(e.stderr)
            f_log.write(f"[FAILURE] {script}\n")
            f_log.write(f"Stdout:\n{e.stdout}\n")
            f_log.write(f"Stderr:\n{e.stderr}\n")
            f_log.write("-" * 30 + "\n")
            f_log.flush()
        except Exception as e:
            print(f"Unexpected error: {e}")
            f_log.write(f"[ERROR] {script}: {e}\n")
            f_log.flush()

print("\nAll scripts finished.")
print(f"Check {log_file} for details.")
