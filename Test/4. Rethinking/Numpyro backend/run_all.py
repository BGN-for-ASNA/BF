import argparse
import os
import subprocess
import sys

parser = argparse.ArgumentParser(description="Run all Rethinking models")
parser.add_argument("--shard", action="store_true",
                    help="Enable data sharding across CPU cores via shard=True in fit()")
parser.add_argument("--start", type=int, default=1,
                    help="Start from model number N (1-indexed, default 1)")
args = parser.parse_args()

# Get the python executable from the venv
python_exe = sys.executable

models = [
    "1.Continuous variable.py",
    "1b.Multiple continuous variables.py",
    "2.Categorical variable.py",
    "3.Continuous interactions.py",
    "4.Binomial.py",
    "5.Binomial with indices.py",
    "6.Poisson.py",
    "6b.Poisson with offset.py",
    "7.Negative binomial.py",
    "8.Multinomial.py",
    "8b.Multinomial fixed income.py",
    "9.Beta binomial.py",
    "10.Zero inflated.py",
    "11.Varying intercepts.py",
    "12.Varying effects.py",
    "13.Gaussian processes.py",
    "14.Dirichlet.py",
    "15.Measuring error.py"
]

script_dir = os.path.dirname(os.path.abspath(__file__))

print("Checking and installing dependencies...")
req_file = os.path.join(script_dir, "requirements.txt")
if os.path.exists(req_file):
    try:
        subprocess.check_call([python_exe, "-m", "pip", "install", "-r", req_file, "--quiet"])
    except subprocess.CalledProcessError:
        print(f"Warning: Failed to install dependencies from {req_file}")
else:
    try:
        subprocess.check_call([python_exe, "-m", "pip", "install", "cmdstanpy", "BayesForge", "--quiet"])
    except subprocess.CalledProcessError:
        print("Warning: Failed to install cmdstanpy and BayesForge")

plots_subdir = "plots_shard" if args.shard else "plots"
plots_dir    = os.path.join(script_dir, plots_subdir)
os.makedirs(plots_dir, exist_ok=True)

models = models[args.start - 1:]
print(f"Starting execution of {len(models)} models (from model {args.start})...")
if args.shard:
    print("  shard=True  — data sharding enabled across CPU cores")
print(f"  Plots → {plots_dir}")

env = os.environ.copy()
env['BF_NSIM']     = os.environ.get('BF_NSIM', '10')
env['BF_PLOTS_DIR'] = plots_subdir          # read by Utils.py
if args.shard:
    env['BF_SHARD'] = '1'                   # read by bf.__init__

log_file = os.path.join(script_dir, "log_shard.txt" if args.shard else "log.txt")

with open(log_file, "w") as f_log:
    f_log.write(f"Execution Log - {len(models)} models\n")
    f_log.write("="*50 + "\n")

    for model in models:
        print(f"\n{'='*50}")
        print(f"Running {model}...")
        print(f"{'='*50}")
        
        try:
            # Run the model script from its own directory
            model_path = os.path.join(script_dir, model)
            result = subprocess.run([python_exe, model_path], env=env, capture_output=True, text=True, check=True, cwd=script_dir)
            print(f"Successfully completed {model}")
            f_log.write(f"[SUCCESS] {model}\n")
            f_log.flush()
        except subprocess.CalledProcessError as e:
            print(f"Error running {model}:")
            print(e.stderr)
            f_log.write(f"[FAILURE] {model}\n")
            f_log.write(f"Error:\n{e.stderr}\n")
            f_log.write("-" * 30 + "\n")
            f_log.flush()
        except Exception as e:
            print(f"Unexpected error: {e}")
            f_log.write(f"[ERROR] {model}: {e}\n")
            f_log.flush()

print("\nAll models finished.")
print(f"Check {log_file} for details.")
