# Simulation-based inference: g-and-k parameter recovery

The g-and-k distribution (location `A`, scale `B`, skewness `g`, kurtosis `k`)
is defined by its quantile function only: simulating from it is trivial, but its
likelihood has no closed form, so it cannot be fitted with `m.fit()`. It is the
classic benchmark for likelihood-free inference (Drovandi & Pettitt 2011;
Prangle 2017).

Each script runs a DGP → inference → recovery pipeline for one simulation-based
approach of BF:

| Script | Approach | Inference per dataset |
|---|---|---|
| `1.ABC.py` | ABC-SMC, `m.abc(method="smc")`, robust octile statistics, 200 000 simulations | one run per dataset |
| `2.ABC random forest.py` | ABC-RF, `m.abc_rf()`, 20 candidate statistics, 10 000 simulations | one run per dataset |
| `3.NPE.py` | NPE, `m.npe(estimator="mdn")`, 20 000 simulations | trained once, amortized (`m.npe_posterior`) |
| `4.Bayesian NPE.py` | Bayesian NPE, `m.npe(bayesian=True)`, 1 000 simulations | trained once, amortized; also reports expected coverage |

For each of `BF_NSIM` datasets, true parameters are drawn uniformly in
`A ∈ (1, 5)`, `B ∈ (0.5, 2)`, `g ∈ (0.5, 3)`, `k ∈ (0.1, 0.8)` (`Utils.TRUE_RANGES`),
500 observations are simulated, and the parameters are estimated with
`Uniform(0, 10)` priors. The posterior means are compared with the true values:
a scatter plot per parameter, with the squared correlation (R²) and the root mean
squared error (RMSE), is saved in `plots/<script>_recovery.png`, together with
the results (`_recovery.csv`) and the metrics (`_metrics.csv`).

## How to run

``` bash
python "Test/16.SBI/g-and-k/run_all.py"
```

### Controlling the number of simulated datasets

`BF_NSIM` sets the number of datasets (default `20`).

``` bash
BF_NSIM=50 python "Test/16.SBI/g-and-k/run_all.py"
```

### Available arguments

-   `--start <N>` : start from script number `N` (1-indexed, default `1`).

A single script can also be run directly from this folder, e.g.
`python "3.NPE.py"`. Execution status and the metrics of each script are
written to `log.txt`.
