# Network Edges Analysis

This directory rigorously tests edge-level network models across an 8-configuration grid (combining binary and count responses, with directed vs. undirected interactions, and zero-inflated vs. standard likelihoods). It compares **BayesForge (BF)** built models against the R package `bisonR` (which uses Stan as a backend).

## Requirements

The required Python packages are listed in `requirements.txt`. Install them using:

```bash
pip install -r requirements.txt
```

*(Note: The underlying R scripts require R to be installed with the `bisonR` and `cmdstanr` packages.)*

## How to Run the Models

A `run_all.py` script is provided to automate the execution of all 8 model configurations.

To run the full suite sequentially:
```bash
python3 run_all.py
```

This Python script triggers the 8 individual `run_*.R` scripts, which build the models in BayesForge (via Rreticulate), sample the posteriors, and compare them directly with `bisonR`.

## Outputs

The outputs are saved within the `results/` folder for each configuration:
- `*_log.txt`: A detailed summary file containing posterior summaries, differences, and Kullback-Leibler (KL) divergences between the BF and bisonR posteriors.
- `timing_summary.csv`: Aggregated execution times mapping the performance differences between the two backends across the test grid.

## Important Note on Posterior Distributions (BISONR vs BayesForge)

When comparing the posteriors for random effects (e.g., `beta_random`), you may notice high Kullback-Leibler (KL) divergences. Visually, Stan's posteriors often appear jagged and multimodal, while BayesForge yields smooth Gaussian curves.

This discrepancy stems from how each backend parameterizes random effects:
1. **bisonR (Stan)** uses a **centered parameterization**: `beta_random ~ normal(random_group_mu, random_group_sigma)`. When the group standard deviation (`random_group_sigma`) is small, this creates a challenging posterior geometry known as "Neal's Funnel". Stan's NUTS sampler struggles to traverse this narrow space, leading to divergent transitions, chains becoming trapped in local regions, and low E-BFMI warnings. The "multimodal" peaks seen in Stan are actually artifacts of different chains getting stuck independently, not a true reflection of the target distribution.
2. **BayesForge (BF)** avoids this entirely by using a **non-centered parameterization**: it samples `beta_raw ~ normal(0, 1)` and then deterministically computes `beta_random = random_group_mu + random_group_sigma * beta_raw`. This breaks the funnel geometry and decouples the parameters, allowing the NUTS sampler to explore the space flawlessly and recover the true, unimodal posterior.

In summary, the high KL divergence between the two backends occurs because BayesForge samples the distribution correctly, whereas the original Stan specification suffers from geometric pathologies that prevent efficient sampling.
