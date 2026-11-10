# BayesForge for Python

<div align="center">
<br>A unified probabilistic programming library, bringing JAX-powered BayesForge to the Python, R and Julia ecosystem.</br>


<i>Run bespoke models on CPU, GPU, or TPU with Julia's native syntax.</i>

[![Website](https://img.shields.io/badge/Website-s--sosa.com/BF-blue?style=flat&logo=google-chrome&logoColor=white)](https://s-sosa.com/BF/) [![bioRxiv](https://img.shields.io/badge/bioRxiv-10.64898%2F2026.01.19.700318v1-BD271E?style=flat&logo=biorxiv&logoColor=white)](https://www.biorxiv.org/content/10.64898/2026.01.19.700318v1) ![Python](https://img.shields.io/badge/Python-3+-3776AB?logo=python&logoColor=white) [![License: GPL (\>= 3)](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

</div>

---

## One Mental Model. Three Languages.

**BayesForge (BF)** provides a unified experience across Julia, Python, and R. Whether you work in R's formula syntax, Python's object-oriented approach, or Julia's mathematical elegance, the model logic remains consistent.

-   ✅ **Zero Context Switching**: Variable names, distribution signatures, and model logic remain consistent across all implementations.
-   ✅ **NumPyro Power**: All interfaces compile down to XLA via JAX for blazing fast inference.
-   ✅ **Rich Diagnostics**: Seamless integration with ArviZ for posterior analysis.

### Compare the Syntax

<table width="100%">
<tr>
<th width="33%">Python Syntax</th>
<th width="33%">Julia Syntax</th>
<th width="33%">R Syntax</th>
</tr>
<tr>
<td valign="top">

```python
def model(height, weight):
    # Priors
    sigma = bf.dist.uniform(0, 50, name='sigma', shape=(1,))
    alpha = bf.dist.normal(178, 20, name='alpha', shape=(1,))
    beta  = bf.dist.normal(0, 1, name='beta', shape=(1,))

    # Likelihood
    mu = alpha + beta * weight
    bf.dist.normal(mu, sigma, obs=height)
```

</td>
<td valign="top">

```julia

@BI function model(weight, height)
    # Priors
    sigma = bf.dist.uniform(0, 50, name='sigma', shape=(1,))
    alpha = bf.dist.normal(178, 20, name='alpha', shape=(1,))
    beta  = bf.dist.normal(0, 1, name='beta', shape=(1,))

    # Likelihood
    mu = alpha + beta * weight
    bf.dist.normal(mu, sigma, obs=height)
end

```

<td valign="top">

```r
model <- function(height, weight){
  # Priors
  sigma = bf.dist.uniform(0, 50, name='sigma', shape=c(1))
  alpha = bf.dist.normal(178, 20, name='alpha', shape=c(1))
  beta  = bf.dist.normal(0, 1, name='beta', shape=c(1))

  # Likelihood
  mu = alpha + beta * weight
  bf.dist.normal(mu, sigma, obs=height)
}
```

</details>

</td>
</tr>
</table>

---

## Built for Speed

Leveraging Just-In-Time (JIT) compilation via JAX, bf outperforms traditional engines on standard hardware and unlocks massive scalability on GPU clusters for large datasets.

**Benchmark: Network Size 800 (Lower is Better)**

| Engine         | Execution Time                 | Relative Performance |
|:---------------|:-------------------------------|:---------------------|
| **STAN (CPU)** | `███████████████████████████████████████` | *Baseline*           |
| **BF (CPU)**   | `██████████`                 | **\~3.7x Faster**     |
| **BF (GPU)**   | `██`                           | **\~13.5X Faster**    |


*\> Comparison of execution time for a Social Relations Model. Source: [Sosa et al. (2026)](https://www.biorxiv.org/content/10.64898/2026.01.19.700318v2).*

------------------------------------------------------------------------

## Installation & Setup

### 1. Install Python

Download and install [Python 3.10 or later](https://www.python.org/downloads/)

### 2. Install Package

#### From pip

``` python
pip install BayesForge
```

#### Development Installation

``` python
pip install git+https://github.com/BGN-for-ASNA/BF.git
```

Or clone the repository and activate it locally:

``` bash
git clone https://github.com/BGN-for-ASNA/BF.git
cd BF
```

Then in Python:

``` python
pip install -e .
```

### 3. Initialize Environment

``` python
from BF import BF
m = BF()
```

### 4. Select Backend

Choose `"cpu"`, `"gpu"`, or `"tpu"` when importing the library.

``` python
# Initialize on CPU (default)
m = BF(platform="cpu")

# Or on GPU (requires JAX GPU installation)
m = BF(platform="gpu")
```

------------------------------------------------------------------------

## Quick Start

``` python
from BayesForge import bf

# Initialize BF
m = bf('cpu')

# Generate some data
x = m.dist.normal(0, 1, shape=(100,), sample=True)
y = m.dist.normal(0.2 + 0.6 * x, 1.2, sample=True)

# Define a Bayesian linear regression model
def linear_model(x, y):
    alpha = m.dist.normal(loc=0, scale=1, name="alpha")
    beta  = m.dist.normal(loc=0, scale=1, name="beta")
    sigma = m.dist.exponential(1, name="sigma")
    mu = alpha + beta * x
    m.dist.normal(mu, sigma, obs=y)


# Fit the model
m.fit(linear_model, num_warmup=1000, num_samples=1000, num_chains=1)

# Display results
m.summary()

# Plot results with @pyplot
m.plot_trace()
```

------------------------------------------------------------------------

## Features

### Data Manipulation

-   One-hot encoding
-   Index variable conversion
-   Scaling and normalization

### Modeling (via NumPyro)

-   **Linear & Generalized Linear Models**: Regression, Binomial, Poisson, Negative Binomial, etc.
-   **Hierarchical/Multilevel Models**: Varying intercepts and slopes.
-   **Time Series & Processes**: Gaussian Processes, Gaussian Random Walks, State Space Models.
-   **Mixture Models**: GMM, Dirichlet Process Mixtures.
-   **Network Models**: Network-based diffusion, Block models.
-   **Bayesian Neural Networks (BNN)**.

### Diagnostics (via ArviZ)

-   Posterior summary statistics and plots.
-   Trace plots, Density plots, Autocorrelation.
-   WAIC and LOO (ELPD) model comparison.
-   R-hat and Effective Sample Size (ESS).

------------------------------------------------------------------------

## Available Distributions

The package provides wrappers for a comprehensive set of distributions from NumPyro.

### Continuous

-   `m.dist.normal`, `m.dist.uniform`, `m.dist.student_t`
-   `m.dist.cauchy`, `m.dist.halfcauchy`, `m.dist.halfnormal`
-   `m.dist.gamma`, `m.dist.inverse_gamma`, `m.dist.exponential`
-   `m.dist.beta`, `m.dist.beta_proportion`
-   `m.dist.laplace`, `m.dist.asymmetric_laplace`
-   `m.dist.log_normal`, `m.dist.log_uniform`
-   `m.dist.pareto`, `m.dist.weibull`, `m.dist.gumbel`
-   `m.dist.chi2`, `m.dist.gompertz`

### Discrete

-   `m.dist.bernoulli`, `m.dist.binomial`
-   `m.dist.poisson`, `m.dist.negative_binomial`
-   `m.dist.geometric`, `m.dist.discrete_uniform`
-   `m.dist.beta_binomial`, `m.dist.zero_inflated_poisson`

### Multivariate

-   `m.dist.multivariate_normal`, `m.dist.multivariate_student_t`
-   `m.dist.dirichlet`, `m.dist.dirichlet_multinomial`
-   `m.dist.multinomial`
-   `m.dist.lkj`, `m.dist.lkj_cholesky`
-   `m.dist.wishart`, `m.dist.wishart_cholesky`

### Time Series & Stochastic Processes

-   `m.dist.gaussian_random_walk`
-   `m.dist.gaussian_state_space`
-   `m.dist.euler_maruyama`
-   `m.dist.car` (Conditional AutoRegressive)

### Mixtures & Truncated

-   `m.dist.mixture`, `m.dist.mixture_same_family`
-   `m.dist.truncated_normal`, `m.dist.truncated_cauchy`
-   `m.dist.lower_truncated_power_law`

*(See package documentation for the full list)*

------------------------------------------------------------------------

## Documentation

For full documentation and examples visit [Official website\](https://s-sosa.com/BF/0.Introduction.html)

------------------------------------------------------------------------

## Platform Support

-   ✅ Linux
-   ✅ macOS
-   ✅ Windows

GPU support available on compatible systems with JAX GPU installation.

------------------------------------------------------------------------

## Related Packages

-   [BFR](https://github.com/BGN-for-ASNA/BIR) - R implementation
-   [BFJ](https://github.com/BGN-for-ASNA/PyBayesForge.jl) - Julia implementation

------------------------------------------------------------------------

::: {align="center"}
**BayesForge**

Based on "The BayesForge library for Python, R, Julia" by [Sosa, McElreath, & Ross (2026)](https://www.biorxiv.org/content/10.64898/2026.01.19.700318v2).


[Official website](https://s-sosa.com/BF) \| [Issues](https://github.com/BGN-for-ASNA/BI/issues) \| [Quick Start](https://s-sosa.com/BF/start/Installation.html)

© 2026 BayesForge Team. Released under GPL-3.0.
:::
::::
:::::