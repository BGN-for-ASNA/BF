"""ABC random forests (ABC-RF).

Raynal et al. (2019): simulate a reference table {(theta_j, s(x_j))} from the
prior predictive, then fit a regression random forest predicting each
parameter from the summary statistics. The forest is used as an adaptive
nearest-neighbour rule: following quantile regression forests (Meinshausen
2006), simulation j receives the weight

    w_j(s_obs) = 1/T * sum_t 1[j in leaf_t(s_obs)] / |leaf_t(s_obs)|,

and the weighted reference table approximates the posterior. No tolerance and
no distance need to be chosen, uninformative statistics are largely ignored by
the forest, and out-of-bag predictions give a free estimate of accuracy.

By default (Raynal et al.) one forest is trained per scalar parameter, which
gives accurate *marginal* posteriors; draws of different parameters are then
independent. ``joint=True`` trains a single multi-output forest whose weights
resample whole rows of the table, preserving posterior correlations.

Random forests come from scikit-learn (imported lazily, optional dependency).
"""

import time

import jax.numpy as jnp
import numpy as np

from BayesForge.Samplers.SBI.simulate import Simulator, SBIPosterior


def _forest(**kwargs):
    try:
        from sklearn.ensemble import RandomForestRegressor
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "ABC random forests need scikit-learn: pip install scikit-learn"
        ) from e
    return RandomForestRegressor(**kwargs)


def _qrf_weights(forest, S, s_obs):
    """Quantile-regression-forest weights of the reference table for s_obs."""
    leaves = forest.apply(S)                       # (N, T)
    leaf_obs = forest.apply(s_obs[None, :])[0]     # (T,)
    same = leaves == leaf_obs[None, :]
    w = (same / same.sum(axis=0, keepdims=True)).mean(axis=1)
    return w / w.sum()


class ABCRF_BF(SBIPosterior):
    """ABC random forest. Behaves like a BF sampler.

    Attributes after ``run``:
        posterior: posterior draws (constrained space).
        forests: {column: fitted RandomForestRegressor} (or {"joint": forest}).
        weights: {column: (N,) weights of the reference table}.
        importance: DataFrame of summary-statistic importances (stats x columns).
        oob: DataFrame of out-of-bag accuracy per column (R2, MAE).
    """

    def __init__(self, simulator: Simulator, num_simulations=10_000, n_estimators=500,
                 min_samples_leaf=5, max_features=1 / 3, joint=False, stat_names=None,
                 num_samples=1000, n_jobs=-1, random_state=0, progress_bar=True):
        super().__init__()
        self.sim = simulator
        self.num_simulations = int(num_simulations)
        self.n_estimators = int(n_estimators)
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.joint = bool(joint)
        self.stat_names = stat_names
        self.num_samples = int(num_samples)
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.progress_bar = progress_bar

    def _log(self, msg):
        if self.progress_bar:
            print(msg)

    def _columns(self, theta):
        """dict of constrained (N, ...) arrays -> (N, P) matrix + column names."""
        cols, names = [], []
        for n in self.sim.param_names:
            v = np.asarray(theta[n]).reshape(theta[n].shape[0], -1)
            cols.append(v)
            if v.shape[1] == 1:
                names.append(n)
            else:
                names += [f"{n}[{i}]" for i in range(v.shape[1])]
        return np.concatenate(cols, axis=1), names

    def _to_dict(self, M):
        out, i = {}, 0
        for n in self.sim.param_names:
            k = int(np.prod(self.sim.shapes[n]))
            out[n] = jnp.asarray(M[:, i:i + k].reshape((M.shape[0],) + self.sim.shapes[n]))
            i += k
        return out

    def run(self, rng_key):
        import pandas as pd

        t0 = time.time()
        theta, _, s = self.sim.simulate_prior(rng_key, self.num_simulations)
        Theta, names = self._columns(theta)
        S = np.asarray(s, dtype=np.float64)
        ok = np.all(np.isfinite(S), axis=1) & np.all(np.isfinite(Theta), axis=1)
        Theta, S = Theta[ok], S[ok]
        s_obs = np.asarray(self.sim.s_obs, dtype=np.float64)
        self.reference_table = (Theta, S)
        self.columns = names
        stat_names = self.stat_names or [f"s{i + 1}" for i in range(S.shape[1])]
        if len(stat_names) != S.shape[1]:
            raise ValueError(f"stat_names has {len(stat_names)} entries but the summary "
                             f"statistics have {S.shape[1]} dimensions.")
        self.stat_names = list(stat_names)

        rng = np.random.default_rng(self.random_state)
        kw = dict(n_estimators=self.n_estimators, min_samples_leaf=self.min_samples_leaf,
                  max_features=self.max_features, oob_score=True, n_jobs=self.n_jobs,
                  random_state=self.random_state)
        self.forests, self.weights, imp, oob = {}, {}, {}, {}

        if self.joint:
            mu, sd = Theta.mean(0), Theta.std(0) + 1e-12
            rf = _forest(**kw).fit(S, (Theta - mu) / sd)
            w = _qrf_weights(rf, S, s_obs)
            self.forests["joint"] = rf
            idx = rng.choice(Theta.shape[0], self.num_samples, p=w)
            draws = Theta[idx]
            pred = np.reshape(rf.oob_prediction_, Theta.shape) * sd + mu
            for j, c in enumerate(names):
                self.weights[c] = w
                oob[c] = self._oob_row(Theta[:, j], pred[:, j])
            imp["joint"] = rf.feature_importances_
        else:
            draws = np.empty((self.num_samples, Theta.shape[1]))
            for j, c in enumerate(names):
                rf = _forest(**kw).fit(S, Theta[:, j])
                w = _qrf_weights(rf, S, s_obs)
                self.forests[c], self.weights[c] = rf, w
                draws[:, j] = Theta[rng.choice(Theta.shape[0], self.num_samples, p=w), j]
                oob[c] = self._oob_row(Theta[:, j], rf.oob_prediction_)
                imp[c] = rf.feature_importances_
                self._log(f"ABC-RF: forest for {c} (OOB R2 = {oob[c]['oob_r2']:.3f})")

        self.importance = pd.DataFrame(imp, index=self.stat_names)
        self.oob = pd.DataFrame(oob).T
        self.posterior = self._to_dict(draws)
        self.extra = {}
        self.n_simulations = self.sim.n_simulations
        self.runtime = time.time() - t0
        self._log(f"ABC-RF: {S.shape[0]} simulations, {self.n_estimators} trees per forest, "
                  f"{self.runtime:.1f}s")
        return self

    @staticmethod
    def _oob_row(y, pred):
        ok = np.isfinite(pred)
        y, pred = y[ok], pred[ok]
        r2 = 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)
        return {"oob_r2": float(r2), "oob_mae": float(np.mean(np.abs(y - pred)))}

    def plot_importance(self, column=None, top=None, ax=None):
        """Bar plot of summary-statistic importances (mean decrease in impurity)."""
        import matplotlib.pyplot as plt
        imp = self.importance
        if column is None:
            column = imp.columns[0]
        v = imp[column].sort_values()
        if top is not None:
            v = v.iloc[-top:]
        if ax is None:
            _, ax = plt.subplots(figsize=(5, 0.3 * len(v) + 1))
        ax.barh(v.index, v.values)
        ax.set_xlabel("Importance")
        ax.set_title(str(column))
        return ax
