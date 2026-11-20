import warnings

import jax
import jax.numpy as jnp
from jax import jit, random
from functools import partial
import numpy as onp

# For type hinting
from jax.typing import ArrayLike
from jax import Array
from typing import Optional



def _sq_dist(X: Array, C: Array) -> Array:
    """
    Squared Euclidean distances between the rows of X (n_samples, n_features)
    and the rows of C (n_clusters, n_features), as an array of shape
    (n_samples, n_clusters).

    RELEASE_AUDIT M2: uses the identity ||x||^2 - 2 x.c + ||c||^2 so memory is
    O(N*K) instead of materialising an (N, K, D) temporary at every iteration
    (N=100k, K=50, D=100 was ~2 Go per Lloyd step). The result is clipped at 0
    because the cross term can push the identity slightly below zero in
    floating point.
    """
    return jnp.maximum(
        jnp.sum(X ** 2, axis=1)[:, None] - 2.0 * (X @ C.T) + jnp.sum(C ** 2, axis=1)[None, :],
        0.0,
    )


@partial(jit, static_argnums=(2, 3))
def _lloyd(X: Array, initial_centroids: Array, n_clusters: int, n_iterations: int,
           tol: float = 0.0):
    """
    One full k-means run: Lloyd's loop, then final labels and inertia.

    Module-level and jitted with (n_clusters, n_iterations) static so the
    compilation is cached across runs and across instances of the same shape
    (n_init runs reuse one compiled kernel). ``tol`` is passed as a runtime
    value rather than a compile-time constant, so changing it does not trigger
    a recompile.

    tol : float
        Convergence tolerance on the largest squared centroid movement of an
        iteration. 0 runs exactly ``n_iterations`` iterations, which is the
        historical behaviour.
    """
    def update_step(state):
        """
        Performs a single iteration of the K-means algorithm (Assignment + Update).
        This function is the body of the `jax.lax.while_loop` below.
        """
        i, centroids, _ = state
        # --- Assignment step ---
        # Squared distances, shape (n_samples, n_clusters), O(N*K) memory
        # via _sq_dist (RELEASE_AUDIT M2).
        distances_sq = _sq_dist(X, centroids)

        # Assign each point to the closest centroid
        labels = jnp.argmin(distances_sq, axis=1)

        # --- Update step ---
        # Create a one-hot encoding of the labels
        # dtype of X: under jax_enable_x64 one_hot defaults to float64 and the
        # while_loop carry would change dtype for float32 data.
        one_hot_labels = jax.nn.one_hot(labels, n_clusters, dtype=X.dtype) # shape (n_samples, n_clusters)

        # Calculate the sum of points for each cluster
        # (k, n_samples) @ (n_samples, n_features) -> (n_clusters, n_features)
        sums = jnp.dot(one_hot_labels.T, X)

        # Calculate the number of points in each cluster
        counts = jnp.sum(one_hot_labels, axis=0) # shape (n_clusters,)

        # Calculate new centroids, handling empty clusters
        # If a cluster is empty (count=0), its centroid remains unchanged
        new_centroids = jnp.where(
            counts[:, None] > 0,          # Condition
            sums / counts[:, None],       # Value if true
            centroids                     # Value if false
        )
        # Largest squared distance any centroid moved this iteration; 0.0 once the
        # assignment is stable, which is what the tolerance below reads.
        shift = jnp.max(jnp.sum(jnp.square(new_centroids - centroids), axis=1))
        return i + 1, new_centroids, shift

    def cond_fn(state):
        i, centroids, shift = state
        # tol <= 0 disables early stopping: the loop then runs exactly
        # n_iterations, as it did before the tolerance existed.
        converged = (tol > 0) & (shift <= tol)
        return (i < n_iterations) & ~converged

    # Run the main K-means loop for at most n_iterations, stopping early once
    # the centroids stop moving by more than tol.
    n_done, final_centroids, final_shift = jax.lax.while_loop(
        cond_fn, update_step,
        (jnp.int32(0), initial_centroids, jnp.array(jnp.inf, dtype=X.dtype)))

    # Final labels and inertia for the output
    final_distances_sq = _sq_dist(X, final_centroids)
    final_labels = jnp.argmin(final_distances_sq, axis=1)
    inertia = jnp.sum(jnp.min(final_distances_sq, axis=1))
    # final_shift is the movement of the last executed iteration: the caller
    # uses it to tell "converged" from "budget exhausted".
    return final_centroids, final_labels, inertia, n_done, final_shift


def _validate_tol(tol) -> float:
    """
    RELEASE_AUDIT m1: ``tol`` is validated at construction *and* on every fit.

    The constructor only fits when X is given, so validating inside fit alone
    let ``JAXKMeans(tol=-1)`` (or NaN/inf) build an estimator that silently
    mis-behaved: a negative tolerance stops the loop before the convergence
    check can mean anything, and NaN never satisfies ``shift <= tol``.
    Accepts a non-negative finite number; 0 disables early stopping.
    """
    if (not isinstance(tol, (int, float)) or isinstance(tol, bool)
            or not onp.isfinite(tol) or tol < 0):
        raise ValueError(
            f"tol must be a non-negative finite number (0 disables early "
            f"stopping); got {tol!r}.")
    return float(tol)


def _kmeans_plus_plus(X: Array, n_clusters: int, key: Array) -> Array:
    """
    k-means++ initialisation (Arthur & Vassilvitskii, 2007) — RELEASE_AUDIT M1.

    The first centre is drawn uniformly from the points; every next centre is
    sampled with probability proportional to D(x)^2, the squared distance of
    each point to the nearest already-chosen centre. Runs eagerly on concrete
    arrays with a key derived from the caller's (fold_in/split upstream).
    """
    n_samples = X.shape[0]
    keys = random.split(key, n_clusters)

    # First centre: uniform over the points.
    idx = random.randint(keys[0], shape=(), minval=0, maxval=n_samples)
    first = X[idx]
    centroids = [first]
    closest_sq = _sq_dist(X, first[None, :])[:, 0]

    for i in range(1, n_clusters):
        total = float(jnp.sum(closest_sq))
        if total > 0.0:
            probs = closest_sq / total
        else:
            # Every point already coincides with a chosen centre: fall back
            # to a uniform draw (sklearn does the same).
            probs = jnp.full((n_samples,), 1.0 / n_samples)
        idx = random.choice(keys[i], n_samples, shape=(), p=probs)
        new_centre = X[idx]
        centroids.append(new_centre)
        closest_sq = jnp.minimum(closest_sq, _sq_dist(X, new_centre[None, :])[:, 0])

    return jnp.stack(centroids)


class JAXKMeans:
    """
    K-Means clustering implemented in JAX with a Scikit-learn like API.

    Parameters
    ----------
    X : ArrayLike of shape (n_samples, n_features)
        Training instances to cluster.

        RELEASE_AUDIT M4 — ``X`` is now optional: ``JAXKMeans(n_clusters=K)``
        builds an *unfitted* estimator (sklearn contract) that you fit later
        with ``fit(X)``. Passing ``X`` to the constructor still fits eagerly,
        on purpose: BayesForge/Main/main.py builds `ml.KMEANS(data,
        n_clusters=K)` and immediately reads `ml.results["centroids"]` to seed
        a GMM's `initial_means`, without any explicit fit() call, and existing
        tests construct with data and read `centroids_` directly. A
        `fit()` call may override the data (`fit(X2)`) or reuse the data the
        instance already holds (`fit()`); it honours (and updates) the
        instance state, and each successive fit folds a new sub-key so two
        fits on the same instance do not reuse the identical initialisation.
    n_clusters : int
        The number of clusters to form as well as the number of centroids
        to generate.
    n_iterations : int, default=100
        Maximum number of iterations of the k-means algorithm for a
        single run.
    random_state : int, optional
        Determines random number generation for centroid initialization. Use an
        int to make the randomness deterministic.
    init : {"k-means++", "random"}, default="k-means++"
        Initialisation strategy (RELEASE_AUDIT M1). "k-means++" samples
        successive centres with probability proportional to the squared
        distance to the nearest chosen centre; "random" keeps the legacy
        behaviour of drawing k points uniformly without replacement.
    n_init : int, default=10
        Number of times the Lloyd loop is run with different seeds (one
        k-means++ or random initialisation each); the run with the lowest
        inertia is kept. Default 10, like scikit-learn.
    tol : float, default=1e-4
        Convergence tolerance. A run stops as soon as no centroid moves by
        more than ``tol`` in squared L2 distance between two iterations, so a fit that
        converges early does not spend the whole ``n_iterations`` budget.
        Set ``tol=0`` to disable early stopping and always run
        ``n_iterations`` iterations, the behaviour of earlier releases.
    """
    def __init__(self, X: Optional[ArrayLike] = None, n_clusters: int = 3,
                 n_iterations: int = 100,
                 random_state: Optional[int] = None, init: str = "k-means++",
                 n_init: int = 10, tol: float = 1e-4):

        # RELEASE_AUDIT M4: X is optional so an unfitted estimator can be
        # built (sklearn contract); when given, the eager fit below keeps the
        # ml.KMEANS -> results path working without an explicit fit call.
        self.X = None if X is None else jnp.asarray(X)
        self.n_clusters = n_clusters
        self.n_iterations = n_iterations
        self.random_state = random_state
        self.init = init
        self.n_init = n_init
        # RELEASE_AUDIT m1: validate at construction too — the eager fit
        # below only runs when X is given, so without this an invalid tol
        # on an unfitted estimator would go unnoticed.
        self.tol = _validate_tol(tol)

        # Attributes that will be set after fitting (ending with an underscore)
        self.centroids_: Optional[Array] = None
        self.labels_: Optional[Array] = None
        self.inertia_: Optional[Array] = None
        # Iterations actually run by the winning n_init run (RELEASE_AUDIT m1).
        self.n_iter_: Optional[int] = None

        # Create a JAX random key
        # None keeps the historical fixed key 0 (reproducible GMM initialisation);
        # an int selects the key so seed sweeps really change the initialisation.
        self.key = random.PRNGKey(0 if random_state is None else int(random_state))

        # RELEASE_AUDIT M4: self.key used to be reused verbatim by every fit,
        # so successive fits reproduced the exact same initialisation. Each fit
        # now folds this counter into the key; a fresh instance with the same
        # random_state stays bit-for-bit reproducible.
        self._fit_count = 0

        self.results = {}
        # None until predict() runs; predict() stores the labels array itself
        # (RELEASE_AUDIT m2), so this never mixes dict and array types.
        self.predictions = None

        # Backward-compatible eager fit (see class docstring, RELEASE_AUDIT M4):
        # main.py's GMM path reads .results right after construction.
        if X is not None:
            self.fit(X)

    def fit(self, X: Optional[ArrayLike] = None, n_clusters: Optional[int] = None,
            n_iterations: Optional[int] = None, init: Optional[str] = None,
            n_init: Optional[int] = None, tol: Optional[float] = None):
        """
        Compute k-means clustering.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features), optional
            Training instances to cluster. Defaults to the data the instance
            already holds (constructor or previous fit) — RELEASE_AUDIT M4.
        n_clusters : int, optional
            Overrides self.n_clusters for this call. The resolved value is
            written back to the instance (RELEASE_AUDIT M4) so the attributes
            never lie about the fitted state.
        n_iterations : int, optional
            Same rule as n_clusters.
        init : {"k-means++", "random"}, optional
            Same rule as n_clusters.
        n_init : int, optional
            Same rule as n_clusters.
        tol : float, optional
            Same rule as n_clusters. A negative tolerance is rejected; 0
            disables early stopping and runs the full ``n_iterations``.

        Returns
        -------
        self
            Fitted estimator.
        """
        # --- Input validation (RELEASE_AUDIT M3): concrete numpy checks done
        # before any JAX tracing, with domain-level messages.
        if X is None:
            X = self.X
        if X is None:
            raise ValueError(
                "KMeans.fit needs data: pass X to fit() or to the constructor."
            )
        X_arr = onp.asarray(X)
        if X_arr.ndim != 2:
            raise ValueError(
                "KMeans expects X to be a 2-D array of shape "
                "(n_samples, n_features); for 1-D data pass X[:, None]."
            )
        if X_arr.shape[0] < 1:
            raise ValueError("KMeans expects at least 1 sample; X has 0 rows.")
        if n_clusters is None:
            n_clusters = self.n_clusters
        if (not isinstance(n_clusters, (int, onp.integer)) or isinstance(n_clusters, bool)
                or not 1 <= int(n_clusters) <= X_arr.shape[0]):
            raise ValueError(
                f"n_clusters must satisfy 1 <= n_clusters <= n_samples "
                f"(got n_clusters={n_clusters}, n_samples={X_arr.shape[0]})."
            )
        if not onp.all(onp.isfinite(X_arr)):
            raise ValueError(
                "X contains NaN or inf values; KMeans cannot fit non-finite data."
            )

        n_iterations = self.n_iterations if n_iterations is None else n_iterations
        init = self.init if init is None else init
        n_init = self.n_init if n_init is None else n_init
        tol = self.tol if tol is None else tol
        if init not in ("k-means++", "random"):
            raise ValueError(f"init must be 'k-means++' or 'random', got {init!r}.")
        if (not isinstance(n_init, (int, onp.integer)) or isinstance(n_init, bool)
                or int(n_init) < 1):
            raise ValueError(f"n_init must be a positive integer, got {n_init!r}.")
        n_clusters = int(n_clusters)
        n_init = int(n_init)
        tol = _validate_tol(tol)

        # RELEASE_AUDIT M4: record the settings actually used by this fit on
        # the instance — after JAXKMeans(X, 3).fit(X2, 5), self.n_clusters is
        # 5 and matches centroids_, not a lying 3.
        self.n_clusters = n_clusters
        self.n_iterations = n_iterations
        self.init = init
        self.n_init = n_init
        self.tol = tol

        # RELEASE_AUDIT M4: remember the data so a later bare fit() reuses it.
        self.X = X

        X_jnp = jnp.asarray(X)
        if not jnp.issubdtype(X_jnp.dtype, jnp.floating):
            X_jnp = X_jnp.astype(float)
        # RELEASE_AUDIT m4: float64 input is silently downcast to float32 when
        # jax_enable_x64 is off (the default). Warn instead of hiding it. The
        # dtype must be read from the *original* input (X_arr): jnp.asarray
        # has already downcast X_jnp to float32 here, so checking X_jnp would
        # never see the float64 and the warning was dead code. The cast itself
        # is the existing M4 behaviour and stays as it was.
        if (onp.issubdtype(X_arr.dtype, onp.floating) and X_arr.dtype.itemsize > 4
                and not jax.config.jax_enable_x64):
            warnings.warn(
                f"X is {X_arr.dtype} but jax_enable_x64 is disabled, so this "
                "fit runs in float32: the data is downcast and precision "
                "beyond float32 is lost. Call "
                "jax.config.update('jax_enable_x64', True) before the first "
                "JAX operation to keep float64, or pass float32 data when the "
                "reduction is intended.",
                UserWarning,
                stacklevel=2,
            )

        # RELEASE_AUDIT M4: derive a per-fit base key (successive fits explore
        # fresh initialisations) and a per-run key for each of the n_init runs
        # (RELEASE_AUDIT M1). Same random_state on a fresh instance -> same
        # keys -> identical result; different random_state -> different inits.
        base_key = random.fold_in(self.key, self._fit_count)
        self._fit_count += 1

        # RELEASE_AUDIT M1: n_init independent runs, keep the lowest inertia
        # (sklearn default is 10; the legacy code ran a single uniform draw).
        best = None
        for run in range(n_init):
            run_key = random.fold_in(base_key, run)
            if init == "k-means++":
                initial_centroids = _kmeans_plus_plus(X_jnp, n_clusters, run_key)
            else:
                # Legacy behaviour: k points drawn uniformly without replacement.
                initial_centroids = random.choice(
                    run_key, X_jnp, shape=(n_clusters,), replace=False)
            centroids, labels, inertia, n_done, shift = _lloyd(
                X_jnp, initial_centroids, n_clusters, n_iterations, tol)
            if best is None or float(inertia) < float(best[2]):
                best = (centroids, labels, inertia, n_done, shift)
        final_centroids, final_labels, inertia, n_done, final_shift = best

        # Store the results in the instance
        self.centroids_ = final_centroids
        self.labels_ = final_labels
        self.inertia_ = inertia
        # RELEASE_AUDIT m1: iterations the winning run actually took before it
        # stopped moving (n_iterations when tol=0 or it never converged).
        self.n_iter_ = int(n_done)
        # RELEASE_AUDIT m1: the budget is a user-set cap, not a convergence
        # guarantee. When tol > 0 the caller asked for convergence, so say so
        # out loud when the last centroid move still exceeded it.
        if tol > 0 and float(final_shift) > tol:
            warnings.warn(
                f"KMeans stopped after {n_iterations} iteration(s) "
                f"without converging: the largest centroid move was "
                f"{float(final_shift):.3e} > tol={tol:.3e}. Raise "
                "n_iterations or relax tol.",
                UserWarning,
                stacklevel=2,
            )
        self.results={
            "centroids": final_centroids,
            "labels": final_labels,
            "inertia": inertia
        }
        return self

    def predict(self, X: ArrayLike) -> Array:
        """
        Predicts the closest cluster for each sample in X based on trained centroids.

        Args:
            X (ArrayLike): The data to predict, shape (n_samples, n_features).
                The centroids come from the fitted model (self.centroids_);
                there is no centroids argument.

        Returns:
            Array: The index of the cluster each sample belongs to, shape (n_samples,).
        """
        # A check to ensure fit has been called.
        if self.centroids_ is None:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        X_jnp = jnp.asarray(X)
        if X_jnp.ndim != 2:
            raise ValueError(
                f"X must be a 2-D array of shape (n_samples, n_features); "
                f"got a {X_jnp.ndim}-D array with shape {X_jnp.shape}.")
        centroids_jnp = jnp.asarray(self.centroids_)

        # O(N*K) squared distances via the shared helper (RELEASE_AUDIT M2).
        distances_sq = _sq_dist(X_jnp, centroids_jnp)
        labels = jnp.argmin(distances_sq, axis=1)
        self.predictions = labels
        return labels

    def plot(self, X: ArrayLike, show_contours: bool = True, show_centroids: bool = True,
             figsize=(10, 8), grid_size: int = 200):
        """
        Generates an advanced plot of the clustering results, including data points,
        centroids, and distance contours.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, 2)
            The data that was used for fitting. The plot is only generated for 2D data.
        show_contours : bool, default=True
            If True, draws contours of the distance to the nearest centroid.
        show_centroids : bool, default=True
            If True, displays the final cluster centroids.
        grid_size : int, default=200
            Number of grid points per axis for the contour field.
        """
        # RELEASE_AUDIT m7: matplotlib is imported here, not at module level,
        # so headless users of the estimator pay no import cost.
        import matplotlib.pyplot as plt

        # --- 1. Pre-computation and Checks ---

        # Check if the model is fitted
        if self.centroids_ is None or self.labels_ is None:
            raise RuntimeError("Model must be fitted before plotting. Call .fit() first.")

        # RELEASE_AUDIT m3: accept a plain list, like fit() and predict() do;
        # the old X.shape[1] raised AttributeError on list input.
        X = jnp.asarray(X)

        # This visualization is only meaningful for 2D data. A silent return
        # left the caller with no figure and no error, so raise instead.
        if X.ndim != 2 or X.shape[1] != 2:
            raise ValueError(
                "plot supports 2-D data of shape (n_samples, 2) only; got shape "
                f"{tuple(X.shape)}.")
        if (not isinstance(grid_size, (int, onp.integer)) or isinstance(grid_size, bool)
                or int(grid_size) < 2):
            raise ValueError(f"grid_size must be an integer >= 2, got {grid_size!r}.")
        grid_size = int(grid_size)

        # RELEASE_AUDIT m3: the point colours come from self.labels_, so a
        # different X silently mislabels the points. Warn instead.
        if self.X is not None:
            X_fit = jnp.asarray(self.X)
            if X_fit.shape != X.shape or not bool(jnp.array_equal(X_fit, X)):
                warnings.warn(
                    "plot() was given data that is not the data this model was fitted on; "
                    "the point colours come from the fitted labels_ and may "
                    "not match these points.",
                    UserWarning,
                    stacklevel=2,
                )

        # Optional dependency check for seaborn
        try:
            import seaborn as sns
        except ImportError:
            sns = None
            print("Warning: Seaborn not found. Using default Matplotlib styles. "
                  "Install with: pip install seaborn")

        # RELEASE_AUDIT m3: 'seaborn-v0_8-whitegrid' needs matplotlib >= 3.6;
        # on older versions style.use() raised OSError/KeyError, which the old
        # try/except (ImportError only) let escape. Check availability first,
        # and apply the style in a context so global matplotlib state is not
        # changed for the rest of the process.
        style = None
        if sns is not None:
            if 'seaborn-v0_8-whitegrid' in plt.style.available:
                style = 'seaborn-v0_8-whitegrid'
            else:
                print("Warning: Matplotlib style 'seaborn-v0_8-whitegrid' is "
                      "not available in this version. Using default styles.")

        # --- 2. Plot Setup ---

        with plt.style.context(style or {}):
            fig, ax = plt.subplots(figsize=figsize)
            fig.patch.set_facecolor('#f0f0f0')
            ax.set_facecolor('#f0f0f0')

            # --- 3. Dynamic Color Palette for Data Points ---

            # Use standard numpy for unique, as it's on the CPU-bound labels array
            unique_labels = onp.unique(self.labels_)
            n_clusters = len(unique_labels)
            if sns is not None:
                palette = sns.color_palette("viridis", n_colors=n_clusters)
            else:
                # plt.cm.viridis is not indexable by position on recent
                # matplotlib, so sample the colormap explicitly.
                palette = [plt.cm.viridis(i / max(n_clusters - 1, 1))
                           for i in range(n_clusters)]

            color_map = {label: palette[i] for i, label in enumerate(unique_labels)}
            point_colors = [color_map[l] for l in onp.array(self.labels_)]

            # Plot the data points
            ax.scatter(X[:, 0], X[:, 1], c=point_colors, s=25, alpha=0.9,
                       edgecolor='white', linewidth=0.5, label='Data Points')

            # --- 4. Plot Centroids ---

            if show_centroids:
                # Plot with a distinctive style
                ax.scatter(self.centroids_[:, 0], self.centroids_[:, 1],
                           marker='X', s=250, c='black',
                           edgecolor='white', linewidth=1.5, label='Centroids')

            # --- 5. Calculate and Plot Distance Contours ---

            if show_contours:
                # Define the grid for the contour plot.
                # RELEASE_AUDIT m3: the old fixed 0.02 step made the grid grow
                # with the data extent (a span of 10^4 meant 500k points per
                # axis); a fixed point count bounds the memory, grid_size.
                x_min, x_max = float(X[:, 0].min()) - 1, float(X[:, 0].max()) + 1
                y_min, y_max = float(X[:, 1].min()) - 1, float(X[:, 1].max()) + 1
                xx, yy = onp.meshgrid(onp.linspace(x_min, x_max, grid_size),
                                      onp.linspace(y_min, y_max, grid_size))

                # Create a JIT-compiled function to score every point on the grid
                @jit
                def score_samples(points, centroids):
                    # Calculate the distance to the *nearest* centroid
                    # O(N*K) via the shared helper (RELEASE_AUDIT M2).
                    distances_sq = _sq_dist(points, centroids)
                    min_dist_sq = jnp.min(distances_sq, axis=1)
                    return min_dist_sq

                # Predict the "score" for each point in the meshgrid
                # onp.c_ concatenates the flattened grids into a list of points
                grid_points = onp.c_[xx.ravel(), yy.ravel()]
                Z = score_samples(jnp.asarray(grid_points), self.centroids_)
                Z = Z.reshape(xx.shape)

                # Plot the contour lines
                ax.contour(xx, yy, Z, levels=10, colors='dimgray',
                           linewidths=0.8, linestyles='--')

            # --- 6. Final Styling ---

            # RELEASE_AUDIT m3: the contours show the distance to the nearest
            # centroid, not the decision boundaries themselves (the boundaries
            # are the level sets where two centroids tie), so say that.
            ax.set_title("K-Means Clustering (contours: distance to the "
                         "nearest centroid)", fontsize=16)
            ax.set_xlabel("Feature 1", fontsize=12)
            ax.set_ylabel("Feature 2", fontsize=12)
            ax.legend()
            ax.grid(True, linestyle=':', color='gray', alpha=0.6)
            # Enforce a square aspect ratio for a correct view of distances
            ax.set_aspect('equal', adjustable='box')

            plt.show()
