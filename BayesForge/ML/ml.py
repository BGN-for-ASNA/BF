"""
Machine-learning handlers on top of the JAX estimators in this package.

Naming note (RELEASE_AUDIT m8): the handler class is named ``ml``, the same
lowercase name as its module, so ``from BayesForge.ML.ml import ml`` (the
import main.py uses) reads oddly and shadows the module name. The class name
is kept for backward compatibility; ``ML = ml`` is provided as an alias for
code that prefers a PEP 8 class name:

    from BayesForge.ML.ml import ML          # or: import BayesForge.ML.ml
    handler = ML()

The constructor of ``JAXKMeans`` takes the data first (``X``), which also
departs from the sklearn convention of an argument-free constructor; that is
kept for the same backward-compatibility reason.
"""
# For type hinting
from jax.typing import ArrayLike
from jax import Array
from typing import Optional
import jax.numpy as jnp
from BayesForge.ML.KMeans import JAXKMeans
class ml():
    """
    A stateful handler to manage and interact with ML models using a fluent API.

    This handler allows selecting, fitting, and predicting with models
    in a chained, stateful manner.

    Example:
        handler = ml()
        handler.KMEANS(X, n_clusters=3).fit()
        predictions = handler.predict(X_test)
        handler.plot(X)  # plot takes the 2D data to draw
    """
    def __init__(self):
        self.model = None
        self.model_name = None
        self.X = None
        self.results = {}
        # Filled by predict() (RELEASE_AUDIT m6); it used to stay None even
        # after a prediction, which only the inner model recorded.
        self.predictions = None

    def fit(self, X: Optional[ArrayLike] = None):
        """
        Fits the model to the provided data.

        Parameters
        ----------
        X : ArrayLike, optional
            Data to fit; defaults to the data given to the model method (e.g. KMEANS).

        Returns
        -------
        self
            For method chaining (results in self.results).
        """
        if self.model is None:
            raise RuntimeError("No model has been initialized. Call a model method first.")

        if X is not None:
            self.X = jnp.asarray(X)
        if self.X is None:
            raise RuntimeError("No data: pass X to fit() or to the model method.")
        self.model.fit(self.X)
        self.results = self.model.results
        return self

    def predict(self, X: ArrayLike) -> Array:
        """
        Prediction method for the fitted model.

        Parameters
        ----------
        X : ArrayLike
            The input data to predict.

        Returns
        -------
        Array
            The predicted cluster labels. Also stored on self.predictions
            (RELEASE_AUDIT m6).
        """
        if self.model is None:
            raise RuntimeError("No model has been fitted. Call fit() first.")

        self.predictions = self.model.predict(X)
        return self.predictions

    def plot(self, X: ArrayLike):
        """
        Plots the model output.
        """
        if self.model is None or self.X is None:
            raise RuntimeError("No model has been fitted. Call fit() first.")
        
        self.model.plot(X)

    def KMEANS(self, X: ArrayLike, n_clusters: int, n_iterations: int = 100, random_state: Optional[int] = None,
               init: str = "k-means++", n_init: int = 10, tol: float = 1e-4):
        """
        Initializes a KMeans model with the given parameters.

        Parameters
        ----------
        X : ArrayLike
            The input data to fit the model.
        n_clusters : int
            The number of clusters to form.
        n_iterations : int, default=100
            The number of iterations for the K-means algorithm.
        random_state : int, optional
            Random seed for reproducibility.
        init : {"k-means++", "random"}, default="k-means++"
            Initialisation strategy (RELEASE_AUDIT M1); "random" keeps the
            legacy uniform draw.
        n_init : int, default=10
            Number of runs with different initialisations; the lowest
            inertia is kept (RELEASE_AUDIT M1).
        tol : float, default=1e-4
            Convergence tolerance passed to JAXKMeans (RELEASE_AUDIT m1);
            0 runs the full n_iterations.

        Returns
        -------
        self
            Returns the instance itself for method chaining.
        """
        self.X = jnp.asarray(X)
        self.model_name = 'KMEANS'
        self.model = JAXKMeans(X, n_clusters=n_clusters, n_iterations=n_iterations,
                               random_state=random_state, init=init, n_init=n_init,
                               tol=tol)
        # Copy, do not alias (RELEASE_AUDIT M4): mutating the model's results
        # dict must not retroactively mutate the handler's results.
        self.results = dict(self.model.results)
        return self


# RELEASE_AUDIT m8: PEP 8 style alias for the lowercase ``ml`` class, which is
# kept under its original name for backward compatibility (main.py imports it).
ML = ml
