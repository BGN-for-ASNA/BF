import functools

import jax
from jax import jit
import jax.numpy as jnp
from jax.scipy.stats import norm

class link:
    """A class to store and manage various mathematical link functions and their inverses.

    Notes
    -----
    - These are plain pointwise transforms: no Jacobians are provided. When a
      variable is transformed with a change-of-variables (volume) correction,
      the log|df/dx| term must be added by the caller.
    - ``LKJcor`` is a density helper for correlation matrices, not a link
      function; it is kept here for backward compatibility (``m.link.LKJcor``).
    - ``inv_logit`` (sigmoid) and ``logit`` (log/log1p) are exact inverses
      analytically but only approximately in finite precision near 0 and 1.
    - All link transforms are jitted so recompilation behaviour is uniform
      across the class; ``eps`` is a static argument of
      ``logit``/``probit``/``cloglog``.
    """
    def __init__(self):
        pass

    @staticmethod
    @functools.partial(jax.jit, static_argnames=("eps",))
    def logit(x, eps=None):
        """
        Computes the logit transformation.

        0 and 1 map to -inf/+inf unless eps is given (e.g. eps=1e-6 clips
        to [eps, 1 - eps]).

        Parameters
        ----------
        x : float or array-like
            Input value(s) in the range (0, 1).
        eps : float, optional
            If not None, x is clipped to [eps, 1 - eps] before the transform.

        Returns
        -------
        float or array-like
            The logit-transformed value(s): log(x) - log(1 - x), computed as
            log(x) - log1p(-x) to avoid overflow near 1.
        """
        if eps is not None:
            x = jnp.clip(x, eps, 1 - eps)
        return jnp.log(x) - jnp.log1p(-x)

    @staticmethod
    @jit
    def inv_logit(x):
        """
        Computes the inverse logit transformation.

        Parameters
        ----------
        x : float or array-like
            Input value(s).

        Returns
        -------
        float or array-like
            The inverse logit-transformed value(s): 1 / (1 + exp(-x)).
        """
        return jax.nn.sigmoid(x)

    @staticmethod
    @jit
    def inv_logit_scale(x):
        """
        Modified inverse logit function scaling between -1 and 1.
        From https://www.science.org/action/downloadSupplement?doi=10.1126%2Fsciadv.aax9070&file=aax9070_sm.pdf

        Parameters
        ----------
        x : float or array-like
            Input value(s). 
        
        Returns
        -------
        float or array-like
            2 * sigmoid(x) - 1, i.e. tanh(x / 2), in the range (-1, 1).
        """
        return jnp.tanh(x / 2)
        
    @staticmethod
    @functools.partial(jax.jit, static_argnames=("eps",))
    def probit(p, eps=None):
        """
        Computes the probit transformation.

        0 and 1 map to -inf/+inf unless eps is given (e.g. eps=1e-6 clips
        to [eps, 1 - eps]).

        Parameters
        ----------
        p : float or array-like
            Input probability value(s) in the range (0, 1).
        eps : float, optional
            If not None, p is clipped to [eps, 1 - eps] before the transform.

        Returns
        -------
        float or array-like
            The probit-transformed value(s), corresponding to the quantile of the 
            standard normal distribution.
        """
        if eps is not None:
            p = jnp.clip(p, eps, 1 - eps)
        return norm.ppf(p)

    @staticmethod
    @jit
    def inv_probit(x):
        """
        Computes the inverse probit transformation.

        Parameters
        ----------
        x : float or array-like
            Input value(s).

        Returns
        -------
        float or array-like
            The probability value(s) from the cumulative distribution function 
            of the standard normal distribution.
        """
        return norm.cdf(x)

    @staticmethod
    @jit
    def log(p):
        """
        Computes the natural logarithm.

        Parameters
        ----------
        p : float or array-like
            Input value(s).

        Returns
        -------
        float or array-like
            The natural logarithm of the input value(s).
        """
        return jnp.log(p)

    @staticmethod
    @jit
    def exp(x):
        """
        Computes the exponential function.

        Parameters
        ----------
        x : float or array-like
            Input value(s).

        Returns
        -------
        float or array-like
            The exponential of the input value(s).
        """
        return jnp.exp(x)

    @staticmethod
    @functools.partial(jax.jit, static_argnames=("eps",))
    def cloglog(p, eps=None):
        """
        Computes the complementary log-log transformation.

        0 and 1 map to -inf/+inf unless eps is given (e.g. eps=1e-6 clips
        to [eps, 1 - eps]).

        Parameters
        ----------
        p : float or array-like
            Input probability value(s) in the range (0, 1).
        eps : float, optional
            If not None, p is clipped to [eps, 1 - eps] before the transform.

        Returns
        -------
        float or array-like
            The complementary log-log transformed value(s): log(-log(1 - p)),
            computed as log(-log1p(-p)) for accuracy near 0.
        """
        if eps is not None:
            p = jnp.clip(p, eps, 1 - eps)
        return jnp.log(-jnp.log1p(-p))

    @staticmethod
    @jit
    def inv_cloglog(x):
        """
        Computes the inverse complementary log-log transformation.

        Parameters
        ----------
        x : float or array-like
            Input value(s).

        Returns
        -------
        float or array-like
            The inverse complementary log-log transformed probability value(s): 
            1 - exp(-exp(x)).
        """
        return 1 - jnp.exp(-jnp.exp(x))

    @staticmethod
    @jit
    def reciprocal(x):
        """
        Computes the reciprocal of the input.

        Parameters
        ----------
        x : float or array-like
            Input value(s).

        Returns
        -------
        float or array-like
            The reciprocal of the input value(s): 1 / x. ``reciprocal(0)``
            returns ``inf`` (JAX semantics); it does not raise.
        """
        return 1 / x

    @staticmethod
    @jit
    def sqrt(p):
        """
        Computes the square root.

        Parameters
        ----------
        p : float or array-like
            Input value(s).

        Returns
        -------
        float or array-like
            The square root of the input value(s).
        """
        return jnp.sqrt(p)

    @staticmethod
    @jit
    def square(x):
        """
        Computes the square of the input.

        Parameters
        ----------
        x : float or array-like
            Input value(s).

        Returns
        -------
        float or array-like
            The square of the input value(s): x ** 2.
        """
        return x ** 2

    @staticmethod
    def LKJcor(corr_matrix, eta):
        """
        Computes the LKJ correlation density.

        The LKJ prior is used for modeling correlation matrices. It is controlled by
        a single parameter, `eta`. Higher values of `eta` imply stronger shrinkage
        towards the identity matrix.

        Parameters
        ----------
        corr_matrix : array-like
            A square, symmetric correlation matrix.
        eta : float
            The shape parameter of the LKJ distribution.

        Returns
        -------
        float
            The LKJ density of the input correlation matrix.
        """
        # LKJ(C | eta) proportional to det(C)^(eta - 1), with numpyro's exact
        # normalising constant (the previous closed form had the exponent fixed at 1).
        from numpyro.distributions import LKJ

        k = corr_matrix.shape[0]
        return jnp.exp(LKJ(k, concentration=eta).log_prob(corr_matrix))
