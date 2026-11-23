import jax
import jax.numpy as jnp
from jax.scipy.special import erf

class activation:
    """
    A collection of common neural network activation functions implemented in JAX.

    This class serves as a namespace for activation functions, which can be
    called directly as static methods (e.g., `activation.relu(x)`).
    """

    # --- Original Functions (with improved docstrings) ---

    @staticmethod
    def tanh(x: jnp.ndarray) -> jnp.ndarray:
        """
        Computes the hyperbolic tangent activation function.

        This function squashes its input values into the range [-1, 1]. It is
        zero-centered, which can help with optimization by making the mean of
        the activations closer to zero. However, it can suffer from the
        vanishing gradient problem for very large or very small inputs.

        Parameters:
        - x (jnp.ndarray): The input array.

        Returns:
        - jnp.ndarray: The array with tanh applied element-wise.
        """
        return jnp.tanh(x)

    @staticmethod
    def relu(x: jnp.ndarray) -> jnp.ndarray:
        """
        Computes the Rectified Linear Unit (ReLU) activation function.

        ReLU is one of the most widely used activation functions. It computes
        f(x) = max(0, x), effectively "turning off" neurons with negative
        outputs. This makes the network sparse and computationally efficient.
        Its main drawback is the "dying ReLU" problem, where neurons can get
        stuck in a state where they always output zero if their weights are
        updated such that their input is always negative.

        Parameters:
        - x (jnp.ndarray): The input array.

        Returns:
        - jnp.ndarray: The array with ReLU applied element-wise.
        """
        return jnp.maximum(0, x)

    @staticmethod
    def sigmoid(x: jnp.ndarray) -> jnp.ndarray:
        """
        Computes the sigmoid (or logistic) activation function.

        This function squashes its input values into the range [0, 1]. It is
        commonly used in the output layer of binary classification models to
        represent a probability. It is less favored for hidden layers due to
        its non-zero-centered output and strong saturation, which leads to
        vanishing gradients.

        Parameters:
        - x (jnp.ndarray): The input array.

        Returns:
        - jnp.ndarray: The array with sigmoid applied element-wise.
        """
        return jax.nn.sigmoid(x)

    @staticmethod
    def softmax(x: jnp.ndarray, axis: int = -1) -> jnp.ndarray:
        """
        Computes the softmax activation function.

        Softmax transforms a vector of real numbers into a probability
        distribution. Each output value is in the range [0, 1], and all
        output values sum to 1. It is almost exclusively used as the output
        activation function for multi-class classification problems.

        Parameters:
        - x (jnp.ndarray): The input array.
        - axis (int): The axis along which the softmax should be computed.

        Returns:
        - jnp.ndarray: The array with softmax applied along the specified axis.
        """
        return jax.nn.softmax(x, axis=axis)

    # --- New and Important Activation Functions ---

    @staticmethod
    def leaky_relu(x: jnp.ndarray, negative_slope: float = 0.01) -> jnp.ndarray:
        """
        Computes the Leaky Rectified Linear Unit (Leaky ReLU).

        Leaky ReLU is a variant of ReLU designed to solve the "dying ReLU"
        problem. Instead of being zero for negative inputs, it allows a small,
        non-zero gradient (controlled by `negative_slope`). This ensures that
        neurons do not become completely inactive.

        f(x) = x if x > 0, else negative_slope * x.

        Usage:
        A common drop-in replacement for ReLU, especially if you suspect
        dying neurons are an issue.

        Parameters:
        - x (jnp.ndarray): The input array.
        - negative_slope (float): The small slope for negative inputs. Default is 0.01.

        Returns:
        - jnp.ndarray: The array with Leaky ReLU applied element-wise.
        """
        return jnp.where(x >= 0, x, negative_slope * x)

    @staticmethod
    def elu(x: jnp.ndarray, alpha: float = 1.0) -> jnp.ndarray:
        """
        Computes the Exponential Linear Unit (ELU).

        ELU is another alternative to ReLU that also aims to solve the dying
        neuron problem and can lead to faster learning. For negative inputs,
        it becomes a smooth, saturating function that pushes the mean
        activation closer to zero, which can speed up convergence.

        f(x) = x if x > 0, else alpha * (exp(x) - 1).

        Usage:
        Often provides better performance than ReLU or Leaky ReLU but is
        slightly more computationally expensive due to the exponential function.

        Parameters:
        - x (jnp.ndarray): The input array.
        - alpha (float): The saturation parameter for negative inputs. Default is 1.0.

        Returns:
        - jnp.ndarray: The array with ELU applied element-wise.
        """
        # expm1 is accurate for small |x|; the clamp keeps the exponential branch
        # out of the overflow-prone region for large positive x.
        return jnp.where(x >= 0, x, alpha * jnp.expm1(jnp.minimum(x, 0)))

    @staticmethod
    def gelu(x: jnp.ndarray) -> jnp.ndarray:
        """
        Computes the Gaussian Error Linear Unit (GELU).

        GELU is a smooth, high-performing activation function that is the
        standard in modern Transformer models like BERT and GPT. It weights
        its input by its value, but this weighting is stochastic and depends
        on the standard Gaussian cumulative distribution function (CDF).
        Intuitively, it's more likely to "drop" (zero-out) inputs that are
        closer to zero.

        f(x) = x * Φ(x), where Φ(x) is the standard normal CDF.

        Usage:
        The state-of-the-art choice for Transformer-based architectures. A
        strong general-purpose choice for many deep networks.

        Parameters:
        - x (jnp.ndarray): The input array.

        Returns:
        - jnp.ndarray: The array with GELU applied element-wise.
        """
        return x * 0.5 * (1.0 + erf(x / jnp.sqrt(2.0)))

    @staticmethod
    def silu(x: jnp.ndarray) -> jnp.ndarray:
        """
        Computes the Sigmoid Linear Unit (SiLU), also known as Swish.

        SiLU is a self-gated activation function, defined as f(x) = x * sigmoid(x).
        The sigmoid part acts as a soft "gate" that modulates the input. This
        function is smooth, non-monotonic, and has been shown to perform as
        well or better than ReLU on many challenging tasks without adding any
        extra parameters.

        Usage:
        An excellent, modern, general-purpose replacement for ReLU that often
        improves performance.

        Parameters:
        - x (jnp.ndarray): The input array.

        Returns:
        - jnp.ndarray: The array with SiLU applied element-wise.
        """
        return x * jax.nn.sigmoid(x)

    @staticmethod
    def softplus(x: jnp.ndarray) -> jnp.ndarray:
        """
        Computes the Softplus activation function.

        Softplus is a smooth approximation of the ReLU function, defined as
        f(x) = log(1 + exp(x)). Its output is always strictly positive.

        Usage:
        While less common as a hidden layer activation, it is very useful in
        the output layer of a model when a strictly positive output is
        required, for example, when predicting the variance (scale) parameter
        of a distribution.

        Parameters:
        - x (jnp.ndarray): The input array.

        Returns:
        - jnp.ndarray: The array with Softplus applied element-wise.

        Note: computed as logaddexp(x, 0), the numerically stable form of
        log(1 + exp(x)) (MAJ-1); the naive form overflowed to inf for
        x > 88 and the inf propagated through the autograd chain.
        """
        return jnp.logaddexp(x, 0.0)

    @staticmethod
    def linear(x: jnp.ndarray) -> jnp.ndarray:
        """
        Computes the linear (or identity) activation function.

        This function simply returns the input without any modification, i.e.,
        f(x) = x.

        Usage:
        This is a critical component. It is the default choice for the output
        layer of any regression model, where the output is an unbounded,
        continuous value.

        Parameters:
        - x (jnp.ndarray): The input array.

        Returns:
        - jnp.ndarray: The identical input array.
        """
        return x

    @staticmethod
    def exp(x: jnp.ndarray) -> jnp.ndarray:
        """
        Computes the exponential activation function.

        f(x) = exp(x). Output is always positive.

        Parameters:
        - x (jnp.ndarray): The input array.

        Returns:
        - jnp.ndarray: The array with exponential applied element-wise.

        Note: the input is clipped at 80 before exponentiating (MAJ-2);
        exp(x) overflows to inf for x > 88.7 in float32 and inf propagates
        through the autograd chain.
        """
        return jnp.exp(jnp.minimum(x, 80.0))