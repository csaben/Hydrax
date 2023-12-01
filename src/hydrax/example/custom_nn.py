import equinox as eqx
import jax
import jax.numpy as jnp
from jax import device_put, grad, jit, random, value_and_grad, vmap
from jaxtyping import Array, Float, Int, PyTree  # https://github.com/google/jaxtyping

class CustomCifar(eqx.Module):
    flatten: eqx.Module
    fc1: eqx.Module
    relu: eqx.Module
    # softmax: eqx.Module

    """A simple nonlinear model.

    description: provides a simple linear model with a softmax output.
    usage: model = LinearModel(input_shape)
    utility: for cifar10

    """

    def __init__(self, key):
        """flatten your image, apply a linear layer, and softmax"""
        self.flatten = jnp.ravel
        self.fc1 = eqx.nn.Linear(32 * 32 * 3, 10, use_bias=False, key=key)
        self.relu = jax.nn.relu
        # we actually want to put the softmax in the loss function (we only want logits from here)
        # self.softmax = jax.nn.log_softmax  # i changed this to log_softmax

    def __call__(self, x: Float[Array, "32 32"]) -> Float[Array, "10"]:
        x = jnp.ravel(x)
        x = self.fc1(x)
        x = self.relu(x)
        # x = self.softmax(x)
        return x
