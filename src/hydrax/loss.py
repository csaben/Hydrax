from typing import Any, TypeAlias

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int  # https://github.com/google/jaxtyping

Model: TypeAlias = Any


def loss(
    model: Model, x: Float[Array, "batch 28 28"], y: Int[Array, " batch"]
) -> Float[Array, ""]:
    """sample loss for an mnist task

    :return: cross entropy logits
    :rtype: Float[Array, ""]
    """
    # Our input has the shape (BATCH_SIZE, 1, 28, 28), but our model operations on
    # a single input input image of shape (1, 28, 28).
    #
    # Therefore, we have to use jax.vmap, which in this case maps our model over the
    # leading (batch) axis.
    pred_y = jax.vmap(model)(x)
    return cross_entropy(y, pred_y)


def cross_entropy(
    y: Int[Array, " batch"], pred_y: Float[Array, "batch 2"]
) -> Float[Array, ""]:
    # y are the true targets, and should be integers 0-1.
    # pred_y are the log-softmax'd predictions that we did in our model.
    pred_y = jax.nn.log_softmax(pred_y, axis=-1)  # Adding softmax here
    pred_y = jnp.take_along_axis(pred_y, jnp.expand_dims(y, 1), axis=1)
    return -jnp.mean(pred_y)
