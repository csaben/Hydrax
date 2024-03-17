from typing import Any, TypeAlias

import jax
import jax.numpy as jnp
from jax import jit
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
    """requires one hot to be done at eval time"""
    num_classes = pred_y.shape[
        -1
    ]  # Assuming the number of classes is the size of the last dimension of pred_y
    y_one_hot = jax.nn.one_hot(y, num_classes)
    # [1,7,1] => [[1,0], [0,1], [1,0]]

    # Apply log softmax to predicted values
    pred_y = jax.nn.log_softmax(pred_y, axis=-1)
    # logic:
    # sample pred y from network: [10,-10]
    # log(softmax(z)) = z_i - ln(sum(z_k))
    # effectively gives [0, -20]

    # Compute the cross-entropy
    loss = -jnp.sum(y_one_hot * pred_y, axis=1)

    # [0,-20] * [1,0] = 0
    # => no loss to be added
    return jnp.mean(loss)
