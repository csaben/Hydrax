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


# next is experimental cross entropy with a mapping fn
@jit
def remap_labels(y):
    """note: if you use this you need the mapping at eval time"""
    # Find the unique labels and sort them
    unique_labels = jnp.unique(y)
    sorted_unique_labels = jnp.sort(unique_labels)

    # Create a mapping from unique labels to a range from 0 to len(unique_labels)-1
    mapping = {label: index for index, label in enumerate(sorted_unique_labels)}

    # Remap the labels
    remapped_labels = jnp.vectorize(mapping.get)(y)
    return remapped_labels


def categorical_cross_entropy(y: Array, pred_y: Array) -> Array:
    # Remap labels
    y_remap = remap_labels(y)

    # Compute loss with remapped labels
    log_pred_y = jax.nn.log_softmax(pred_y, axis=-1)
    true_log_probs = jnp.take_along_axis(
        log_pred_y, jnp.expand_dims(y_remap, axis=-1), axis=-1
    )
    loss = -jnp.mean(true_log_probs)
    return loss
