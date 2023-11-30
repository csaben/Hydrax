from typing import Union

import equinox as eqx
import jax
import jax.numpy as jnp
import optax  # for some reason pip install works but not conda
from jaxtyping import Array, Float, Int
from nn.linear import LinearModel
from nn.loss import loss
from nn.nonlinear import NonLinearModel

# from utils._dataloader import DataLoader


def evaluate(model: Union[LinearModel, NonLinearModel], testloader) -> Float[Array, ""]:
    """This function evaluates the model on the test dataset,
    computing both the average loss and the average accuracy.
    """
    avg_loss = 0
    avg_acc = 0
    for x, y in testloader:
        # Note that all the JAX operations happen inside `loss` and `compute_accuracy`,
        # and both have JIT wrappers, so this is fast.
        avg_loss += loss(model, x, y)
        avg_acc += compute_accuracy(model, x, y)

    return avg_loss / len(testloader), avg_acc / len(testloader)


@eqx.filter_jit
def compute_accuracy(
    model: Union[LinearModel, NonLinearModel],
    x: Float[Array, "batch 1 28 28"],
    y: Int[Array, " batch"],
) -> Float[Array, ""]:
    """This function takes as input the current model
    and computes the average accuracy on a batch.
    """
    # vmap here is used to vectorise the model over the batch axis.
    pred_y = jax.vmap(model)(x)
    pred_labels = jnp.argmax(pred_y, axis=1)
    return jnp.mean(pred_labels == y)


@eqx.filter_vmap(in_axes=(0, None))
def init_models(key, model_type: Union[LinearModel, NonLinearModel]):
    return model_type(key)


@eqx.filter_vmap
def parallel_init(model, lr):
    optim = optax.adamw(lr)  # Initialise the optimiser
    return optim.init(eqx.filter(model, eqx.is_array))
