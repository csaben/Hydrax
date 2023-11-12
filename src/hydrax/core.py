from typing import Any, TypeAlias

import jax
import jax.numpy as jnp
import optax

from hydrax.jax_filters import init_models, parallel_init
from hydrax.sampling import multi_iterator, parallel_step, unbatch_model

Model: TypeAlias = Any
loss: TypeAlias = Any
array: jnp.array = Any


def train(
    x_train,
    y_train,
    ensemble,
    optim: optax.GradientTransformation,
    steps: int,
    print_every: int,
    keys,
    lr=1e-3,
    verbose=True,
) -> Model:
    if verbose:
        print(f"Training with {ensemble.__class__.__name__}")
    model_states = ensemble  # just for readability
    opt_states = parallel_init(ensemble, lr)
    dataset_indices = jnp.linspace(0, 783, 784).astype(jnp.int32)

    # Usage in your loop
    data_iterator = multi_iterator((x_train, y_train), dataset_indices)
    for step in range(steps):
        x, y = next(data_iterator)
        model_states, opt_states, train_loss, len_of_data, pre_hash = parallel_step(
            optim, model_states, opt_states, x, y, loss
        )
        if verbose and ((step % print_every) == 0 or (step == steps - 1)):
            for i, loss_val in enumerate(train_loss):
                unbatch_model(model_states, i)
                print(
                    f"Model {i+1} Loss at step {step}: {loss_val} and len of data: {len_of_data}"
                )
    if verbose:
        print(train_loss)
    return model_states


def get_trained_models(
    x_train: array,
    y_train: array,
    model_type: Model,
    verbose=True,
    start_slice: str = "0_784",
    random_key=0,
):
    """sample usage of hydrax with MNIST. Default subslice amount is of step size 1.

    :param x_train: a jnp array of appropriate shape
    :type x_train: a jnp array of appropriate shape
    :param y_train: a jnp array of appropriate shape
    :type y_train: a jnp array of appropriate shape
    :return: ensemble of models trained on each respective slice
    :rtype: _type_
    """
    start_idx, end_idx = map(int, start_slice.split("_"))

    # we will test on the full test set
    lEARNING_RATE = 3e-4
    STEPS = 300
    PRINT_EVERY = 60

    optim = optax.adamw(lEARNING_RATE)  # Initialise the optimiser
    key = jax.random.PRNGKey(random_key)
    keys = jax.random.split(key, 784)  #
    models = init_models(keys, model_type)
    trained_models = train(
        x_train,
        y_train,
        models,
        optim,
        steps=STEPS,
        print_every=PRINT_EVERY,
        keys=keys,
        lr=lEARNING_RATE,
        verbose=True,
    )

    return trained_models
