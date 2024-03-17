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
    loss: loss,
    steps: int,
    keys,
    print_every: int = 60,
    lr=1e-3,
    indices=(0, 783),
    verbose=True,
) -> Model:
    if verbose:
        print(f"Training with {ensemble.__class__.__name__}")
    model_states = ensemble  # just for readability
    start_idx, end_idx = indices
    # SMOL
    lr = 1e-7
    opt_states = parallel_init(ensemble, lr)
    # (start_idx, end_idx - start_idx - 1, end_idx)
    # i.e. mnist, (0, 783, 784)
    # dataset_indices = jnp.linspace(0, 783, 784).astype(jnp.int32)
    print(start_idx, end_idx - start_idx - 1, end_idx)
    dataset_indices = jnp.linspace(start_idx, end_idx - start_idx - 1, end_idx).astype(
        jnp.int32
    )

    # Usage in your loop
    data_iterator = multi_iterator((x_train, y_train), dataset_indices)
    for step in range(steps):
        x, y = next(data_iterator)
        # nan_mask = jnp.isnan(x)
        # nan_indices = jnp.where(nan_mask)
        # print("NaN locations BEFORE parallel step:", nan_indices)
        # X NOT NAN HERE
        # with jax.debug_nans():
        model_states, opt_states, train_loss, len_of_data, pre_hash = parallel_step(
            optim, model_states, opt_states, x, y, loss
        )

        # nan_mask = jnp.isnan(train_loss)
        # nan_indices = jnp.where(nan_mask)
        # print("NaN locations after parallel step:", nan_indices)
        # LOSS NAN HERE AT 4th MODEL
        if verbose and ((step % print_every) == 0 or (step == steps - 1)):
            for i, loss_val in enumerate(train_loss):
                # ith_model = unbatch_model(model_states, i)
                # ith_perf = ith_model.evaluate(xtest, ytest)
                # ith_perf = ith_model.evaluate(x_train, y_train)
                # print(xtest.shape)
                # print(xtest.T.shape)
                # print(ith_model)
                # ith_output = ith_model(xtest)
                # ith_perf = jnp.mean(jnp.equal(jnp.argmax(ith_output, axis=-1), ytest))
                # if jnp.isnan(loss_val):
                #     print(loss_val)
                #     print("loss value nan at model", i + 1, "at step", step)
                #     import sys

                #     sys.exit()
                # print(
                #     f"Model {i+1} Loss at step {step}: {loss_val:.2f} and len of data: {len_of_data} and perf: {ith_perf:.2f}%"
                # )
                # FIXME: one day it would be nice to see performance here, you could get it from the parallel step if you
                # implement it in there
                print(
                    f"Model {i+1} Loss at step {step}: {loss_val:.2f} and len of data: {len_of_data}"
                )
    # if verbose:
    #     print(train_loss)
    return model_states


def get_trained_models(
    x_train: array,
    y_train: array,
    model_type: Model,
    verbose=True,
    start_slice: str = "0_784",
    random_key=0,
    num_models=784,
    loss=loss,
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
    LEARNING_RATE = 3e-4
    STEPS = 300
    PRINT_EVERY = 60

    optim = optax.adamw(LEARNING_RATE)  # Initialise the optimiser
    key = jax.random.PRNGKey(random_key)
    keys = jax.random.split(key, num_models)  #
    models = init_models(keys, model_type)
    trained_models = train(
        x_train,
        y_train,
        models,
        optim,
        loss,
        steps=STEPS,
        print_every=PRINT_EVERY,
        keys=keys,
        lr=LEARNING_RATE,
        indices=(start_idx, end_idx),
        verbose=True,
    )

    return trained_models
