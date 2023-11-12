from functools import partial
from typing import List, Union

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax  # for some reason pip install works but not conda
from nn.linear import LinearModel
from nn.loss import loss
from nn.nn_utils import init_models, parallel_init
from nn.nonlinear import NonLinearModel


@eqx.filter_vmap(in_axes=(None, eqx.if_array(0), 0, 0, 0))
def parallel_step(optim, model, opt_state, x, y):
    loss_value, grads = eqx.filter_value_and_grad(loss)(model, x, y)
    updates, opt_state = optim.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)

    len_of_data = x.shape[0]  # sanity check, first few should be 1,2...32, 32, 32

    return model, opt_state, loss_value, len_of_data, x


def unbatch_model(ensemble: Union[LinearModel, NonLinearModel], idx):
    key = jax.random.PRNGKey(0)

    weights = ensemble.fc1.weight[idx, :, :]
    # create a linear model to fill
    # TODO: if it breaks bc not lambda change back
    # fmt: off
    def where(m):
        return m.fc1.weight
    # fmt: on
    # https://docs.kidger.site/equinox/api/manipulation/#:~:text=equinox.tree_at(where%3A,%C2%A4
    model = type(ensemble)
    tmp_model = model(key)
    model = eqx.tree_at(where, tmp_model, weights)


def get_first_seed(dataset_index):
    return jr.split(jr.PRNGKey(dataset_index))[0, 0]


@jax.jit
def get_example(data_x, data_y, dataset_index, i):
    first_seed = get_first_seed(dataset_index)

    def true_fn(idx, seed, dataset_index):
        # well hold on, if i is variable then there should be a way
        # to finesse
        return idx

    def false_fn(idx, seed, dataset_index):
        point_seed = (
            first_seed + idx
        )  # For determinism within the slice + no lock in to first 32
        point_index = jr.randint(
            jr.PRNGKey(point_seed), shape=(), minval=0, maxval=dataset_index
        )
        return point_index

    idx = jax.lax.cond(
        # if batch_idx > current_ds_size => sample from anywhere in slice
        i <= dataset_index,
        # True,
        # return idx from anywhere in slice (behaves as expected)
        true_fn,
        # else forcibly sample the ds[batch_idx]
        false_fn,
        # use operands: curr_batch_idx, seed, dataset_index
        i,
        first_seed,
        dataset_index,
    )

    x_i = jax.lax.dynamic_index_in_dim(data_x, idx, keepdims=False)
    y_i = jax.lax.dynamic_index_in_dim(data_y, idx, keepdims=False)
    return x_i, y_i


def multi_iterator(dataset, dataset_indices):
    batch_size = 32  # 784  # 32
    dataset_indices = jnp.array(dataset_indices)
    # another divergence
    data_x, data_y = dataset  # (points, labels) => (X_train, Y_train)
    # discrepancy: shapes of my X_train is 2d, not 1d
    len(data_x)

    get_example_from_dataset = partial(get_example, data_x, data_y)

    # sample a batch of data from one dataset
    get_batch = jax.vmap(get_example_from_dataset, in_axes=(None, 0))

    # sample a batch of data from _each_ dataset
    get_multibatch = jax.vmap(get_batch, in_axes=(0, None))

    def iterate_multibatch():
        """construct an iterator which runs forever, at each step returning a batch of batches"""
        i = 0
        while True:
            indices = jnp.arange(i, i + batch_size, dtype=jnp.int32)
            yield get_multibatch(dataset_indices, indices)
            i += batch_size

    loader_iter = iterate_multibatch()
    return loader_iter


def train(
    X_train_normalized,
    Y_train,
    ensemble,
    optim: optax.GradientTransformation,
    steps: int,
    print_every: int,
    keys,
    lr=1e-3,
    verbose=True,
) -> List[Union[LinearModel, NonLinearModel]]:
    print(f"Training with {ensemble.__class__.__name__}")
    model_states = ensemble  # just for readability
    opt_states = parallel_init(ensemble, lr)
    dataset_indices = jnp.linspace(0, 783, 784).astype(jnp.int32)

    # Usage in your loop
    # data_iterator = multi_iterator((pre_sliced_x, pre_sliced_y), dataset_indices)
    data_iterator = multi_iterator((X_train_normalized, Y_train), dataset_indices)
    for step in range(steps):
        x, y = next(data_iterator)
        model_states, opt_states, train_loss, len_of_data, pre_hash = parallel_step(
            optim,
            model_states,
            opt_states,
            x,
            y,
        )
        if verbose and ((step % print_every) == 0 or (step == steps - 1)):
            for i, loss_val in enumerate(train_loss):
                unbatch_model(model_states, i)
                # # TODO: make this compatible
                # # evaluate it
                # # acc = compute_accuracy(model, x, y)

                print(
                    # f"Model {i+1} Loss at step {step}: {loss_val} and accuracy: {acc} and len of data: {len_of_data}"
                    f"Model {i+1} Loss at step {step}: {loss_val} and len of data: {len_of_data}"
                )
    print(train_loss)
    return model_states


def get_trained_models(
    verbose=True, model_type=NonLinearModel, start_slice="0_784", random_key=0
):
    # TODO: clean up the lines between here and call to train()
    start_idx, end_idx = map(int, start_slice.split("_"))

    import os

    from config.locations import mnist_data

    cwd = os.getcwd()
    os.chdir(mnist_data)
    X_train_normalized = np.load("preprocessed/X_train_normalized.npy")[
        start_idx:end_idx
    ]
    # X_test_normalized = np.load("MNIST/MNIST/preprocessed/X_test_normalized.npy")
    Y_train = np.load("preprocessed/Y_train.npy")[start_idx:end_idx]
    os.chdir(cwd)
    # # we will test on the full test set
    lEARNING_RATE = 3e-4
    STEPS = 300
    PRINT_EVERY = 60

    # test it
    optim = optax.adamw(lEARNING_RATE)  # Initialise the optimiser
    key = jax.random.PRNGKey(random_key)
    keys = jax.random.split(key, 784)  #
    models = init_models(keys, model_type)
    import time

    timer = time.time()

    trained_models = train(
        X_train_normalized,
        Y_train,
        models,
        optim,
        steps=STEPS,
        print_every=PRINT_EVERY,
        keys=keys,
        lr=lEARNING_RATE,
        verbose=True,
    )
    print(f"Training took {time.time() - timer} seconds")

    return trained_models
