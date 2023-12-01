from functools import partial
from typing import Any, List, TypeAlias, Union

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

Model: TypeAlias = Any
loss: TypeAlias = Any


@eqx.filter_vmap(in_axes=(None, eqx.if_array(0), 0, 0, 0, None))
def parallel_step(optim, model, opt_state, x, y, loss):
    loss_value, grads = eqx.filter_value_and_grad(loss)(model, x, y)
    updates, opt_state = optim.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)

    len_of_data = x.shape[0]  # sanity check, first few should be 1,2...32, 32, 32

    return model, opt_state, loss_value, len_of_data, x


def unbatch_model(ensemble: Model, idx):
    key = jax.random.PRNGKey(0)

    weights = ensemble.fc1.weight[idx, :, :]

    # create a linear model to fill
    def where(m):
        return m.fc1.weight

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
