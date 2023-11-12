from typing import Any, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp

from hydrax.jax_filters import init_models

Model: TypeAlias = Any


# get original model 784 shape
def get_original_model(size: int, model_type: Model):
    """Get the pytree associated with your model type"""
    key = jax.random.PRNGKey(0)
    keys = jax.random.split(key, size)  #
    models = init_models(keys, model_type)
    return models, size


def unbatch_model(ensemble: Model, idx):
    key = jax.random.PRNGKey(0)

    weights = ensemble.fc1.weight[idx, :, :]

    def where(m):
        return m.fc1.weight

    # https://docs.kidger.site/equinox/api/manipulation/#:~:text=equinox.tree_at(where%3A,%C2%A4
    model = type(ensemble)
    tmp_model = model(key)
    model = eqx.tree_at(where, tmp_model, weights)
    return model
