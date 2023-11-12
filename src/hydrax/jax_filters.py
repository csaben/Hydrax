from typing import Any, TypeAlias

import equinox as eqx
import optax

Model: TypeAlias = Any


@eqx.filter_vmap(in_axes=(0, None))
def init_models(key, model_type: Model) -> Model:
    return model_type(key)


@eqx.filter_vmap
def parallel_init(model, lr):
    optim = optax.adamw(lr)  # Initialise the optimiser
    return optim.init(eqx.filter(model, eqx.is_array))
