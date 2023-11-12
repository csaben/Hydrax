# Hydrax

 ![Static Badge](https://img.shields.io/badge/python-3.10-blue?logo=python)
 ![Static Badge](https://img.shields.io/badge/jax-0.4.17-green)

A jax implementation for training models in parallel with different slices of data from a larger dataset.


# Installation




1. `git clone git@github.com:csaben/Hydrax.git`
2. `cd Hyrdax`
3. `pip install -e .`

# Examples

```python
from hyrax import get_trained_models
import equinox as eqx

...

x_train, y_train = get_data() # your own fn for dataloading

# after setting dataset up and defining a eqx model, train models in parallel
models = get_trained_models(x_train, y_train, model_type="NonLinearModel", verbose=False start_slice="0_784")

# save models as a batch of model
eqx.tree_serialise_leaves(next_filename, models)
```

# Dependencies




1. Equinox
2. Jax
3. Flax


