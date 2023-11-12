# Hydrax
![Static Badge](https://img.shields.io/badge/python-3.10-blue?logo=python)
![Static Badge](https://img.shields.io/badge/jax-0.4.17-green)

A jax implementation for training models in parallel with different slices of data from a larger dataset.

---

# Installation
1. `git clone git@github.com:csaben/Hydrax.git`
2. `cd Hyrdax`
3. `pip install -e .`

# Examples
```python
from hyrax import get_trained_models
import equinox as eqx

...

# after setting dataset up and defining a eqx model, train models in parallel
models = get_trained_models(verbose=False, model_type="NonLinearModel", start_slice=0)

# save models as a batch of model
eqx.tree_serialise_leaves(next_filename, models)
```

# Dependencies
1. Equinox
2. Jax
3. Flax
