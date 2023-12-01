# Hydrax

 ![Static Badge](https://img.shields.io/badge/python-3.10-blue?logo=python)
 ![Static Badge](https://img.shields.io/badge/jax-0.4.17-green)

A jax implementation for training models in parallel with different slices of data from a larger dataset. 

Like a multiheaded mythological hydra, Hydrax enables each model head to select its own subset of data to learn from a base dataset. More specifically, Hydrax allows for variable sized slices of data to be used inside of each model with them all trained in parallel! This works great for researcher trying to get the most out of a single GPU, but generalizes to larger setups.

The sampling method used in Hydrax makes it such that each model doesn't get stuck continuously sampling the same slice of data if the model number is larger than the slice size.

# Installation




1. `git clone git@github.com:csaben/Hydrax.git`
2. `cd Hydrax`
3. `pip install -e .`

# Examples
Let's imagine you have 784 datapoints. To then train 784 models from model 1 to model 784, each getting it's model number worth of datapoints (sliced from 0:model_number), we would do as follows:

```python
from hydrax import get_trained_models
import equinox as eqx

...
import equinox as eqx
from hydrax import get_trained_models

# cross entropy loss
from hydrax.loss import loss

# basic eqx model
from hydrax.nn.nonlinear import NonLinearModel

start_slice = "0_784"
start_idx, end_idx = map(int, start_slice.split("_"))
x_train, y_train = load_data() # your dataloading fn for your dataset
x_train, y_train = x_train[start_idx: end_idx], y_train[start_idx: end_idx]

models = get_trained_models(
    x_train,
    y_train,
    num_models=784,
    model_type=NonLinearModel,
    verbose=True,
    start_slice="0_784",
    loss=loss,
)

# save models as a batch of model
eqx.tree_serialise_leaves(next_filename, models)
```
The above example is with an MNIST like dataset in mind.

# Dependencies




1. Equinox
2. Jax
3. Flax

# Contributing
Hydrax currently only supports tasks specifically oriented towards classification (binary crossentropy is built-in). I intend to incorporate NLP support soon, but pull requests are welcome!
