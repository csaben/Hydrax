import equinox as eqx

# custom eqx model
from custom_nn import CustomCifar as NonLinearModel
from datasets import load_dataset

from hydrax import get_trained_models

# cross entropy loss
from hydrax.loss import loss

# Load the MNIST dataset
dataset = load_dataset("cifar10")
dataset = dataset.with_format("jax")
dataset = dataset.shuffle(seed=42)
# print(dataset["train"].shape)
# (50000, 2)
# print(dataset["test"].shape)
# (10000, 2)

train_dataset = dataset["train"]
test_dataset = dataset["test"]
# print(x_train.shape, y_train.shape)
# (1024, 32, 32, 3) (1024,)

# depends on pixel size (32,32)
start_slice = "0_1024"  #
start_idx, end_idx = map(int, start_slice.split("_"))

x_train, y_train = train_dataset["img"], train_dataset["label"]
x_train, y_train = x_train[start_idx:end_idx], y_train[start_idx:end_idx]


models = get_trained_models(
    x_train,
    y_train,
    num_models=1024,
    model_type=NonLinearModel,
    verbose=True,
    start_slice=start_slice,
    loss=loss,
)
