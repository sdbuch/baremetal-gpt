import os

import jax
import jax.numpy as jnp
from datasets import load_dataset

# Configure your HF_HOME and HF_TOKEN as necessary externally

# Load and array-ify data
dataset = load_dataset("ylecun/mnist")
train = dataset["train"]
test = dataset["test"]
train_imgs = jnp.stack([jnp.array(im) for im in train["image"]])
train_labels = jnp.array(train["label"])
test_imgs = jnp.stack([jnp.array(im) for im in test["image"]])
test_labels = jnp.array(test["label"])


# Preprocess the data
def preprocess_img(im: jax.Array) -> jax.Array:
    # Convert to bf16
    im = im.astype(jnp.bfloat16)
    # Rescale to [0, 1]
    im /= 255.0
    return im


train_imgs, test_imgs = jax.tree.map(jax.vmap(preprocess_img), (train_imgs, test_imgs))
mu = train_imgs.mean()
sigma = train_imgs.std()
train_imgs, test_imgs = jax.tree.map(
    lambda im: (im - mu) / sigma, (train_imgs, test_imgs)
)

# Save data
os.makedirs("data/mnist", exist_ok=True)
jnp.savez("data/mnist/mnist_train.npz", images=train_imgs, labels=train_labels)
jnp.savez("data/mnist/mnist_test.npz", images=test_imgs, labels=test_labels)
