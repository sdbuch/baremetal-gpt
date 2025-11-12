import os

import jax
import jax.numpy as jnp
from datasets import load_dataset

# Configure your HF_HOME and HF_TOKEN as necessary externally

PATCH_SIZE = 4

# Load and array-ify data
dataset = load_dataset("ylecun/mnist")
train = dataset["train"]
test = dataset["test"]
train_imgs = jnp.stack([jnp.array(im) for im in train["image"]])
train_labels = jnp.array(train["label"])
test_imgs = jnp.stack([jnp.array(im) for im in test["image"]])
test_labels = jnp.array(test["label"])


# Preprocess the data
assert train_imgs.shape[-1] % PATCH_SIZE == 0, "PATCH_SIZE needs to divide image size"


@jax.jit
def preprocess_img(im: jax.Array) -> jax.Array:
    # TODO: if adapting this to multichannel img, update this
    # Convert to bf16
    im = im.astype(jnp.bfloat16)
    # Rescale to [0, 1]
    im /= 255.0
    # Patchify (non-overlapping)
    num_patch_H = im.shape[-2] // PATCH_SIZE
    num_patch_W = im.shape[-1] // PATCH_SIZE
    im = im.reshape(num_patch_H, PATCH_SIZE, num_patch_W, PATCH_SIZE)
    im = jnp.permute_dims(im, (0, 2, 1, 3))
    im = im.reshape(-1, PATCH_SIZE**2)
    return im


train_imgs, test_imgs = jax.tree.map(jax.vmap(preprocess_img), (train_imgs, test_imgs))
mu = train_imgs.mean()
sigma = train_imgs.std()
train_imgs, test_imgs = jax.tree.map(
    lambda im: (im - mu) / sigma, (train_imgs, test_imgs)
)

# Save data
os.makedirs(f"data/mnist/patch_{PATCH_SIZE}x{PATCH_SIZE}", exist_ok=True)
jnp.savez(
    f"data/mnist/patch_{PATCH_SIZE}x{PATCH_SIZE}/mnist_train.npz",
    images=train_imgs,
    labels=train_labels,
)
jnp.savez(
    f"data/mnist/patch_{PATCH_SIZE}x{PATCH_SIZE}/mnist_test.npz",
    images=test_imgs,
    labels=test_labels,
)
