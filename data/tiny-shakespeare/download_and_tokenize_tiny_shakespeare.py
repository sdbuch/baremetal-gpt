import os

import jax.numpy as jnp
import requests
import tiktoken

"""From https://github.com/karpathy/nanoGPT/blob/master/data/shakespeare/prepare.py"""

# download the tiny shakespeare dataset
input_file_path = os.path.join(os.path.dirname(__file__), "input.txt")
if not os.path.exists(input_file_path):
  data_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
  with open(input_file_path, "w", encoding="utf-8") as f:
    f.write(requests.get(data_url).text)

with open(input_file_path, "r", encoding="utf-8") as f:
  data = f.read()
n = len(data)
train_data = data[: int(n * 0.9)]
val_data = data[int(n * 0.9) :]

# encode with tiktoken gpt2 bpe
enc = tiktoken.get_encoding("gpt2")
train_ids = enc.encode_ordinary(train_data)
val_ids = enc.encode_ordinary(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = jnp.array(train_ids, dtype=jnp.int32)
val_ids = jnp.array(val_ids, dtype=jnp.int32)
jnp.save(f"data/tiny-shakespeare/train.npy", train_ids)
jnp.save(f"data/tiny-shakespeare/val.npy", val_ids)

# train.bin has 301,966 tokens
# val.bin has 36,059 tokens
