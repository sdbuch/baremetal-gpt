from enum import Enum

import jax
import jax.numpy as jnp
import tiktoken

from bmgpt.config import InferenceConfig, TokenizerType


class Tokenizer:
  def __init__(self):
    pass

  def encode(self, text: str) -> jax.Array:
    pass

  def decode(self, ids: jax.Array) -> str:
    pass


class IdentityTokenizer(Tokenizer):
  def encode(self, text: str):
    assert text.isnumeric()
    return jnp.array([int(c) for c in text])

  def decode(self, ids: jax.Array):
    return "".join(str(n) for n in ids)


class TiktokenTokenizer(Tokenizer):
  def __init__(self, vocab_str: str):
    self.vocab_str = vocab_str
    self.enc = tiktoken.get_encoding(vocab_str)

  def encode(self, text: str):
    return jnp.array(self.enc.encode_ordinary(text))

  def decode(self, ids: jax.Array):
    return self.enc.decode([int(n) for n in ids])


def get_tokenizer_factory(config: InferenceConfig):
  match config.tokenizer:
    case TokenizerType.IDENTITY:
      return IdentityTokenizer()
    case TokenizerType.GPT2:
      return TiktokenTokenizer("gpt2")
