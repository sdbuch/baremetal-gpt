import base64
from enum import Enum
from pathlib import Path

import jax
import jax.numpy as jnp
import tiktoken

from bmgpt.config import InferenceConfig, TokenizerType


class Tokenizer:
  def __init__(self):
    pass

  def encode(self, text: str) -> jax.Array:
    return jnp.array(())

  def decode(self, ids: jax.Array) -> str:
    return ""


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


class TiktokenCustomTokenizer(TiktokenTokenizer):
  def __init__(self, vocab_file: Path):
    # Load the vocabulary and create the tokenizer as close as possible to DCLM
    # Rust preprocessing code (tokshuf.rs)
    # Could make this more general in the future
    mergeable_ranks = {}
    with open(vocab_file, "r") as f:
      for line in f:
        token_b64, rank_str = line.strip().split(" ")
        token_bytes = base64.b64decode(token_b64)
        rank = int(rank_str)
        mergeable_ranks[token_bytes] = rank

    pattern = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"

    tokenizer = tiktoken.Encoding(
      name="llama3-dclm",
      pat_str=pattern,
      mergeable_ranks=mergeable_ranks,
      special_tokens={},
    )
    self.enc = tokenizer
    self.vocab_size = len(mergeable_ranks)


def get_tokenizer_factory(config: InferenceConfig):
  match config.tokenizer:
    case TokenizerType.IDENTITY:
      return IdentityTokenizer()
    case TokenizerType.GPT2:
      return TiktokenTokenizer("gpt2")
    case TokenizerType.LLAMA3:
      # NOTE: (IMPORTANT) This is using the DCLM llama3 tokenizer
      # It has some nonstandard quirks:
      # It uses ID 0 (which is !) as end-of-text (llama3's EoT is 128001)
      # It uses ID 128258 as a PAD token (although vocab_size is 128256)
      return TiktokenCustomTokenizer(Path("tokenizers/meta-llama-3-8B.tiktoken"))
