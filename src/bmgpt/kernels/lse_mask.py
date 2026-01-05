# Copyright 2023 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""VocabMask for fused logsumexp kernel."""

import numpy as np
from jax.experimental.pallas.ops.tpu.splash_attention.splash_attention_mask import (
  _ComputableMask,
)


class VocabMask(_ComputableMask):
  """Mask that allows attention only to valid vocabulary tokens (with fused xent)"""

  max_valid_id: int

  def __init__(
    self,
    shape: tuple[int, int],
    max_valid_id: int,
    shard_count: int = 1,
  ):
    self.max_valid_id = max_valid_id

    def vocab_mask_function(q_ids, kv_ids):
      return kv_ids <= max_valid_id

    super().__init__(
      shape=shape,
      mask_function=vocab_mask_function,
      shard_count=shard_count,
    )

  def __eq__(self, other: object):
    if not isinstance(other, type(self)):
      return NotImplemented
    return (
      self.shape == other.shape
      and self.max_valid_id == other.max_valid_id
      and np.array_equal(self.q_sequence, other.q_sequence)
    )

  def __hash__(self):
    return hash(
      (
        type(self),
        self.shape,
        self.max_valid_id,
        self.q_sequence.tobytes() if self.q_sequence is not None else None,
      )
    )
