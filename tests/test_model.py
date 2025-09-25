import jax
import jax.numpy as jnp

from bmgpt.config import Config, DType, config_post_init
from bmgpt.model import (
    Transformer,
    _apply_rope,
    _attn,
    _precompute_rope_sincos,
    _transformer,
    init_model_params,
)


def test_manual_attn_matches_jax_no_kv():
    config_args = {
        "mesh_shape": [1],
        "sharding_data": None,
        "use_rope": False,
        "update_cache": False,
        # "compute_dtype": DType.BFLOAT16,
    }
    config = Config(**config_args)
    config_no_fa = Config(**(config_args | {"use_fa": False}))
    config_post_init(config)
    key = jax.random.key(config.seed)
    model = init_model_params(key, config)
    input = jnp.arange(8)
    out, _ = _transformer(config, model, input, None, 0)
    out_no_fa, _ = _transformer(config_no_fa, model, input, None, 0)
    print(out)
    print(out_no_fa)
    assert jnp.allclose(out, out_no_fa)


def test_rope():
    pass
