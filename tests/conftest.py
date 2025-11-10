import pytest

from bmgpt.config import Config, config_post_init


@pytest.fixture(scope="session", autouse=True)
def setup_config():
    c = Config(mesh_shape=[1])
    config_post_init(c)
    yield
