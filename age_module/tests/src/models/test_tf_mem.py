import pytest
import tensorflow as tf
from age_module.src.models.tf_mem import limit_gpu_gb


@pytest.fixture(scope="session")
def test_limit_gpu_gb():
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        limit_gpu_gb(1.0)  # Set GPU limit to 1 GB for testing purposes
        config = tf.config.experimental.get_virtual_device_configuration(gpu)
        assert config[0].memory_limit == 1024, "GPU memory limit not set to 1 GB"

    yield
