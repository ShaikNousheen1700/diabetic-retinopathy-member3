import numpy as np
import pytest
from tensorflow import keras

from src import config
from src.model import build_densenet121, unfreeze_top


@pytest.fixture(scope="module")
def model_and_base():
    return build_densenet121(weights=None)       # random weights: no download needed


def test_output_is_probability_per_class(model_and_base):
    model, _ = model_and_base
    x = np.random.default_rng(0).integers(0, 256, size=(2, 224, 224, 3)).astype("float32")
    probs = model.predict(x, verbose=0)
    assert probs.shape == (2, config.NUM_CLASSES)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, rtol=1e-5)


def test_normalisation_matches_keras_preprocess_input(model_and_base):
    """The layers inside the model must do exactly what densenet.preprocess_input does."""
    model, _ = model_and_base
    norm = keras.Sequential([model.get_layer("rescale_0_1"), model.get_layer("imagenet_norm")])
    x = np.random.default_rng(1).integers(0, 256, size=(1, 8, 8, 3)).astype("float32")
    expected = keras.applications.densenet.preprocess_input(x.copy())
    np.testing.assert_allclose(norm(x).numpy(), expected, atol=1e-4)


def test_phase1_base_frozen_phase2_top_block_trainable(model_and_base):
    model, base = model_and_base
    assert not base.trainable
    head_params = sum(np.prod(w.shape) for w in model.trainable_weights)
    assert head_params == 1024 * 5 + 5            # only the Dense(5) layer learns in phase 1

    n = unfreeze_top(base)
    assert n > 0
    names = [l.name for l in base.layers]
    start = names.index(config.FINE_TUNE_FROM)
    assert not any(l.trainable for l in base.layers[:start])
    assert not any(l.trainable for l in base.layers if isinstance(l, keras.layers.BatchNormalization))
