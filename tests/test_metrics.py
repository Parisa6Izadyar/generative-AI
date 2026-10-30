import warnings

import numpy as np
import pytest

from gan_vae.metrics import frechet_feature_distance, prediction_summary


def test_distance_identity_and_known_translation():
    real = np.random.default_rng(4).normal(size=(80, 8))
    assert frechet_feature_distance(real, real) == pytest.approx(0, abs=1e-7)
    assert frechet_feature_distance(real, real + 2) == pytest.approx(32, abs=1e-7)


def test_distance_known_one_dimensional_variance():
    real = np.array([[-1.0], [1.0]])
    assert frechet_feature_distance(real, real * 3) == pytest.approx(8)


def test_distance_symmetric_for_singular_covariances():
    rng = np.random.default_rng(8)
    real = np.column_stack([rng.normal(size=(12, 4)), np.zeros((12, 12))])
    fake = np.column_stack([rng.normal(size=(12, 4)) + 1, np.zeros((12, 12))])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        forward = frechet_feature_distance(real, fake)
        reverse = frechet_feature_distance(fake, real)
    assert forward >= 0
    assert forward == pytest.approx(reverse, abs=1e-7)


@pytest.mark.parametrize("real,fake", [
    (np.ones((1, 3)), np.ones((3, 3))),
    (np.ones((4, 2)), np.ones((4, 3))),
    (np.full((4, 3), np.nan), np.ones((4, 3))),
    (np.ones(3), np.ones(3)),
])
def test_invalid_features_are_rejected(real, fake):
    with pytest.raises(ValueError, match="Features"):
        frechet_feature_distance(real, fake)


def test_entropy_detects_class_collapse_and_balanced_predictions():
    collapsed = prediction_summary(np.tile(np.eye(10)[3], (20, 1)))
    balanced = prediction_summary(np.eye(10).repeat(2, axis=0))
    assert collapsed["predicted_class_entropy_nats"] == 0
    assert collapsed["predicted_classes_present"] == 1
    assert balanced["normalized_class_entropy"] == pytest.approx(1)
    assert balanced["predicted_class_counts"] == [2] * 10


@pytest.mark.parametrize("probabilities", [
    np.empty((0, 10)), np.ones((4, 10)), np.full((4, 10), np.nan),
    np.ones((4, 2)) / 2, np.tile([-0.1, 1.1] + [0] * 8, (4, 1)),
])
def test_invalid_probabilities_are_rejected(probabilities):
    with pytest.raises(ValueError, match="probabilities"):
        prediction_summary(probabilities)
