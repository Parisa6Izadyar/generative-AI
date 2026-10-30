"""Diagnostics in the frozen MNIST classifier's feature space."""

import numpy as np
import torch


def prediction_summary(probabilities):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if (probabilities.ndim != 2 or probabilities.shape[0] == 0
            or probabilities.shape[1] != 10 or not np.isfinite(probabilities).all()
            or np.any(probabilities < 0) or np.any(probabilities > 1)
            or not np.allclose(probabilities.sum(1), 1, atol=1e-6)):
        raise ValueError("Expected a nonempty finite array of ten-class probabilities.")
    counts = np.bincount(probabilities.argmax(1), minlength=10)
    proportions = counts / counts.sum()
    positive = proportions > 0
    entropy = float(-(proportions[positive] * np.log(proportions[positive])).sum())
    return {
        "mean_max_probability": float(probabilities.max(1).mean()),
        "predicted_class_counts": counts.tolist(),
        "predicted_class_entropy_nats": entropy,
        "normalized_class_entropy": entropy / float(np.log(10)),
        "predicted_classes_present": int(positive.sum()),
    }


def _psd_sqrt(matrix):
    matrix = (matrix + matrix.T) / 2
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    tolerance = 1e-10 * max(1.0, float(np.max(np.abs(eigenvalues))))
    if eigenvalues.min() < -tolerance:
        raise ValueError("Covariance matrix is not positive semidefinite.")
    # Rank-deficient empirical covariances are valid; clip round-off below zero.
    return (eigenvectors * np.sqrt(np.maximum(eigenvalues, 0))) @ eigenvectors.T


def frechet_feature_distance(real, generated):
    """Squared Gaussian 2-Wasserstein distance in this project's CNN features.

    This uses a symmetric PSD square root rather than discarding imaginary
    components of a square root of a nonsymmetric covariance product.
    """
    real, generated = (np.asarray(x, dtype=np.float64) for x in (real, generated))
    if (real.ndim != 2 or generated.ndim != 2 or real.shape[1] == 0
            or real.shape[1] != generated.shape[1]
            or min(real.shape[0], generated.shape[0]) < 2
            or not np.isfinite(real).all() or not np.isfinite(generated).all()):
        raise ValueError("Features need at least two finite rows and matching dimensions.")
    difference = real.mean(0) - generated.mean(0)
    covariance_real = np.atleast_2d(np.cov(real, rowvar=False))
    covariance_generated = np.atleast_2d(np.cov(generated, rowvar=False))
    root_real = _psd_sqrt(covariance_real)
    root_product = _psd_sqrt(root_real @ covariance_generated @ root_real)
    mean_term = float(difference @ difference)
    covariance_term = float(np.trace(covariance_real + covariance_generated - 2 * root_product))
    distance = mean_term + covariance_term
    scale = max(1.0, mean_term + float(np.trace(covariance_real + covariance_generated)))
    if distance < -1e-7 * scale or not np.isfinite(distance):
        raise ValueError("Feature distance is numerically invalid.")
    return max(0.0, distance)


@torch.inference_mode()
def judge_outputs(judge, images, device, batch_size=256):
    if len(images) == 0 or batch_size < 1:
        raise ValueError("Images and evaluation batch size must be nonempty.")
    judge.eval()
    probabilities, features = [], []
    for batch in images.split(batch_size):
        logits, hidden = judge(batch.to(device))
        probabilities.append(logits.softmax(1).cpu().numpy())
        features.append(hidden.cpu().numpy())
    return np.concatenate(probabilities), np.concatenate(features)


def compare_images(judge, image_sets, reference, device, batch_size=256):
    reference_probabilities, reference_features = judge_outputs(judge, reference, device, batch_size)
    results = {"real_reference": prediction_summary(reference_probabilities)}
    for name, images in image_sets.items():
        if len(images) != len(reference):
            raise ValueError("Each image set must match the reference sample count.")
        probabilities, features = judge_outputs(judge, images, device, batch_size)
        results[name] = {
            **prediction_summary(probabilities),
            "cnn_frechet_distance": frechet_feature_distance(reference_features, features),
        }
    return results
