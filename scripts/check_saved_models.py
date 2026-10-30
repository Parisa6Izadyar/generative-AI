"""Recompute a recorded run's metrics from its saved models."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

from gan_vae.data import load_mnist
from gan_vae.experiment import generate_images, load_checkpoint, seed_everything
from gan_vae.metrics import compare_images
from gan_vae.training import judge_accuracy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("results/full"))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    args = parser.parse_args()
    expected = json.loads((args.run / "metrics.json").read_text())
    for name, digest in expected["source_sha256"].items():
        assert hashlib.sha256((ROOT / "gan_vae" / name).read_bytes()).hexdigest() == digest, name
    vae, generator, judge, config = load_checkpoint(args.run / "models.pt")
    seed_everything(config.seed, config.threads)
    _, test, reference, comparison, manifest = load_mnist(
        args.data_root, config.train_samples, config.eval_samples, config.seed
    )
    assert manifest == expected["data"], "Dataset manifest changed"
    generated = generate_images(vae, generator, config.eval_samples, config.seed + 40,
                                "cpu", config.eval_batch_size)
    noise = torch.rand(reference.shape, generator=torch.Generator().manual_seed(config.seed + 41)) * 2 - 1
    actual = compare_images(judge, {**generated, "real_comparison": comparison, "uniform_noise": noise},
                            reference, "cpu", config.eval_batch_size)
    assert actual.keys() == expected["evaluation"].keys()
    for name, row in actual.items():
        assert row.keys() == expected["evaluation"][name].keys()
        for key, value in row.items():
            np.testing.assert_allclose(value, expected["evaluation"][name][key], rtol=1e-6, atol=1e-7,
                                       err_msg=f"{name}.{key}")
    accuracy = judge_accuracy(judge, DataLoader(test, batch_size=config.eval_batch_size), "cpu")
    np.testing.assert_allclose(accuracy, expected["judge_test_accuracy"], rtol=0, atol=1e-12)
    print(f"Saved models reproduce all recorded diagnostics on {config.eval_samples} images per model.")
    print(f"Classifier test accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()
