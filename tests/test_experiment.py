from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from gan_vae.data import synthetic_data
from gan_vae.experiment import (
    generate_images, load_checkpoint, preset_config, run_experiment, training_loader,
)
from gan_vae.metrics import judge_outputs
from gan_vae.models import Discriminator, Generator, MNISTJudge
from gan_vae.training import train_gan, train_judge


def test_gan_training_updates_both_networks_with_finite_losses():
    train, *_ = synthetic_data()
    generator, discriminator = Generator(), Discriminator()
    before_g = generator.layers[0].weight.detach().clone()
    before_d = discriminator.layers[1].weight.detach().clone()
    history = train_gan(generator, discriminator, training_loader(train, 23, 42), 1, "cpu")
    assert not torch.equal(before_g, generator.layers[0].weight)
    assert not torch.equal(before_d, discriminator.layers[1].weight)
    assert np.isfinite(list(history[0].values())).all()


def test_judge_is_frozen_and_batching_preserves_outputs():
    train, _, images, *_ = synthetic_data()
    judge = MNISTJudge()
    train_judge(judge, training_loader(train, 23, 42), 1, "cpu")
    assert not judge.training
    assert not any(p.requires_grad for p in judge.parameters())
    first = judge_outputs(judge, images, "cpu", batch_size=7)
    second = judge_outputs(judge, images, "cpu", batch_size=32)
    for left, right in zip(first, second):
        np.testing.assert_allclose(left, right, atol=2e-6)


def test_full_smoke_run_reload_and_reproducibility(tmp_path):
    config = preset_config("smoke")
    first = run_experiment(config, tmp_path / "first")
    second = run_experiment(config, tmp_path / "second")
    assert first["evaluation"] == second["evaluation"]
    assert first["history"] == second["history"]
    assert "synthetic" in first["run_type"]
    vae, generator, judge, loaded_config = load_checkpoint(tmp_path / "first/models.pt")
    assert loaded_config == config
    generated = generate_images(vae, generator, 32, config.seed + 40, "cpu", 7)
    loaded_again = load_checkpoint(tmp_path / "first/models.pt")
    repeated = generate_images(loaded_again[0], loaded_again[1], 32, config.seed + 40, "cpu", 32)
    for name in generated:
        torch.testing.assert_close(generated[name], repeated[name], atol=1e-6, rtol=1e-5)
    assert not any(p.requires_grad for p in judge.parameters())
    assert json.loads((tmp_path / "first/metrics.json").read_text())["config"]["preset"] == "smoke"
    assert (tmp_path / "first/interpolation.png").stat().st_size > 0
    with pytest.raises(ValueError, match="not empty"):
        run_experiment(config, tmp_path / "first")
    # A saved CUDA run must remain usable for inference on a CPU-only machine.
    checkpoint = torch.load(tmp_path / "first/models.pt", weights_only=True)
    checkpoint["config"]["device"] = "cuda"
    torch.save(checkpoint, tmp_path / "cuda_metadata.pt")
    load_checkpoint(tmp_path / "cuda_metadata.pt", "cpu")


@pytest.mark.parametrize("field,value", [("seed", -1), ("eval_samples", 5001), ("batch_size", 0)])
def test_invalid_configuration_is_rejected(field, value):
    with pytest.raises(ValueError):
        replace(preset_config(), **{field: value}).validate()


def test_cli_help_and_no_download_error(tmp_path):
    root = Path(__file__).resolve().parents[1]
    help_result = subprocess.run([sys.executable, "-m", "gan_vae", "--help"], cwd=root,
                                 capture_output=True, text=True, timeout=30)
    assert help_result.returncode == 0 and "smoke" in help_result.stdout
    missing = subprocess.run([sys.executable, "-m", "gan_vae", "train", "--data-root", str(tmp_path / "empty"),
                              "--output", str(tmp_path / "run")], cwd=root,
                             capture_output=True, text=True, timeout=30)
    assert missing.returncode != 0 and "--download" in missing.stderr
    assert not (tmp_path / "run/metrics.json").exists()


def test_resume_matches_uninterrupted_training(monkeypatch, tmp_path):
    import gan_vae.progress as progress

    config = replace(preset_config("smoke"), judge_epochs=2, vae_epochs=2, gan_epochs=2)
    expected = run_experiment(config, tmp_path / "continuous")
    save_checkpoint = progress.save_checkpoint

    def interrupt_after_saved_epoch(path, checkpoint):
        save_checkpoint(path, checkpoint)
        if path.stem == "gan" and checkpoint["completed_epochs"] == 1:
            raise RuntimeError("Simulated interruption after a completed epoch")

    monkeypatch.setattr(progress, "save_checkpoint", interrupt_after_saved_epoch)
    with pytest.raises(RuntimeError, match="Simulated interruption"):
        run_experiment(config, tmp_path / "interrupted")
    monkeypatch.setattr(progress, "save_checkpoint", save_checkpoint)
    actual = run_experiment(config, tmp_path / "interrupted", resume=True)
    assert expected["history"] == actual["history"]
    assert expected["evaluation"] == actual["evaluation"]
    with pytest.raises(ValueError, match="unchanged"):
        run_experiment(replace(config, seed=43), tmp_path / "interrupted", resume=True)


def test_resume_requires_progress_and_excludes_overwrite(tmp_path):
    with pytest.raises(ValueError, match="No saved"):
        run_experiment(preset_config("smoke"), tmp_path / "missing", resume=True)
    with pytest.raises(ValueError, match="either resume or overwrite"):
        run_experiment(preset_config("smoke"), tmp_path / "missing", resume=True, overwrite=True)
