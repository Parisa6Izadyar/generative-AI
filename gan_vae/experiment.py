"""A fixed-budget experiment with separate training and evaluation stages."""

from dataclasses import asdict, dataclass, replace
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import load_mnist, synthetic_data
from .metrics import compare_images
from .models import Discriminator, Generator, MNISTJudge, VAE
from .plots import save_comparison
from .progress import prepare_progress, train_stage
from .training import judge_accuracy, train_gan, train_judge, train_vae


@dataclass(frozen=True)
class ExperimentConfig:
    preset: str = "quick"
    seed: int = 42
    train_samples: int = 10_000
    eval_samples: int = 1_000
    judge_epochs: int = 3
    vae_epochs: int = 5
    gan_epochs: int = 5
    latent_dim: int = 20
    batch_size: int = 128
    eval_batch_size: int = 256
    threads: int = 2
    device: str = "cpu"

    def validate(self):
        for name in ("seed", "train_samples", "eval_samples", "judge_epochs", "vae_epochs",
                     "gan_epochs", "latent_dim", "batch_size", "eval_batch_size", "threads"):
            value = getattr(self, name)
            if type(value) is not int or value < (0 if name == "seed" else 1):
                raise ValueError(f"{name} must be an integer in its valid positive range.")
        if self.seed >= 2 ** 32 - 100:
            raise ValueError("Seed must be below 2**32 - 100.")
        if self.train_samples > 60_000 or not 2 <= self.eval_samples <= 5_000:
            raise ValueError("MNIST allows at most 60,000 training and 2–5,000 reference samples.")
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("Device must be cpu or cuda.")
        if self.device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable. Use --device cpu.")


def preset_config(preset="quick", seed=42, device="cpu"):
    if preset == "quick":
        config = ExperimentConfig(seed=seed, device=device)
    elif preset == "full":
        config = ExperimentConfig(preset="full", seed=seed, device=device, train_samples=60_000,
                                  eval_samples=5_000, judge_epochs=5, vae_epochs=50, gan_epochs=50)
    elif preset == "smoke":
        config = ExperimentConfig(preset="smoke", seed=seed, device=device, train_samples=64,
                                  eval_samples=32, judge_epochs=1, vae_epochs=1, gan_epochs=1,
                                  batch_size=32)
    else:
        raise ValueError("Unknown preset.")
    config.validate()
    return config


def seed_everything(seed, threads=2):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def training_loader(dataset, batch_size, seed):
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False,
                      num_workers=0, generator=torch.Generator().manual_seed(seed))


@torch.inference_mode()
def generate_images(vae, generator, count, seed, device, batch_size=256):
    if count < 1 or batch_size < 1:
        raise ValueError("Image count and batch size must be positive.")
    vae.eval()
    generator.eval()
    # Draw on CPU so changing the evaluation batch size never changes the noise.
    noise = torch.randn(count, vae.latent_dim, generator=torch.Generator().manual_seed(seed))
    vae_images, gan_images = [], []
    for batch in noise.split(batch_size):
        batch = batch.to(device)
        vae_images.append(vae.decode(batch).cpu())
        gan_images.append(generator(batch).cpu())
    return {"vae": torch.cat(vae_images), "gan": torch.cat(gan_images)}


@torch.inference_mode()
def latent_interpolation(vae, generator, seed, device):
    endpoints = torch.randn(2, vae.latent_dim, generator=torch.Generator().manual_seed(seed))
    alpha = torch.linspace(0, 1, 10).unsqueeze(1)
    noise = ((1 - alpha) * endpoints[0] + alpha * endpoints[1]).to(device)
    return {"vae": vae.decode(noise).cpu(), "gan": generator(noise).cpu()}


def load_checkpoint(path, device="cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("schema_version") != 1:
        raise ValueError("Unsupported checkpoint schema.")
    config = ExperimentConfig(**checkpoint["config"])
    replace(config, device=device).validate()
    if device not in {"cpu", "cuda"} or (device == "cuda" and not torch.cuda.is_available()):
        raise ValueError("Requested inference device is unavailable.")
    vae, generator, judge = VAE(config.latent_dim), Generator(config.latent_dim), MNISTJudge()
    for name, model in (("vae", vae), ("generator", generator), ("judge", judge)):
        model.load_state_dict(checkpoint[name])
        model.to(device).eval().requires_grad_(False)
    return vae, generator, judge, config


def run_experiment(config, output="results/reproduced", data_root="data", download=False,
                   overwrite=False, resume=False):
    config.validate()
    output = Path(output)
    if resume and overwrite:
        raise ValueError("Choose either resume or overwrite.")
    if output.exists() and any(output.iterdir()) and not overwrite and not resume:
        raise ValueError(f"{output} is not empty. Choose another --output or pass --overwrite.")
    start = time.perf_counter()
    seed_everything(config.seed, config.threads)
    if config.preset == "smoke":
        train, test, reference, comparison, manifest = synthetic_data(config.seed)
    else:
        train, test, reference, comparison, manifest = load_mnist(
            data_root, config.train_samples, config.eval_samples, config.seed, download
        )
    print(f"Dataset: {manifest['dataset']}; train {len(train)}, reference {len(reference)}", flush=True)
    source_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in sorted(Path(__file__).parent.glob("*.py"))}
    progress = prepare_progress(output, asdict(config), manifest, source_hashes, resume, overwrite)
    histories = {}
    # Each training stage has its own random stream and shuffle order.
    seed_everything(config.seed + 10, config.threads)
    judge = MNISTJudge().to(config.device)
    histories["judge"] = train_stage("judge", {"judge": judge},
                                     training_loader(train, config.batch_size, config.seed + 11),
                                     train_judge, config.judge_epochs, config.device, progress, resume)
    accuracy = judge_accuracy(judge, DataLoader(test, batch_size=config.eval_batch_size), config.device)
    print(f"Judge held-out accuracy: {accuracy:.4f} on {len(test)} images", flush=True)
    seed_everything(config.seed + 20, config.threads)
    vae = VAE(config.latent_dim).to(config.device)
    histories["vae"] = train_stage("vae", {"vae": vae},
                                   training_loader(train, config.batch_size, config.seed + 21),
                                   train_vae, config.vae_epochs, config.device, progress, resume)
    seed_everything(config.seed + 30, config.threads)
    generator, discriminator = Generator(config.latent_dim).to(config.device), Discriminator().to(config.device)
    histories["gan"] = train_stage("gan", {"generator": generator, "discriminator": discriminator},
                                   training_loader(train, config.batch_size, config.seed + 31),
                                   train_gan, config.gan_epochs, config.device, progress, resume)
    generated = generate_images(vae, generator, config.eval_samples, config.seed + 40,
                                config.device, config.eval_batch_size)
    noise_baseline = torch.rand(reference.shape, generator=torch.Generator().manual_seed(config.seed + 41)) * 2 - 1
    image_sets = {**generated, "real_comparison": comparison, "uniform_noise": noise_baseline}
    results = compare_images(judge, image_sets, reference, config.device, config.eval_batch_size)
    metrics = {
        "schema_version": 1,
        "run_type": "execution check on synthetic tensors" if config.preset == "smoke" else "fixed-budget MNIST comparison",
        "config": asdict(config), "data": manifest,
        "judge_test_accuracy": accuracy, "judge_test_samples": len(test),
        "evaluation": results, "history": histories,
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        **{name: version(name) for name in ("torch", "torchvision", "numpy", "matplotlib")}},
        "source_sha256": source_hashes,
        "limitations": [
            "One seed; no estimate of variation across independent training runs.",
            "CNN feature distance is evaluator-specific and is not standard Inception FID.",
            "Classifier confidence is not calibrated image quality.",
            "Predicted-class entropy does not measure within-class diversity or rule out memorization.",
            "Equal epochs and latent dimension do not equalize parameters, compute or optimization difficulty.",
            "Fixed budgets and final checkpoints; no test-driven hyperparameter or checkpoint selection.",
        ],
    }
    output.mkdir(parents=True, exist_ok=True)
    # Only tensors and primitive metadata are stored; inference uses weights_only=True.
    torch.save({"schema_version": 1, "config": asdict(config),
                **{name: {key: value.detach().cpu() for key, value in model.state_dict().items()}
                   for name, model in (("vae", vae), ("generator", generator), ("judge", judge))}},
               output / "models.pt")
    save_comparison(generated, results, histories,
                    latent_interpolation(vae, generator, config.seed + 42, config.device), output)
    metrics["invocation_seconds"] = round(time.perf_counter() - start, 3)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n")
    for name in ("vae", "gan"):
        row = results[name]
        print(f"{name.upper()}: CNN distance {row['cnn_frechet_distance']:.4f}, "
              f"mean confidence {row['mean_max_probability']:.4f}, "
              f"class entropy {row['predicted_class_entropy_nats']:.4f}", flush=True)
    print(f"Finished. Results: {output.resolve()}", flush=True)
    return metrics
