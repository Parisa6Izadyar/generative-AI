"""Command-line training and inference."""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Compare a GAN and VAE on MNIST.")
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="Run the quick or full MNIST experiment.")
    train.add_argument("--preset", choices=("quick", "full"), default="quick")
    train.add_argument("--data-root", type=Path, default=Path("data"))
    train.add_argument("--download", action="store_true")
    smoke = commands.add_parser("smoke", help="Check execution on synthetic tensors without a download.")
    for command, default_output in ((train, "results/reproduced"), (smoke, "results/smoke")):
        command.add_argument("--output", type=Path, default=Path(default_output))
        existing = command.add_mutually_exclusive_group()
        existing.add_argument("--overwrite", action="store_true")
        existing.add_argument("--resume", action="store_true")
        command.add_argument("--seed", type=int, default=42)
        command.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    generate = commands.add_parser("generate", help="Generate samples from a saved local checkpoint.")
    generate.add_argument("--checkpoint", type=Path, required=True)
    generate.add_argument("--output", type=Path, default=Path("results/generated"))
    generate.add_argument("--seed", type=int, default=123)
    generate.add_argument("--count", type=int, default=64)
    args = parser.parse_args()
    from .experiment import generate_images, load_checkpoint, preset_config, run_experiment, seed_everything
    from .plots import save_grid

    try:
        if args.command == "generate":
            if not 1 <= args.count <= 10_000 or not 0 <= args.seed < 2 ** 32 - 100:
                raise ValueError("Count must be 1–10,000 and seed must be in [0, 2**32 - 100).")
            seed_everything(args.seed)
            vae, generator, _, _ = load_checkpoint(args.checkpoint)
            images = generate_images(vae, generator, args.count, args.seed, "cpu")
            args.output.mkdir(parents=True, exist_ok=True)
            for name, samples in images.items():
                save_grid(samples[:64], args.output / f"{name}_samples.png", name.upper())
            print(f"Generated {args.count} images per model; grids show up to 64. Results: {args.output.resolve()}")
        else:
            config = preset_config("smoke" if args.command == "smoke" else args.preset, args.seed, args.device)
            run_experiment(config, args.output, getattr(args, "data_root", "data"),
                           getattr(args, "download", False), args.overwrite, args.resume)
    except (ValueError, OSError, RuntimeError, KeyError, TypeError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
