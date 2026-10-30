"""Save generated samples and diagnostics without opening GUI windows."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def save_grid(images, path, title, columns=8):
    pixels = images.detach().cpu().squeeze(1).numpy()
    pixels = np.clip((pixels + 1) / 2, 0, 1)
    rows = (len(pixels) + columns - 1) // columns
    canvas = np.ones((rows * 30, columns * 30))
    for index, image in enumerate(pixels):
        row, column = divmod(index, columns)
        canvas[row * 30:row * 30 + 28, column * 30:column * 30 + 28] = image
    fig, axis = plt.subplots(figsize=(6, max(1.5, rows * 0.7)))
    axis.imshow(canvas, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    axis.set_title(title)
    axis.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_comparison(image_sets, results, history, interpolation, output):
    output.mkdir(parents=True, exist_ok=True)
    for name in ("vae", "gan"):
        save_grid(image_sets[name][:64], output / f"{name}_samples.png", name.upper())
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for axis, name in zip(axes, ("vae", "gan")):
        axis.bar(range(10), results[name]["predicted_class_counts"], color="#376a92")
        axis.set(title=f"{name.upper()}: classifier predictions", xlabel="Predicted digit", ylabel="Images")
        axis.set_xticks(range(10))
    fig.tight_layout()
    fig.savefig(output / "predicted_classes.png", dpi=140)
    fig.savefig(output / "predicted_classes.svg", metadata={"Date": None})
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for field in ("total", "reconstruction", "kl"):
        axes[0].plot(range(1, len(history["vae"]) + 1), [x[field] for x in history["vae"]], label=field)
    for field in ("generator", "discriminator"):
        axes[1].plot(range(1, len(history["gan"]) + 1), [x[field] for x in history["gan"]], label=field)
    axes[0].set(title="VAE training", ylabel="Loss per image", xlabel="Epoch")
    axes[1].set(title="GAN training", ylabel="Binary cross-entropy", xlabel="Epoch")
    for axis in axes:
        axis.legend(frameon=False)
        axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output / "training.png", dpi=140)
    plt.close(fig)

    save_grid(torch.cat([interpolation["vae"], interpolation["gan"]]),
              output / "interpolation.png", "Latent interpolation: VAE above, GAN below", columns=10)
