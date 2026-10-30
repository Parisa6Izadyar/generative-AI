"""Training steps with no evaluation data in the optimization path."""

import math

import torch
from torch import nn

from .models import vae_loss


def _check_loss(loss):
    if not math.isfinite(float(loss.detach())):
        raise ValueError("Training produced a non-finite loss.")


def train_judge(model, loader, epochs, device, start_epoch=0, history=None,
                optimizer_state=None, on_epoch_end=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
    history = list(history or [])
    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss, correct, count = 0.0, 0, 0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(images)
            loss = nn.functional.cross_entropy(logits, labels)
            _check_loss(loss)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(images)
            correct += int((logits.argmax(1) == labels).sum())
            count += len(images)
        history.append({"loss": total_loss / count, "accuracy": correct / count})
        print(f"Judge {epoch + 1}/{epochs}: loss {total_loss / count:.4f}, accuracy {correct / count:.4f}", flush=True)
        if on_epoch_end:
            on_epoch_end(epoch + 1, optimizer, history)
    model.eval()
    model.requires_grad_(False)
    return history


@torch.inference_mode()
def judge_accuracy(model, loader, device):
    model.eval()
    correct, count = 0, 0
    for images, labels in loader:
        logits, _ = model(images.to(device))
        correct += int((logits.argmax(1).cpu() == labels).sum())
        count += len(images)
    return correct / count


def train_vae(model, loader, epochs, device, start_epoch=0, history=None,
              optimizer_state=None, on_epoch_end=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
    history = list(history or [])
    for epoch in range(start_epoch, epochs):
        model.train()
        totals = torch.zeros(3, dtype=torch.float64)
        count = 0
        for images, _ in loader:
            images = images.to(device)
            optimizer.zero_grad(set_to_none=True)
            reconstruction, mu, logvar = model(images)
            losses = vae_loss(reconstruction, images, mu, logvar)
            _check_loss(losses[0])
            losses[0].backward()
            optimizer.step()
            totals += torch.tensor([float(x.detach()) for x in losses], dtype=torch.float64) * len(images)
            count += len(images)
        values = (totals / count).tolist()
        history.append(dict(zip(("total", "reconstruction", "kl"), values)))
        print(f"VAE {epoch + 1}/{epochs}: total {values[0]:.4f}, SSE {values[1]:.4f}, KL {values[2]:.4f}", flush=True)
        if on_epoch_end:
            on_epoch_end(epoch + 1, optimizer, history)
    return history


def train_gan(generator, discriminator, loader, epochs, device, start_epoch=0,
              history=None, optimizer_state=None, on_epoch_end=None):
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    if optimizer_state is not None:
        optimizer_g.load_state_dict(optimizer_state["generator"])
        optimizer_d.load_state_dict(optimizer_state["discriminator"])
    loss_function = nn.BCEWithLogitsLoss()
    history = list(history or [])
    for epoch in range(start_epoch, epochs):
        generator.train()
        discriminator.train()
        total_g, total_d, count = 0.0, 0.0, 0
        for images, _ in loader:
            images = images.to(device)
            noise = torch.randn(len(images), generator.latent_dim, device=device)
            fake = generator(noise)
            optimizer_d.zero_grad(set_to_none=True)
            real_logits = discriminator(images)
            fake_logits = discriminator(fake.detach())
            loss_d = (loss_function(real_logits, torch.ones_like(real_logits))
                      + loss_function(fake_logits, torch.zeros_like(fake_logits))) / 2
            _check_loss(loss_d)
            loss_d.backward()
            optimizer_d.step()

            # Gradients still pass through D to G while D's weights stay fixed.
            discriminator.requires_grad_(False)
            optimizer_g.zero_grad(set_to_none=True)
            fake_logits = discriminator(fake)
            loss_g = loss_function(fake_logits, torch.ones_like(fake_logits))
            _check_loss(loss_g)
            loss_g.backward()
            optimizer_g.step()
            discriminator.requires_grad_(True)
            total_g += float(loss_g.detach()) * len(images)
            total_d += float(loss_d.detach()) * len(images)
            count += len(images)
        history.append({"generator": total_g / count, "discriminator": total_d / count})
        print(f"GAN {epoch + 1}/{epochs}: D {total_d / count:.4f}, G {total_g / count:.4f}", flush=True)
        if on_epoch_end:
            on_epoch_end(epoch + 1, {"generator": optimizer_g, "discriminator": optimizer_d}, history)
    return history
