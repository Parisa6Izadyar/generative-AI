"""Networks used in the original MNIST experiment."""

import torch
from torch import nn
from torch.nn import functional as F


class MNISTJudge(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, images):
        features = F.relu(self.fc1(self.conv_layers(images).flatten(1)))
        return self.fc2(features), features


class VAE(nn.Module):
    def __init__(self, latent_dim=20):
        super().__init__()
        self.latent_dim = latent_dim
        self.fc1 = nn.Linear(784, 400)
        self.fc_mu = nn.Linear(400, latent_dim)
        self.fc_logvar = nn.Linear(400, latent_dim)
        self.fc3 = nn.Linear(latent_dim, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, images):
        hidden = F.relu(self.fc1(images.flatten(1)))
        return self.fc_mu(hidden), self.fc_logvar(hidden)

    @staticmethod
    def reparameterize(mu, logvar):
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def decode(self, noise):
        return torch.tanh(self.fc4(F.relu(self.fc3(noise)))).reshape(-1, 1, 28, 28)

    def forward(self, images):
        mu, logvar = self.encode(images)
        return self.decode(self.reparameterize(mu, logvar)), mu, logvar


class Generator(nn.Module):
    def __init__(self, latent_dim=20):
        super().__init__()
        self.latent_dim = latent_dim
        self.layers = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 512), nn.LeakyReLU(0.2),
            nn.Linear(512, 784), nn.Tanh(),
        )

    def forward(self, noise):
        return self.layers(noise).reshape(-1, 1, 28, 28)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(), nn.Linear(784, 512), nn.LeakyReLU(0.2),
            nn.Linear(512, 256), nn.LeakyReLU(0.2), nn.Linear(256, 1),
        )

    def forward(self, images):
        # Training uses BCEWithLogitsLoss, so the network returns raw logits.
        return self.layers(images)


def vae_loss(reconstruction, images, mu, logvar):
    """Return per-image pixel SSE + KL, preserving the original objective."""
    reconstruction_loss = (reconstruction - images).square().flatten(1).sum(1).mean()
    kl = -0.5 * (1 + logvar - mu.square() - logvar.exp()).sum(1).mean()
    return reconstruction_loss + kl, reconstruction_loss, kl
