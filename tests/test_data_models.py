import numpy as np
import pytest
import torch

from gan_vae.data import load_mnist, normalize_pixels, select_indices
from gan_vae.models import Discriminator, Generator, MNISTJudge, VAE, vae_loss


def test_pixel_normalization_has_correct_range():
    pixels = torch.tensor([0, 255], dtype=torch.uint8).reshape(2, 1, 1).expand(2, 28, 28)
    normalized = normalize_pixels(pixels)
    assert normalized.shape == (2, 1, 28, 28)
    assert torch.all(normalized[0] == -1)
    assert torch.all(normalized[1] == 1)


@pytest.mark.parametrize("pixels", [torch.zeros(2, 28, 28), torch.zeros(2, 27, 28, dtype=torch.uint8)])
def test_invalid_pixels_are_rejected(pixels):
    with pytest.raises(ValueError, match="uint8"):
        normalize_pixels(pixels)


def test_sample_selection_is_seeded_without_replacement():
    first = select_indices(1000, 80, 42)
    assert np.array_equal(first, select_indices(1000, 80, 42))
    assert not np.array_equal(first, select_indices(1000, 80, 43))
    assert len(np.unique(first)) == 80
    with pytest.raises(ValueError):
        select_indices(10, 11, 42)


def test_official_partitions_and_reference_subsets_stay_separate(monkeypatch, tmp_path):
    class LocalMNIST:
        def __init__(self, root, train, download):
            self.data = torch.arange(80, dtype=torch.uint8).reshape(80, 1, 1).expand(80, 28, 28)
            if not train:
                self.data = self.data + 100
            self.targets = torch.arange(80) % 10

        def __len__(self):
            return len(self.data)

    monkeypatch.setattr("torchvision.datasets.MNIST", LocalMNIST)
    train, test, reference, comparison, manifest = load_mnist(tmp_path, 20, 12, 42)
    assert float(train.tensors[0].max()) < float(test.tensors[0].min())
    assert set(reference[:, 0, 0, 0].tolist()).isdisjoint(comparison[:, 0, 0, 0].tolist())
    assert manifest["reference_samples"] == 12


def test_model_shapes_ranges_and_vae_gradient():
    images = torch.randn(7, 1, 28, 28).tanh()
    vae = VAE(latent_dim=9)
    reconstruction, mu, logvar = vae(images)
    assert reconstruction.shape == images.shape
    assert mu.shape == logvar.shape == (7, 9)
    loss, _, _ = vae_loss(reconstruction, images, mu, logvar)
    loss.backward()
    assert torch.isfinite(vae.fc_mu.weight.grad).all()
    generated = Generator(9)(torch.randn(7, 9))
    assert generated.shape == images.shape
    assert float(generated.detach().abs().max()) <= 1
    assert Discriminator()(generated).shape == (7, 1)
    logits, features = MNISTJudge()(images)
    assert logits.shape == (7, 10) and features.shape == (7, 128)


def test_vae_loss_preserves_per_image_scaling_for_short_batches():
    images = torch.zeros(3, 1, 28, 28)
    mu, logvar = torch.zeros(3, 4), torch.zeros(3, 4)
    total, reconstruction, kl = vae_loss(torch.ones_like(images), images, mu, logvar)
    assert total.item() == reconstruction.item() == 784
    assert kl.item() == 0
