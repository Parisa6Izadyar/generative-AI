# Experiment methodology

## Data and evaluation protocol

MNIST provides 60,000 training and 10,000 test images. The loader preserves that separation and maps uint8 pixels to `[-1, 1]` with `pixels / 127.5 - 1`. The classifier, VAE and GAN see only the selected training partition during optimization. Generative training does not use digit labels.

The full preset uses all training images. The quick preset takes 10,000 indices from a seeded permutation, without replacement. The same selected training images feed all three models. Batches include the final incomplete batch, and epoch losses are weighted by image count.

After training, a seeded permutation of the test partition supplies two disjoint equally sized subsets: the real feature reference and a real-versus-real control. Generated-image and uniform-noise sample counts match the reference. The full preset uses 5,000 images per subset; the quick preset uses 1,000. The classifier's test accuracy uses the entire official test set.

The model architectures and training budgets are fixed in advance. Final weights supply all results; test metrics do not select a model, epoch or hyperparameter. There is no validation-driven tuning in this comparison. Reusing the same benchmark repeatedly to tune a model would require a separate validation protocol and a fresh final evaluation.

## Networks

| Model | Architecture | Training objective and optimizer |
| --- | --- | --- |
| VAE | 784 → 400 → Gaussian mean and log variance of dimension 20; decoder 20 → 400 → 784 with Tanh | Per-image pixel SSE + KL divergence; Adam, learning rate 0.001 |
| GAN generator | 20 → 256 → 512 → 784 with LeakyReLU and final Tanh | Non-saturating binary cross-entropy objective; Adam, learning rate 0.0002, betas (0.5, 0.999) |
| GAN discriminator | 784 → 512 → 256 → 1 with LeakyReLU and raw output logits | Mean of real and detached-fake binary cross-entropies; same Adam settings |
| Classifier | Two 3×3 convolution blocks, 16 then 32 channels, each followed by ReLU and 2×2 max pooling; 128-unit hidden layer and ten logits | Cross-entropy; Adam, learning rate 0.001 |

The VAE preserves the original squared-error objective. Its reconstruction term sums across pixels and averages across images; it is not a pixel-mean MSE. The KL term sums across latent dimensions and averages across images. The coefficient on KL is 1.

The VAE has 652,824 parameters. The GAN has 539,152 generator parameters and 533,505 discriminator parameters. The separate classifier has 206,922 parameters.

The discriminator returns logits to `BCEWithLogitsLoss`, which combines sigmoid and binary cross-entropy using a numerically stable formulation. Its parameters are frozen during the generator update while gradients still pass through it to the generated pixels. The classifier is frozen after training and remains in evaluation mode.

Equal epochs and latent dimensions leave different parameter counts, optimization dynamics and computational costs. This experiment does not control for equal FLOPs or attempt to find the strongest architecture for either model family.

## Diagnostics

### Classifier predictions

Mean maximum softmax probability summarizes classifier confidence. It does not measure the probability that an image is realistic, and high confidence on random noise is possible. The uniform-noise control helps expose that limitation for this evaluator.

Predicted labels supply ten class counts and Shannon entropy `-sum(p * log(p))`, in nats. Zero-count classes contribute zero. The maximum is `log(10)`, approximately 2.3026; normalized entropy divides by that value. High entropy describes balance across predicted labels, without establishing within-class diversity, accurate labels or freedom from memorization.

### CNN Fréchet distance

The evaluator's hidden layer supplies 128 ReLU features. For real and generated feature samples, compute their means and sample covariance matrices in float64. The reported value is the squared Gaussian 2-Wasserstein distance:

`||mu_real - mu_generated||² + trace(C_real + C_generated - 2 * sqrt(sqrt(C_real) @ C_generated @ sqrt(C_real)))`.

Covariance matrices are symmetrized before their eigendecomposition. Tiny negative eigenvalues caused by round-off are clipped to zero; materially negative eigenvalues raise an error. This handles valid rank-deficient empirical covariances without taking the real part of an unchecked complex result. Small negative final distances from round-off are clipped to zero after a scale-dependent validity check.

This is an evaluator-specific distance, not standard Fréchet Inception Distance. Different classifier weights change the feature space and the score. A zero value means matching Gaussian feature summaries, which is weaker than equality of full image distributions. Estimates also depend on sample size. Keep evaluator, sample count and protocol fixed for any numeric comparison. Historical scores from the old unseeded classifier are not directly comparable to the new run.

The disjoint real-versus-real control gives finite-sample context. It is not a calibrated quality threshold or a confidence interval. A multi-seed experiment with additional perceptual and memorization checks would be needed for stronger conclusions.

## Reproducibility

The default seed is 42. Training stages use separate deterministic seeds: classifier +10, VAE +20 and GAN +30, with independent DataLoader generators at +11, +21 and +31. Evaluation latent vectors use +40, random pixels +41, and interpolation endpoints +42. Sampling on CPU fixes the latent vectors independently of the evaluation batch size.

Python, NumPy and PyTorch are seeded. Deterministic PyTorch algorithms are enabled; cuDNN benchmarking is disabled. Training uses zero DataLoader workers and two CPU threads. The run records dependency versions, source hashes, raw MNIST file checksums and hashes of selected index arrays. These controls support repeatability within the tested environment; PyTorch does not guarantee identical results across platforms, devices or releases.

The final `models.pt` checkpoint contains state dictionaries and primitive configuration metadata for generation and evaluation. Separate epoch checkpoints under `progress/` also retain optimizer states, completed histories, PyTorch RNG state and the DataLoader generator state. Loading uses `weights_only=True`.

Epoch checkpoints use an atomic file replacement, so an interrupted write leaves the preceding completed epoch available. Resume verifies configuration, data and source hashes, then restores the saved states. It repeats any unfinished epoch, without trying to resume inside a batch. Use the same environment when resuming. The automated interruption test compares a resumed GAN run with uninterrupted training and checks exact agreement of histories and evaluation metrics.

The manifest's `invocation_seconds` measures the current command, including output writing up to the metrics file. For a resumed run it does not represent total accumulated training time.

## References

- [Kingma and Welling, Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)
- [Goodfellow et al., Generative Adversarial Nets](https://arxiv.org/abs/1406.2661)
- [Heusel et al., GANs Trained by a Two Time-Scale Update Rule](https://arxiv.org/abs/1706.08500), source of the standard Inception-based FID definition
- [PyTorch reproducibility notes](https://docs.pytorch.org/docs/2.9/notes/randomness.html)
- [PyTorch BCEWithLogitsLoss](https://docs.pytorch.org/docs/2.9/generated/torch.nn.BCEWithLogitsLoss.html)
