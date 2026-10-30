# Validation record

The full benchmark ran on GitHub Actions with Ubuntu 24.04, Python 3.12 and CPU PyTorch. [Full benchmark run 41140533383](https://github.com/Parisa6Izadyar/generative-AI/actions/runs/41140533383) records source commit `72cede4312eedf8c943016beadf411125bbf3167`. The final documentation and figures do not change the trained model code.

## Automated checks

`python -m pytest -q`: **29 tests passed** locally and in the hosted test workflow. All **14 notebook cells** executed through nbclient using synthetic tensors. [Test run 41140533320](https://github.com/Parisa6Izadyar/generative-AI/actions/runs/41140533320) checks the same source commit as the full benchmark.

The tests cover normalization, partition isolation, disjoint reference subsets, invalid probabilities, singular covariances, finite training updates, frozen evaluation, deterministic repeated runs, saved-model reloads and CLI errors. An interruption test restores a saved epoch and verifies exact agreement with uninterrupted training histories and evaluation metrics.

## Full MNIST benchmark

```bash
python -m gan_vae train --preset full --download --output results/full
```

Training completed all 5 classifier epochs and all 50 VAE and 50 GAN epochs. The classifier, VAE and GAN used 60,000 official training images. The classifier was evaluated on all 10,000 official test images. Each generative model supplied 5,000 images for the same classifier diagnostics and feature-distance calculation. Real-reference and real-comparison subsets contain 5,000 disjoint test images each.

| Image set | CNN Fréchet distance | Mean classifier confidence | Predicted-class entropy (nats) |
| --- | ---: | ---: | ---: |
| VAE | 224.3125 | 79.89% | 2.2605 |
| GAN | 56.9913 | 92.05% | 2.2304 |
| Real vs real control | 1.5449 | 99.00% | 2.3013 |
| Uniform random pixels | 1123.5248 | 72.82% | 0.8625 |

Classifier test accuracy: **99.05%**.

After training, `scripts/check_saved_models.py` reloaded `models.pt`, checked data and source hashes, regenerated the same latent samples and recomputed all diagnostics. It compared each metric with the saved JSON using relative tolerance 1e-6 and absolute tolerance 1e-7; classifier test accuracy was checked to absolute tolerance 1e-12. The inference CLI also generated new grids without training.

The tracked benchmark contains metrics and visual diagnostics. Dataset downloads, model weights and epoch checkpoints remain outside version control; the hosted workflow retains its output artifact for 30 days. `invocation_seconds` records the current command duration and does not accumulate time across restarted sessions.

## Scope of the evidence

The quick MNIST preset and offline synthetic checks exercise execution with smaller budgets. Their scores are separate from the full benchmark. The recorded comparison uses one seed, without a training-variance estimate or hyperparameter search. The custom CNN distance is not standard Inception FID. Confidence and class entropy describe classifier predictions; they do not calibrate perceptual quality or rule out memorization. Parameter counts and computation are not equalized across architectures.
