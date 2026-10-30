# Review notes

The original notebook contains completed classifier, VAE and GAN training outputs. Its saved figures and February 2026 presentation are retained under the `legacy_` directories. The earlier source and notebook remain accessible in Git history at commit `6bd774865523533ccf3caa2f5e8fd94da7decebe`.

## Corrections

| Original issue | Current behavior |
| --- | --- |
| The README ended inside an unclosed installation block | Complete setup, CLI, Codespaces, notebook and inference instructions |
| Importing the exported script downloaded MNIST and began 105 training epochs | Importable modules with explicit train, generate and smoke commands |
| Dependencies were unpinned and omitted the imported SciPy package | Pinned direct dependencies; distance calculation now uses NumPy, with no SciPy dependency |
| No seed, saved weights or machine-readable run manifest | Separate seeded training stages, inference checkpoints, epoch-level resume, settings, source/data hashes and JSON results |
| Custom classifier features were called FID in plots and slides | Current results say CNN Fréchet distance and explain the distinction from Inception FID |
| A singular-matrix warning appeared in the saved distance calculation; complex components were discarded without a tolerance check | Float64 symmetric PSD calculation with singular-covariance and analytic-distance tests |
| Confidence was described as visual realism and class entropy as evidence about mode collapse | Both are described as classifier diagnostics, with real-image and random-noise controls |
| Titles assumed smooth VAE convergence and sharp GAN images before inspecting the run | Neutral titles that identify the model and plotted quantity |
| Evaluation sent thousands of images through the CNN at once and used separate random draws across metrics | Batched evaluation with one matched generated-image set for all metrics |
| Hard-coded latent dimension 20 appeared outside model construction | Sampling and interpolation read the configured latent dimension |
| Training dropped the final incomplete batch and averaged batch means | All training images participate; epoch summaries are weighted by image count |
| The notebook and exported Python file duplicated the full training code | Notebook and compatibility entry point call the shared package |
| No automated regression checks or execution workflow | Tests cover partition isolation, losses, singular covariances, frozen evaluator, deterministic runs, CLI errors and checkpoint reloads; CI executes all notebook cells using synthetic tensors |

The original training/test MNIST separation was already correct. This review does not describe it as a discovered data leak. The changes preserve the original MLP widths, latent dimension, learning rates and VAE objective. They improve execution and evaluation without claiming an architecture innovation.

## Historical results

The original notebook recorded classifier test accuracy 98.55%, mean generated-image confidence 78.67% for the VAE and 90.25% for the GAN, predicted-class entropy 2.2407 and 2.2056 nats, and custom feature distances 162.69 and 64.61. Confidence/entropy used 5,000 generated images, while the distance used a separate sample of 5,120.

Those are archived outputs from an unseeded run with no saved classifier or generator checkpoints. They are not a verified reproduction of the current benchmark. In particular, the new classifier defines a different feature space, so a change in the distance cannot be reported as a percentage improvement over the archived score.

The historical presentation calls the custom score “Fréchet Inception Distance” and makes broad statements about guaranteed VAE training stability and complete class representation. The current comparison supports neither guarantee. Its conclusions are limited to the recorded architectures, training budget, evaluator and seed.
