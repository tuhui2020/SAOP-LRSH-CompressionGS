# SAOP-LRSH for CompressionGS

This repository contains the course-project implementation of **SAOP-LRSH**: **Scale-Aware Opacity Pruning with Low-Rank Spherical Harmonics Compression** for compact 3D Gaussian Splatting.

The code is based on the original 3DGS / CompressionGS-style training pipeline and keeps the baseline training, rendering, and metric scripts. The improved method is implemented as an extension of the quantized training path.

For the required course-specific description, commands, environment, and experiment-log locations, see:

```text
README_IMPROVEMENT.md
```

## Main Contributions

- Adds scale-aware opacity pruning to remove low-contribution Gaussians more aggressively during the late training stage.
- Adds a low-rank SH compression option, `lowrank_sh`, to reduce spherical-harmonic storage while preserving rendering quality.
- Provides reproducible single-scene commands and a final 8-experiment batch script for `chair`, `ficus`, `lego`, and `materials`.

## Repository Structure

```text
.
├── README.md
├── README_IMPROVEMENT.md
├── train.py
├── train_kmeans.py
├── render.py
├── metrics.py
├── scene/
├── gaussian_renderer/
├── scripts/
│   └── run_batch_op07_q995_30k.py
├── submodules/
└── environment.yml
```

## Key Files

- `train.py`: original 3DGS-style training entry.
- `train_kmeans.py`: quantized training and the SAOP-LRSH improvement entry.
- `scene/gaussian_model.py`: quantized model loading, low-rank SH decode, and pruning support.
- `scripts/run_batch_op07_q995_30k.py`: final reported 30k-iteration batch.
- `README_IMPROVEMENT.md`: reproducibility instructions required by the course.

## Quick Start

Install dependencies following `README_IMPROVEMENT.md`, prepare the NeRF Synthetic dataset, then run a baseline or improved command from the repository root.

Final batch command:

```bash
/data/miniconda3/envs/gaussian_splatting/bin/python -u scripts/run_batch_op07_q995_30k.py
```

## Experiment Data

Large outputs are not stored in this GitHub repository. They are packaged separately as:

```text
experiment_data_package.tar.gz
```

The data package contains CSV/JSON metric files, logs, completeness manifests, and final experiment records used in the report.

## Note

This repository is for a SLAM Technology final project. It contains both the baseline-compatible code path and the modified SAOP-LRSH code path, so the original and improved methods can be compared under the same training and evaluation scripts.
