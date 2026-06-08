# README_IMPROVEMENT

## Project

This repository is an improved implementation based on the CompGS / Gaussian Splatting compression codebase. The submitted branch adds an opacity-pruning and SH-compression variant for reproducible 3DGS compression experiments.

The implementation keeps the original training entry points and adds the improved path in `train_kmeans.py` and `scene/gaussian_model.py`.

## Method Changes

1. Scale-aware opacity pruning
   - Adds explicit opacity pruning after densification.
   - Supports a scale-aware condition so low-opacity Gaussians are pruned more aggressively when their scale is also above a chosen quantile.
   - Main arguments: `--opacity_prune_start_iter`, `--opacity_prune_interval`, `--opacity_prune_threshold`, `--scale_aware_pruning`, `--scale_prune_quantile`.

2. Low-rank SH compression
   - Adds `lowrank_sh` as a quantized attribute option.
   - Replaces full SH coefficient quantization with a rank-limited low-rank representation before k-means storage.
   - Main arguments: `--quant_params rot scale dc lowrank_sh`, `--lowrank_sh_rank`.

3. Regularized pruning schedule
   - Keeps opacity regularization compatible with the pruning schedule.
   - Main arguments: `--opacity_reg`, `--lambda_reg`.

## Key Modified Files

- `train_kmeans.py`
  - Adds low-rank SH state construction.
  - Adds pruning schedule controls.
  - Adds command-line arguments for rank, opacity threshold, scale-aware pruning, and lambda regularization.

- `scene/gaussian_model.py`
  - Adds low-rank SH decode support during quantized model loading.
  - Adds scale-aware pruning support.

- `scripts/run_batch_op07_q995_30k.py`
  - Reproduces the final 8-experiment batch used in the report.

## Environment

The experiments were run with the local conda environment:

```bash
/data/miniconda3/envs/gaussian_splatting/bin/python
```

Dataset root used in the experiments:

```bash
/data/datasets/nerf_synthetic
```

The tested scenes are:

```text
chair, ficus, lego, materials
```

## Reproduction Commands

Run from the repository root:

```bash
cd /root/gaussian-splatting_lowrank_opacity
```

Final 8-experiment batch:

```bash
/data/miniconda3/envs/gaussian_splatting/bin/python -u scripts/run_batch_op07_q995_30k.py
```

Single-scene improved low-rank SH example:

```bash
/data/miniconda3/envs/gaussian_splatting/bin/python -u train_kmeans.py \
  -s /data/datasets/nerf_synthetic/chair \
  -m /root/lowrank_opacity_results/compgs_chair_30k_lowranksh_r32_opthr007_q995_lambda1e4_prune20k \
  --eval -w \
  --iterations 30000 --total_iterations 30000 \
  --test_iterations 1000 3000 5000 7000 10000 15000 18500 19000 20000 25000 30000 \
  --save_iterations 30000 \
  --densify_until_iter 15000 \
  --max_prune_iter 20000 \
  --opacity_prune_start_iter 18500 \
  --opacity_prune_interval 500 \
  --kmeans_st_iter 20000 \
  --kmeans_freq 500 \
  --kmeans_iters 1 \
  --kmeans_ncls 4096 \
  --kmeans_ncls_sh 4096 \
  --kmeans_ncls_dc 4096 \
  --quant_params rot scale dc lowrank_sh \
  --lowrank_sh_rank 32 \
  --opacity_reg \
  --lambda_reg 1e-4 \
  --opacity_prune_threshold 0.07 \
  --scale_aware_pruning \
  --scale_prune_quantile 0.995
```

Single-scene no-low-rank ablation example:

```bash
/data/miniconda3/envs/gaussian_splatting/bin/python -u train_kmeans.py \
  -s /data/datasets/nerf_synthetic/chair \
  -m /root/lowrank_opacity_results/batch30k_opthr007_q995_lambda1e4_prune20k/compgs_chair_30k_nolowranksh_opthr007_q995_lambda1e4_prune20k \
  --eval -w \
  --iterations 30000 --total_iterations 30000 \
  --test_iterations 1000 3000 5000 7000 10000 15000 18500 19000 20000 25000 30000 \
  --save_iterations 30000 \
  --densify_until_iter 15000 \
  --max_prune_iter 20000 \
  --opacity_prune_start_iter 18500 \
  --opacity_prune_interval 500 \
  --kmeans_st_iter 20000 \
  --kmeans_freq 500 \
  --kmeans_iters 1 \
  --kmeans_ncls 4096 \
  --kmeans_ncls_sh 4096 \
  --kmeans_ncls_dc 4096 \
  --quant_params rot scale dc sh \
  --opacity_reg \
  --lambda_reg 1e-4 \
  --opacity_prune_threshold 0.07 \
  --scale_aware_pruning \
  --scale_prune_quantile 0.995
```

For `lego` and `materials`, the final reported configuration uses `--lambda_reg 1e-7`. For `chair` and `ficus`, it uses `--lambda_reg 1e-4`.

## Evaluation Commands

Render and evaluate a trained model:

```bash
/data/miniconda3/envs/gaussian_splatting/bin/python -u render.py \
  -s /data/datasets/nerf_synthetic/chair \
  -m /root/lowrank_opacity_results/compgs_chair_30k_lowranksh_r32_opthr007_q995_lambda1e4_prune20k \
  --iteration 30000 --skip_train --load_quant -w

/data/miniconda3/envs/gaussian_splatting/bin/python -u metrics.py \
  -m /root/lowrank_opacity_results/compgs_chair_30k_lowranksh_r32_opthr007_q995_lambda1e4_prune20k
```

## Experiment Data

Historical experiment records are stored in:

```text
/root/experiment_data_group
```

The final 8-experiment batch is stored in:

```text
/root/lowrank_opacity_results/batch30k_opthr007_q995_lambda1e4_prune20k
```

Important result files:

- `/root/experiment_data_group/experiments_summary.csv`
- `/root/experiment_data_group/final_results_30k.csv`
- `/root/experiment_data_group/ALL_EXPERIMENTS_COMPLETENESS.md`
- `/root/lowrank_opacity_results/batch30k_opthr007_q995_lambda1e4_prune20k/summary.csv`
- `/root/lowrank_opacity_results/batch30k_opthr007_q995_lambda1e4_prune20k/full_8_with_compgs_comparison.csv`
- `/root/lowrank_opacity_results/batch30k_opthr007_q995_lambda1e4_prune20k/DATA_COMPLETENESS.md`

## Final Reported Batch Configuration

```text
iterations: 30000
densify_until_iter: 15000
kmeans_st_iter: 20000
kmeans_freq: 500
kmeans_iters: 1
kmeans_ncls: 4096
kmeans_ncls_sh: 4096
kmeans_ncls_dc: 4096
opacity_prune_start_iter: 18500
opacity_prune_interval: 500
max_prune_iter: 20000
opacity_prune_threshold: 0.07
scale_aware_pruning: true
scale_prune_quantile: 0.995
lowrank_sh_rank: 32
lambda_reg chair/ficus: 1e-4
lambda_reg lego/materials: 1e-7
```

## Upload Notes

Do not upload large training outputs, datasets, rendered images, or point-cloud binaries into the GitHub code repository. These belong in the separate experiment data ZIP.

The GitHub repository should contain source code, scripts, environment files, and this improvement README. The experiment package should contain CSV/JSON metrics, rendered comparison images, logs, and completeness manifests.
