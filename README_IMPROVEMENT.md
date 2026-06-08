# README_IMPROVEMENT

## 1. Project and Branch

Project name:

```text
SAOP-LRSH: Scale-Aware Opacity Pruning with Low-Rank SH Compression
```

This repository is prepared for the SLAM Technology final project requirement:

```text
Fork original repository, create an improvement branch, and provide reproducible baseline and improved commands.
```

The uploaded code contains both:

- Baseline-compatible 3DGS / CompressionGS training code.
- Modified SAOP-LRSH implementation.

The method description corresponds to the report sections:

- Section 5: Identified Limitations
- Section 6: Proposed Improvements
- Section 7: Experiments and Ablations

## 2. Improvement Summary

### 2.1 Scale-Aware Opacity Pruning

Motivation: the baseline compression pipeline can retain many low-opacity Gaussians after densification. These Gaussians increase storage and deployment size, while their contribution to final rendering is limited.

Implementation:

- Adds opacity-threshold pruning after the main densification stage.
- Adds scale-aware filtering, so pruning can target Gaussians that are both low-opacity and large-scale.
- Main arguments:

```text
--opacity_prune_start_iter
--opacity_prune_interval
--opacity_prune_threshold
--scale_aware_pruning
--scale_prune_quantile
```

### 2.2 Low-Rank SH Compression

Motivation: full spherical harmonics coefficients are expensive to store in compressed 3DGS models. Direct k-means compression of full SH features can keep model size high.

Implementation:

- Adds `lowrank_sh` as a quantized attribute.
- Builds a rank-limited SH representation before k-means storage.
- Decodes low-rank SH features when loading quantized models.
- Main arguments:

```text
--quant_params rot scale dc lowrank_sh
--lowrank_sh_rank
```

## 3. Modified Files

```text
train_kmeans.py
scene/gaussian_model.py
scripts/run_batch_op07_q995_30k.py
README.md
README_IMPROVEMENT.md
```

Important implementation points:

- `train_kmeans.py`
  - Adds `build_lowrank_sh_state`.
  - Adds `--lowrank_sh_rank`.
  - Adds opacity-pruning schedule arguments.
  - Prevents invalid mixed use of `lowrank_sh` with `sh` or `add_sh_dc`.

- `scene/gaussian_model.py`
  - Adds quantized low-rank SH decode.
  - Adds scale-aware pruning support.

- `scripts/run_batch_op07_q995_30k.py`
  - Runs the final 8 reported experiments.
  - Saves per-scene summaries and aggregate CSV/JSON records.

## 4. Environment

The experiments were run in the following local environment:

```text
Python: 3.10.20
PyTorch: 2.8.0+cu128
PyTorch CUDA runtime: 12.8
CUDA toolkit / nvcc: 12.8, V12.8.93
Conda env path: /data/miniconda3/envs/gaussian_splatting
```

Install with the provided environment file when possible:

```bash
conda env create -f environment.yml
conda activate gaussian_splatting
```

The local experiment commands used this Python executable:

```bash
/data/miniconda3/envs/gaussian_splatting/bin/python
```

Dataset path used in the experiments:

```text
/data/datasets/nerf_synthetic
```

Tested scenes:

```text
chair, ficus, lego, materials
```

## 5. Baseline Reproduction Command

Run from the repository root. This command keeps the baseline full-SH quantized path and does not enable the low-rank SH replacement:

```bash
/data/miniconda3/envs/gaussian_splatting/bin/python -u train_kmeans.py -s /data/datasets/nerf_synthetic/chair -m /root/lowrank_opacity_results/baseline_compgs_chair_30k --eval -w --iterations 30000 --total_iterations 30000 --test_iterations 1000 3000 5000 7000 10000 15000 20000 25000 30000 --save_iterations 30000 --densify_until_iter 15000 --kmeans_st_iter 20000 --kmeans_freq 500 --kmeans_iters 1 --kmeans_ncls 4096 --kmeans_ncls_sh 4096 --kmeans_ncls_dc 4096 --quant_params rot scale dc sh --opacity_reg --lambda_reg 1e-4
```

Baseline rendering and metrics:

```bash
/data/miniconda3/envs/gaussian_splatting/bin/python -u render.py -s /data/datasets/nerf_synthetic/chair -m /root/lowrank_opacity_results/baseline_compgs_chair_30k --iteration 30000 --skip_train --load_quant -w && /data/miniconda3/envs/gaussian_splatting/bin/python -u metrics.py -m /root/lowrank_opacity_results/baseline_compgs_chair_30k
```

## 6. Improved SAOP-LRSH Command

Run from the repository root. This command enables both scale-aware opacity pruning and low-rank SH compression:

```bash
/data/miniconda3/envs/gaussian_splatting/bin/python -u train_kmeans.py -s /data/datasets/nerf_synthetic/chair -m /root/lowrank_opacity_results/saop_lrsh_chair_30k --eval -w --iterations 30000 --total_iterations 30000 --test_iterations 1000 3000 5000 7000 10000 15000 18500 19000 20000 25000 30000 --save_iterations 30000 --densify_until_iter 15000 --max_prune_iter 20000 --opacity_prune_start_iter 18500 --opacity_prune_interval 500 --kmeans_st_iter 20000 --kmeans_freq 500 --kmeans_iters 1 --kmeans_ncls 4096 --kmeans_ncls_sh 4096 --kmeans_ncls_dc 4096 --quant_params rot scale dc lowrank_sh --lowrank_sh_rank 32 --opacity_reg --lambda_reg 1e-4 --opacity_prune_threshold 0.07 --scale_aware_pruning --scale_prune_quantile 0.995
```

Improved rendering and metrics:

```bash
/data/miniconda3/envs/gaussian_splatting/bin/python -u render.py -s /data/datasets/nerf_synthetic/chair -m /root/lowrank_opacity_results/saop_lrsh_chair_30k --iteration 30000 --skip_train --load_quant -w && /data/miniconda3/envs/gaussian_splatting/bin/python -u metrics.py -m /root/lowrank_opacity_results/saop_lrsh_chair_30k
```

## 7. Final Batch Command

The final reported 8-experiment batch is:

```bash
/data/miniconda3/envs/gaussian_splatting/bin/python -u scripts/run_batch_op07_q995_30k.py
```

This batch runs:

```text
chair lowranksh
ficus lowranksh
lego lowranksh
materials lowranksh
chair nolowranksh
ficus nolowranksh
lego nolowranksh
materials nolowranksh
```

Final reported hyperparameters:

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

## 8. Experiment Logs and Data

Training logs for the final batch:

```text
/root/lowrank_opacity_results/batch30k_opthr007_q995_lambda1e4_prune20k/logs
```

Final batch metrics:

```text
/root/lowrank_opacity_results/batch30k_opthr007_q995_lambda1e4_prune20k/summary.csv
/root/lowrank_opacity_results/batch30k_opthr007_q995_lambda1e4_prune20k/summary.json
/root/lowrank_opacity_results/batch30k_opthr007_q995_lambda1e4_prune20k/full_8_with_compgs_comparison.csv
/root/lowrank_opacity_results/batch30k_opthr007_q995_lambda1e4_prune20k/full_8_with_compgs_comparison.md
```

Historical experiment records:

```text
/root/experiment_data_group/experiments_summary.csv
/root/experiment_data_group/final_results_30k.csv
/root/experiment_data_group/final_results_7k_debug.csv
/root/experiment_data_group/ALL_EXPERIMENTS_COMPLETENESS.md
```

Packaged experiment data for course submission:

```text
/root/SLAM2026Final_upload/experiment_data_package.tar.gz
```

## 9. Notes on Data Packaging

Large files are intentionally excluded from the GitHub repository:

```text
*.ply
*.pth
*.bin
*.npy
logs/
output/
outputs/
```

They are stored in the separate experiment data package. This keeps the GitHub repository focused on reproducible code while preserving the raw experiment records required by the report.

