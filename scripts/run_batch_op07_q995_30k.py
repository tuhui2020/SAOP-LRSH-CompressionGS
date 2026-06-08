import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


CODE_ROOT = Path("/root/gaussian-splatting_lowrank_opacity")
OUT_ROOT = Path("/root/lowrank_opacity_results")
BATCH_ROOT = OUT_ROOT / "batch30k_opthr007_q995_lambda1e4_prune20k"
LOG_ROOT = BATCH_ROOT / "logs"
PYTHON = "/data/miniconda3/envs/gaussian_splatting/bin/python"
DATA_ROOT = Path("/data/datasets/nerf_synthetic")
SCENES = ["chair", "ficus", "lego", "materials"]

OPACITY_THRESHOLD = 0.07
SCALE_QUANTILE = 0.995
LAMBDA_BY_SCENE = {
    "chair": "1e-4",
    "ficus": "1e-4",
    "lego": "1e-7",
    "materials": "1e-7",
}
ITERATIONS = 30000
PRUNE_END = 20000
PRUNE_START = 18500
PRUNE_INTERVAL = 500
KMEANS_START = 20000
PORT = "6044"


def run(cmd, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n[{datetime.now().isoformat(timespec='seconds')}] running: {' '.join(cmd)}", flush=True)
    with log_path.open("w", buffering=1) as log:
        process = subprocess.Popen(
            cmd,
            cwd=CODE_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            log.write(line)
        rc = process.wait()
    if rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)


def model_name(scene, variant):
    lambda_tag = "lambda1e7" if LAMBDA_BY_SCENE[scene] == "1e-7" else "lambda1e4"
    if variant == "lowranksh":
        return f"compgs_{scene}_30k_lowranksh_r32_opthr007_q995_{lambda_tag}_prune20k"
    return f"compgs_{scene}_30k_nolowranksh_opthr007_q995_{lambda_tag}_prune20k"


def model_path(scene, variant):
    existing_chair = OUT_ROOT / "compgs_chair_30k_lowranksh_r32_opthr007_q995_lambda1e4_prune20k"
    if scene == "chair" and variant == "lowranksh" and existing_chair.exists():
        return existing_chair
    return BATCH_ROOT / model_name(scene, variant)


def summary_path(scene, variant):
    lambda_tag = "lambda1e7" if LAMBDA_BY_SCENE[scene] == "1e-7" else "lambda1e4"
    return BATCH_ROOT / f"{scene}_30k_{variant}_opthr007_q995_{lambda_tag}_prune20k_summary.json"


def train_cmd(scene, variant, model):
    quant_params = ["rot", "scale", "dc", "lowrank_sh"] if variant == "lowranksh" else ["rot", "scale", "dc", "sh"]
    cmd = [
        PYTHON,
        "-u",
        "train_kmeans.py",
        "-s",
        str(DATA_ROOT / scene),
        "-m",
        str(model),
        "--eval",
        "-w",
        "--iterations",
        str(ITERATIONS),
        "--total_iterations",
        str(ITERATIONS),
        "--test_iterations",
        "1000",
        "3000",
        "5000",
        "7000",
        "10000",
        "15000",
        "18500",
        "19000",
        "20000",
        "25000",
        "30000",
        "--save_iterations",
        "30000",
        "--densify_until_iter",
        "15000",
        "--max_prune_iter",
        str(PRUNE_END),
        "--opacity_prune_start_iter",
        str(PRUNE_START),
        "--opacity_prune_interval",
        str(PRUNE_INTERVAL),
        "--kmeans_st_iter",
        str(KMEANS_START),
        "--kmeans_freq",
        "500",
        "--kmeans_iters",
        "1",
        "--kmeans_ncls",
        "4096",
        "--kmeans_ncls_sh",
        "4096",
        "--kmeans_ncls_dc",
        "4096",
        "--quant_params",
        *quant_params,
        "--opacity_reg",
        "--lambda_reg",
        LAMBDA_BY_SCENE[scene],
        "--opacity_prune_threshold",
        str(OPACITY_THRESHOLD),
        "--scale_aware_pruning",
        "--scale_prune_quantile",
        str(SCALE_QUANTILE),
        "--port",
        PORT,
    ]
    if variant == "lowranksh":
        cmd.extend(["--lowrank_sh_rank", "32"])
    return cmd


def train(scene, variant, model):
    ply = model / "point_cloud" / f"iteration_{ITERATIONS}" / "point_cloud.ply"
    if ply.exists():
        print(f"[skip] train exists: {model}", flush=True)
        return
    run(train_cmd(scene, variant, model), LOG_ROOT / f"{model.name}_train.log")


def render(scene, model):
    if (model / "test" / f"ours_{ITERATIONS}").exists():
        print(f"[skip] render exists: {model}", flush=True)
        return
    cmd = [
        PYTHON,
        "-u",
        "render.py",
        "-s",
        str(DATA_ROOT / scene),
        "-m",
        str(model),
        "--iteration",
        str(ITERATIONS),
        "--skip_train",
        "--load_quant",
        "-w",
    ]
    run(cmd, LOG_ROOT / f"{model.name}_render.log")


def metrics(model):
    if (model / "results.json").exists():
        print(f"[skip] metrics exists: {model}", flush=True)
        return
    cmd = [PYTHON, "-u", "metrics.py", "-m", str(model)]
    run(cmd, LOG_ROOT / f"{model.name}_metrics.log")


def count_gaussians(model):
    ply = model / "point_cloud" / f"iteration_{ITERATIONS}" / "point_cloud.ply"
    if not ply.exists():
        return None
    with ply.open("rb") as f:
        for raw in f:
            line = raw.decode("ascii", errors="ignore").strip()
            if line.startswith("element vertex "):
                return int(line.split()[-1])
            if line == "end_header":
                break
    return None


def deploy_size(model):
    root = model / "point_cloud" / f"iteration_{ITERATIONS}"
    files = [
        root / "point_cloud.ply",
        root / "kmeans_inds.bin",
        root / "kmeans_centers.pth",
        root / "kmeans_args.npy",
    ]
    return sum(path.stat().st_size for path in files if path.exists())


def summarize_one(scene, variant, model):
    result_path = model / "results.json"
    values = {}
    if result_path.exists():
        values = json.loads(result_path.read_text()).get(f"ours_{ITERATIONS}", {})
    size = deploy_size(model)
    summary = {
        "scene": scene,
        "variant": variant,
        "method": "lowrank_sh_r32_opacity_threshold_scaleaware" if variant == "lowranksh" else "sh_opacity_threshold_scaleaware",
        "iterations": ITERATIONS,
        "rank": 32 if variant == "lowranksh" else None,
        "opacity_prune_threshold": OPACITY_THRESHOLD,
        "scale_prune_quantile": SCALE_QUANTILE,
        "lambda_reg": float(LAMBDA_BY_SCENE[scene]),
        "model": str(model),
        "psnr": values.get("PSNR"),
        "ssim": values.get("SSIM"),
        "lpips": values.get("LPIPS"),
        "gaussians": count_gaussians(model),
        "deploy_size_bytes": size,
        "deploy_size_mb": round(size / 1_000_000, 4),
    }
    path = summary_path(scene, variant)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    write_aggregate()
    return summary


def experiments():
    for scene in SCENES:
        yield scene, "lowranksh"
    for scene in SCENES:
        yield scene, "nolowranksh"


def read_summaries():
    rows = []
    for scene, variant in experiments():
        path = summary_path(scene, variant)
        if path.exists():
            rows.append(json.loads(path.read_text()))
        elif scene == "chair" and variant == "lowranksh":
            old = OUT_ROOT / "chair_30k_lowranksh_r32_opthr007_q995_lambda1e4_prune20k_summary.json"
            if old.exists():
                data = json.loads(old.read_text())
                data["variant"] = "lowranksh"
                rows.append(data)
    return rows


def write_aggregate():
    rows = read_summaries()
    json_path = BATCH_ROOT / "summary.json"
    csv_path = BATCH_ROOT / "summary.csv"
    json_path.write_text(json.dumps(rows, indent=2) + "\n")
    fields = [
        "scene",
        "variant",
        "psnr",
        "ssim",
        "lpips",
        "gaussians",
        "deploy_size_mb",
        "opacity_prune_threshold",
        "scale_prune_quantile",
        "lambda_reg",
        "model",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})


def main():
    BATCH_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    for scene, variant in experiments():
        model = model_path(scene, variant)
        print(f"\n=== {scene} / {variant} / {model} ===", flush=True)
        train(scene, variant, model)
        render(scene, model)
        metrics(model)
        summarize_one(scene, variant, model)
    write_aggregate()
    print(f"\n[done] aggregate: {BATCH_ROOT / 'summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
