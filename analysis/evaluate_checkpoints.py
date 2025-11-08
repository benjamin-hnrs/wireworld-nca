import os
import glob
import gc
import random
import multiprocessing
import torch
import numpy as np
import pandas as pd

from src.config.io import load_cfg
from src.config.finalise import finalise
from src.nca.nca import NCA
from src.training.discrete_trainer import DiscreteTrainer
from src.training.continuous_trainer import ContinuousTrainer
from src.training.metrics import mse_similarity, exact_rgb_accuracy, palette_snapped_exact_rgb_accuracy
from src.utils.utils import load_palette_from_image, state_to_rgba, to_rgba

RESULTS_DIR = "results"
CHECKPOINT_PATTERN = "checkpoints/checkpoint_*.pth"
N_EVAL_RUNS = 100  # number of repeated evaluations per checkpoint (take mean)

def find_latest_checkpoint(subdir):
    checkpoints = glob.glob(os.path.join(subdir, CHECKPOINT_PATTERN))
    if not checkpoints:
        return None
    return max(checkpoints, key=os.path.getmtime)

def evaluate_checkpoint(subdir):
    # load config
    cfg_path = os.path.join(subdir, "config.yaml")
    if not os.path.exists(cfg_path):
        print(f"Config not found in {subdir}")
        return None
    cfg = finalise(load_cfg(cfg_path))

    # read rollout window from config
    rollout_min = cfg.training.rollout.min
    rollout_max = cfg.training.rollout.max + 1
    rollout_steps = rollout_max * 2
    print(f"Rollout window from config: min={rollout_min}, max={rollout_max} -> running {rollout_steps} steps")

    # load model
    model = NCA(
        config=cfg,
        num_visible=cfg.model.num_visible,
        num_hidden=cfg.model.num_hidden,
        device=cfg.compute.device,
        fire_rate=cfg.model.fire_rate,
        alive_threshold=cfg.model.alive_threshold,
        default_step_size=cfg.model.step_size,
    ).to(cfg.compute.device)
    model.eval()

    checkpoint_path = find_latest_checkpoint(subdir) if 'find_latest_checkpoint' in globals() else max(glob.glob(os.path.join(subdir, CHECKPOINT_PATTERN)), key=os.path.getmtime) if glob.glob(os.path.join(subdir, CHECKPOINT_PATTERN)) else None
    if not checkpoint_path:
        print(f"No checkpoint found in {subdir}")
        return None
    model.load_state_dict(torch.load(checkpoint_path, map_location=cfg.compute.device))

    print(f"Config loaded from {cfg_path}. Evaluating checkpoint {checkpoint_path} in {subdir} with {N_EVAL_RUNS} runs")

    # choose trainer for metrics
    if cfg.training.mode == "continuous":
        trainer = ContinuousTrainer(model=model, config=cfg)
    else:
        trainer = DiscreteTrainer(model=model, config=cfg)

    target = trainer.target  # base target (1, C, H, W)
    palette = load_palette_from_image(cfg.target.target_image)

    runs_dfs = []
    per_run_summaries = []

    for run_idx in range(N_EVAL_RUNS):
        # make runs differ
        py_seed = random.randint(0, 2**31 - 1)
        np.random.seed(py_seed)
        torch.manual_seed(py_seed)

        H = cfg.target.target_size + 2 * cfg.target.target_padding
        W = H
        seed = torch.zeros((1, cfg.model.num_visible + cfg.model.num_hidden + 1,
                            H, W), device=cfg.compute.device)
        cx = (H // 2) + int(np.random.randint(-1, 2))
        cy = (W // 2) + int(np.random.randint(-1, 2))
        cx = np.clip(cx, 0, H - 1)
        cy = np.clip(cy, 0, W - 1)
        seed[..., cx, cy] = 1.0

        target_b = target.expand(seed.shape[0], -1, -1, -1)

        if trainer.config.training.mode == "discrete":
            y_rgba = state_to_rgba(target_b, trainer.palette, trainer.model.num_visible, trainer.config.model.alive_threshold, alpha_mode='clamped')
        else:
            y_rgba = to_rgba(target_b)

        x = seed.clone()
        metrics = []

        with torch.inference_mode():
            for step in range(rollout_steps):
                x = model(x)

                if trainer.config.training.mode == "discrete":
                    x_rgba = state_to_rgba(x, trainer.palette, trainer.model.num_visible, trainer.config.model.alive_threshold, alpha_mode='clamped')
                else:
                    x_rgba = to_rgba(x)

                mse = mse_similarity(x_rgba, y_rgba)
                acc = exact_rgb_accuracy(x_rgba, y_rgba, use_uint8_compare=True)
                acc_snap = palette_snapped_exact_rgb_accuracy(x_rgba, y_rgba, trainer.palette)

                metrics.append({"step": step, "mse": mse, "acc": acc, "acc_snap": acc_snap})

        run_df = pd.DataFrame(metrics)
        runs_dfs.append(run_df)

        # compute required stats for this run
        max_mse = run_df["mse"].max()
        max_acc = run_df["acc"].max()
        max_acc_snap = run_df["acc_snap"].max()

        # training window mean (between rollout_min and rollout_max inclusive)
        window_mask = (run_df["step"] >= rollout_min) & (run_df["step"] <= rollout_max)
        mean_mse_window = run_df.loc[window_mask, "mse"].mean() if window_mask.any() else np.nan
        mean_acc_window = run_df.loc[window_mask, "acc"].mean() if window_mask.any() else np.nan
        mean_acc_snap_window = run_df.loc[window_mask, "acc_snap"].mean() if window_mask.any() else np.nan

        # tail mean (last 10% of rollout)
        tail_start = int(0.9 * len(run_df))
        tail_mask = run_df["step"] >= tail_start
        mean_mse_tail = run_df.loc[tail_mask, "mse"].mean() if tail_mask.any() else np.nan
        mean_acc_tail = run_df.loc[tail_mask, "acc"].mean() if tail_mask.any() else np.nan
        mean_acc_snap_tail = run_df.loc[tail_mask, "acc_snap"].mean() if tail_mask.any() else np.nan

        # did this run ever reach perfect snapped accuracy (1.0)?
        reached_mask = run_df["acc_snap"] >= 1.0
        reached_acc_snap_1 = bool(reached_mask.any())
        step_reached_acc_snap_1 = int(run_df.loc[reached_mask, "step"].iloc[0]) if reached_acc_snap_1 else np.nan
        # did it maintain perfect snapped accuracy from that point to the end?
        maintained_acc_snap_after_reach = False
        if reached_acc_snap_1:
            post = run_df[run_df["step"] >= step_reached_acc_snap_1]
            maintained_acc_snap_after_reach = bool((post["acc_snap"] >= 1.0).all())

        # did this run ever reach perfect (non-snapped) acc == 1.0?
        reached_mask_acc = run_df["acc"] >= 1.0
        reached_acc_1 = bool(reached_mask_acc.any())
        step_reached_acc_1 = int(run_df.loc[reached_mask_acc, "step"].iloc[0]) if reached_acc_1 else np.nan
        # did it maintain perfect acc from that point to the end?
        maintained_acc_after_reach = False
        if reached_acc_1:
            post_acc = run_df[run_df["step"] >= step_reached_acc_1]
            maintained_acc_after_reach = bool((post_acc["acc"] >= 1.0).all())

        per_run = {
            "run": run_idx + 1,
            "max_mse": max_mse,
            "max_acc": max_acc,
            "max_acc_snap": max_acc_snap,
            "mean_mse_window": mean_mse_window,
            "mean_acc_window": mean_acc_window,
            "mean_acc_snap_window": mean_acc_snap_window,
            "mean_mse_tail": mean_mse_tail,
            "mean_acc_tail": mean_acc_tail,
            "mean_acc_snap_tail": mean_acc_snap_tail,
            "reached_acc_1": reached_acc_1,
            "step_reached_acc_1": step_reached_acc_1,
            "maintained_acc_after_reach": maintained_acc_after_reach,
            "reached_acc_snap_1": reached_acc_snap_1,
            "step_reached_acc_snap_1": step_reached_acc_snap_1,
            "maintained_acc_snap_after_reach": maintained_acc_snap_after_reach,
        }
        per_run_summaries.append(per_run)

        final_row = run_df.iloc[-1]
        print(f"  Run {run_idx+1}/{N_EVAL_RUNS} - final step {int(final_row['step'])}: mse={final_row['mse']:.6e}, acc={final_row['acc']:.4f}, acc_snap={final_row['acc_snap']:.4f}")
        print(f"    -> run stats: max_mse={max_mse:.6e}, mean_mse_window={mean_mse_window:.6e}, mean_mse_tail={mean_mse_tail:.6e}")
        if reached_acc_snap_1:
            print(f"    -> reached acc_snap=1.0 at step {step_reached_acc_snap_1}, maintained to end: {maintained_acc_snap_after_reach}")
        else:
            print("    -> did NOT reach acc_snap=1.0 during rollout")

        # save individual run CSV
        run_df.to_csv(os.path.join(subdir, f"eval_metrics_run{run_idx+1}.csv"), index=False)

        # cleanup
        del x, x_rgba, seed, target_b, run_df, metrics
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    if not runs_dfs:
        print(f"No runs collected for {subdir}")
        return None

    # per-step mean across runs (optional)
    combined = pd.concat(runs_dfs, keys=range(len(runs_dfs)), names=["run", "row"]).reset_index()
    if "row" in combined.columns:
        combined = combined.drop(columns=["row"])
    mean_by_step = combined.groupby("step").mean().reset_index()
    mean_by_step.to_csv(os.path.join(subdir, "eval_metrics_mean_by_step.csv"), index=False)
    print(f"Saved mean-by-step metrics to {os.path.join(subdir, 'eval_metrics_mean_by_step.csv')}")

    # save per-run summary and print checkpoint-level means
    runs_summary_df = pd.DataFrame(per_run_summaries)
    runs_summary_df.to_csv(os.path.join(subdir, "eval_runs_summary.csv"), index=False)

    # numeric means
    mean_summary = runs_summary_df.mean(numeric_only=True) if len(runs_summary_df) > 0 else pd.Series()

    # counts for acc_snap == 1.0 reaches / maintained-to-end
    total_runs = len(runs_summary_df)
    reached_count = int(runs_summary_df['reached_acc_snap_1'].sum()) if 'reached_acc_snap_1' in runs_summary_df.columns else 0
    maintained_count = int(runs_summary_df['maintained_acc_snap_after_reach'].sum()) if 'maintained_acc_snap_after_reach' in runs_summary_df.columns else 0
    # counts for acc == 1.0 reaches / maintained-to-end
    reached_count_acc = int(runs_summary_df['reached_acc_1'].sum()) if 'reached_acc_1' in runs_summary_df.columns else 0
    maintained_count_acc = int(runs_summary_df['maintained_acc_after_reach'].sum()) if 'maintained_acc_after_reach' in runs_summary_df.columns else 0

    print(f"\nAggregate means over {total_runs} runs for checkpoint {os.path.basename(checkpoint_path)}:")
    if not mean_summary.empty:
        print(f"  max_mse (mean over runs)          : {mean_summary.get('max_mse', np.nan):.6e}")
        print(f"  mean_mse_window (steps {rollout_min}-{rollout_max}): {mean_summary.get('mean_mse_window', np.nan):.6e}")
        print(f"  mean_mse_tail (last 10%)          : {mean_summary.get('mean_mse_tail', np.nan):.6e}")
        print(f"  max_acc (mean over runs)          : {mean_summary.get('max_acc', np.nan):.4f}")
        print(f"  mean_acc_window                    : {mean_summary.get('mean_acc_window', np.nan):.4f}")
        print(f"  mean_acc_tail                      : {mean_summary.get('mean_acc_tail', np.nan):.4f}")
        print(f"  max_acc_snap (mean over runs)     : {mean_summary.get('max_acc_snap', np.nan):.4f}")
        print(f"  mean_acc_snap_window               : {mean_summary.get('mean_acc_snap_window', np.nan):.4f}")
        print(f"  mean_acc_snap_tail                 : {mean_summary.get('mean_acc_snap_tail', np.nan):.4f}")
    else:
        print("  (no numeric run summaries available)")

    # print counts of perfect snapped-accuracy behaviour
    print(f"  runs reaching acc_snap == 1.0      : {reached_count}/{total_runs}")
    print(f"  runs maintaining to end after reach: {maintained_count}/{total_runs}")
    # print counts of perfect non-snapped accuracy behaviour
    print(f"  runs reaching acc == 1.0           : {reached_count_acc}/{total_runs}")
    print(f"  runs maintaining acc to end       : {maintained_count_acc}/{total_runs}")

    # cleanup
    del model, trainer, target, palette, runs_dfs, combined, mean_by_step, runs_summary_df
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

def main():
    tasks = []
    for first_level in os.listdir(RESULTS_DIR):
        first_path = os.path.join(RESULTS_DIR, first_level)
        if os.path.isdir(first_path):
            for second_level in os.listdir(first_path):
                second_path = os.path.join(first_path, second_level)
                if os.path.isdir(second_path):
                    tasks.append(second_path)

    for path in tasks:
        p = multiprocessing.Process(target=evaluate_checkpoint, args=(path,))
        p.start()
        p.join()

if __name__ == "__main__":
    main()