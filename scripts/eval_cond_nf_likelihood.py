#!/usr/bin/env python
"""Held-out log-likelihood of a trained conditional NF.

The conditional-NF trainer optimises `-log_prob` over pixels of the TRAIN split
only, so its loss curve says nothing about generalization. This script evaluates
the exact same quantity on both splits -- the train frames the flow was fitted on
and nerfstudio's held-out frames (`train_split_fraction`, 0.9 by default, leaves
~10% of frames unseen by both the NeRF and the flow) -- and reports the gap.

Deliberately reuses `train_conditional_nf`'s own `build_training_cameras`,
`precompute_dino_cache` and `sample_batch`, so what is measured here IS the
training objective: the same continuous sub-pixel sampling, the same DINO patch
lookup for the condition, the same frozen-NeRF depth render for the 3-D target.
The only differences are `field.eval()` and no backward pass.

Example:
    python scripts/eval_cond_nf_likelihood.py \\
        --nerf-config outputs/bonsai/nerfacto/TIMESTAMP/config.yml \\
        --scene-dir data/mipnerf360/bonsai \\
        --cond-nf-checkpoint checkpoints/conditional_nf/bonsai/latest.pt \\
        --output-path eval/bonsai_cond_nf_ll.json
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from nerfstudio.fields.nf_field import ConditionalNFField
from nerfstudio.utils.dino_features import DinoExtractor
from nerfstudio.utils.eval_utils import eval_setup
from scripts.train_conditional_nf import build_training_cameras, precompute_dino_cache, sample_batch


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--nerf-config", type=Path, required=True, help="Trained frozen nerfacto config.yml (supplies the depth AND the dataparser frame)")
    p.add_argument("--scene-dir", type=Path, required=True, help="Scene dir containing transforms.json (the same data the NeRF was trained on)")
    p.add_argument("--cond-nf-checkpoint", type=Path, required=True, help="Conditional-NF .pt (architecture is read from the checkpoint)")
    p.add_argument("--output-path", type=Path, required=True, help="Where to write the results JSON")
    p.add_argument("--splits", nargs="+", default=["train", "test"], help='Dataparser splits to evaluate ("train" -> i_train, "test"/"val" -> the held-out remainder)')
    p.add_argument("--num-batches", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=4096, help="Matches the trainer's default, so a train-split mean here is directly comparable to -loss at the end of training")
    p.add_argument("--dino-cache-dir", type=Path, default=None, help="Defaults to <scene-dir>/dino_cache for the train split and <scene-dir>/dino_cache_<split> otherwise -- one cache per split, since the memmap's shape is keyed on the image COUNT")
    p.add_argument("--seed", type=int, default=0, help="Same seed for every split, so the splits see the same pixel-sampling randomness")
    p.add_argument("--worst-k", type=int, default=20, help="How many of the lowest-log_prob samples to dump in full (with their rendered depth and source image) -- these are what a huge std is usually made of")
    p.add_argument("--save-samples", dest="save_samples", action="store_true", default=True, help="Also write <output-path stem>_samples.npz with the raw per-sample log_prob/depth/camera index for every split, so the report can be re-analysed without re-running (default: on)")
    p.add_argument("--no-save-samples", dest="save_samples", action="store_false")
    return p.parse_args()


# --- statistics ---------------------------------------------------------------
# Pure numpy and deliberately free of any torch/nerfstudio dependency, so it can be
# exercised on synthetic data without a GPU.

QUANTILES = (0.0, 0.01, 0.1, 1.0, 5.0, 25.0, 50.0, 75.0, 95.0, 99.0, 99.9, 99.99, 100.0)
TAIL_FRACTIONS = (0.001, 0.01, 0.1, 1.0)   # percent


def spearman(a, b):
    """Rank correlation, without scipy: Pearson on the ranks. Ties get ordinal
    (not averaged) ranks, which is immaterial for continuous log-densities."""
    if len(a) < 2:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = math.sqrt(float((ra ** 2).sum()) * float((rb ** 2).sum()))
    return float((ra * rb).sum() / denom) if denom else float("nan")


def tail_contribution(vals, mean, var):
    """How much of the mean and of the variance the lowest-p% of samples carry.

    A log-density has no floor, so a single sample landing far off the manifold the
    flow was fitted on can own essentially all of the variance -- exactly what
    happened in the first counter report (one sample in 81,920 at ~ -13,400 gave
    std 47.5 around a mean of 9.67). Reporting this makes that visible instead of
    leaving it to be inferred from batch_means.
    """
    n = len(vals)
    order = np.sort(vals)
    total_sq_dev = float(((vals - mean) ** 2).sum())
    out = {}
    for pct in TAIL_FRACTIONS:
        k = max(1, int(round(n * pct / 100.0)))
        low = order[:k]
        # what the mean would be with these dropped, vs. the reported mean
        rest_mean = float(order[k:].mean()) if k < n else float("nan")
        out[f"lowest_{pct:g}pct"] = {
            "n": k,
            "threshold": float(low[-1]),
            "mean_shift": mean - rest_mean if k < n else float("nan"),
            "variance_share": (float(((low - mean) ** 2).sum()) / total_sq_dev
                               if total_sq_dev > 0 else 0.0),
        }
    return out


def describe(vals, mean, var):
    """Order statistics + tail accounting for one split's log-probs."""
    n = len(vals)
    qs = np.percentile(vals, QUANTILES)
    med = float(np.median(vals))
    q25, q75 = float(np.percentile(vals, 25)), float(np.percentile(vals, 75))
    out = {
        "median_log_prob": med,
        "iqr_log_prob": q75 - q25,
        "mad_log_prob": float(np.median(np.abs(vals - med))),
        "min_log_prob": float(vals.min()),
        "max_log_prob": float(vals.max()),
        "quantiles": {f"p{q:g}": float(v) for q, v in zip(QUANTILES, qs)},
        "tail_contribution": tail_contribution(vals, mean, var),
    }
    for pct in (0.1, 1.0):
        k = int(n * pct / 100.0)
        trimmed = np.sort(vals)[k:n - k] if k and n - 2 * k > 0 else vals
        out[f"trimmed_mean_log_prob_{pct:g}pct"] = float(trimmed.mean())
    # Histogram over the central 99.8% so a single blow-up cannot flatten every bin;
    # the clipped samples are counted separately rather than silently dropped.
    lo, hi = float(np.percentile(vals, 0.1)), float(np.percentile(vals, 99.9))
    if not (hi > lo):
        lo, hi = float(vals.min()), float(vals.min()) + 1.0
    counts, edges = np.histogram(vals, bins=60, range=(lo, hi))
    out["histogram"] = {
        "bin_edges": [float(e) for e in edges],
        "counts": [int(c) for c in counts],
        "underflow": int((vals < lo).sum()),
        "overflow": int((vals > hi).sum()),
    }
    return out


def load_conditional_nf(checkpoint_path: Path, device: torch.device):
    """Rebuild a ConditionalNFField with the architecture recorded in its own
    checkpoint (same contract as the explorer / gradio app)."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    field = ConditionalNFField(
        context_dim=ckpt["context_dim"],
        num_dims=ckpt.get("num_dims", 6),
        num_blocks=ckpt["num_blocks"],
        hidden_dim=ckpt["hidden_dim"],
        cond_prior=ckpt["cond_prior"],
        use_cond_in_coupling=True,
        use_batchnorm=ckpt["use_batchnorm"],
        reduce_dim=ckpt.get("reduce_dim"),
        reduce_divide_factor=ckpt.get("reduce_divide_factor", 8),
        device=str(device),
    )
    field.load_state_dict(ckpt["model_state"])
    field.eval()
    return field, ckpt["dino_model_name"], int(ckpt.get("step", -1))


def evaluate_split(split, args, config, nerf_model, field, extractor, device):
    """Log-likelihood statistics over `num_batches * batch_size` sampled pixels of
    one split. Non-finite values are counted and excluded rather than allowed to
    poison the mean (a single -inf would).

    Every finite sample is kept (400k float64 is ~3 MB) so the order statistics and
    the tail accounting are exact, together with the rendered depth, the source
    camera and |P| for each one -- a huge std here is normally a handful of samples
    whose frozen-NeRF depth put the 3-D point nowhere near the surfaces the flow was
    fitted on, and those three columns are what makes that visible.
    """
    cameras, image_filenames = build_training_cameras(
        args.scene_dir, config.pipeline.datamanager.dataparser, split=split
    )
    cameras = cameras.to(device)

    # One cache per split: precompute_dino_cache keys its .npy on the model name
    # and the image/grid SHAPE, not the split, so pointing two splits at the same
    # directory would make each one clobber the other's cache (the image counts
    # differ -> the shape check fails -> recompute).
    if args.dino_cache_dir is not None:
        cache_dir = args.dino_cache_dir if split == "train" else args.dino_cache_dir.with_name(f"{args.dino_cache_dir.name}_{split}")
    else:
        cache_dir = args.scene_dir / ("dino_cache" if split == "train" else f"dino_cache_{split}")
    print(f"[{split}] {len(image_filenames)} images | DINO cache -> {cache_dir}", flush=True)

    # The DINO precompute peaks GPU memory (no flash-attention on torch < 2.0);
    # park the frozen NeRF on the CPU while it runs, exactly as the trainer does.
    nerf_model.cpu()
    torch.cuda.empty_cache()
    dino_caches = precompute_dino_cache(image_filenames, cache_dir, extractor)
    nerf_model.to(device)

    torch.manual_seed(args.seed)
    total = total_sq = 0.0
    n = n_nonfinite = 0
    batch_means, batch_mins, batch_maxes = [], [], []
    keep = {"log_prob": [], "depth": [], "cam_idx": [], "point_norm": []}
    for b in range(args.num_batches):
        with torch.no_grad():
            ray_bundle, conditions = sample_batch(cameras, dino_caches, args.batch_size, device)
            outputs = nerf_model(ray_bundle)
            points = ray_bundle.origins + ray_bundle.directions * outputs["depth"]
            x = torch.cat([points, ray_bundle.directions], dim=-1)
            log_prob = field.log_prob(x, conditions).reshape(-1)

        finite = torch.isfinite(log_prob)
        n_nonfinite += int((~finite).sum())
        vals = log_prob[finite].double()
        if vals.numel():
            total += float(vals.sum())
            total_sq += float((vals ** 2).sum())
            n += int(vals.numel())
            batch_means.append(float(vals.mean()))
            batch_mins.append(float(vals.min()))
            batch_maxes.append(float(vals.max()))
            keep["log_prob"].append(vals.cpu().numpy())
            keep["depth"].append(outputs["depth"].reshape(-1)[finite].float().cpu().numpy())
            keep["cam_idx"].append(ray_bundle.camera_indices.reshape(-1)[finite].cpu().numpy())
            keep["point_norm"].append(points.reshape(-1, 3)[finite].norm(dim=-1).float().cpu().numpy())
        print(f"[{split}] batch {b + 1}/{args.num_batches} | mean log-prob "
              f"{batch_means[-1] if batch_means else float('nan'):.4f}"
              f" | min {batch_mins[-1] if batch_mins else float('nan'):.4f}", flush=True)

    mean = total / n if n else math.nan
    # population variance from the running sums; clamped because catastrophic
    # cancellation can push it a hair below zero for a near-constant sample.
    var = max(total_sq / n - mean ** 2, 0.0) if n else math.nan
    out = {
        "split": split,
        "mean_log_prob": mean,
        "std_log_prob": math.sqrt(var) if n else math.nan,
        "var_log_prob": var,
        "n_samples": n,
        "n_nonfinite": n_nonfinite,
        "n_images": len(image_filenames),
        "batch_means": batch_means,
        "batch_mins": batch_mins,
        "batch_maxes": batch_maxes,
    }
    if not n:
        return out, None

    lp = np.concatenate(keep["log_prob"])
    depth = np.concatenate(keep["depth"]).astype(np.float64)
    cam = np.concatenate(keep["cam_idx"]).astype(np.int64)
    pnorm = np.concatenate(keep["point_norm"]).astype(np.float64)
    out.update(describe(lp, mean, var))

    # Is a bad depth what throws a sample off the manifold? A strongly positive rank
    # correlation (low log-prob <-> extreme depth) says yes; ~0 says look elsewhere.
    out["depth_stats"] = {f"p{q:g}": float(v)
                          for q, v in zip(QUANTILES, np.percentile(depth, QUANTILES))}
    out["spearman_log_prob_vs_depth"] = spearman(lp, depth)
    out["spearman_log_prob_vs_point_norm"] = spearman(lp, pnorm)

    worst = np.argsort(lp)[: max(0, args.worst_k)]
    out["worst_samples"] = [
        {"log_prob": float(lp[i]), "depth": float(depth[i]), "point_norm": float(pnorm[i]),
         "image_index": int(cam[i]), "image": Path(str(image_filenames[cam[i]])).name}
        for i in worst
    ]

    # Per source image: pins the blow-ups on specific frames if they cluster.
    per_image = []
    for i in range(len(image_filenames)):
        m = cam == i
        if not m.any():
            continue
        per_image.append({
            "image_index": i,
            "image": Path(str(image_filenames[i])).name,
            "n": int(m.sum()),
            "mean": float(lp[m].mean()),
            "median": float(np.median(lp[m])),
            "min": float(lp[m].min()),
        })
    out["per_image"] = per_image
    return out, {"log_prob": lp.astype(np.float32), "depth": depth.astype(np.float32),
                 "cam_idx": cam.astype(np.int32), "point_norm": pnorm.astype(np.float32)}


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading frozen NeRF from {args.nerf_config} ...", flush=True)
    config, pipeline, _, _ = eval_setup(args.nerf_config, test_mode="inference")
    nerf_model = pipeline.model.to(device).eval()
    for p in nerf_model.parameters():
        p.requires_grad_(False)

    field, dino_model_name, step = load_conditional_nf(args.cond_nf_checkpoint, device)
    print(f"conditional-NF checkpoint step = {step}", flush=True)
    extractor = DinoExtractor(model_name=dino_model_name, device=str(device))

    results, samples = {}, {}
    for s in args.splits:
        results[s], samples[s] = evaluate_split(
            s, args, config, nerf_model, field, extractor, device)

    summary = {
        "scene_dir": str(args.scene_dir),
        "nerf_config": str(args.nerf_config),
        "cond_nf_checkpoint": str(args.cond_nf_checkpoint),
        "cond_nf_step": step,
        "dino_model": dino_model_name,
        "batch_size": args.batch_size,
        "num_batches": args.num_batches,
        "seed": args.seed,
        "splits": results,
    }
    # Positive gap = the flow assigns higher density to what it was fitted on, i.e.
    # the usual overfitting direction. Negative would mean the splits are crossed.
    # `train_minus_test` keeps its original mean-based definition so older reports
    # stay comparable; the median/trimmed versions beside it are the ones to read,
    # since a single catastrophic sample can move a mean by a sixth of a nat.
    if "train" in results and "test" in results:
        tr, te = results["train"], results["test"]
        summary["train_minus_test"] = tr["mean_log_prob"] - te["mean_log_prob"]
        for key, name in (("median_log_prob", "median_train_minus_test"),
                          ("trimmed_mean_log_prob_0.1pct", "trimmed_train_minus_test")):
            if key in tr and key in te:
                summary[name] = tr[key] - te[key]

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(summary, indent=2))

    if args.save_samples and any(v is not None for v in samples.values()):
        npz = args.output_path.with_name(args.output_path.stem + "_samples.npz")
        np.savez_compressed(npz, **{f"{s}_{k}": v for s, d in samples.items()
                                    if d is not None for k, v in d.items()})
        summary["samples_npz"] = str(npz)
        args.output_path.write_text(json.dumps(summary, indent=2))
        print(f"raw per-sample arrays -> {npz}", flush=True)

    for s, r in results.items():
        print(f"{s:>5}: median {r.get('median_log_prob', float('nan')):.4f}  "
              f"trimmed {r.get('trimmed_mean_log_prob_0.1pct', float('nan')):.4f}  "
              f"| raw mean {r['mean_log_prob']:.4f} (std {r['std_log_prob']:.4f})  "
              f"| {r['n_samples']} samples from {r['n_images']} images, "
              f"{r['n_nonfinite']} non-finite", flush=True)
        worst_tail = r.get("tail_contribution", {}).get("lowest_0.001pct")
        if worst_tail:
            print(f"       lowest {worst_tail['n']} sample(s) (<= "
                  f"{worst_tail['threshold']:.1f}) carry "
                  f"{100 * worst_tail['variance_share']:.1f}% of the variance and "
                  f"{worst_tail['mean_shift']:+.4f} of the mean", flush=True)
    for name in ("train_minus_test", "median_train_minus_test", "trimmed_train_minus_test"):
        if name in summary:
            print(f"{name} = {summary[name]:.4f}", flush=True)
    print(f"-> {args.output_path}", flush=True)


if __name__ == "__main__":
    main()
