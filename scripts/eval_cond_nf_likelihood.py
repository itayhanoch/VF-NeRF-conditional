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
    return p.parse_args()


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
    """Mean/std log-likelihood over `num_batches * batch_size` sampled pixels of
    one split. Non-finite values are counted and excluded rather than allowed to
    poison the mean (a single -inf would)."""
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
    batch_means = []
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
        print(f"[{split}] batch {b + 1}/{args.num_batches} | mean log-prob "
              f"{batch_means[-1] if batch_means else float('nan'):.4f}", flush=True)

    mean = total / n if n else math.nan
    # population variance from the running sums; clamped because catastrophic
    # cancellation can push it a hair below zero for a near-constant sample.
    var = max(total_sq / n - mean ** 2, 0.0) if n else math.nan
    return {
        "split": split,
        "mean_log_prob": mean,
        "std_log_prob": math.sqrt(var) if n else math.nan,
        "var_log_prob": var,
        "n_samples": n,
        "n_nonfinite": n_nonfinite,
        "n_images": len(image_filenames),
        "batch_means": batch_means,
    }


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

    results = {s: evaluate_split(s, args, config, nerf_model, field, extractor, device)
               for s in args.splits}

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
    if "train" in results and "test" in results:
        summary["train_minus_test"] = results["train"]["mean_log_prob"] - results["test"]["mean_log_prob"]

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(summary, indent=2))
    for s, r in results.items():
        print(f"{s:>5}: mean log-prob {r['mean_log_prob']:.4f} "
              f"(std {r['std_log_prob']:.4f}, {r['n_samples']} samples from "
              f"{r['n_images']} images, {r['n_nonfinite']} non-finite)", flush=True)
    if "train_minus_test" in summary:
        print(f"train - test = {summary['train_minus_test']:.4f}", flush=True)
    print(f"-> {args.output_path}", flush=True)


if __name__ == "__main__":
    main()
