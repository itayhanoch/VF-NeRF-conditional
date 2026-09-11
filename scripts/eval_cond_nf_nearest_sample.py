#!/usr/bin/env python
"""Nearest-sample error of a trained conditional NF -- the flow run the other way.

`eval_cond_nf_likelihood.py` asks "how likely is the real answer?" (log_prob of the
true (point, direction) under the pixel's DINO feature). This asks "does the flow
GENERATE the real answer?": for each sampled pixel, draw K candidates from
P((point, direction) | feature) and measure how close the closest one lands to the
frozen-NeRF surface point along that pixel's ray, and how far its direction is from
the ray direction. Densities are hard to read (a slightly-off but very peaked flow
scores terribly even when a sample would land a hair away); a distance in scene
units and an angle in degrees are not.

Two per-pixel variants are recorded, each with identical statistics:
  min   -- the best of the K samples (distance and angle minimised separately);
           "did any sample land near the truth?"
  top   -- the single sample the flow itself ranks highest by log_prob, i.e. what
           the explorer / gradio app would render first; "is its best guess right?"

Uses the same `sample_batch` / DINO cache / depth render as the trainer and the
likelihood eval, so the pixels, features and 3-D targets are exactly the training
distribution. `--depth-range` repeats every statistic over only the samples whose
NeRF depth is in range (the bad-depth outliers removed), as in the likelihood eval.

Cost: num_pixels * num_samples flow samples (40k x 100 = 4M), drawn in chunks of
`pixel_chunk * num_samples`; the flow is a few small MLPs on a 6-D input, so this
is a minute or two per split on a T4. The NeRF depth render is only num_pixels rays.

Example:
    python scripts/eval_cond_nf_nearest_sample.py \\
        --nerf-config outputs/bonsai/nerfacto/TIMESTAMP/config.yml \\
        --scene-dir data/mipnerf360/bonsai \\
        --cond-nf-checkpoint checkpoints/conditional_nf/bonsai/latest.pt \\
        --output-path eval/bonsai_cond_nf_nearest.json --depth-range 0.3 3.0
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from nerfstudio.utils.dino_features import DinoExtractor
from nerfstudio.utils.eval_utils import eval_setup
from scripts.eval_cond_nf_likelihood import basic_stats, load_conditional_nf
from scripts.train_conditional_nf import build_training_cameras, precompute_dino_cache, sample_batch


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--nerf-config", type=Path, required=True, help="Trained frozen nerfacto config.yml (supplies the depth AND the dataparser frame)")
    p.add_argument("--scene-dir", type=Path, required=True, help="Scene dir containing transforms.json (the same data the NeRF was trained on)")
    p.add_argument("--cond-nf-checkpoint", type=Path, required=True, help="Conditional-NF .pt (architecture is read from the checkpoint)")
    p.add_argument("--output-path", type=Path, required=True, help="Where to write the results JSON")
    p.add_argument("--splits", nargs="+", default=["train", "test"], help='Dataparser splits to evaluate ("train" -> i_train, "test"/"val" -> the held-out remainder)')
    p.add_argument("--num-pixels", type=int, default=40000, help="Random (image, sub-pixel) draws per split")
    p.add_argument("--num-samples", type=int, default=100, help="K candidates drawn from the flow per pixel")
    p.add_argument("--ray-batch", type=int, default=4096, help="Rays per frozen-NeRF depth render")
    p.add_argument("--pixel-chunk", type=int, default=512, help="Pixels per flow call (pixel_chunk * num_samples rows through the flow at once)")
    p.add_argument("--depth-range", type=float, nargs=2, default=None, metavar=("LO", "HI"), help="Also report every statistic over only the pixels whose rendered depth is within [LO, HI]")
    p.add_argument("--hit-frac", type=float, default=0.05, help="A pixel is a 'hit' when the variant's distance is below hit_frac * backoff, where backoff is the median camera distance from the scene centre (the explorer's constant backoff)")
    p.add_argument("--dino-cache-dir", type=Path, default=None, help="Defaults to <scene-dir>/dino_cache for the train split and <scene-dir>/dino_cache_<split> otherwise")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-samples", dest="save_samples", action="store_true", default=True, help="Also write <output-path stem>_samples.npz with the per-pixel arrays (default: on)")
    p.add_argument("--no-save-samples", dest="save_samples", action="store_false")
    return p.parse_args()


def default_backoff_distance(cameras) -> float:
    """Median camera distance from the mean camera centre -- the explorer's and
    gradio app's constant backoff (copied: app/gradio_app.py imports gradio)."""
    centers = cameras.camera_to_worlds[..., :3, 3]
    scene_center = centers.mean(dim=0)
    return float((centers - scene_center).norm(dim=-1).median())


VARIANTS = ("min", "top")


def sample_errors(field, conditions, points, directions, num_samples, pixel_chunk):
    """Per-pixel nearest-sample errors for one batch of targets.

    Returns dict of [B] tensors: {min,top}_dist / {min,top}_angle_deg, where
    `min` is the closest of the K samples (distance and angle separately) and
    `top` is the sample with the highest log_prob under its own condition.
    Non-finite samples are ignored (distance/angle -> inf, log_prob -> -inf).
    """
    out = {f"{v}_{m}": [] for v in VARIANTS for m in ("dist", "angle_deg")}
    for i in range(0, conditions.shape[0], pixel_chunk):
        cond = conditions[i:i + pixel_chunk]
        P, d = points[i:i + pixel_chunk], directions[i:i + pixel_chunk]
        B = cond.shape[0]
        # cond_prior=True makes the flow draw exactly context.shape[0] samples, so
        # the K-fold tiling is what sets the sample count.
        ctx = cond.repeat_interleave(num_samples, dim=0)                 # [B*K, C]
        s = field.sample(num_samples=B * num_samples, context=ctx)      # [B*K, 6]
        lp = field.log_prob(s, ctx).reshape(B, num_samples)             # [B, K]
        s = s.reshape(B, num_samples, 6)
        ok = torch.isfinite(s).all(dim=-1) & torch.isfinite(lp)         # [B, K]

        dist = (s[..., :3] - P[:, None, :]).norm(dim=-1)                # [B, K]
        sd = s[..., 3:]
        sd = sd / sd.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        cosang = (sd * d[:, None, :]).sum(dim=-1).clamp(-1.0, 1.0)
        angle = torch.rad2deg(torch.acos(cosang))                       # [B, K]

        inf = torch.full_like(dist, float("inf"))
        dist = torch.where(ok, dist, inf)
        angle = torch.where(ok, angle, inf)
        lp = torch.where(ok, lp, torch.full_like(lp, float("-inf")))

        top = lp.argmax(dim=1)                                          # [B]
        ar = torch.arange(B, device=s.device)
        out["min_dist"].append(dist.min(dim=1).values)
        out["min_angle_deg"].append(angle.min(dim=1).values)
        out["top_dist"].append(dist[ar, top])
        out["top_angle_deg"].append(angle[ar, top])
    return {k: torch.cat(v) for k, v in out.items()}


def variant_stats(arrays, mask, hit_threshold):
    """The per-variant statistics block over the pixels selected by `mask`."""
    out = {}
    for v in VARIANTS:
        dist = arrays[f"{v}_dist"][mask]
        ang = arrays[f"{v}_angle_deg"][mask]
        finite = np.isfinite(dist)
        out[v] = {
            "dist": basic_stats(dist),
            "angle_deg": basic_stats(ang),
            "n_hit": int((dist[finite] < hit_threshold).sum()),
            "frac_hit": float((dist[finite] < hit_threshold).mean()) if finite.any() else float("nan"),
            "n_nonfinite": int((~finite).sum()),
        }
    return out


def evaluate_split(split, args, config, nerf_model, field, extractor, backoff, device):
    cameras, image_filenames = build_training_cameras(
        args.scene_dir, config.pipeline.datamanager.dataparser, split=split
    )
    cameras = cameras.to(device)
    if args.dino_cache_dir is not None:
        cache_dir = args.dino_cache_dir if split == "train" else args.dino_cache_dir.with_name(f"{args.dino_cache_dir.name}_{split}")
    else:
        cache_dir = args.scene_dir / ("dino_cache" if split == "train" else f"dino_cache_{split}")
    print(f"[{split}] {len(image_filenames)} images | DINO cache -> {cache_dir}", flush=True)
    nerf_model.cpu()
    torch.cuda.empty_cache()
    dino_caches = precompute_dino_cache(image_filenames, cache_dir, extractor)
    nerf_model.to(device)

    torch.manual_seed(args.seed)
    keep = {k: [] for k in ("depth", "cam_idx")}
    keep.update({f"{v}_{m}": [] for v in VARIANTS for m in ("dist", "angle_deg")})
    n_batches = math.ceil(args.num_pixels / args.ray_batch)
    done = 0
    for b in range(n_batches):
        bs = min(args.ray_batch, args.num_pixels - done)
        with torch.no_grad():
            ray_bundle, conditions = sample_batch(cameras, dino_caches, bs, device)
            depth = nerf_model(ray_bundle)["depth"].reshape(-1)                      # [B]
            points = ray_bundle.origins + ray_bundle.directions * depth[:, None]     # [B, 3]
            errs = sample_errors(field, conditions, points, ray_bundle.directions,
                                 args.num_samples, args.pixel_chunk)
        keep["depth"].append(depth.float().cpu().numpy())
        keep["cam_idx"].append(ray_bundle.camera_indices.reshape(-1).cpu().numpy())
        for k, v in errs.items():
            keep[k].append(v.float().cpu().numpy())
        done += bs
        md = keep["min_dist"][-1]
        print(f"[{split}] {done}/{args.num_pixels} pixels | batch median min-dist "
              f"{float(np.median(md[np.isfinite(md)])) if np.isfinite(md).any() else float('nan'):.4f}"
              f" | median min-angle {float(np.median(keep['min_angle_deg'][-1])):.2f} deg", flush=True)

    arrays = {k: np.concatenate(v) for k, v in keep.items()}
    depth = arrays["depth"].astype(np.float64)
    hit_threshold = args.hit_frac * backoff
    all_mask = np.ones(len(depth), dtype=bool)
    out = {
        "split": split,
        "n_pixels": int(len(depth)),
        "n_images": len(image_filenames),
        "num_samples": args.num_samples,
        "backoff": backoff,
        "hit_frac": args.hit_frac,
        "hit_threshold": hit_threshold,
        "variants": variant_stats(arrays, all_mask, hit_threshold),
    }
    if args.depth_range is not None:
        lo, hi = args.depth_range
        kept = (depth >= lo) & (depth <= hi)
        out["depth_range"] = [lo, hi]
        out["depth_filtered"] = {
            "n_kept": int(kept.sum()),
            "frac_kept": float(kept.mean()),
            "variants": variant_stats(arrays, kept, hit_threshold) if kept.any() else {},
        }
    arrays = {k: (v.astype(np.int32) if k == "cam_idx" else v.astype(np.float32)) for k, v in arrays.items()}
    return out, arrays


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

    train_cameras, _ = build_training_cameras(args.scene_dir, config.pipeline.datamanager.dataparser, split="train")
    backoff = default_backoff_distance(train_cameras)
    print(f"backoff (median camera distance from scene centre) = {backoff:.4f}; "
          f"hit threshold = {args.hit_frac:g} * backoff = {args.hit_frac * backoff:.4f}", flush=True)

    results, samples = {}, {}
    for s in args.splits:
        results[s], samples[s] = evaluate_split(
            s, args, config, nerf_model, field, extractor, backoff, device)

    summary = {
        "scene_dir": str(args.scene_dir),
        "nerf_config": str(args.nerf_config),
        "cond_nf_checkpoint": str(args.cond_nf_checkpoint),
        "cond_nf_step": step,
        "dino_model": dino_model_name,
        "num_pixels": args.num_pixels,
        "num_samples": args.num_samples,
        "hit_frac": args.hit_frac,
        "depth_range": list(args.depth_range) if args.depth_range is not None else None,
        "seed": args.seed,
        "splits": results,
    }
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(summary, indent=2))

    if args.save_samples:
        npz = args.output_path.with_name(args.output_path.stem + "_samples.npz")
        np.savez_compressed(npz, **{f"{s}_{k}": v for s, d in samples.items() for k, v in d.items()})
        summary["samples_npz"] = str(npz)
        args.output_path.write_text(json.dumps(summary, indent=2))
        print(f"raw per-pixel arrays -> {npz}", flush=True)

    for s, r in results.items():
        for v in VARIANTS:
            st = r["variants"][v]
            print(f"{s:>5} {v:<4}: dist mean {st['dist']['mean']:.4f} var {st['dist']['var']:.4f} "
                  f"median {st['dist']['median']:.4f} | hits (< {r['hit_threshold']:.4f}) "
                  f"{st['n_hit']} / {r['n_pixels']} ({100 * st['frac_hit']:.1f}%) | angle mean "
                  f"{st['angle_deg']['mean']:.2f} var {st['angle_deg']['var']:.2f} "
                  f"median {st['angle_deg']['median']:.2f} deg", flush=True)
            f = r.get("depth_filtered", {})
            if f.get("variants"):
                ft = f["variants"][v]
                print(f"      depth in {r['depth_range']} (kept {f['n_kept']}, {100 * f['frac_kept']:.1f}%): "
                      f"dist mean {ft['dist']['mean']:.4f} var {ft['dist']['var']:.4f} "
                      f"median {ft['dist']['median']:.4f} | hits {ft['n_hit']} ({100 * ft['frac_hit']:.1f}%) "
                      f"| angle mean {ft['angle_deg']['mean']:.2f} var {ft['angle_deg']['var']:.2f} "
                      f"median {ft['angle_deg']['median']:.2f} deg", flush=True)
    print(f"-> {args.output_path}", flush=True)


if __name__ == "__main__":
    main()
