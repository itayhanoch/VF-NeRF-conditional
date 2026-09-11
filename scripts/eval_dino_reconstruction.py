#!/usr/bin/env python
"""How faithfully does a frozen NeRF reproduce DINOv2 features?

The whole conditional-NF fork rests on an assumption that is never actually
checked: that a NeRF render is DINO-comparable to the real photograph, so a
feature extracted from a rendered novel view means the same thing as one
extracted from a training frame. PSNR/SSIM/LPIPS (the notebook's `ns-eval` cell)
measure pixels, not features, and say nothing about it.

This measures it directly. For each frame of a split, render the NeRF from that
frame's own camera at native resolution, extract the DINOv2 patch grid of both
the render and the ground-truth image, and take the per-patch cosine similarity
between them. Both grids come from the same `extract_patch_grid` path (identical
padding / normalization / resolution), so patch (i, j) of one corresponds
exactly to patch (i, j) of the other.

A cosine value on its own says little -- every patch of one scene looks alike to
DINO -- so the same frame also gives the FLOOR: each ground-truth patch against a
random (shuffled) patch of the render, i.e. the similarity two unrelated patches
of the same scene reach by chance. The gap between the two ("separation") is the
scale on which any other DINO-similarity number for this scene should be read;
if they overlap, DINO on NeRF renders cannot tell right from wrong here.

Reported per split: the mean and variance OVER IMAGES -- each frame contributes
its own mean patch similarity, so the variance says how much fidelity varies from
viewpoint to viewpoint. "train" is the split the NeRF was fitted on; "test" is
nerfstudio's held-out remainder (`train_split_fraction`, 0.9 by default).

Example:
    python scripts/eval_dino_reconstruction.py \\
        --nerf-config outputs/bonsai/nerfacto/TIMESTAMP/config.yml \\
        --scene-dir data/mipnerf360/bonsai \\
        --output-path eval/bonsai_dino_recon.json --max-frames 40
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from nerfstudio.utils.dino_features import DinoExtractor, load_image_chw_01
from nerfstudio.utils.eval_utils import eval_setup


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--nerf-config", type=Path, required=True, help="Trained frozen nerfacto config.yml")
    p.add_argument("--scene-dir", type=Path, required=True, help="Scene dir containing transforms.json (the same data the NeRF was trained on)")
    p.add_argument("--output-path", type=Path, required=True, help="Where to write the results JSON")
    p.add_argument("--splits", nargs="+", default=["train", "test"], help='Dataparser splits ("train" -> i_train, "test"/"val" -> the held-out remainder)')
    p.add_argument("--max-frames", type=int, default=40, help="Evenly-spaced cap on frames per split (0 = every frame). Full-resolution rendering is the cost here, a few seconds per frame")
    p.add_argument("--dino-model", default="dinov2_vits14")
    p.add_argument("--seed", type=int, default=0, help="Seed for the random-patch floor's per-frame shuffle")
    return p.parse_args()


def patch_cosine(a, b):
    """Per-patch cosine similarity of two [C, N] feature stacks -> [N]."""
    a = a / a.norm(dim=0, keepdim=True).clamp_min(1e-8)
    b = b / b.norm(dim=0, keepdim=True).clamp_min(1e-8)
    return (a * b).sum(dim=0)


def frame_indices(n: int, max_frames: int):
    """Evenly-spaced subset of 0..n-1, mirroring the dataparser's own
    `np.linspace` split idiom so the frames are spread over the trajectory
    instead of clustered at its start."""
    if max_frames <= 0 or max_frames >= n:
        return list(range(n))
    return sorted(set(np.linspace(0, n - 1, max_frames, dtype=int).tolist()))


def evaluate_split(split, args, config, nerf_model, extractor, device):
    dp = config.pipeline.datamanager.dataparser
    dp.data = Path(args.scene_dir)
    outs = dp.setup().get_dataparser_outputs(split=split)
    cameras = outs.cameras.to(device)
    indices = frame_indices(len(outs.image_filenames), args.max_frames)
    print(f"[{split}] {len(indices)} of {len(outs.image_filenames)} frames", flush=True)

    per_frame = []
    n_patches = None
    gen = torch.Generator(device="cpu")
    for k, i in enumerate(indices):
        path = Path(outs.image_filenames[i])
        gt = load_image_chw_01(path)
        h, w = int(cameras.height[i]), int(cameras.width[i])
        assert tuple(gt.shape[-2:]) == (h, w), (
            f"{path.name} is {tuple(gt.shape[-2:])} but its camera is {(h, w)} -- the "
            f"dataparser's downscale-factor and the images_N/ folder disagree, so the two "
            f"patch grids would not correspond")

        t0 = time.time()
        with torch.no_grad():
            rb = cameras.generate_rays(camera_indices=int(i), keep_shape=True)
            render = nerf_model.get_outputs_for_camera_ray_bundle(rb)["rgb"].clamp(0, 1)
        render = render.permute(2, 0, 1).contiguous().cpu()  # [3,H,W], matching load_image_chw_01
        torch.cuda.empty_cache()  # the render's peak is not needed during the DINO passes

        with torch.no_grad():
            g_gt, _ = extractor.extract_patch_grid(gt)
            g_render, _ = extractor.extract_patch_grid(render)
        g_gt, g_render = g_gt.float(), g_render.float()
        assert g_gt.shape == g_render.shape
        c = g_gt.shape[0]
        gt_flat, render_flat = g_gt.reshape(c, -1), g_render.reshape(c, -1)  # [C, Hp*Wp]
        n_patches = int(gt_flat.shape[1])

        same = patch_cosine(gt_flat, render_flat)                          # [Hp*Wp]
        # floor: the same GT patches against a shuffled render -- unrelated patches
        # of the same scene. Seeded per frame so the report is reproducible.
        gen.manual_seed(args.seed + int(i))
        perm = torch.randperm(n_patches, generator=gen).to(render_flat.device)
        rand = patch_cosine(gt_flat, render_flat[:, perm])                 # [Hp*Wp]
        per_frame.append({"index": int(i), "filename": path.name,
                          "mean_patch_cos": float(same.mean()),
                          "mean_random_patch_cos": float(rand.mean()),
                          "frac_same_beats_random": float((same > rand).float().mean())})
        del g_gt, g_render, gt_flat, render_flat, same, rand
        torch.cuda.empty_cache()
        print(f"[{split}] {k + 1}/{len(indices)} {path.name} "
              f"mean patch cos {per_frame[-1]['mean_patch_cos']:.4f} "
              f"(random patch {per_frame[-1]['mean_random_patch_cos']:.4f}) "
              f"({time.time() - t0:.1f}s)", flush=True)

    means = np.array([f["mean_patch_cos"] for f in per_frame], dtype=np.float64)
    rands = np.array([f["mean_random_patch_cos"] for f in per_frame], dtype=np.float64)
    beats = np.array([f["frac_same_beats_random"] for f in per_frame], dtype=np.float64)
    nan = math.nan
    return {
        "split": split,
        # over IMAGES: each frame contributes one number, so the variance is
        # viewpoint-to-viewpoint spread, not patch-to-patch spread.
        "mean_cos": float(means.mean()) if means.size else nan,
        "var_cos": float(means.var()) if means.size else nan,
        "std_cos": float(means.std()) if means.size else nan,
        "min_cos": float(means.min()) if means.size else nan,
        "max_cos": float(means.max()) if means.size else nan,
        # the floor: GT patch vs a random render patch of the same frame
        "mean_random_cos": float(rands.mean()) if rands.size else nan,
        "var_random_cos": float(rands.var()) if rands.size else nan,
        "std_random_cos": float(rands.std()) if rands.size else nan,
        # how far the correct patch sits above chance; ~0 means DINO cannot tell
        # a right render from a wrong one in this scene
        "separation": float(means.mean() - rands.mean()) if means.size else nan,
        "mean_frac_same_beats_random": float(beats.mean()) if beats.size else nan,
        "n_frames": len(per_frame),
        "n_frames_available": len(outs.image_filenames),
        "n_patches_per_frame": n_patches,
        "frames": per_frame,
    }


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading frozen NeRF from {args.nerf_config} ...", flush=True)
    config, pipeline, _, _ = eval_setup(args.nerf_config, test_mode="inference")
    nerf_model = pipeline.model.to(device).eval()
    for param in nerf_model.parameters():
        param.requires_grad_(False)

    extractor = DinoExtractor(model_name=args.dino_model, device=str(device))

    results = {s: evaluate_split(s, args, config, nerf_model, extractor, device)
               for s in args.splits}
    summary = {
        "scene_dir": str(args.scene_dir),
        "nerf_config": str(args.nerf_config),
        "dino_model": args.dino_model,
        "max_frames": args.max_frames,
        "seed": args.seed,
        "splits": results,
    }
    if "train" in results and "test" in results:
        # Negative = held-out frames reconstruct worse in feature space, the
        # expected direction (higher cosine is better).
        summary["test_minus_train"] = results["test"]["mean_cos"] - results["train"]["mean_cos"]

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(summary, indent=2))
    for s, r in results.items():
        print(f"{s:>5}: same-patch cos {r['mean_cos']:.4f} (var {r['var_cos']:.5f})  "
              f"random-patch cos {r['mean_random_cos']:.4f} (var {r['var_random_cos']:.5f})  "
              f"separation {r['separation']:.4f}  same>random {100 * r['mean_frac_same_beats_random']:.1f}%  "
              f"over {r['n_frames']} frames", flush=True)
    print(f"-> {args.output_path}", flush=True)


if __name__ == "__main__":
    main()
