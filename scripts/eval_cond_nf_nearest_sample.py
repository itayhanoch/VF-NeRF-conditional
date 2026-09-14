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

DINO round-trip (`--dino-views N`): for the N best pixels of each pool -- the
`min` pool ranked by the min sample's distance and angle, the `top` pool by the
top sample's, each pixel scored by its distance rank plus its angle rank, over
the depth-filtered pixels -- that pool's sample is rendered through the frozen
NeRF as a view looking along the sampled direction at the sampled point --
a centred `--view-crop` window at the reference camera's native pixel pitch, so a
DINO patch of the render covers the same footprint as the source patch -- and the
DINOv2 feature of the patch the point lands in (the principal point) is compared
by cosine with the feature that conditioned the sample: the explorer's `pt cos`
as a statistic. Read the numbers against eval_dino_reconstruction.py's same-patch
(ceiling) and random-patch (floor) cosines of the same scene and split.

Cost: num_pixels * num_samples flow samples (40k x 100 = 4M), drawn in chunks of
`pixel_chunk * num_samples`; the flow is a few small MLPs on a 6-D input, so this
is a minute or two per split on a T4. The NeRF depth render is only num_pixels rays.
The round-trip adds 2 * N crop renders (364^2 = 132k rays each) + DINO passes:
~3 min per split for N = 200. Because the pixels are the flow's best cases, the
cosine answers "when the sample is geometrically right, does the appearance match?".

Example:
    python scripts/eval_cond_nf_nearest_sample.py \\
        --nerf-config outputs/bonsai/nerfacto/TIMESTAMP/config.yml \\
        --scene-dir data/mipnerf360/bonsai \\
        --cond-nf-checkpoint checkpoints/conditional_nf/bonsai/latest.pt \\
        --output-path eval/bonsai_cond_nf_nearest.json --depth-range 0.3 3.0 --dino-views 200
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.utils.dino_features import DinoExtractor, load_image_chw_01, patch_pixel_box, pixel_to_patch_cell
from nerfstudio.utils.eval_utils import eval_setup
from scripts.eval_cond_nf_likelihood import basic_stats, load_conditional_nf
from scripts.train_conditional_nf import build_training_cameras, precompute_dino_cache, sample_batch

WORLD_UP = torch.tensor([0.0, 0.0, 1.0])   # the explorer's / gradio app's camera up


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
    p.add_argument("--dino-views", type=int, default=200, help="DINO round-trip: for the min pool and the top pool separately, render the sample of this many best pixels (smallest combined rank of distance and angle over the depth-filtered pixels) and report the cosine between the landing patch's DINO feature and the source feature; 0 disables")
    p.add_argument("--view-crop", type=int, default=364, help="Side of the square render window in pixels at the reference camera's native pitch (a multiple of the 14-px DINO patch; 364 = 26 patches)")
    p.add_argument("--views-png-rows", type=int, default=6, help="Pixels shown in the per-split montage <output-path stem>_views_<split>.png (source crop | min render | top render); 0 disables")
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
    `top` is the sample with the highest log_prob under its own condition, plus
    the [B, 6] samples themselves, {min,top}_sample (`min_sample` is the
    closest-by-distance sample with its own direction), for the DINO round-trip.
    Non-finite samples are ignored (distance/angle -> inf, log_prob -> -inf).
    """
    out = {f"{v}_{m}": [] for v in VARIANTS for m in ("dist", "angle_deg", "sample")}
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
        mn = dist.argmin(dim=1)                                         # [B]
        ar = torch.arange(B, device=s.device)
        out["min_dist"].append(dist[ar, mn])
        out["min_angle_deg"].append(angle.min(dim=1).values)
        out["min_sample"].append(s[ar, mn])
        out["top_dist"].append(dist[ar, top])
        out["top_angle_deg"].append(angle[ar, top])
        out["top_sample"].append(s[ar, top])
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


def build_crop_camera(position, direction, reference_cameras, backoff, crop, world_up=WORLD_UP):
    """The explorer's `build_camera_from_point_direction` (origin = P - forward *
    backoff, looking along `direction`, world-Z up), but as a centred `crop` x
    `crop` window at the reference camera's native pixel pitch: fx/fy are kept,
    the principal point is the window centre, so P projects to (crop/2, crop/2)
    and one DINO patch of the render covers the same footprint as a source patch.
    """
    device = position.device
    forward = direction / direction.norm().clamp_min(1e-8)
    up_ref = world_up.to(device)
    if torch.abs(torch.dot(forward, up_ref)) > 0.99:
        up_ref = torch.tensor([1.0, 0.0, 0.0], device=device)
    right = torch.cross(forward, up_ref, dim=-1)
    right = right / right.norm().clamp_min(1e-8)
    up = torch.cross(right, forward, dim=-1)
    origin = position - forward * backoff
    rotation = torch.stack([right, up, -forward], dim=-1)
    c2w = torch.cat([rotation, origin.unsqueeze(-1)], dim=-1)
    half = crop / 2.0
    return Cameras(
        camera_to_worlds=c2w.unsqueeze(0),
        fx=reference_cameras.fx[0:1], fy=reference_cameras.fy[0:1],
        cx=torch.full_like(reference_cameras.cx[0:1], half),
        cy=torch.full_like(reference_cameras.cy[0:1], half),
        width=torch.full_like(reference_cameras.width[0:1], crop),
        height=torch.full_like(reference_cameras.height[0:1], crop),
        camera_type=reference_cameras.camera_type[0:1],
    ).to(device)


def patch_cell(grid, h, w, x, y):
    """Nearest patch-grid cell for pixel (x, y) of an (h, w) image -> [EMBED_DIM];
    the binning of `sample_batch` / the explorer, so the cell read here is the
    one the condition would be read from."""
    hp, wp = grid.shape[-2:]
    py, px = pixel_to_patch_cell(y, x, h, w, hp, wp)
    return grid[:, py, px].reshape(-1), py, px


def select_best_pixels(arrays, variant, n, depth_range):
    """The `n` pixels whose `variant` sample is best on distance AND angle: the
    pool is every pixel with finite depth (inside `depth_range` when given) and
    finite distance / angle; each pixel gets its rank by distance plus its rank
    by angle, and the `n` smallest sums win (best first). Returns (indices, pool size)."""
    depth = arrays["depth"]
    dist, ang = arrays[f"{variant}_dist"], arrays[f"{variant}_angle_deg"]
    cand = np.isfinite(depth) & np.isfinite(dist) & np.isfinite(ang)
    if depth_range is not None:
        cand &= (depth >= depth_range[0]) & (depth <= depth_range[1])
    pool = np.flatnonzero(cand)
    if not pool.size:
        return np.zeros(0, dtype=np.int64), 0
    rank_d = np.argsort(np.argsort(dist[pool], kind="stable"), kind="stable")
    rank_a = np.argsort(np.argsort(ang[pool], kind="stable"), kind="stable")
    order = np.argsort(rank_d + rank_a, kind="stable")[:n]
    return pool[order], int(pool.size)


def dino_round_trip(split, args, arrays, cond, cameras, image_filenames, dino_caches,
                    nerf_model, extractor, out_png, device):
    """Render, for each variant, that variant's sample of the `args.dino_views`
    best pixels (smallest combined rank of distance and angle -- the min pool
    and the top pool are chosen separately) and score the DINO feature of the
    landing patch against the source feature.

    Returns (json block, {view_{v}_pixel_idx, view_{v}_cos}).
    """
    crop = args.view_crop
    sel, n_pool = {}, {}
    for v in VARIANTS:
        sel[v], n_pool[v] = select_best_pixels(arrays, v, args.dino_views, args.depth_range)
    print(f"[{split}] DINO round-trip: " + ", ".join(
        f"{v}: best {len(sel[v])} of {n_pool[v]}" for v in VARIANTS) +
        f" pixels, one {crop}x{crop} render each", flush=True)

    cos = {v: np.full(len(sel[v]), np.nan, dtype=np.float32) for v in VARIANTS}
    renders = {v: {} for v in VARIANTS}
    grid_hw = None
    for v in VARIANTS:
        n = len(sel[v])
        for j, i in enumerate(sel[v]):
            i = int(i)
            t = float(arrays["depth"][i])
            cond_i = torch.from_numpy(cond[i].astype(np.float32)).to(device)
            s6 = torch.from_numpy(arrays[f"{v}_sample"][i]).to(device)
            if torch.isfinite(s6).all() and float(s6[3:].norm()) >= 1e-8:
                cam = build_crop_camera(s6[:3], s6[3:], cameras, t, crop)
                with torch.no_grad():
                    rb = cam.generate_rays(camera_indices=0)                     # [crop, crop]
                    rgb = nerf_model.get_outputs_for_camera_ray_bundle(rb)["rgb"].clamp(0, 1)
                    grid, _ = extractor.extract_patch_grid(rgb.permute(2, 0, 1).contiguous().cpu())
                cell, py, px = patch_cell(grid, crop, crop, crop / 2.0, crop / 2.0)
                grid_hw = tuple(int(g) for g in grid.shape[-2:])
                cos[v][j] = float(F.cosine_similarity(cell.float().to(device), cond_i, dim=0))
                if j < args.views_png_rows:
                    renders[v][j] = (rgb.cpu().numpy(), py, px)
            if (j + 1) % 25 == 0 or (j + 1) == n:
                print(f"[{split}] round-trip {v} {j + 1}/{n} | running median cos "
                      f"{_finite_median(cos[v][:j + 1]):.3f}", flush=True)

    def _with_extrema(vals):
        st = basic_stats(vals)
        vals = np.asarray(vals, dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        st["min"] = float(vals.min()) if vals.size else float("nan")
        st["max"] = float(vals.max()) if vals.size else float("nan")
        return st

    block = {
        "selection": "best_by_combined_rank_of_dist_and_angle",
        "n_requested": args.dino_views,
        "crop_px": crop,
        "patch_grid": list(grid_hw) if grid_hw else None,
        "backoff": "pixel_depth",
        "restricted_to_depth_range": args.depth_range is not None,
        "variants": {v: {
            "n_pixels": int(len(sel[v])),
            "n_candidates": n_pool[v],
            "cos": basic_stats(cos[v]),
            "n_nonfinite": int((~np.isfinite(cos[v])).sum()),
            "selected_dist": _with_extrema(arrays[f"{v}_dist"][sel[v]]),
            "selected_angle_deg": _with_extrema(arrays[f"{v}_angle_deg"][sel[v]]),
        } for v in VARIANTS},
    }
    if any(renders.values()) and out_png is not None:
        if save_views_montage(split, args, arrays, sel, cos, renders, image_filenames, dino_caches, out_png):
            block["png"] = str(out_png)
    extra = {}
    for v in VARIANTS:
        extra[f"view_{v}_pixel_idx"] = sel[v].astype(np.int32)
        extra[f"view_{v}_cos"] = cos[v]
    return block, extra


def _finite_median(a):
    a = np.asarray(a, dtype=np.float64)
    a = a[np.isfinite(a)]
    return float(np.median(a)) if a.size else float("nan")


def save_views_montage(split, args, arrays, sel, cos, renders, image_filenames, dino_caches, out_png):
    """Row r = the r-th best pixel of each pool; per pool a column pair: the
    source crop (red box = the 14-px cell the condition was read from, dot = the
    pixel) | that pool's render (red box = the centre cell whose feature was
    scored). Returns False (and skips the figure) when matplotlib is missing."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:
        print(f"  ! no montage: {e!r}", flush=True)
        return False

    grids, h, w = dino_caches
    hp, wp = grids.shape[1], grids.shape[2]
    crop = args.view_crop
    rows = min(args.views_png_rows, max(len(sel[v]) for v in VARIANTS))
    ncol = 2 * len(VARIANTS)
    fig, axes = plt.subplots(rows, ncol, figsize=(4.0 * ncol, 4.2 * rows), squeeze=False)
    for r in range(rows):
        for k, v in enumerate(VARIANTS):
            ax_src, ax_nv = axes[r, 2 * k], axes[r, 2 * k + 1]
            label = "min-dist" if v == "min" else "top-logp"
            if r >= len(sel[v]):
                ax_src.axis("off"); ax_nv.axis("off")
                continue
            i = int(sel[v][r])
            y, x = (float(c) for c in arrays["pixel_yx"][i])
            cam = int(arrays["cam_idx"][i])
            img = load_image_chw_01(image_filenames[cam]).permute(1, 2, 0).numpy()
            x0 = int(min(max(0, x - crop / 2), max(0, w - crop)))
            y0 = int(min(max(0, y - crop / 2), max(0, h - crop)))
            ax_src.imshow(img[y0:y0 + crop, x0:x0 + crop], extent=(x0, x0 + crop, y0 + crop, y0))
            py, px = pixel_to_patch_cell(y, x, h, w, hp, wp)
            bx, by, sx, sy = patch_pixel_box(py, px, h, w)
            ax_src.add_patch(plt.Rectangle((bx, by), sx, sy, fill=False, color="red", lw=1.5))
            ax_src.plot([x], [y], marker=".", color="red", ms=4)
            ax_src.set_title(f"{label} #{r + 1}: {Path(image_filenames[cam]).name} [{split}]\n"
                             f"px ({x:.0f}, {y:.0f})  depth {float(arrays['depth'][i]):.3f}", fontsize=8.5)
            ax_src.axis("off")
            hit = renders[v].get(r)
            if hit is None:
                ax_nv.set_title(f"{label} sample: no render", fontsize=8.5)
                ax_nv.axis("off")
                continue
            rgb, cpy, cpx = hit
            ax_nv.imshow(rgb)
            bx, by, sx, sy = patch_pixel_box(cpy, cpx, crop, crop)
            ax_nv.add_patch(plt.Rectangle((bx, by), sx, sy, fill=False, color="red", lw=1.5))
            ax_nv.set_title(f"{label} sample  cos {cos[v][r]:.3f}\n"
                            f"dist {float(arrays[f'{v}_dist'][i]):.4f}  angle {float(arrays[f'{v}_angle_deg'][i]):.1f} deg",
                            fontsize=8.5)
            ax_nv.axis("off")
    fig.suptitle(f"{split}: DINO round-trip on the best {args.dino_views} pixels of each pool (combined "
                 f"distance + angle rank) -- cosine between the centre patch of the render and the source "
                 f"patch feature ({crop}px crops at native pitch)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.12 / rows))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=100)
    plt.close(fig)
    print(f"-> {out_png}", flush=True)
    return True


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
    keep = {k: [] for k in ("depth", "cam_idx", "point", "direction", "pixel_yx")}
    keep.update({f"{v}_{m}": [] for v in VARIANTS for m in ("dist", "angle_deg", "sample")})
    conds = []   # fp16 [B, C] per batch, kept only for the round-trip (never written out)
    n_batches = math.ceil(args.num_pixels / args.ray_batch)
    done = 0
    for b in range(n_batches):
        bs = min(args.ray_batch, args.num_pixels - done)
        with torch.no_grad():
            ray_bundle, conditions, coords = sample_batch(cameras, dino_caches, bs, device, return_coords=True)
            depth = nerf_model(ray_bundle)["depth"].reshape(-1)                      # [B]
            points = ray_bundle.origins + ray_bundle.directions * depth[:, None]     # [B, 3]
            errs = sample_errors(field, conditions, points, ray_bundle.directions,
                                 args.num_samples, args.pixel_chunk)
        keep["depth"].append(depth.float().cpu().numpy())
        keep["cam_idx"].append(ray_bundle.camera_indices.reshape(-1).cpu().numpy())
        keep["point"].append(points.float().cpu().numpy())
        keep["direction"].append(ray_bundle.directions.float().cpu().numpy())
        keep["pixel_yx"].append(coords.float().cpu().numpy())
        conds.append(conditions.half().cpu().numpy())
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
    if args.dino_views > 0:
        cond = np.concatenate(conds)
        png = (args.output_path.with_name(f"{args.output_path.stem}_views_{split}.png")
               if args.views_png_rows > 0 else None)
        out["dino_views"], extra = dino_round_trip(
            split, args, arrays, cond, cameras, image_filenames, dino_caches,
            nerf_model, extractor, png, device)
        arrays.update(extra)
    int_keys = {"cam_idx"} | {f"view_{v}_pixel_idx" for v in VARIANTS}
    arrays = {k: (v.astype(np.int32) if k in int_keys else v.astype(np.float32)) for k, v in arrays.items()}
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
        dv = r.get("dino_views")
        if dv:
            for v in VARIANTS:
                b = dv["variants"][v]
                c = b["cos"]
                print(f"      round-trip {v:<4} (best {b['n_pixels']} of {b['n_candidates']}; dist <= "
                      f"{b['selected_dist']['max']:.4f}, angle <= {b['selected_angle_deg']['max']:.2f} deg): "
                      f"cos mean {c['mean']:.4f} var {c['var']:.4f} median {c['median']:.4f} "
                      f"(n {c['n']}, non-finite {b['n_nonfinite']})", flush=True)
    print(f"-> {args.output_path}", flush=True)


if __name__ == "__main__":
    main()
