#!/usr/bin/env python
"""Measure picked point pairs through the frozen NeRF -- a scale reference per scene.

Every evaluation number in this repo is in nerfstudio's normalized scene units.
This script anchors them: for each pair of pixels picked on a frame with
app/pick_scale_pairs.py (the two ends of a plate, a book, a tile), it shoots the
frame's camera rays through both pixels AND through their pixel midpoint, reads
the frozen NeRF's depth along each, and reports

  * |AB|      -- the 3-D distance between the two end points,
  * A-M, M-B  -- each end point's distance to the NeRF-rendered midpoint pixel,
                 next to |AB|/2 (on a flat object the three agree; a large gap
                 says the depth is unreliable there or the object is not flat),
  * the same in the original COLMAP units (divided by the dataparser scale),
  * the depth floor / ceiling (0.3 / 3.0) and the scene backoff expressed in
    units of |AB| -- "the depth floor is 1.4 plate-widths".

Cameras come from the checkpoint's own dataparser (both splits, so a TEST frame
works too), exactly as the explorer does.

Example:
    python scripts/measure_scene_scale.py \\
        --nerf-config outputs/counter/nerfacto/TIMESTAMP/config.yml \\
        --scene-dir data/mipnerf360/counter \\
        --pairs scale_pairs.json --scene counter \\
        --output-json eval/counter_scale.json --output-png eval/counter_scale.png
"""
import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from nerfstudio.utils.eval_utils import eval_setup

DEPTH_RANGE = (0.3, 3.0)   # the likelihood evals' depth filter, for the context lines


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--nerf-config", type=Path, required=True, help="Trained frozen nerfacto config.yml")
    p.add_argument("--scene-dir", type=Path, required=True, help="Scene dir containing transforms.json")
    p.add_argument("--pairs", type=Path, required=True, help="JSON: either {scene: [rows]} or a bare list of rows; a row is [image, x1, y1, x2, y2, TAG, label] or the equivalent dict")
    p.add_argument("--scene", default=None, help="Key to pick out of a {scene: [...]} pairs file (default: the scene dir's name)")
    p.add_argument("--output-json", type=Path, required=True)
    p.add_argument("--output-png", type=Path, default=None, help="Per-pair crops with A, B, M and the distances (default: next to the JSON)")
    p.add_argument("--downscale", type=int, default=2, help="images_<N>/ folder the pixels were picked on")
    return p.parse_args()


def default_backoff_distance(cameras) -> float:
    centers = cameras.camera_to_worlds[..., :3, 3]
    scene_center = centers.mean(dim=0)
    return float((centers - scene_center).norm(dim=-1).median())


def load_pairs(path: Path, scene: str):
    data = json.loads(path.read_text())
    rows = data.get(scene, []) if isinstance(data, dict) else data
    out = []
    for r in rows:
        if isinstance(r, dict):
            out.append({"image": r["image"], "x1": int(r["x1"]), "y1": int(r["y1"]),
                        "x2": int(r["x2"]), "y2": int(r["y2"]),
                        "tag": r.get("tag", ""), "label": r.get("label", "")})
        else:
            name, x1, y1, x2, y2 = r[0], int(r[1]), int(r[2]), int(r[3]), int(r[4])
            rest = list(r[5:])
            out.append({"image": name, "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        "tag": rest[0] if rest else "", "label": rest[1] if len(rest) > 1 else ""})
    return out


def main():
    args = parse_args()
    scene = args.scene or args.scene_dir.name
    out_png = args.output_png or args.output_json.with_suffix(".png")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pairs = load_pairs(args.pairs, scene)
    if not pairs:
        raise SystemExit(f"no pairs for scene {scene!r} in {args.pairs}")

    print(f"Loading frozen NeRF from {args.nerf_config} ...", flush=True)
    config, pipeline, _, _ = eval_setup(args.nerf_config, test_mode="inference")
    nerf_model = pipeline.model.to(device).eval()
    for p in nerf_model.parameters():
        p.requires_grad_(False)

    dp = config.pipeline.datamanager.dataparser
    dp.data = Path(args.scene_dir)
    outs = dp.setup().get_dataparser_outputs(split="train")
    cameras = outs.cameras.to(device)
    cam_by_name = {Path(p).name: (cameras, i, "train") for i, p in enumerate(outs.image_filenames)}
    try:
        outs_te = dp.setup().get_dataparser_outputs(split="test")
        cams_te = outs_te.cameras.to(device)
        for i, p in enumerate(outs_te.image_filenames):
            cam_by_name.setdefault(Path(p).name, (cams_te, i, "test"))
    except Exception as e:
        print(f"  ! no test-split cameras ({e!r})", flush=True)
    dp_scale = float(getattr(outs, "dataparser_scale", 1.0) or 1.0)
    backoff = default_backoff_distance(cameras)
    img_dir = args.scene_dir / f"images_{args.downscale}"

    results = []
    for k, pr in enumerate(pairs):
        name = pr["image"]
        hit = cam_by_name.get(name)
        rec = dict(pr, index=k, ok=False)
        if hit is None:
            rec["error"] = f"{name} has no camera in either split"
            print(f"pair {k}: ! {rec['error']}", flush=True)
            results.append(rec)
            continue
        cams, i, split = hit
        img_path = img_dir / name
        with Image.open(img_path) as im:
            img_w, img_h = im.size
        sy = float(cams.height[i]) / img_h
        sx = float(cams.width[i]) / img_w
        mx, my = (pr["x1"] + pr["x2"]) / 2.0, (pr["y1"] + pr["y2"]) / 2.0
        coords = torch.tensor([[pr["y1"] * sy, pr["x1"] * sx],
                               [pr["y2"] * sy, pr["x2"] * sx],
                               [my * sy, mx * sx]], dtype=torch.float32)
        rb = cams.generate_rays(camera_indices=torch.tensor([[i]] * 3), coords=coords).to(device)
        with torch.no_grad():
            depth = nerf_model(rb)["depth"].reshape(-1)                    # [3]
        P = rb.origins.reshape(-1, 3) + rb.directions.reshape(-1, 3) * depth[:, None]
        dA, dB, dM = (float(v) for v in depth)
        PA, PB, PM = P[0], P[1], P[2]
        ab = float((PA - PB).norm())
        am = float((PA - PM).norm())
        mb = float((PM - PB).norm())
        mid_gap = float((PM - (PA + PB) / 2).norm())
        ok = all(math.isfinite(v) and v > 1e-4 for v in (dA, dB, dM))
        rec.update({
            "ok": ok, "split": split, "camera_index": i,
            "midpoint_px": [mx, my],
            "depth": {"A": dA, "B": dB, "M": dM},
            "points": {"A": PA.tolist(), "B": PB.tolist(), "M": PM.tolist()},
            "dist_AB": ab, "dist_AM": am, "dist_MB": mb, "half_AB": ab / 2, "midpoint_gap": mid_gap,
            "dist_AB_orig": ab / dp_scale, "dist_AM_orig": am / dp_scale, "dist_MB_orig": mb / dp_scale,
            "depth_floor_in_AB": DEPTH_RANGE[0] / ab if ab > 0 else math.nan,
            "depth_ceiling_in_AB": DEPTH_RANGE[1] / ab if ab > 0 else math.nan,
            "backoff_in_AB": backoff / ab if ab > 0 else math.nan,
        })
        print(f"pair {k} [{pr['label'] or '-'}] {name} ({split}): |AB| {ab:.4f} ({ab / dp_scale:.4f} orig)  "
              f"A-M {am:.4f}  M-B {mb:.4f}  AB/2 {ab / 2:.4f}  midpoint gap {mid_gap:.4f}  "
              f"depth A/B/M {dA:.3f}/{dB:.3f}/{dM:.3f}" + ("" if ok else "  ! non-finite depth"), flush=True)
        results.append(rec)

    # --- figure: one crop per pair with A, B, M and the distances
    good = [r for r in results if r["ok"]]
    if good:
        n = len(good)
        fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 5.4), squeeze=False)
        for ax, r in zip(axes[0], good):
            img = np.asarray(Image.open(img_dir / r["image"]).convert("RGB"))
            h, w = img.shape[:2]
            x1, y1, x2, y2 = r["x1"], r["y1"], r["x2"], r["y2"]
            L = max(math.hypot(x2 - x1, y2 - y1), 40.0)
            m = 0.4 * L
            cx0, cy0 = (x1 + x2) / 2, (y1 + y2) / 2
            half = L / 2 + m
            xa, xb = int(max(0, cx0 - half)), int(min(w, cx0 + half))
            ya, yb = int(max(0, cy0 - half)), int(min(h, cy0 + half))
            ax.imshow(img[ya:yb, xa:xb], extent=(xa, xb, yb, ya))
            ax.plot([x1, x2], [y1, y2], color="red", lw=1.2)
            rad = max(3.0, 0.03 * L)
            for x, y, t in ((x1, y1, "A"), (x2, y2, "B")):
                ax.add_patch(plt.Circle((x, y), rad, fill=False, color="red", lw=1.5))
                ax.text(x + rad, y - rad, t, color="red", fontsize=10, weight="bold")
            mx, my = r["midpoint_px"]
            ax.plot([mx], [my], marker="+", color="lime", ms=12, mew=1.8)
            ax.text(mx + rad, my + 2.5 * rad, "M", color="lime", fontsize=10, weight="bold")
            d = r["depth"]
            title = r["label"] or f"pair {r['index']}"
            ax.set_title(f"{title} -- {r['image']} [{r['split']}]\n"
                         f"|AB| {r['dist_AB']:.3f} ({r['dist_AB_orig']:.3f} orig)   A-M {r['dist_AM']:.3f}   "
                         f"M-B {r['dist_MB']:.3f}   AB/2 {r['half_AB']:.3f}\n"
                         f"depth A/B/M {d['A']:.3f} / {d['B']:.3f} / {d['M']:.3f}   "
                         f"0.3 = {r['depth_floor_in_AB']:.1f}x|AB|  3.0 = {r['depth_ceiling_in_AB']:.1f}x|AB|  "
                         f"backoff = {r['backoff_in_AB']:.1f}x|AB|", fontsize=8.5)
            ax.axis("off")
        fig.suptitle(f"{scene}: scale reference pairs through the frozen NeRF (normalized units; "
                     f"dataparser scale {dp_scale:.3f}, backoff {backoff:.3f})", fontsize=10)
        fig.tight_layout()
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=110)
        plt.close(fig)
        print(f"-> {out_png}", flush=True)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps({
        "scene": scene,
        "scene_dir": str(args.scene_dir),
        "nerf_config": str(args.nerf_config),
        "dataparser_scale": dp_scale,
        "backoff": backoff,
        "depth_range": list(DEPTH_RANGE),
        "downscale": args.downscale,
        "png": str(out_png) if good else None,
        "pairs": results,
    }, indent=2))
    print(f"-> {args.output_json}", flush=True)


if __name__ == "__main__":
    main()
