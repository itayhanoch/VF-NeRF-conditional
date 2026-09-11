#!/usr/bin/env python
"""What does a frozen-NeRF depth of 0.3 or 3.0 look like in this scene?

The conditional-NF likelihood evals filter samples by their rendered depth
(`--depth-range LO HI`), because the catastrophic log-probs are rays whose depth
is nonsense. Depth here is in nerfstudio's normalized scene units (the median
depth of a scene is typically ~1), so the bounds need a picture to be chosen
sensibly. This script gives one per scene:

  1. render one training frame from its own camera and read the NeRF depth at
     the centre pixel, D;
  2. keep the orientation, slide the camera along the centre ray so the centre
     point sits at each requested depth (0.3 and 3.0 by default), render again
     and read the centre depth back;
  3. write a one-row PNG: the frame + a crosshair, one panel per target depth,
     and the original view's depth map with contour lines at the targets.

Note these mip-NeRF 360 rooms are indoors: a camera 3.0 units behind a point is
usually outside the room, so that panel is expected to show a wall's back face or
floaters -- that is the information.

Example:
    python scripts/depth_probe.py \\
        --nerf-config outputs/bonsai/nerfacto/TIMESTAMP/config.yml \\
        --scene-dir data/mipnerf360/bonsai \\
        --output-png eval/bonsai_depth_probe.png --output-json eval/bonsai_depth_probe.json
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

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.utils.eval_utils import eval_setup


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--nerf-config", type=Path, required=True, help="Trained frozen nerfacto config.yml")
    p.add_argument("--scene-dir", type=Path, required=True, help="Scene dir containing transforms.json")
    p.add_argument("--output-png", type=Path, required=True)
    p.add_argument("--output-json", type=Path, default=None, help="Defaults to <output-png>.json")
    p.add_argument("--frame-index", type=int, default=None, help="Train-split frame to probe (default: the middle one)")
    p.add_argument("--depths", type=float, nargs="+", default=[0.3, 3.0], help="Target centre depths, normalized scene units")
    p.add_argument("--render-downscale", type=float, default=3.0, help="Render at 1/this of the camera's resolution")
    return p.parse_args()


def render(nerf_model, cam: Cameras):
    with torch.no_grad():
        rb = cam.generate_rays(camera_indices=0)
        o = nerf_model.get_outputs_for_camera_ray_bundle(rb)
    return rb, o["rgb"].clamp(0, 1).cpu().numpy(), o["depth"][..., 0].cpu().numpy()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_json = args.output_json or args.output_png.with_suffix(".json")

    print(f"Loading frozen NeRF from {args.nerf_config} ...", flush=True)
    config, pipeline, _, _ = eval_setup(args.nerf_config, test_mode="inference")
    nerf_model = pipeline.model.to(device).eval()
    for p in nerf_model.parameters():
        p.requires_grad_(False)

    dp = config.pipeline.datamanager.dataparser
    dp.data = Path(args.scene_dir)
    outs = dp.setup().get_dataparser_outputs(split="train")
    dp_scale = float(getattr(outs, "dataparser_scale", 1.0) or 1.0)
    n = len(outs.image_filenames)
    i = args.frame_index if args.frame_index is not None else n // 2
    name = Path(outs.image_filenames[i]).name

    cam = outs.cameras[i:i + 1].to(device)
    cam.rescale_output_resolution(1.0 / args.render_downscale)
    h, w = int(cam.height[0, 0]), int(cam.width[0, 0])
    cy, cx = h // 2, w // 2

    rb, rgb0, depth0 = render(nerf_model, cam)
    D = float(depth0[cy, cx])
    o = rb.origins[cy, cx]
    d = rb.directions[cy, cx]
    P = o + d * D
    print(f"frame {i} ({name}): centre depth {D:.4f} (= {D / dp_scale:.4f} in COLMAP units; "
          f"dataparser scale {dp_scale:.4f})", flush=True)

    panels = [("original", rgb0, D)]
    targets = []
    for t in args.depths:
        # same rotation, camera slid along the centre ray so P is `t` in front
        c2w = cam.camera_to_worlds[0].clone()
        c2w[:, 3] = P - d * t
        cam_t = Cameras(
            camera_to_worlds=c2w.unsqueeze(0), fx=cam.fx, fy=cam.fy, cx=cam.cx, cy=cam.cy,
            width=cam.width, height=cam.height, camera_type=cam.camera_type,
        ).to(device)
        _, rgb_t, depth_t = render(nerf_model, cam_t)
        Dt = float(depth_t[cy, cx])
        print(f"  camera moved to {t:g} behind the centre point: rendered centre depth {Dt:.4f}"
              + ("" if math.isfinite(Dt) and abs(Dt - t) < 0.1 * max(t, 1e-6)
                 else "  (!= target: the point is occluded from there, or the camera is inside geometry)"),
              flush=True)
        panels.append((f"camera at {t:g}", rgb_t, Dt))
        targets.append({"target_depth": t, "rendered_center_depth": Dt,
                        "camera_origin": (P - d * t).tolist()})

    ncol = len(panels) + 1
    fig, axes = plt.subplots(1, ncol, figsize=(4.2 * ncol, 4.4), squeeze=False)
    axes = axes[0]
    for ax, (label, img, dv) in zip(axes, panels):
        ax.imshow(img)
        ax.plot([cx], [cy], marker="+", color="red", ms=16, mew=1.5)
        ax.set_title(f"{label}\ncentre depth {dv:.3f} ({dv / dp_scale:.3f} orig)", fontsize=9)
        ax.axis("off")
    ax = axes[-1]
    finite = np.isfinite(depth0)
    vmax = float(np.percentile(depth0[finite], 99)) if finite.any() else 1.0
    im = ax.imshow(np.where(finite, depth0, np.nan), cmap="viridis", vmin=0.0, vmax=vmax)
    levels = sorted(t for t in args.depths if t < vmax)
    if levels:
        cs = ax.contour(np.where(finite, depth0, vmax), levels=levels, colors="red", linewidths=0.8)
        ax.clabel(cs, fmt="%g", fontsize=7)
    ax.plot([cx], [cy], marker="+", color="white", ms=14, mew=1.5)
    ax.set_title(f"depth map (clipped at p99 = {vmax:.2f})\ncontours at {', '.join(f'{t:g}' for t in args.depths)}", fontsize=9)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.suptitle(f"{args.scene_dir.name}: {name} (train frame {i}) -- NeRF depth in normalized units, "
                 f"dataparser scale {dp_scale:.3f}", fontsize=10)
    fig.tight_layout()
    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_png, dpi=100)
    plt.close(fig)

    depth_pcts = {f"p{q:g}": float(v) for q, v in zip((1, 5, 25, 50, 75, 95, 99),
                                                       np.percentile(depth0[finite], (1, 5, 25, 50, 75, 95, 99)))}
    out_json.write_text(json.dumps({
        "scene_dir": str(args.scene_dir),
        "nerf_config": str(args.nerf_config),
        "frame_index": i,
        "frame": name,
        "render_size": [h, w],
        "dataparser_scale": dp_scale,
        "center_depth": D,
        "center_point": P.tolist(),
        "view_depth_percentiles": depth_pcts,
        "targets": targets,
        "png": str(args.output_png),
    }, indent=2))
    print(f"-> {args.output_png}\n-> {out_json}", flush=True)


if __name__ == "__main__":
    main()
