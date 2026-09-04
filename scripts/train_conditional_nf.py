#!/usr/bin/env python
"""Standalone conditional-NF trainer.

Trains a ConditionalNFField to model P((3D point, 3D direction) | DINO feature),
using a FROZEN, already-trained nerfacto checkpoint purely for inference (never
fine-tuned here) to get each sampled ray's surface point via its depth render.

Deliberately bypasses nerfstudio's Trainer/Pipeline/DataManager entirely -- this is
a plain PyTorch training loop meant to run on a single Colab/Kaggle GPU,
checkpointing directly to a (typically mounted) directory.

DINO features are precomputed once, at each image's native resolution (capped at
`dino_features.MAX_DINO_SIDE`), into ONE memory-mapped `.npy` on disk -- only the
per-batch sampled slice is ever resident. Each training step samples a continuous
sub-pixel (not a patch centre), so the frozen NeRF's depth renders give 3-D
targets that densely cover the surfaces; the condition is the DINOv2 token of the
14x14 patch that pixel falls in (nearest cell, matching the interactive explorer).

Example:
    python scripts/train_conditional_nf.py \\
        --nerf-config outputs/bonsai/nerfacto/TIMESTAMP/config.yml \\
        --scene-dir data/mipnerf360/bonsai \\
        --checkpoint-dir checkpoints/conditional_nf/bonsai
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch.optim import RAdam
from torch.optim.lr_scheduler import CosineAnnealingLR

from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.fields.nf_field import ConditionalNFField
from nerfstudio.utils.dino_features import DinoExtractor, load_image_chw_01
from nerfstudio.utils.eval_utils import eval_setup


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--nerf-config", type=Path, required=True, help="Path to a trained frozen nerfacto config.yml")
    p.add_argument("--scene-dir", type=Path, required=True, help="Scene dir containing transforms.json (the same data the NeRF was trained on)")
    p.add_argument("--checkpoint-dir", type=Path, required=True, help="Where to save conditional-NF checkpoints (point this at a mounted Drive path on Colab)")
    p.add_argument("--dino-cache-dir", type=Path, default=None, help="Defaults to <scene-dir>/dino_cache")
    p.add_argument("--dino-model", default="dinov2_vits14")
    p.add_argument("--num-blocks", type=int, default=8)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--cond-prior", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--use-batchnorm", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--max-steps", type=int, default=30000)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--grad-clip-norm", type=float, default=5000.0)
    p.add_argument("--checkpoint-every", type=int, default=500)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def build_training_cameras(scene_dir: Path, dataparser_config=None):
    """Cameras + image paths in the frozen NeRF's own dataparser frame.

    Pass the checkpoint's dataparser config so center-method / scene-scale /
    downscale-factor match what the NeRF (and hence these targets) were trained
    in; falls back to stock defaults when not given.
    """
    if dataparser_config is None:
        dataparser_config = NerfstudioDataParserConfig()
    dataparser_config.data = Path(scene_dir)
    outputs = dataparser_config.setup().get_dataparser_outputs(split="train")
    return outputs.cameras, outputs.image_filenames


def precompute_dino_cache(image_filenames, cache_dir: Path, extractor: DinoExtractor):
    """Native-resolution DINOv2 patch grids for every training image, kept in ONE
    memory-mapped .npy on disk (only the per-batch sampled slice is resident).

    Layout [N, Hp, Wp, EMBED_DIM] fp16 -- the last-axis-contiguous form makes
    sample_batch's gather a plain 3-array advanced index. Returns
    (grids_mmap, H, W) where (H, W) is the shared pixel size of the images.
    Recompute is skipped when a cache with the matching shape already exists.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    n = len(image_filenames)

    img0 = load_image_chw_01(image_filenames[0])
    h, w = int(img0.shape[-2]), int(img0.shape[-1])
    with torch.no_grad():
        grid0, _ = extractor.extract_patch_grid(img0)  # [C, Hp, Wp]
    c, hp, wp = (int(v) for v in grid0.shape)

    npy = cache_dir / f"dino_grids_{extractor.model_name}_{h}x{w}_{hp}x{wp}.npy"
    if npy.exists():
        mm = np.load(npy, mmap_mode="r")
        if mm.shape == (n, hp, wp, c):
            print(f"  reusing DINO cache {npy.name} {mm.shape}")
            return mm, h, w
        print(f"  DINO cache {npy.name} is {mm.shape}, want {(n, hp, wp, c)} -- recomputing")
        del mm

    mm = np.lib.format.open_memmap(npy, mode="w+", dtype=np.float16, shape=(n, hp, wp, c))
    mm[0] = grid0.permute(1, 2, 0).to(torch.float16).cpu().numpy()
    for i in range(1, n):
        img = load_image_chw_01(image_filenames[i])
        assert tuple(img.shape[-2:]) == (h, w), (
            f"{image_filenames[i].name} is {tuple(img.shape[-2:])}, expected {(h, w)} -- "
            "all training images must share a resolution for the mem-mapped cache")
        with torch.no_grad():
            grid, _ = extractor.extract_patch_grid(img)
        mm[i] = grid.permute(1, 2, 0).to(torch.float16).cpu().numpy()
        if (i + 1) % 20 == 0 or (i + 1) == n:
            print(f"  DINO cache: {i + 1}/{n}")
    mm.flush()
    del mm
    return np.load(npy, mmap_mode="r"), h, w


def sample_batch(cameras, dino_caches, batch_size, device):
    """Random (image, sub-pixel) -> (RayBundle, condition[B, C]) on `device`.

    Pixels are drawn continuously over each image, not snapped to patch centres,
    so the frozen NeRF's depth renders give 3-D targets that densely cover the
    surfaces. The condition is the DINOv2 token of the 14x14 patch that pixel
    falls in -- the same nearest-cell lookup kaggle_explorer.py / gradio_app.py
    use at inference.
    """
    grids, h, w = dino_caches  # grids: np.memmap [N, Hp, Wp, C] fp16
    n, hp, wp, _ = grids.shape

    img_idx = torch.randint(0, n, (batch_size,))
    ys = torch.rand(batch_size) * h
    xs = torch.rand(batch_size) * w
    py = (ys * (hp / h)).long().clamp_(0, hp - 1)
    px = (xs * (wp / w)).long().clamp_(0, wp - 1)

    cond_np = grids[img_idx.numpy(), py.numpy(), px.numpy()].astype(np.float32)  # [B, C]
    conditions = torch.from_numpy(cond_np)

    coords = torch.stack([ys, xs], dim=-1)  # (y, x) fractional pixel, matches Cameras.generate_rays
    camera_indices = img_idx.unsqueeze(-1)
    ray_bundle = cameras.generate_rays(camera_indices=camera_indices, coords=coords)
    return ray_bundle.to(device), conditions.to(device)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    print(f"Loading frozen NeRF from {args.nerf_config} ...")
    config, pipeline, _, _ = eval_setup(args.nerf_config, test_mode="inference")
    nerf_model = pipeline.model.to(device)
    nerf_model.eval()
    for param in nerf_model.parameters():
        param.requires_grad_(False)

    # The DINO precompute peaks GPU memory (no flash-attention on torch < 2.0);
    # park the frozen NeRF on the CPU while it runs, then bring it back for the
    # training loop.
    nerf_model = nerf_model.cpu()
    torch.cuda.empty_cache()

    print(f"Loading training cameras/images from {args.scene_dir} ...")
    cameras, image_filenames = build_training_cameras(
        args.scene_dir, config.pipeline.datamanager.dataparser
    )
    cameras = cameras.to(device)

    dino_cache_dir = args.dino_cache_dir or (args.scene_dir / "dino_cache")
    extractor = DinoExtractor(model_name=args.dino_model, device=str(device))
    print(f"Precomputing/loading DINO feature cache for {len(image_filenames)} images -> {dino_cache_dir} ...")
    dino_caches = precompute_dino_cache(image_filenames, dino_cache_dir, extractor)

    context_dim = extractor.embed_dim
    del extractor
    torch.cuda.empty_cache()
    nerf_model = nerf_model.to(device)

    field = ConditionalNFField(
        context_dim=context_dim,
        num_blocks=args.num_blocks,
        hidden_dim=args.hidden_dim,
        cond_prior=args.cond_prior,
        use_cond_in_coupling=True,
        use_batchnorm=args.use_batchnorm,
        device=str(device),
    )
    optimizer = RAdam(field.parameters(), lr=args.lr, eps=0.1)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.max_steps)

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt_meta = dict(
        context_dim=context_dim,
        num_dims=6,
        num_blocks=args.num_blocks,
        hidden_dim=args.hidden_dim,
        cond_prior=args.cond_prior,
        use_batchnorm=args.use_batchnorm,
        dino_model_name=args.dino_model,
    )

    print(f"Starting training for {args.max_steps} steps on {device} ...")
    field.train()
    start = time.time()
    for step in range(1, args.max_steps + 1):
        with torch.no_grad():
            ray_bundle, conditions = sample_batch(cameras, dino_caches, args.batch_size, device)
            outputs = nerf_model(ray_bundle)
            depth = outputs["depth"]
            points = ray_bundle.origins + ray_bundle.directions * depth
            directions = ray_bundle.directions
            x = torch.cat([points, directions], dim=-1)

        log_prob = field.log_prob(x, conditions)
        loss = -log_prob.mean()

        if not torch.isfinite(loss):
            print(f"[step {step}] non-finite loss ({loss.item()!r}); skipping this step")
            optimizer.zero_grad()
            continue

        optimizer.zero_grad()
        loss.backward()
        if args.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(field.parameters(), max_norm=args.grad_clip_norm)
        optimizer.step()
        scheduler.step()

        if step % args.log_every == 0 or step == 1:
            elapsed = time.time() - start
            print(f"step {step}/{args.max_steps} | loss {loss.item():.4f} | elapsed {elapsed:.1f}s")

        if step % args.checkpoint_every == 0 or step == args.max_steps:
            payload = dict(step=step, model_state=field.state_dict(), optimizer_state=optimizer.state_dict(), **ckpt_meta)
            ckpt_path = args.checkpoint_dir / f"cond_nf_step_{step:07d}.pt"
            torch.save(payload, ckpt_path)
            torch.save(payload, args.checkpoint_dir / "latest.pt")
            print(f"Saved checkpoint -> {ckpt_path}")

    print("Training complete.")


if __name__ == "__main__":
    main()
