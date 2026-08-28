#!/usr/bin/env python
"""Standalone conditional-NF trainer.

Trains a ConditionalNFField to model P((3D point, 3D direction) | DINO feature),
using a FROZEN, already-trained nerfacto checkpoint purely for inference (never
fine-tuned here) to get each sampled ray's surface point via its depth render.

Deliberately bypasses nerfstudio's Trainer/Pipeline/DataManager entirely -- this is
a plain PyTorch training loop meant to run on a single Colab GPU, checkpointing
directly to a (typically Drive-mounted) directory.

Example:
    python scripts/train_conditional_nf.py \\
        --nerf-config outputs/bonsai/nerfacto/TIMESTAMP/config.yml \\
        --scene-dir data/mipnerf360/bonsai \\
        --checkpoint-dir checkpoints/conditional_nf/bonsai
"""
import argparse
import time
from pathlib import Path

import torch
from torch.optim import RAdam
from torch.optim.lr_scheduler import CosineAnnealingLR

from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.fields.nf_field import ConditionalNFField
from nerfstudio.utils.dino_features import DinoExtractor, get_or_compute_cache, load_image_chw_01
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
    p.add_argument("--max-steps", type=int, default=20000)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--grad-clip-norm", type=float, default=5000.0)
    p.add_argument("--checkpoint-every", type=int, default=500)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def build_training_cameras(scene_dir: Path):
    dataparser = NerfstudioDataParserConfig(data=scene_dir).setup()
    outputs = dataparser.get_dataparser_outputs(split="train")
    return outputs.cameras, outputs.image_filenames


def precompute_dino_cache(image_filenames, cache_dir: Path, extractor: DinoExtractor):
    """Returns a list of [EMBED_DIM,H,W] feature-map tensors, aligned with image_filenames."""
    caches = []
    for i, path in enumerate(image_filenames):
        img = load_image_chw_01(path)
        feat = get_or_compute_cache(img, path.stem, cache_dir, extractor)
        caches.append(feat)
        if (i + 1) % 20 == 0 or (i + 1) == len(image_filenames):
            print(f"  DINO cache: {i + 1}/{len(image_filenames)}")
    return caches


def sample_batch(cameras, dino_caches, batch_size, device):
    """Random (image, pixel) triples -> (RayBundle, condition[B,C]) on `device`."""
    num_images = len(dino_caches)
    img_idx = torch.randint(0, num_images, (batch_size,))
    ys = torch.empty(batch_size, dtype=torch.long)
    xs = torch.empty(batch_size, dtype=torch.long)
    embed_dim = dino_caches[0].shape[0]
    conditions = torch.empty(batch_size, embed_dim)

    for i in range(num_images):
        mask = img_idx == i
        n = int(mask.sum())
        if n == 0:
            continue
        h, w = dino_caches[i].shape[-2:]
        y = torch.randint(0, h, (n,))
        x = torch.randint(0, w, (n,))
        ys[mask] = y
        xs[mask] = x
        conditions[mask] = dino_caches[i][:, y, x].permute(1, 0).cpu()

    coords = torch.stack([ys.float() + 0.5, xs.float() + 0.5], dim=-1)  # (y, x), matches Cameras.generate_rays' convention
    camera_indices = img_idx.unsqueeze(-1)
    ray_bundle = cameras.generate_rays(camera_indices=camera_indices, coords=coords)
    return ray_bundle.to(device), conditions.to(device)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    print(f"Loading frozen NeRF from {args.nerf_config} ...")
    _, pipeline, _, _ = eval_setup(args.nerf_config, test_mode="inference")
    nerf_model = pipeline.model.to(device)
    nerf_model.eval()
    for param in nerf_model.parameters():
        param.requires_grad_(False)

    print(f"Loading training cameras/images from {args.scene_dir} ...")
    cameras, image_filenames = build_training_cameras(args.scene_dir)
    cameras = cameras.to(device)

    dino_cache_dir = args.dino_cache_dir or (args.scene_dir / "dino_cache")
    extractor = DinoExtractor(model_name=args.dino_model, device=str(device))
    print(f"Precomputing/loading DINO feature cache for {len(image_filenames)} images -> {dino_cache_dir} ...")
    dino_caches = precompute_dino_cache(image_filenames, dino_cache_dir, extractor)

    context_dim = extractor.embed_dim
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
