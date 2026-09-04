"""DINOv2 per-pixel feature extraction + on-disk caching.

Model loading and the patch-token extraction call follow
itayhanoch/conditional-normalizing-flows-toy's src/condflow/dino_features/model.py
(MIT license), which established the correct torch.hub call for this DINOv2 release.

Resizing strategy deliberately differs from that repo: rather than forcing every
image through a fixed 518x518 (squashing aspect ratio and, for any image far from
518px, a large downsample-then-blur-upsample round trip that coarsens spatial
precision), this module reflection-pads each image up to the nearest multiple of
14 in each dimension and runs DINOv2 at that near-native resolution (down to a
`MAX_DINO_SIDE` ceiling -- see below). DINOv2 supports arbitrary patch-divisible
input sizes via interpolated position embeddings -- the toy repo's fixed-518
choice was a simplicity shortcut for its own visualization script, not a hard
model constraint. This matters here because the extracted feature at a pixel
needs to correspond as precisely as possible to that pixel's own NeRF ray, not a
blurred neighborhood of it.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

HUB_REPO = "facebookresearch/dinov2"
HUB_ENTRYPOINT = "dinov2_vits14"
PATCH_SIZE = 14
EMBED_DIM = 384
# Ceiling on the DINOv2 forward-pass long side (112 * 14). torch < 2.0 has no
# flash-attention, so the attention fallback materialises the full [1, heads, N, N]
# patch-vs-patch matrix; at native `images_2` this is manageable (bonsai ~1559 px
# -> ~8.4k patches -> ~3-4 GB) but a much larger frame OOMs. An image whose long
# side is <= this runs at native resolution, so its patch tokens exactly match the
# un-rescaled image; a larger one is bilinearly downscaled for the forward pass
# only (raise this, or use a higher downscale factor, to get those native too).
MAX_DINO_SIDE = 1568
# DINOv2's own ImageNet-pretraining normalization (dinov2/data/transforms.py).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _pad_to_patch_multiple(image_chw: torch.Tensor, patch_size: int = PATCH_SIZE):
    """[3,H,W] -> [3,H',W'] reflection-padded (right/bottom edges only) so H',W'
    are each the next multiple of `patch_size`. Returns (padded_image, (h, w))
    where (h, w) is the ORIGINAL shape, for cropping back later."""
    _, h, w = image_chw.shape
    pad_h = (patch_size - h % patch_size) % patch_size
    pad_w = (patch_size - w % patch_size) % patch_size
    if pad_h == 0 and pad_w == 0:
        return image_chw, (h, w)
    # F.pad's reflect mode requires pad width < the corresponding input dimension --
    # true for any real training image (pad_h/pad_w are both < patch_size=14, far
    # smaller than any real photo), but falls back to edge-replication for a
    # degenerate image smaller than the patch size in either dimension, where
    # reflect padding isn't mathematically valid.
    mode = "reflect" if pad_h < h and pad_w < w else "replicate"
    padded = F.pad(image_chw.unsqueeze(0), (0, pad_w, 0, pad_h), mode=mode).squeeze(0)
    return padded, (h, w)


def _normalize_for_dinov2(rgb_chw_01: torch.Tensor) -> torch.Tensor:
    """[3,H,W] in [0,1] -> normalized with DINOv2's ImageNet stats (required --
    DINOv2 was pretrained on inputs preprocessed this exact way)."""
    mean = torch.tensor(IMAGENET_MEAN, device=rgb_chw_01.device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=rgb_chw_01.device).view(3, 1, 1)
    return (rgb_chw_01 - mean) / std


class DinoExtractor:
    """Loads a frozen DINOv2 model once; extracts per-pixel feature maps on demand."""

    def __init__(self, model_name: str = HUB_ENTRYPOINT, device: str = "cuda"):
        if model_name != HUB_ENTRYPOINT:
            raise NotImplementedError(f"Only {HUB_ENTRYPOINT} is wired up (embed_dim assumptions elsewhere assume 384-dim); got {model_name!r}.")
        self.model_name = model_name
        self.device = device
        self.embed_dim = EMBED_DIM
        self.model = torch.hub.load(HUB_REPO, HUB_ENTRYPOINT, pretrained=True).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    def extract_patch_grid(self, image_chw_01: torch.Tensor):
        """[3,H,W] float in [0,1] -> ([EMBED_DIM, Hp, Wp] patch-token grid, (h, w)).

        (h, w) is the image's ORIGINAL pixel shape (before any MAX_DINO_SIDE
        downscale or patch-multiple padding); callers map a pixel to its patch
        cell with `Hp/h`, `Wp/w`. Uses
        model.get_intermediate_layers(x, n=1, reshape=True, return_class_token=False)
        -- the hub model's documented way to get the last block's normalized patch
        tokens pre-reshaped to a spatial grid.
        """
        img = image_chw_01.to(self.device)
        h, w = int(img.shape[-2]), int(img.shape[-1])
        if max(h, w) > MAX_DINO_SIDE:
            # downscale for the forward pass only; the returned (h, w) stays native
            scale = MAX_DINO_SIDE / max(h, w)
            img = F.interpolate(
                img.unsqueeze(0), scale_factor=scale, mode="bilinear",
                align_corners=False, recompute_scale_factor=False,
            ).squeeze(0)
        padded, _ = _pad_to_patch_multiple(img)
        normalized = _normalize_for_dinov2(padded)
        with torch.no_grad():
            (patch_tokens,) = self.model.get_intermediate_layers(normalized.unsqueeze(0), n=1, reshape=True, return_class_token=False)
        return patch_tokens.squeeze(0), (h, w)  # [EMBED_DIM, Hp, Wp], (h, w)

    def upsample_to_pixels(self, patch_grid: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """[C,Hp,Wp] -> [C,h,w]: bilinear-upsample the patch grid straight to the
        original pixel size. (Interpolating to (h, w) directly, rather than to
        Hp*14 x Wp*14 then cropping, also covers the case where the grid came from
        a MAX_DINO_SIDE-downscaled forward pass, i.e. Hp*14 < h.)"""
        return F.interpolate(
            patch_grid.unsqueeze(0), size=(int(h), int(w)), mode="bilinear", align_corners=False
        ).squeeze(0)

    def extract_pixel_map(self, image_chw_01: torch.Tensor) -> torch.Tensor:
        """[3,H,W] float in [0,1] -> [EMBED_DIM,H,W] full-resolution per-pixel feature map,
        pixel-aligned with the input image."""
        patch_grid, (h, w) = self.extract_patch_grid(image_chw_01)
        return self.upsample_to_pixels(patch_grid, h, w)

    def extract_at_pixel(self, image_chw_01: torch.Tensor, x: int, y: int) -> torch.Tensor:
        """[3,H,W] float in [0,1] + pixel coords (x=col, y=row) -> [EMBED_DIM] feature
        at that one pixel. Used for the Gradio external-image live-extraction path --
        costs the same as extracting the whole map (DINOv2's forward pass is whole-image
        regardless), so this simply extracts+upsamples the full map and indexes it, to
        keep condition-vector semantics identical to the cached training-image path."""
        pixel_map = self.extract_pixel_map(image_chw_01)
        return pixel_map[:, y, x].clone()


def cache_path(cache_dir: Path, model_name: str, image_stem: str) -> Path:
    return Path(cache_dir) / model_name / f"{image_stem}.pt"


def get_or_compute_cache(
    image_chw_01: torch.Tensor,
    image_stem: str,
    cache_dir: Path,
    extractor: DinoExtractor,
    device: Optional[str] = None,
) -> torch.Tensor:
    """Load a cached [EMBED_DIM,H,W] fp16 per-pixel feature map for one training image,
    computing (and saving) it on first use. Cache key includes the model name so
    switching DINOv2 variants can't silently reuse stale features; resolution is
    implicit in the saved tensor's own shape, checked against the current image."""
    path = cache_path(cache_dir, extractor.model_name, image_stem)
    _, h, w = image_chw_01.shape
    if path.exists():
        cached = torch.load(path, map_location="cpu")
        if cached.shape[-2:] == (h, w):
            return cached.to(device or extractor.device).float()
        # Stale cache from a different resolution -- recompute below.
    pixel_map = extractor.extract_pixel_map(image_chw_01)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(pixel_map.to(torch.float16).cpu(), path)
    return pixel_map.to(device or extractor.device).float()


def load_image_chw_01(image_path: Path) -> torch.Tensor:
    """Read an image file -> [3,H,W] float32 tensor in [0,1] (RGB, drops alpha)."""
    from PIL import Image

    img = Image.open(image_path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()
