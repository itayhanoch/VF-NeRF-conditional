"""Unit tests for the pixel-alignment math in nerfstudio.utils.dino_features --
deliberately network-independent (no DINOv2 model download): only tests
_pad_to_patch_multiple, patch_pixel_box and upsample_to_pixels's crop logic,
using a synthetic patch grid in place of a real DINOv2 forward pass. Extraction-through-a-real-
model is exercised manually/in Colab (network access + a GPU are both assumed
there), not in this suite.
"""
import math

import torch

from nerfstudio.utils.dino_features import (
    EMBED_DIM,
    MAX_DINO_SIDE,
    PATCH_SIZE,
    DinoExtractor,
    _pad_to_patch_multiple,
    patch_pixel_box,
)


def test_pad_to_patch_multiple_shapes():
    for h, w in [(100, 100), (101, 99), (14, 14), (1, 1), (37 * 14, 37 * 14)]:
        image = torch.rand(3, h, w)
        padded, (orig_h, orig_w) = _pad_to_patch_multiple(image, patch_size=PATCH_SIZE)
        assert (orig_h, orig_w) == (h, w)
        assert padded.shape[-2] % PATCH_SIZE == 0
        assert padded.shape[-1] % PATCH_SIZE == 0
        assert padded.shape[-2] >= h and padded.shape[-2] - h < PATCH_SIZE
        assert padded.shape[-1] >= w and padded.shape[-1] - w < PATCH_SIZE


def test_pad_to_patch_multiple_preserves_original_content():
    image = torch.rand(3, 100, 130)
    padded, (h, w) = _pad_to_patch_multiple(image, patch_size=PATCH_SIZE)
    assert torch.equal(padded[:, :h, :w], image), "padding must only extend the right/bottom edges, not alter existing pixels"


def test_pad_to_patch_multiple_noop_when_already_aligned():
    image = torch.rand(3, 2 * PATCH_SIZE, 3 * PATCH_SIZE)
    padded, (h, w) = _pad_to_patch_multiple(image, patch_size=PATCH_SIZE)
    assert padded.shape == image.shape
    assert torch.equal(padded, image)


def test_upsample_to_pixels_crops_to_original_shape():
    # A DinoExtractor without loading the real model -- only exercising the pure
    # tensor-math method, which needs no model state.
    extractor = object.__new__(DinoExtractor)  # bypass __init__ (no network/model needed)
    h, w = 100, 130
    hp, wp = -(-h // PATCH_SIZE), -(-w // PATCH_SIZE)  # ceil division, matches _pad_to_patch_multiple's padded grid
    patch_grid = torch.rand(EMBED_DIM, hp, wp)
    pixel_map = DinoExtractor.upsample_to_pixels(extractor, patch_grid, h, w)
    assert pixel_map.shape == (patch_grid.shape[0], h, w)


def test_upsample_to_pixels_is_spatially_smooth_within_a_patch():
    """A coarse patch grid upsampled back to pixel resolution should vary smoothly
    within each 14x14 block (bilinear interpolation), not jump discontinuously --
    a cheap proxy check that the upsample geometry lines up with patch boundaries
    as intended, without needing a real DINOv2 forward pass."""
    extractor = object.__new__(DinoExtractor)
    hp, wp = 4, 4
    patch_grid = torch.zeros(1, hp, wp)
    patch_grid[0, 1, 1] = 10.0  # one distinctive patch
    h, w = hp * PATCH_SIZE, wp * PATCH_SIZE
    pixel_map = DinoExtractor.upsample_to_pixels(extractor, patch_grid, h, w)
    # the peak should land within (or very near) patch (1,1)'s pixel block
    peak_y, peak_x = divmod(int(pixel_map[0].argmax()), w)
    assert PATCH_SIZE <= peak_y < 3 * PATCH_SIZE
    assert PATCH_SIZE <= peak_x < 3 * PATCH_SIZE


def test_patch_pixel_box_tiles_a_native_resolution_image():
    # 518 = 37 * 14: already patch-aligned and well under MAX_DINO_SIDE, so every
    # cell is exactly 14x14 and the grid tiles the image with no gap or overlap.
    h = w = 37 * PATCH_SIZE
    hp = wp = 37
    for py in range(hp):
        for px in range(wp):
            x0, y0, sx, sy = patch_pixel_box(py, px, h, w)
            assert (sx, sy) == (PATCH_SIZE, PATCH_SIZE)
            assert (x0, y0) == (px * PATCH_SIZE, py * PATCH_SIZE)
    # last cell's far edge lands exactly on the image extent
    x0, y0, sx, sy = patch_pixel_box(hp - 1, wp - 1, h, w)
    assert (x0 + sx, y0 + sy) == (w, h)


def test_patch_pixel_box_covers_the_padded_extent_when_unaligned():
    # 100x130 pads to 112x140 (8 x 10 cells); the final row/column boxes run past
    # the native size into the padded strip, which is what those cells cover.
    h, w = 100, 130
    hp, wp = 8, 10
    x0, y0, sx, sy = patch_pixel_box(hp - 1, wp - 1, h, w)
    assert (x0 + sx, y0 + sy) == (wp * PATCH_SIZE, hp * PATCH_SIZE) == (140, 112)
    assert x0 + sx > w and y0 + sy > h


def test_patch_pixel_box_scales_up_past_max_dino_side():
    # Long side above MAX_DINO_SIDE -> the forward pass is downscaled by
    # MAX_DINO_SIDE/long_side, so one patch covers 14/scale NATIVE pixels.
    h, w = 2000, 3000
    scale = MAX_DINO_SIDE / w
    expected_side = PATCH_SIZE / scale
    x0, y0, sx, sy = patch_pixel_box(3, 5, h, w)
    assert sx == sy == expected_side > PATCH_SIZE
    assert (x0, y0) == (5 * expected_side, 3 * expected_side)
    # still tiles: consecutive cells abut
    nx0, _, _, _ = patch_pixel_box(3, 6, h, w)
    assert math.isclose(nx0, x0 + sx)


def test_patch_pixel_box_broadcasts_over_tensors():
    # The explorer feeds it whole meshgrids of cell indices at once.
    h = w = 10 * PATCH_SIZE
    ys, xs = torch.meshgrid(torch.arange(10), torch.arange(10), indexing="ij")
    x0, y0, sx, sy = patch_pixel_box(ys.reshape(-1), xs.reshape(-1), h, w)
    assert x0.shape == y0.shape == (100,)
    assert torch.allclose(x0, xs.reshape(-1).float() * PATCH_SIZE)
    assert torch.allclose(y0, ys.reshape(-1).float() * PATCH_SIZE)
    assert (sx, sy) == (PATCH_SIZE, PATCH_SIZE)
