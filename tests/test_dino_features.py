"""Unit tests for the pixel-alignment math in nerfstudio.utils.dino_features --
deliberately network-independent (no DINOv2 model download): only tests
_pad_to_patch_multiple and upsample_to_pixels's crop logic, using a synthetic
patch grid in place of a real DINOv2 forward pass. Extraction-through-a-real-
model is exercised manually/in Colab (network access + a GPU are both assumed
there), not in this suite.
"""
import torch

from nerfstudio.utils.dino_features import EMBED_DIM, PATCH_SIZE, DinoExtractor, _pad_to_patch_multiple


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
