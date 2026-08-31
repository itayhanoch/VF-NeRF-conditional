#!/usr/bin/env python
"""Local interactive UI for the trained conditional-NF novel-view generator.

Pick a training-set image (or upload an external one), click a point, and its
DINOv2 feature becomes the condition for the conditional Normalizing Flow: 100
(position, direction) candidates are sampled, ranked by likelihood, and the
top ones can be rendered into novel views through the frozen NeRF.

Run after downloading a frozen NeRF checkpoint (config.yml) and a trained
conditional-NF checkpoint (.pt, from scripts/train_conditional_nf.py) locally:

    python app/gradio_app.py \\
        --nerf-config outputs/bonsai/nerfacto/TIMESTAMP/config.yml \\
        --cond-nf-checkpoint checkpoints/conditional_nf/bonsai/latest.pt \\
        --scene-dir data/mipnerf360/bonsai
"""
import argparse
from pathlib import Path
from typing import List, Tuple

import gradio as gr
import numpy as np
import torch

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.fields.nf_field import ConditionalNFField
from nerfstudio.utils.dino_features import DinoExtractor, load_image_chw_01
from nerfstudio.utils.eval_utils import eval_setup

TOP_K_SHOWN = 20  # of the sampled candidates, how many (ranked) to show for selection
DEFAULT_WORLD_UP = torch.tensor([0.0, 0.0, 1.0])  # nerfstudio's default post-auto-orient world-up axis


def default_backoff_distance(cameras: Cameras) -> float:
    """Scene-scale-aware default camera backoff: median training-camera distance
    to the scene's camera centroid."""
    centers = cameras.camera_to_worlds[..., :3, 3]
    scene_center = centers.mean(dim=0)
    dists = (centers - scene_center).norm(dim=-1)
    return dists.median().item()


def build_camera_from_point_direction(
    position: torch.Tensor,
    direction: torch.Tensor,
    reference_cameras: Cameras,
    backoff_distance: float,
    world_up: torch.Tensor = DEFAULT_WORLD_UP,
) -> Cameras:
    """A sampled (position, direction) pair, matching the training-time convention
    `point = ray_origin + ray_direction * depth` (direction points FROM camera
    TOWARD the point) -> a one-camera nerfstudio Cameras object, reusing the
    training scene's own intrinsics.

    nerfstudio's camera convention (verified from cameras.py/camera_utils.py):
    local -Z = forward, local +Y = up, local +X = right; world-up after the
    default orientation_method="up" auto-orientation is world +Z.
    """
    device = position.device
    world_up = world_up.to(device)

    forward = direction / direction.norm().clamp_min(1e-8)
    up_ref = world_up
    if torch.abs(torch.dot(forward, up_ref)) > 0.99:  # near-degenerate: looking ~straight up/down
        up_ref = torch.tensor([1.0, 0.0, 0.0], device=device)
    right = torch.cross(forward, up_ref)
    right = right / right.norm().clamp_min(1e-8)
    up = torch.cross(right, forward)

    camera_origin = position - forward * backoff_distance
    rotation = torch.stack([right, up, -forward], dim=-1)  # columns: local +X,+Y,+Z in world coords
    c2w = torch.cat([rotation, camera_origin.unsqueeze(-1)], dim=-1)  # [3,4]

    return Cameras(
        camera_to_worlds=c2w.unsqueeze(0),
        fx=reference_cameras.fx[0:1],
        fy=reference_cameras.fy[0:1],
        cx=reference_cameras.cx[0:1],
        cy=reference_cameras.cy[0:1],
        width=reference_cameras.width[0:1],
        height=reference_cameras.height[0:1],
        camera_type=reference_cameras.camera_type[0:1],
    ).to(device)


def load_conditional_nf(checkpoint_path: Path, device: torch.device) -> Tuple[ConditionalNFField, str]:
    ckpt = torch.load(checkpoint_path, map_location=device)
    field = ConditionalNFField(
        context_dim=ckpt["context_dim"],
        num_dims=ckpt.get("num_dims", 6),
        num_blocks=ckpt["num_blocks"],
        hidden_dim=ckpt["hidden_dim"],
        cond_prior=ckpt["cond_prior"],
        use_cond_in_coupling=True,
        use_batchnorm=ckpt["use_batchnorm"],
        device=str(device),
    )
    field.load_state_dict(ckpt["model_state"])
    field.eval()
    return field, ckpt["dino_model_name"]


class InteractiveApp:
    def __init__(self, nerf_config: Path, cond_nf_checkpoint: Path, scene_dir: Path, device: str):
        self.device = torch.device(device)

        print(f"Loading frozen NeRF from {nerf_config} ...")
        config, pipeline, _, _ = eval_setup(nerf_config, test_mode="inference")
        self.nerf_model = pipeline.model.to(self.device)
        self.nerf_model.eval()
        for p in self.nerf_model.parameters():
            p.requires_grad_(False)

        print(f"Loading training cameras/images from {scene_dir} ...")
        # Build the reference cameras in the SAME frame the frozen NeRF (and hence the
        # conditional NF) was trained in -- the checkpoint's own dataparser config, not
        # stock NerfstudioDataParserConfig defaults (center-method / scene-scale /
        # downscale-factor differ, which would misplace both the cameras and the pixel
        # coordinates the DINO patch grid is indexed by).
        dataparser = config.pipeline.datamanager.dataparser
        dataparser.data = scene_dir
        outputs = dataparser.setup().get_dataparser_outputs(split="train")
        self.reference_cameras = outputs.cameras.to(self.device)
        self.image_filenames = [Path(p) for p in outputs.image_filenames]
        self.backoff_distance = default_backoff_distance(self.reference_cameras)
        print(f"Default render backoff distance: {self.backoff_distance:.3f} (scene units)")

        print(f"Loading conditional-NF checkpoint from {cond_nf_checkpoint} ...")
        self.field, dino_model_name = load_conditional_nf(cond_nf_checkpoint, self.device)

        self.extractor = DinoExtractor(model_name=dino_model_name, device=device)
        self._grid_cache: dict = {}   # image path -> (patch grid [C,Hp,Wp] cpu, H, W)

    def image_choices(self) -> List[str]:
        return [str(p) for p in self.image_filenames]

    # --- condition extraction -------------------------------------------------

    def _patch_grid(self, path: Path):
        """[EMBED_DIM, Hp, Wp] DINOv2 patch-token grid for one image + its (H, W)."""
        key = str(path)
        if key not in self._grid_cache:
            img = load_image_chw_01(path)
            with torch.no_grad():
                grid, (h, w) = self.extractor.extract_patch_grid(img)
            self._grid_cache[key] = (grid.float().cpu(), int(h), int(w))
        return self._grid_cache[key]

    @staticmethod
    def _pixel_to_patch_feature(grid: torch.Tensor, h: int, w: int, x: int, y: int) -> torch.Tensor:
        """Pixel (x=col, y=row) -> the DINO feature of the patch cell it falls in.

        Indexes the patch grid directly (same as the NF trainer's sample_batch); no
        per-pixel upsampled map, and matches training-time condition semantics.
        """
        hp, wp = grid.shape[-2:]
        px = min(max(int(x * wp / w), 0), wp - 1)
        py = min(max(int(y * hp / h), 0), hp - 1)
        return grid[:, py, px].reshape(-1).clone()

    @staticmethod
    def _click_xy(evt: gr.SelectData):
        if evt is None or evt.index is None or evt.index[0] is None:
            raise gr.Error("Click a point on the image first.")
        return int(evt.index[0]), int(evt.index[1])  # (col, row) = (x, y)

    def training_image_condition(self, image_path_str: str, evt: gr.SelectData) -> torch.Tensor:
        x, y = self._click_xy(evt)
        grid, h, w = self._patch_grid(Path(image_path_str))
        return self._pixel_to_patch_feature(grid, h, w, x, y)

    def external_image_condition(self, pil_image, evt: gr.SelectData) -> torch.Tensor:
        x, y = self._click_xy(evt)
        arr = np.asarray(pil_image.convert("RGB"), dtype=np.float32) / 255.0
        img = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
        with torch.no_grad():
            grid, (h, w) = self.extractor.extract_patch_grid(img)
        return self._pixel_to_patch_feature(grid.float().cpu(), int(h), int(w), x, y)

    # --- sample / rank / render -------------------------------------------------

    def sample_and_rank(self, condition: torch.Tensor, num_samples: int = 100, top_k: int = TOP_K_SHOWN):
        condition = condition.reshape(-1).to(self.device)
        assert condition.numel() == self.field.context_dim, (
            f"condition is {tuple(condition.shape)}, expected [{self.field.context_dim}]")
        with torch.no_grad():
            samples = self.field.sample(num_samples=num_samples, context=condition)
            cond_batch = condition.unsqueeze(0).expand(num_samples, -1)
            log_p = self.field.log_prob(samples, cond_batch).squeeze(-1)
        order = torch.argsort(log_p, descending=True)[:top_k]
        samples, log_p = samples[order], log_p[order]

        rows = []
        for i in range(samples.shape[0]):
            pos, dirn = samples[i, :3].tolist(), samples[i, 3:].tolist()
            rows.append([i, *[round(v, 3) for v in pos], *[round(v, 3) for v in dirn], round(log_p[i].item(), 3)])
        return samples.cpu(), rows

    def render_selected(self, samples: torch.Tensor, selected_labels: List[str],
                        render_downscale: float = 3.0) -> List[np.ndarray]:
        images = []
        for label in selected_labels:
            idx = int(label)
            pos = samples[idx, :3].to(self.device)
            dirn = samples[idx, 3:].to(self.device)
            camera = build_camera_from_point_direction(pos, dirn, self.reference_cameras, self.backoff_distance)
            if render_downscale and render_downscale != 1.0:
                camera.rescale_output_resolution(1.0 / render_downscale)
            ray_bundle = camera.generate_rays(camera_indices=0)
            with torch.no_grad():
                out = self.nerf_model.get_outputs_for_camera_ray_bundle(ray_bundle)
            rgb = (out["rgb"].cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            images.append(rgb)
        return images


def build_ui(app: InteractiveApp, render_downscale_default: float = 3.0) -> gr.Blocks:
    columns = ["#", "x", "y", "z", "dx", "dy", "dz", "log_likelihood"]

    with gr.Blocks(title="Conditional-NF Novel View Explorer") as demo:
        gr.Markdown("# Conditional-NF Novel View Explorer")
        condition_state = gr.State(value=None)
        samples_state = gr.State(value=None)

        with gr.Tabs():
            with gr.TabItem("Training-set image"):
                image_dropdown = gr.Dropdown(choices=app.image_choices(), label="Training image")
                train_image = gr.Image(label="Click a point", type="filepath", interactive=False)
                image_dropdown.change(lambda p: p, inputs=image_dropdown, outputs=train_image)
                train_image.select(app.training_image_condition, inputs=image_dropdown, outputs=condition_state)

            with gr.TabItem("External image"):
                gr.Markdown("Upload any image and click a point -- only its DINO feature at that pixel is used; rendering still reuses the training scene's camera intrinsics.")
                external_image = gr.Image(label="Click a point", type="pil", sources=["upload"])
                external_image.select(app.external_image_condition, inputs=external_image, outputs=condition_state)

        num_samples = gr.Slider(minimum=20, maximum=500, value=100, step=10, label="Samples to draw")
        sample_btn = gr.Button("Sample")
        results_table = gr.Dataframe(
            headers=columns, label=f"Top {TOP_K_SHOWN} candidates, ranked by log-likelihood",
            interactive=False, col_count=(len(columns), "fixed"), datatype="number")
        selection = gr.CheckboxGroup(choices=[], label="Select 1-5 candidates to render (by #)")

        def do_sample(condition, n):
            if condition is None:
                raise gr.Error("Click a point on an image first.")
            samples, rows = app.sample_and_rank(condition, num_samples=int(n))
            choices = [str(r[0]) for r in rows]
            return samples, rows, gr.CheckboxGroup(choices=choices, value=[])

        sample_btn.click(do_sample, inputs=[condition_state, num_samples], outputs=[samples_state, results_table, selection])

        render_downscale = gr.Slider(minimum=1.0, maximum=6.0, value=render_downscale_default, step=0.5,
                                     label="Render downscale (higher = faster, lower-res)")
        render_btn = gr.Button("Render selected")
        gallery = gr.Gallery(label="Novel views", columns=5, height="auto", object_fit="contain")

        def do_render(samples, selected, downscale):
            if not selected:
                raise gr.Error("Select at least one candidate.")
            if len(selected) > 5:
                raise gr.Error("Select at most 5 candidates.")
            return app.render_selected(samples, selected, render_downscale=float(downscale))

        render_btn.click(do_render, inputs=[samples_state, selection, render_downscale], outputs=gallery)

    return demo


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--nerf-config", type=Path, required=True)
    p.add_argument("--cond-nf-checkpoint", type=Path, required=True)
    p.add_argument("--scene-dir", type=Path, required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--render-downscale", type=float, default=3.0,
                   help="Downscale factor for the rendered novel views (higher = faster)")
    p.add_argument("--share", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    app = InteractiveApp(args.nerf_config, args.cond_nf_checkpoint, args.scene_dir, args.device)
    demo = build_ui(app, render_downscale_default=args.render_downscale)
    demo.queue(default_concurrency_limit=1)
    demo.launch(share=args.share, show_error=True)


if __name__ == "__main__":
    main()
