# VF-NeRF: Viewshed Field For Rigid NeRF Registration
## ECCV 2024
<a href="https://leosegre.github.io/VF_NeRF/"><img src="https://img.shields.io/static/v1?label=Project&message=Website&color=blue"></a>
<a href="https://arxiv.org/abs/2404.03349"><img src="https://img.shields.io/badge/arXiv-2404.03349-b31b1b.svg"></a>

By [Leo Segre](https://scholar.google.co.il/citations?hl=iw&user=A7FWhoIAAAAJ) and [Shai Avidan](https://scholar.google.co.il/citations?hl=iw&user=hpItE1QAAAAJ)

This repo is the official implementation of "[VF-NeRF: Viewshed Field For Rigid NeRF Registration](https://arxiv.org/pdf/2404.03349.pdf)".

<p align="center">
<img src="images/merged_office.png" width="384">
</p>

## Citation
If you find this useful, please cite this work as follows:
```bibtex
@misc{segre2024vfnerf,
      title={VF-NeRF: Viewshed Fields for Rigid NeRF Registration}, 
      author={Leo Segre and Shai Avidan},
      year={2024},
      eprint={2404.03349},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

# About

This fork repurposes VF-NeRF's Normalizing-Flow machinery away from NeRF-to-NeRF
registration and toward **conditional novel-view generation**: a conditional
RealNVP flow learns `P((3D point, 3D direction) | DINO feature)` against a single,
**frozen** pretrained nerfacto NeRF (never fine-tuned during flow training), so that
clicking a point in an image -- via its DINOv2 feature -- can generate plausible
novel-view rays through that scene. The flow architecture is ported from
[itayhanoch/conditional-normalizing-flows-toy](https://github.com/itayhanoch/conditional-normalizing-flows-toy),
with its coupling-layer masking realigned to match the original VF-NeRF's
position-vs-direction block split.

The original registration pipeline (`reg_pipeline*.py`, `scripts/fgr.py`, the
`register-nerfacto` method, etc.) has been removed; see the original
[VF-NeRF paper](https://arxiv.org/pdf/2404.03349.pdf) and
[leosegre/VF_NeRF](https://github.com/leosegre/VF_NeRF) for that use case.

## Workflow

1. **Train a frozen NeRF backbone** on your scene (or a nerfstudio example scene)
   with plain `ns-train nerfacto` -- see "Preparing your data" below. This NeRF is
   trained once and never touched again.
2. **Train the conditional Normalizing Flow** against that frozen NeRF, conditioned
   on DINOv2 features -- `scripts/train_conditional_nf.py`, meant to run on a
   Colab GPU (see [`notebooks/train_conditional_nf_colab.ipynb`](notebooks/train_conditional_nf_colab.ipynb)).
   This is a standalone training loop, decoupled from nerfstudio's `Trainer`/`Pipeline`.
3. **Explore interactively**, locally, once both checkpoints are downloaded --
   `app/gradio_app.py` (`pip install -e ".[ui]"` first): pick a training image (or
   upload an external one), click a point, and the conditional NF samples 100
   `(position, direction)` candidates ranked by likelihood; pick 1-5 to render as
   novel views through the frozen NeRF.

## 1. Installation: Setup the environment

### Prerequisites

You must have an NVIDIA video card with CUDA installed on the system. This library has been tested with version 11.7 of CUDA. You can find more information about installing CUDA [here](https://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html)

### Create environment

Nerfstudio requires `python >= 3.7`. We recommend using conda to manage dependencies. Make sure to install [Conda](https://docs.conda.io/en/latest/miniconda.html) before proceeding.

```bash
conda create --name vf_nerf -y python=3.8
conda activate vf_nerf
python -m pip install --upgrade pip
```

### Dependencies

Install pytorch with CUDA (this repo has been tested witt CUDA 11.7) and [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn)

```bash
pip install torch==1.13.1 torchvision functorch --extra-index-url https://download.pytorch.org/whl/cu117
pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```

### Installing vf_nerf (Based on [Nerfstudio](https://docs.nerf.studio/))

```bash
git clone https://github.com/leosegre/VF_NeRF.git
cd VF_NeRF
pip install --upgrade pip setuptools
pip install -e .
```

### Installing Normalizing-flows

```bash
cd normalizing-flows
pip install -e .
cd ..
```

## 2. Preparing your data

Assuming you have a video or a set of images, run COLMAP to get a valid `transforms.json`:
````bash
ns-process-data {video,images} --data {DATA_PATH} --output-dir {PROCESSED_DATA_DIR}
````
Or use one of nerfstudio's bundled example scenes to try the pipeline end-to-end
without your own capture:
````bash
ns-download-data nerfstudio --capture-name=poster
````

## 3. Train the frozen NeRF backbone

```bash
ns-train nerfacto --data {PROCESSED_DATA_DIR}
```
This uses the `TCNNNerfactoField` (tiny-cuda-nn) backbone, matching upstream
nerfstudio/VF-NeRF -- see "Dependencies" above for the tiny-cuda-nn build step
(also required on Colab; the training notebook handles it).

## 4. Train the conditional Normalizing Flow

```bash
python scripts/train_conditional_nf.py \
    --nerf-config {outputs_dir}/{scene}/nerfacto/{timestamp}/config.yml \
    --scene-dir {PROCESSED_DATA_DIR} \
    --checkpoint-dir checkpoints/conditional_nf/{scene}
```
This is a standalone script (not `ns-train`) meant to run on a Colab GPU for real
training runs -- see [`notebooks/train_conditional_nf_colab.ipynb`](notebooks/train_conditional_nf_colab.ipynb),
which handles Drive-mounted checkpointing so training survives disconnects.

## 5. Explore interactively

```bash
pip install -e ".[ui]"
python app/gradio_app.py \
    --nerf-config {outputs_dir}/{scene}/nerfacto/{timestamp}/config.yml \
    --cond-nf-checkpoint checkpoints/conditional_nf/{scene}/latest.pt \
    --scene-dir {PROCESSED_DATA_DIR}
```


# Built On
<a href="https://github.com/nerfstudio-project/nerfstudio">
<!-- pypi-strip -->
<picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://docs.nerf.studio/_images/logo.png" />
<!-- /pypi-strip -->
    <img alt="nerfstudio logo" src="https://docs.nerf.studio/_images/logo.png" width="150px" />
<!-- pypi-strip -->
</picture>
<!-- /pypi-strip -->
</a>

- A collaboration friendly studio for NeRFs

