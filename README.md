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

### Example scene: `bonsai` (recommended)

To try the pipeline end-to-end without your own capture, this repo defaults to
the `bonsai` scene from the [Mip-NeRF 360 dataset](https://jonbarron.info/mipnerf360/)
-- a real, casually-captured, 360°, object-centric scene (a bonsai on a stand,
on a small table), used here as a stand-in for VF-NeRF's own "table" scene,
which was never publicly released. Unlike nerfstudio's own example captures
(see below), it's hosted on Google Cloud Storage, not personal Google Drive, so
it isn't subject to the same throttling/availability problems:
````bash
python scripts/downloads/download_mipnerf360.py --scene bonsai --save-dir data/mipnerf360
````
This only fetches `bonsai`'s own files (~1GB) out of the ~12.5GB archive that
bundles all 9 Mip-NeRF 360 scenes together (via HTTP range requests), and
converts its bundled COLMAP reconstruction straight to `transforms.json` --
no local COLMAP run needed. Other indoor, bounded, object-centric scenes from
the same release are available with `--scene {counter,kitchen,room}`.

### Alternative: nerfstudio's own example scenes

````bash
ns-download-data nerfstudio --capture-name=poster
````
> **Google Drive throttling**: nerfstudio's example captures are hosted on Google
> Drive, which frequently blocks scripted/automated downloads (`gdown`) with a
> "cannot retrieve the public link" error -- confirmed against several captures.
> At least one (`poster`) currently returns a plain 404 even from a normal
> logged-in browser, meaning the file itself appears to have been removed, not
> just throttled -- this is why `bonsai` is the recommended default above.

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

## Training via the Colab CLI (alternative to the notebook)

Instead of clicking through [`notebooks/train_conditional_nf_colab.ipynb`](notebooks/train_conditional_nf_colab.ipynb)
in a browser, Google's official [`google-colab-cli`](https://github.com/googlecolab/google-colab-cli)
(shipped June 2026) drives a real Colab GPU runtime from your own terminal --
no browser round-trips, and (unlike older `colab-ssh`/ngrok tunneling tricks,
now against Colab's terms of service) it's Google's own sanctioned tool.

```bash
pip install google-colab-cli   # or: uv tool install google-colab-cli

colab new --gpu T4              # provision a runtime (plain --gpu isn't documented
                                 # as Pro-only; --high-mem explicitly requires Colab Pro/Pro+)
colab console                   # drop into a raw shell on that VM
```

From the `colab console` shell, this is an ordinary terminal on the remote VM --
run the same commands you would locally:

```bash
git clone https://github.com/itayhanoch/VF-NeRF-conditional.git
cd VF-NeRF-conditional
pip install torch==1.13.1 torchvision functorch --extra-index-url https://download.pytorch.org/whl/cu117
pip install ninja "git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch"
pip install -e . && pip install -e ./normalizing-flows

python scripts/downloads/download_mipnerf360.py --scene bonsai --save-dir data/mipnerf360
ns-train nerfacto --data data/mipnerf360/bonsai
python scripts/train_conditional_nf.py \
    --nerf-config outputs/bonsai/nerfacto/TIMESTAMP/config.yml \
    --scene-dir data/mipnerf360/bonsai \
    --checkpoint-dir checkpoints/conditional_nf/bonsai
```

Useful CLI commands from your own terminal (outside `colab console`), run against
the same session with `-s NAME` or the most recent one by default:
- `colab drivemount` -- mount Google Drive, so checkpoints can land in the same
  Drive path the notebook uses, for resuming across sessions (see "GPU budget" below).
- `colab install -r requirements.txt` -- install deps via uv/pip without dropping into the console.
- `colab upload`/`colab download` -- move files to/from the VM directly.
- `colab status` / `colab sessions` -- check what's running and its hardware.
- `colab stop` -- tear the session down when done.

Auth is a one-time `--auth {oauth2,adc}` login (default `adc`), not per-command.

### GPU budget: splitting nerfacto and conditional-NF training across sessions

A free-tier T4 is meaningfully slower than the GPUs typical benchmarks are run
on -- training the frozen nerfacto backbone for the default 30,000 iterations
plus the one-time tiny-cuda-nn compile step can plausibly take 30-70+ minutes,
which may not comfortably fit in a single free-tier session alongside the
conditional-NF training too. Since both the notebook and `ns-train`'s own
checkpointing land on Google Drive, the two steps don't need to happen in the
same session:
1. **Session 1**: `colab new --gpu T4` -> `colab drivemount` -> train nerfacto only, to a Drive-backed output dir.
2. **Session 2** (later, fresh daily quota): `colab new --gpu T4` -> `colab drivemount` -> point `--nerf-config` at the checkpoint saved in session 1, and run only `scripts/train_conditional_nf.py` (much cheaper -- small MLPs, no tiny-cuda-nn, no hash grid).

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

