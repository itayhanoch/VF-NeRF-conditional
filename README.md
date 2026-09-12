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
   `(position, direction)` candidates ranked by likelihood; pick any to render as
   novel views through the frozen NeRF. To probe on Kaggle instead, use
   `app/pick_points.py` locally to click the points and paste the list into
   [`notebooks/train_vf_nerf_kaggle.ipynb`](notebooks/train_vf_nerf_kaggle.ipynb)
   cell 6a (cell 6b then renders them).

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
drives a real Colab GPU runtime from your own terminal. `colab exec -f <script.py>`
runs a **local** Python script *on* the remote VM and streams its output back --
it does not open an interactive shell, so the setup below uses a launcher script
that writes and detaches a background bash job, rather than a chain of individual
commands (an interactive `pip install torch==1.13.1 ...` one-liner run this way
hits real, previously-debugged failures: Colab's system Python is too new for
the `cu117` wheels this repo needs, the CUDA 11.7 dev headers tiny-cuda-nn's
build needs aren't preinstalled, `setuptools>=81` silently breaks `pkg_resources`,
and training on full-resolution images OOMs on a free-tier T4 -- all of which the
scripts below already work around).

```bash
pip install google-colab-cli   # or: uv tool install google-colab-cli

colab new --gpu T4   # opens a browser OAuth prompt once; note the session alias it prints
```

**1. Launch setup + training** ([`scripts/colab/launch_train.py`](scripts/colab/launch_train.py)
builds an isolated Python 3.10 venv, installs the CUDA 11.7 toolchain, compiles
tiny-cuda-nn, clones the repo, downloads
the `bonsai` scene, generates downscaled training images, and launches
`ns-train nerfacto` -- all via a single `nohup`'d background job on the VM, so it
survives the `exec` call returning):

```bash
colab exec -s <session> -f scripts/colab/launch_train.py --timeout 60
```

**2. Poll progress** ([`scripts/colab/check_status.py`](scripts/colab/check_status.py)
prints whether the job is still running, its last log lines, and any traceback
count -- repeat this every few minutes rather than leaving the terminal idle,
since an idle Colab connection is what disconnects the session):

```bash
colab exec -s <session> -f scripts/colab/check_status.py --timeout 30
```

**3. Download the checkpoint the moment it finishes** -- this script does **not**
mount Google Drive, so the checkpoint only exists on the ephemeral VM disk until
downloaded. Do this immediately, with no chat/thinking pause in between (a
disconnect between "training finished" and "checkpoint downloaded" is a real,
unrecoverable loss -- it has happened twice in this project):

```bash
colab download -s <session> /content/VF-NeRF-conditional/outputs "./checkpoints/nerfacto_bonsai"
```

Other useful commands, run from your own terminal against the same session
with `-s <session>`:
- `colab status` / `colab sessions` -- check what's running and its hardware.
- `colab log -s <session>` -- raw session log, if `check_status.py` isn't enough.
- `colab stop -s <session>` -- tear the session down when done.
- If `colab new` fails with `TooManyAssignmentsError`, a previous session is
  still holding a GPU/CPU reservation -- go to colab.research.google.com ->
  Runtime -> Manage sessions and terminate it there before retrying.

Once the frozen NeRF checkpoint is downloaded, `scripts/train_conditional_nf.py`
(step 4 below) can run the same way: write a second launcher script that installs
into the same venv and calls it directly (much cheaper than nerfacto training --
small MLPs, no tiny-cuda-nn, no hash grid), or just use the notebook for that
step, which mounts Drive and persists checkpoints as it trains instead of only
at the end.

## 5. Explore interactively

```bash
pip install -e ".[ui]"
python app/gradio_app.py \
    --nerf-config {outputs_dir}/{scene}/nerfacto/{timestamp}/config.yml \
    --cond-nf-checkpoint checkpoints/conditional_nf/{scene}/latest.pt \
    --scene-dir {PROCESSED_DATA_DIR}
```

### Picking probe points for the Kaggle notebook

If you probe the conditional NF on Kaggle instead (see
[`notebooks/train_vf_nerf_kaggle.ipynb`](notebooks/train_vf_nerf_kaggle.ipynb) --
Kaggle's JupyterLab has no `ipympl` widget frontend, so there is no in-notebook
click capture), pick the points here first. This tool needs only
`matplotlib` + `Pillow` (+ `remotezip` for the one-time image download), not the
full stack:

```bash
pip install matplotlib pillow remotezip

python app/pick_points.py                      # bonsai training frames (downloads them once)
python app/pick_points.py --also-scenes counter kitchen room  # + those scenes as external picks
python app/pick_points.py --extra ~/photo.jpg  # also pick on your own image(s)
python app/pick_points.py --extra-only --extra a.jpg b.png   # only your images (no download)
```

It opens the images in a window; page with the on-screen buttons (or `n`/`p`),
click the objects you want to probe, and it prints a `COORDS` block to paste into
the notebook's cell 6a. Each entry is `["name.ext", x, y, "TAG"]` -- the image's
basename plus where it came from. A bonsai frame is matched by filename in the
notebook's `images_2/`; any other name is an external image (attach a Kaggle
Dataset containing it; it is looked up under `/kaggle/input/**`). Referencing
frames by filename rather than position keeps the picker and the notebook
aligned. Coordinates are in that image's own pixels.

#### Train / test tags

Frames of the primary `--scene` are tagged `TRAIN` or `TEST` in the window title
(green / orange-red) and as the **4th field of every emitted row**, so the
provenance travels with the coordinates instead of living in a comment:

```python
COORDS = [
    ["DSCF5565.JPG", 981, 329, "TRAIN"],     # bonsai frame 0
    ["DSCF5575.JPG", 205, 163, "TEST"],      # bonsai frame 10
    ["piano.jpeg", 197, 257, "EXTERNAL"],    # attach a Kaggle Dataset containing 'piano.jpeg'
]
```

A `TEST` frame is one the frozen NeRF and the flow never saw, but it still has a
camera pose, and cell 6b's explorer loads the held-out split's cameras too: a
TEST probe gets the frozen NeRF's real depth along its ray as the render backoff
(logged as `depth(test)`) and its own 3-D source point, exactly like a TRAIN
frame. Only EXTERNAL images have no camera -- they fall back to the scene-wide
constant backoff and get no source point.

The tag is provenance, not an input -- nothing branches on it, but cell 6b echoes
it in its log, on each montage, and into `dino_consistency.json`, so results can
be split by TRAIN / TEST / EXTERNAL after the fact. The render-backoff override
that used to be the 4th field is now the 5th
(`[ref, x, y, "TRAIN", 0.42]`); a bare number in the 4th slot is still read as a
backoff, so blocks picked before tags existed keep working.

The tags reproduce the dataparser's split exactly (`train_split_fraction = 0.9`,
equally spaced train indices, remainder held out). That split is positional over
`transforms.json`'s `frames`, i.e. **COLMAP order, which is not alphabetical** --
for `counter` and `kitchen` an alphabetical guess would mis-tag 10 and 4 frames.
So the picker reads the order from the reconstruction: it uses `transforms.json`
if `scripts/downloads/download_mipnerf360.py` already produced one, otherwise it
fetches just that scene's `sparse/0/images.bin` once (~30-45 MB, cached). Pass
`--no-split-tags` to skip the fetch, or `--train-split-fraction` if you changed
the dataparser's. Anything that goes wrong (offline, no `remotezip`) just prints
a warning and leaves the frames untagged.

External images are never tagged -- they belong to no split.

#### Downscaling external images

DINOv2 runs at native resolution only up to a 1568px long side; above that it
features a shrunk copy, so a patch on a 4444px photo covers far more of the scene
than one on a 1559px training frame. Normalise your external images once, in
place, before picking:

```bash
python scripts/downscale_external_images.py --dry-run   # look first
python scripts/downscale_external_images.py             # rewrite at 1/4 size
```

Originals are parked in a sibling `external images_originals/`; the rewritten
files keep their basenames, so `external images/` stays the folder you upload as
the Kaggle Dataset and the picked coordinates are in the same pixel space the
notebook loads. Re-running is a no-op. The script warns about any image that ends
up under ~518px, where a DINO patch smears over a large part of the subject.

For an external image only its DINOv2 feature at the clicked pixel is used -- the
sampled novel views are still rendered as bonsai views through the frozen bonsai
NeRF, so it is most useful when the external image shows similar content.

### Scene scale reference

Every evaluation number is in nerfstudio's normalized scene units. To give them a
physical sense, pick the two ends of a recognisable object (a plate, a tile) on a
frame of each scene -- twice per scene is plenty:

```bash
pip install matplotlib pillow remotezip
python app/pick_scale_pairs.py --scenes counter kitchen room   # click A, then B on the same frame
```

Type an object name into the `label` box before the second click to tag the pair;
the toolbar zoom is kept while you stay on a frame, so zoom in to place the points.
It prints a `SCALE_PAIRS` block (`["name.ext", x1, y1, x2, y2, "TRAIN", "plate"]`
rows per scene) to paste into
[`notebooks/measure_scene_scale.ipynb`](notebooks/measure_scene_scale.ipynb), which
restores the frozen NeRFs, shoots each frame's camera rays through A, B and their
pixel midpoint M, and reports `|AB|` (normalized and original COLMAP units), `A-M`
and `M-B` against `|AB|/2` (a flatness / depth-reliability check), the three depths,
and the depth filter bounds and scene backoff in units of `|AB|`
(`scripts/measure_scene_scale.py`; archived as `report_scene_scale.zip`).

### Reading a probe montage (cell 6b)

One row per probe: the source frame (red box = the 14x14 DINO patch the condition
was read from, magnified in the corner inset), then the `RENDER_TOP`
highest-likelihood samples rendered through the frozen NeRF, each carrying:

- a **green wash** over the pixels passing VF-NeRF's view-likelihood mask (the
  flow's log-density at every pixel's NeRF-depth surface point, min-max
  normalised per view, cut at `VF_SCORE_THRESHOLD`), opacity ramping 0.2 -> 0.55
  with that likelihood;
- a **red box** on the DINO patch the sampled point lands in, with its cosine
  similarity to the source patch (`pt cos`) in the title -- read it against the
  scene's same-patch / random-patch cosines from the DINO reconstruction report;
- a **green circle** where the source pixel's own 3-D point projects into the
  view (dashed if the render occludes it) and `|S-P|`, the distance between
  that point and the sampled one, in normalized / original units.

## 6. Evaluation reports

All evaluations run from cells of
[`notebooks/train_vf_nerf_kaggle.ipynb`](notebooks/train_vf_nerf_kaggle.ipynb);
each cell zips its outputs to `/kaggle/working/report_<name>.zip` the moment it
finishes. The scripts behind them run anywhere the stack is installed:

| cell | script | what it answers |
|---|---|---|
| 3c | `scripts/depth_probe.py` | what NeRF depth 0.3 / 3.0 looks like per scene (the outlier filter bounds): one frame, the same camera slid to those depths, and the depth map with contours |
| 5b | `scripts/eval_cond_nf_likelihood.py --depth-range 0.3 3.0 --shuffled` | held-out log-likelihood, raw and with bad-depth samples filtered out, against a **shuffled-condition baseline** (same targets, another pixel's DINO feature -- does the flow use the feature at all?) |
| 5c | `scripts/eval_cond_nf_nearest_sample.py --dino-views 200` | the flow run the other way: 100 samples per pixel, distance / angle of the closest one (`min`) and of the flow's own top-ranked one (`top`) to the true point / direction, hit rate under 5 % of the scene backoff. **DINO round-trip**: for 200 random pixels per split the `min` and `top` samples are rendered through the frozen NeRF (a 364-px window at native pixel pitch, looking along the sampled direction at the sampled point) and the DINO feature of the patch the point lands in is scored by cosine against the source feature -- read against 6c's same-patch ceiling and random-patch floor; montage `*_views_<split>.png` |
| 6c | `scripts/eval_dino_reconstruction.py` | can DINO on a NeRF render be trusted: same-patch cosine between a real frame and its render (ceiling) vs. a random render patch (floor) |

The standalone [`notebooks/eval_cond_nf_likelihood.ipynb`](notebooks/eval_cond_nf_likelihood.ipynb)
and [`notebooks/eval_dino_reconstruction.ipynb`](notebooks/eval_dino_reconstruction.ipynb)
run 5b and 6c against restored checkpoints without training anything;
[`notebooks/measure_scene_scale.ipynb`](notebooks/measure_scene_scale.ipynb) does the same
for the scene scale reference above (`report_scene_scale.zip`).


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

