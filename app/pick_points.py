#!/usr/bin/env python
"""Local interactive point picker for the conditional-NF probe workflow.

Opens a matplotlib window with the primary scene's training images (default
bonsai -- change with --scene), plus any --also-scenes Mip-NeRF 360 scenes and
any --extra images you pass. Click points on the objects you want to probe;
the script prints a ready-to-paste `COORDS` block for the Kaggle notebook's
cell 6a (and, equivalently, for `kaggle_explorer.py` / `app/gradio_app.py`).

Each `COORDS` entry is `["name.ext", x, y, "TAG"]` -- the image's basename plus
where it came from:
  * a --scene frame -> found in the notebook's images_<downscale>/ folder, and
    tagged "TRAIN" or "TEST" for the split the notebook's dataparser makes
  * any other image (an --also-scenes frame or an --extra file) -> "EXTERNAL";
    on Kaggle, attach a Dataset containing that file (it is looked up by name
    under /kaggle/input/**)
The tag is documentation -- cell 6b reads it but does not act on it; an optional
5th field still sets the render backoff. Referencing frames by filename (not by
position) keeps the picker and the notebook aligned. `[frame_int, x, y]` is still
accepted by cell 6b for the old index form.

Primary-scene frames are tagged TRAIN or TEST -- the split the Kaggle notebook's
dataparser makes (equally spaced 90% train, remainder held out). The tag shows in
the window title and is the 4th field of every emitted `COORDS` row, so you can
tell -- while picking, and later when reading the block back -- whether a frame is
one the NeRF/NF actually saw. Deriving it needs
the COLMAP frame order, so the scene's sparse/0/images.bin is fetched once
(~30-45 MB) unless transforms.json is already there; --no-split-tags skips that.

Coordinates are in that image's own pixel space. For anything other than a
--scene frame, only its DINOv2 feature at the clicked pixel is used -- the
sampled novel views are still rendered through the frozen --scene NeRF, so
it's most useful when the other image shows similar content (e.g. picking on
counter/kitchen/room while probing a bonsai-trained NeRF).

Deps: matplotlib, Pillow, and (only for the one-time scene downloads) remotezip
-- NOT the full nerfstudio stack.

    python app/pick_points.py                              # bonsai frames (downloads once)
    python app/pick_points.py --scene counter               # counter frames instead
    python app/pick_points.py --also-scenes counter kitchen room  # + those scenes as external picks
    python app/pick_points.py --extra ~/photo.jpg           # + one external image
    python app/pick_points.py --extra-only --extra a.jpg b.png    # only the external images

Controls: click on the image = add a point. Buttons along the bottom
(Prev / Next / Undo / Reset / Print / Done) and a "go to" box to jump to an
index or an --extra filename. Keys also work: n/p or arrows = change image,
u = undo, r = reset, s = print, q = quit.

Run external images through scripts/downscale_external_images.py first, so their
DINO features come out at a resolution comparable to the training frames'.
"""
import argparse
import json
import math
import struct
from pathlib import Path, PurePosixPath

import matplotlib  # native backend on purpose; there's an 'agg' guard in main()
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.widgets import Button, TextBox
import numpy as np
from PIL import Image

# Mip-NeRF 360 archive (same source scripts/downloads/download_mipnerf360.py uses).
ARCHIVE_URL = "http://storage.googleapis.com/gresearch/refraw360/360_v2.zip"
# Kept in sync by hand with scripts/downloads/download_mipnerf360.py's SCENES --
# not imported from there, since that module pulls in the full nerfstudio stack
# (colmap_to_json) that this script deliberately avoids.
SCENES = ("bonsai", "counter", "kitchen", "room")
# NerfstudioDataParserConfig.train_split_fraction's default, which the Kaggle
# notebook deliberately keeps (cell 5: 1.0 leaves 0 eval cameras, which crashes
# nerfacto's periodic eval and ns-render).
TRAIN_SPLIT_FRACTION = 0.9
# dino_features.MAX_DINO_SIDE -- above this long side DINOv2 extracts features
# from a bilinearly shrunk copy of the image. Kept in sync by hand; importing it
# would drag in torch.
MAX_DINO_SIDE = 1568
# Window-title colour per split; None (external, or untagged) falls back to black.
SPLIT_COLOR = {"train": "darkgreen", "test": "orangered"}


def download_scene_images(scene_dir: Path, scene: str) -> None:
    """Fetch just <scene>/images/* from the remote archive via HTTP range
    requests into scene_dir/images/. No nerfstudio, no COLMAP conversion --
    picking points only needs the pixels, not transforms.json."""
    from concurrent.futures import ThreadPoolExecutor

    try:
        from remotezip import RemoteZip
    except ImportError:
        raise SystemExit("need `remotezip` for the download:  pip install remotezip")

    images_dir = scene_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    print(f"downloading {scene}/images/ from {ARCHIVE_URL} (partial, HTTP range) ...")
    with RemoteZip(ARCHIVE_URL) as z:
        members = [n for n in z.namelist()
                   if n.startswith(f"{scene}/images/") and not n.endswith("/")]
    if not members:
        raise SystemExit(f"no images for scene {scene!r} in the archive")

    n_workers = 8
    chunks = [members[i::n_workers] for i in range(n_workers)]

    def grab(chunk):
        with RemoteZip(ARCHIVE_URL) as z:
            for m in chunk:
                (images_dir / Path(m).name).write_bytes(z.read(m))
        return len(chunk)

    done = 0
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        for got in ex.map(grab, [c for c in chunks if c]):
            done += got
            print(f"  {done}/{len(members)}")


def ensure_scene(scene_dir: Path, downscale: int, scene: str) -> list:
    """Make sure <scene_dir>/images_<downscale>/ exists, downloading + resizing
    as needed, and return the sorted list of its image paths."""
    scene_dir = scene_dir.expanduser().resolve()
    src = scene_dir / "images"
    dst = scene_dir / f"images_{downscale}"

    if not src.is_dir() or not any(src.iterdir()):
        download_scene_images(scene_dir, scene)

    src_imgs = sorted(p for p in src.glob("*") if p.is_file())
    if not src_imgs:
        raise SystemExit(f"no images in {src}")

    if not dst.is_dir() or len(list(dst.glob("*"))) != len(src_imgs):
        # identical recipe to the Kaggle notebook's cell 1
        dst.mkdir(exist_ok=True)
        print(f"generating {dst.name}/ ({len(src_imgs)} images, /{downscale} LANCZOS) ...")
        for i, p in enumerate(src_imgs):
            im = Image.open(p)
            w, h = im.size
            im.resize((w // downscale, h // downscale), Image.LANCZOS).save(dst / p.name)
            if (i + 1) % 50 == 0 or i + 1 == len(src_imgs):
                print(f"  {i + 1}/{len(src_imgs)}")

    return sorted(dst.glob("*"))


def download_colmap_images_bin(scene_dir: Path, scene: str) -> Path:
    """Fetch just <scene>/sparse/0/images.bin from the remote archive.

    That one file is all the train/test tags need -- the rest of the
    reconstruction (cameras.bin, points3D.bin) only matters for poses, which
    this script never uses. ~30-45 MB per scene, once, cached on disk;
    scripts/downloads/download_mipnerf360.py fetches it too, along with the rest,
    if you would rather have a full transforms.json here.
    """
    try:
        from remotezip import RemoteZip
    except ImportError:  # RuntimeError, not SystemExit: an untagged session still works
        raise RuntimeError("`remotezip` is not installed (pip install remotezip)")

    member = f"{scene}/sparse/0/images.bin"
    dst = scene_dir / "sparse" / "0" / "images.bin"
    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"fetching {member} for the train/test tags (one time, ~30-45 MB) ...")
    # Write-then-rename: a run killed mid-write must not leave a truncated file
    # behind, since a short name list would silently shift the split.
    part = dst.with_suffix(".bin.part")
    with RemoteZip(ARCHIVE_URL) as z:
        part.write_bytes(z.read(member))
    part.replace(dst)
    print(f"  wrote {dst} ({dst.stat().st_size / 1e6:.0f} MB)")
    return dst


def read_colmap_image_names(path: Path) -> list:
    """COLMAP images.bin -> the registered image names, in file order.

    Same iteration order as nerfstudio's read_images_binary
    (nerfstudio/data/utils/colmap_parsing_utils.py), which is the order
    colmap_to_json writes transforms.json's `frames` in, which is the order the
    dataparser appends image_filenames in -- and the split indexes into that.
    Only the names are wanted, so each record's 2D observations are seek()ed past
    instead of parsed: a few milliseconds rather than a multi-second numpy parse
    of a ~70 MB file, and no nerfstudio import.
    """
    names = []
    with open(path, "rb") as f:
        (num_reg_images,) = struct.unpack("<Q", f.read(8))
        for _ in range(num_reg_images):
            f.read(64)  # "<idddddddi": image_id, qvec[4], tvec[3], camera_id
            chars = []
            while True:
                c = f.read(1)
                if c in (b"\x00", b""):
                    break
                chars.append(c)
            names.append(b"".join(chars).decode("utf-8"))
            (num_points2D,) = struct.unpack("<Q", f.read(8))
            f.seek(24 * num_points2D, 1)  # per point: x, y (2 doubles) + point3D_id (int64)
    return names


def ensure_split_tags(scene_dir: Path, scene: str, frac: float):
    """{basename: "train" | "test"} for one scene, or None if it can't be worked out.

    Reproduces exactly the split the Kaggle notebook's dataparser makes
    (nerfstudio/data/dataparsers/nerfstudio_dataparser.py: equally spaced train
    indices, remainder eval). That split is POSITIONAL over transforms.json's
    `frames` -- i.e. COLMAP images.bin order, not sorted(images_<ds>/*) -- so the
    order has to come from the reconstruction itself and cannot be guessed from
    the filenames on disk.

    Never fatal: anything missing (offline, no remotezip, malformed file) just
    prints a warning and leaves the session untagged.
    """
    scene_dir = scene_dir.expanduser().resolve()
    transforms = scene_dir / "transforms.json"
    images_bin = scene_dir / "sparse" / "0" / "images.bin"
    try:
        if transforms.is_file():
            meta = json.loads(transforms.read_text())
            names = [PurePosixPath(f["file_path"]).name for f in meta["frames"]]
        else:
            if not images_bin.is_file():
                download_colmap_images_bin(scene_dir, scene)
            names = read_colmap_image_names(images_bin)
    except Exception as e:
        print(f"! could not work out the {scene} train/test split "
              f"({type(e).__name__}: {e}) -- frames will be untagged "
              f"(--no-split-tags silences this)")
        return None

    if not names:
        print(f"! the {scene} reconstruction lists no images -- frames will be untagged")
        return None

    n = len(names)
    n_train = math.ceil(n * frac)
    # np.linspace rather than a hand-rolled range, so the int truncation matches
    # the dataparser's exactly.
    train = set(np.linspace(0, n - 1, n_train, dtype=int).tolist())
    return {name: ("train" if i in train else "test") for i, name in enumerate(names)}


def build_items(args) -> list:
    """Ordered list of browsable images: primary-scene frames, then
    --also-scenes frames, then --extra.

    Each item is (name, path, is_external, origin, split). `name` (the file's
    basename) is what goes into COORDS -- referencing frames by filename, not
    position, keeps the picker and the Kaggle notebook aligned even if their
    image sets differ slightly. `origin` is the scene name for dataset/
    also-scenes frames, or "extra" for --extra files. `split` is "train"/"test"
    for a primary --scene frame, else None.

    Only the primary scene is tagged: --also-scenes frames and --extra files are
    external either way (their DINO feature is all that is used, the views are
    still rendered through the primary scene's NeRF), so their own splits are
    meaningless here -- and tagging them would mean another images.bin per scene.
    """
    items, seen = [], {}

    def add(scene: str, paths, is_external: bool, splits):
        for p in paths:
            if p.name in seen:
                raise SystemExit(f"{scene!r} frame {p.name!r} clashes with an existing image name; rename it")
            items.append((p.name, p, is_external, scene, splits.get(p.name) if splits else None))
            seen[p.name] = True

    if not args.extra_only:
        # Images first, then the (much smaller) reconstruction, so a fresh scene
        # shows its long download before its short one.
        paths = ensure_scene(args.data_root / args.scene, args.downscale, args.scene)
        splits = None
        if not args.no_split_tags:
            splits = ensure_split_tags(args.data_root / args.scene, args.scene,
                                       args.train_split_fraction)
            if splits:
                missing = [p.name for p in paths if p.name not in splits]
                if missing:
                    print(f"! {len(missing)} of {len(paths)} {args.scene} frames are not in the "
                          f"COLMAP reconstruction, so they are in neither split and the notebook "
                          f"never sees them; left untagged (e.g. {missing[0]})")
        add(args.scene, paths, False, splits)
        for scene in args.also_scenes:
            add(scene, ensure_scene(args.data_root / scene, args.downscale, scene), True, None)
    for e in args.extra or []:
        p = Path(e).expanduser().resolve()
        if not p.is_file():
            raise SystemExit(f"--extra file not found: {p}")
        if p.name in seen:
            raise SystemExit(f"--extra {p.name!r} clashes with an existing image name; rename it")
        items.append((p.name, p, True, "extra", None))
        seen[p.name] = True
    if not items:
        raise SystemExit("nothing to show (use --extra, or drop --extra-only)")
    return items


def format_block(coords: list, items: list) -> str:
    if not coords:
        return "COORDS = []  # nothing picked"
    origin = {name: o for name, _, _, o, _ in items}
    is_ext = {name: e for name, _, e, _, _ in items}
    split = {name: s for name, _, _, _, s in items}
    # NB: positions in sorted(images_<ds>/*), NOT dataparser indices (those follow
    # COLMAP order). The TRAIN/TEST tag is resolved by filename, which is exact.
    frame_no = {name: n for n, name in
                enumerate(name for name, _, is_external, _, _ in items if not is_external)}
    rows = []
    for name, x, y in coords:
        o = origin.get(name)
        if not is_ext.get(name):
            tag = {"train": "TRAIN", "test": "TEST"}.get(split.get(name), "UNKNOWN")
            comment = f"{o} frame {frame_no.get(name, '?')}"
        elif o == "extra":
            tag = "EXTERNAL"
            comment = f"attach a Kaggle Dataset containing {name!r}"
        else:
            tag = "EXTERNAL"
            comment = (f"{o} scene - attach a Kaggle Dataset containing {name!r}, or fetch "
                       f"it via scripts/downloads/download_mipnerf360.py --scene {o}")
        rows.append((f'["{name}", {x}, {y}, "{tag}"],', comment))
    width = max(len(r) for r, _ in rows)
    return "\n".join(["COORDS = ["]
                     + [f"    {r:<{width}}  # {c}" for r, c in rows]
                     + ["]"])


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--scene", default="bonsai", choices=SCENES,
                    help="primary scene -- the one the frozen NeRF is/will be trained on; "
                         "its frames are the non-external COORDS entries")
    ap.add_argument("--also-scenes", nargs="+", metavar="SCENE", default=[], choices=SCENES,
                    help="additional Mip-NeRF 360 scenes to browse/pick on (auto-downloaded "
                         "like --scene); their frames are treated as external (DINO feature only)")
    ap.add_argument("--data-root", type=Path, default=Path("data/mipnerf360"),
                    help="root folder holding <scene>/images/ for --scene and --also-scenes")
    ap.add_argument("--downscale", type=int, default=2)
    ap.add_argument("--max-points", type=int, default=None,
                    help="optional safety cap on how many points you can drop in one session "
                         "(default: unlimited)")
    ap.add_argument("--extra", nargs="+", metavar="IMG",
                    help="local image file(s) to pick on in addition to the dataset frames")
    ap.add_argument("--extra-only", action="store_true",
                    help="browse only --extra images; skip the scene downloads")
    ap.add_argument("--train-split-fraction", type=float, default=TRAIN_SPLIT_FRACTION,
                    help="fraction of --scene frames the notebook trains on; the rest are "
                         "held out. Must match the dataparser's train_split_fraction, or the "
                         "TRAIN/TEST tags will be wrong (default: %(default)s)")
    ap.add_argument("--no-split-tags", action="store_true",
                    help="do not tag --scene frames TRAIN/TEST (skips the one-time "
                         "sparse/0/images.bin fetch)")
    ap.add_argument("--out", type=Path, default=None, help="also write the block here")
    args = ap.parse_args()

    if matplotlib.get_backend().lower() == "agg":
        raise SystemExit(
            "matplotlib has no interactive backend (got 'agg'). Run this from a real "
            "terminal on your machine (not a headless / notebook context), e.g.\n"
            "    python app/pick_points.py"
        )
    if args.extra_only and not args.extra:
        raise SystemExit("--extra-only needs --extra")
    if args.scene in args.also_scenes:
        raise SystemExit(f"--scene {args.scene!r} also appears in --also-scenes; drop one")
    if not 0.0 < args.train_split_fraction <= 1.0:
        raise SystemExit(f"--train-split-fraction must be in (0, 1], got {args.train_split_fraction}")

    items = build_items(args)
    n_ext = sum(is_ext for _, _, is_ext, _, _ in items)
    tags = [s for _, _, is_ext, _, s in items if not is_ext]
    detail = f" ({tags.count('train')} train + {tags.count('test')} test)" if any(tags) else ""
    print(f"{len(items)} images: {len(items) - n_ext} {args.scene} frames{detail} "
          f"+ {n_ext} external")

    # DINOv2 shrinks anything over MAX_DINO_SIDE for its forward pass, so a patch on
    # an oversized image covers far more content than one on a ~1559px training
    # frame -- and nothing in the window would tell you.
    for name, path, _is_ext, o, _ in items:
        if o != "extra":
            continue
        long_side = max(Image.open(path).size)
        if long_side > MAX_DINO_SIDE:
            print(f"! {name} is {long_side}px on its long side (> {MAX_DINO_SIDE}); DINOv2 will "
                  f"feature it at reduced resolution -- run scripts/downscale_external_images.py")

    state = {"i": 0}
    coords: list = []  # [ [name, x, y], ... ]  name = the image's basename

    # free up keys we bind below from matplotlib's default toolbar shortcuts
    for _k in ("keymap.pan", "keymap.back", "keymap.forward"):
        plt.rcParams[_k] = []

    fig, ax = plt.subplots(figsize=(13, 9))
    fig.subplots_adjust(bottom=0.16)
    try:
        fig.canvas.manager.set_window_title("VF-NeRF point picker")
    except Exception:
        pass

    def show():
        ax.clear()
        name, path, is_ext, origin, split = items[state["i"]]
        img = Image.open(path)
        ax.imshow(img)
        r = max(img.size) // 45
        for k, (cname, x, y) in enumerate(coords):
            if cname == name:
                ax.add_patch(Circle((x, y), radius=r, fill=False, color="red", lw=2))
                ax.text(x + r, y - r, str(k), color="red", fontsize=13, weight="bold")
        if is_ext:
            kind = f"EXTERNAL ({origin})"
        else:
            kind = f"{origin} {split.upper()}" if split else origin
        cap = args.max_points if args.max_points is not None else "∞"
        ax.set_title(
            f"[{state['i']}/{len(items) - 1}]  {kind}  {name}   points: {len(coords)}/{cap}"
            f"\nclick = add point   (buttons below, or keys n/p u r s q)",
            color=SPLIT_COLOR.get(split, "black"),
        )
        ax.set_xlabel("x (px)")
        ax.set_ylabel("y (px)")
        fig.canvas.draw_idle()

    def step(d):
        state["i"] = (state["i"] + d) % len(items)
        show()

    def goto(i):
        if 0 <= i < len(items):
            state["i"] = i
            show()
        else:
            print(f"index {i} out of range 0..{len(items) - 1}")

    def add_point(x, y):
        if args.max_points is not None and len(coords) >= args.max_points:
            print(f"already at {args.max_points} points (undo / reset first)")
            return
        name = items[state["i"]][0]
        xi, yi = int(round(x)), int(round(y))
        coords.append([name, xi, yi])
        print(f"point {len(coords) - 1}: [{name!r}, {xi}, {yi}]")
        show()

    def undo(*_):
        if coords:
            print("undo", coords.pop())
            show()

    def reset(*_):
        coords.clear()
        print("reset")
        show()

    def dump(*_):
        print("\n" + format_block(coords, items) + "\n")

    def on_click(event):
        if event.inaxes is ax and event.xdata is not None and event.button == 1:
            add_point(event.xdata, event.ydata)

    def on_key(event):
        if event.key in ("n", "right"):
            step(1)
        elif event.key in ("p", "left"):
            step(-1)
        elif event.key == "u":
            undo()
        elif event.key == "r":
            reset()
        elif event.key == "s":
            dump()
        elif event.key in ("q", "escape"):
            plt.close(fig)

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)

    # on-screen controls (kept referenced so callbacks stay alive)
    widgets = []
    specs = [("< Prev", lambda e: step(-1)), ("Next >", lambda e: step(1)),
             ("Undo", undo), ("Reset", reset), ("Print", dump),
             ("Done", lambda e: plt.close(fig))]
    for j, (label, cb) in enumerate(specs):
        b = Button(fig.add_axes([0.04 + j * 0.11, 0.04, 0.10, 0.055]), label)
        b.on_clicked(cb)
        widgets.append(b)

    def on_goto(text):
        text = text.strip()
        if not text:
            return
        if text.isdigit():
            goto(int(text))
            return
        hit = next((k for k, (name, _, _, _, _) in enumerate(items)
                    if text.lower() in name.lower()), None)
        if hit is None:
            print(f"no image matching {text!r}")
        else:
            goto(hit)

    tb = TextBox(fig.add_axes([0.78, 0.04, 0.12, 0.055]), "go to ")
    tb.on_submit(on_goto)
    widgets.append(tb)

    show()
    plt.show()

    block = format_block(coords, items)
    print("\n" + block + "\n")
    ext_origin = {name: o for name, _, is_ext, o, _ in items if is_ext}
    split_of = {name: s for name, _, _, _, s in items}
    if coords:
        n_train = sum(1 for name, _, _ in coords if split_of.get(name) == "train")
        n_test = sum(1 for name, _, _ in coords if split_of.get(name) == "test")
        n_picked_ext = sum(1 for name, _, _ in coords if name in ext_origin)
        parts = [f"{n_train} train", f"{n_test} test", f"{n_picked_ext} external"]
        untagged = len(coords) - n_train - n_test - n_picked_ext
        if untagged:
            parts.append(f"{untagged} untagged")
        print(f"{len(coords)} point(s) picked: " + ", ".join(parts))
    picked_ext = sorted({name for name, _, _ in coords if name in ext_origin})
    if picked_ext:
        print("external images picked -- put these in a Kaggle Dataset and attach it")
        print("(or, for a Mip-NeRF 360 scene, let the notebook fetch it via "
              "scripts/downloads/download_mipnerf360.py):")
        for name in picked_ext:
            o = ext_origin[name]
            tag = f" ({o} scene)" if o != "extra" else ""
            print(f"  {name}{tag}")
    if args.out:
        args.out.write_text(block + "\n")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
