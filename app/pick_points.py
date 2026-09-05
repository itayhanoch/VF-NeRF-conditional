#!/usr/bin/env python
"""Local interactive point picker for the conditional-NF probe workflow.

Opens a matplotlib window with the primary scene's training images (default
bonsai -- change with --scene), plus any --also-scenes Mip-NeRF 360 scenes and
any --extra images you pass. Click points on the objects you want to probe;
the script prints a ready-to-paste `COORDS` block for the Kaggle notebook's
cell 6a (and, equivalently, for `kaggle_explorer.py` / `app/gradio_app.py`).

Each `COORDS` entry is `["name.ext", x, y]` -- the image's basename:
  * a --scene frame -> found in the notebook's images_<downscale>/ folder
  * any other image (an --also-scenes frame or an --extra file) -> on Kaggle,
    attach a Dataset containing that file (it is looked up by name under
    /kaggle/input/**)
Referencing frames by filename (not by position) keeps the picker and the
notebook aligned. `[frame_int, x, y]` is still accepted by cell 6b for the old
index form.

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
"""
import argparse
from pathlib import Path

import matplotlib  # native backend on purpose; there's an 'agg' guard in main()
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.widgets import Button, TextBox
from PIL import Image

# Mip-NeRF 360 archive (same source scripts/downloads/download_mipnerf360.py uses).
ARCHIVE_URL = "http://storage.googleapis.com/gresearch/refraw360/360_v2.zip"
# Kept in sync by hand with scripts/downloads/download_mipnerf360.py's SCENES --
# not imported from there, since that module pulls in the full nerfstudio stack
# (colmap_to_json) that this script deliberately avoids.
SCENES = ("bonsai", "counter", "kitchen", "room")


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


def build_items(args) -> list:
    """Ordered list of browsable images: primary-scene frames, then
    --also-scenes frames, then --extra.

    Each item is (name, path, is_external, origin). `name` (the file's
    basename) is what goes into COORDS -- referencing frames by filename, not
    position, keeps the picker and the Kaggle notebook aligned even if their
    image sets differ slightly. `origin` is the scene name for dataset/
    also-scenes frames, or "extra" for --extra files."""
    items, seen = [], {}

    def add_scene(scene: str, is_external: bool):
        for p in ensure_scene(args.data_root / scene, args.downscale, scene):
            if p.name in seen:
                raise SystemExit(f"{scene!r} frame {p.name!r} clashes with an existing image name; rename it")
            items.append((p.name, p, is_external, scene))
            seen[p.name] = True

    if not args.extra_only:
        add_scene(args.scene, False)
        for scene in args.also_scenes:
            add_scene(scene, True)
    for e in args.extra or []:
        p = Path(e).expanduser().resolve()
        if not p.is_file():
            raise SystemExit(f"--extra file not found: {p}")
        if p.name in seen:
            raise SystemExit(f"--extra {p.name!r} clashes with an existing image name; rename it")
        items.append((p.name, p, True, "extra"))
        seen[p.name] = True
    if not items:
        raise SystemExit("nothing to show (use --extra, or drop --extra-only)")
    return items


def format_block(coords: list, items: list) -> str:
    if not coords:
        return "COORDS = []  # nothing picked"
    origin = {name: o for name, _, _, o in items}
    is_ext = {name: e for name, _, e, _ in items}
    frame_no = {name: n for n, name in
                enumerate(name for name, _, is_external, _ in items if not is_external)}
    lines = ["COORDS = ["]
    for name, x, y in coords:
        o = origin.get(name)
        if not is_ext.get(name):
            comment = f"{o} frame {frame_no.get(name, '?')}"
        elif o == "extra":
            comment = f"EXTERNAL - attach a Kaggle Dataset containing {name!r}"
        else:
            comment = (f"EXTERNAL ({o} scene) - attach a Kaggle Dataset containing {name!r}, "
                       f"or fetch it via scripts/downloads/download_mipnerf360.py --scene {o}")
        lines.append(f'    ["{name}", {x}, {y}],  # {comment}')
    lines.append("]")
    return "\n".join(lines)


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

    items = build_items(args)
    n_ext = sum(is_ext for _, _, is_ext, _ in items)
    print(f"{len(items)} images ({len(items) - n_ext} {args.scene} frames + {n_ext} external)")

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
        name, path, is_ext, origin = items[state["i"]]
        img = Image.open(path)
        ax.imshow(img)
        r = max(img.size) // 45
        for k, (cname, x, y) in enumerate(coords):
            if cname == name:
                ax.add_patch(Circle((x, y), radius=r, fill=False, color="red", lw=2))
                ax.text(x + r, y - r, str(k), color="red", fontsize=13, weight="bold")
        kind = f"EXTERNAL ({origin})" if is_ext else origin
        cap = args.max_points if args.max_points is not None else "∞"
        ax.set_title(
            f"[{state['i']}/{len(items) - 1}]  {kind}  {name}   points: {len(coords)}/{cap}"
            f"\nclick = add point   (buttons below, or keys n/p u r s q)"
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
        hit = next((k for k, (name, _, _, _) in enumerate(items)
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
    ext_origin = {name: o for name, _, is_ext, o in items if is_ext}
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
