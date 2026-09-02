#!/usr/bin/env python
"""Local interactive point picker for the conditional-NF probe workflow.

Opens a matplotlib window with the bonsai training images (and any --extra
images you pass). Click points on the objects you want to probe; the script
prints a ready-to-paste `COORDS` block for the Kaggle notebook's cell 6a (and,
equivalently, for `kaggle_explorer.py` / `app/gradio_app.py`).

Each `COORDS` entry is `["name.ext", x, y]` -- the image's basename:
  * a bonsai frame  -> found in the notebook's images_<downscale>/ folder
  * any other image -> on Kaggle, attach a Dataset containing that file (it is
    looked up by name under /kaggle/input/**)
Referencing frames by filename (not by position) keeps the picker and the
notebook aligned. `[frame_int, x, y]` is still accepted by cell 6b for the old
index form.

Coordinates are in that image's own pixel space. For an external image only its
DINOv2 feature at the clicked pixel is used -- the sampled novel views are still
rendered as bonsai views through the frozen bonsai NeRF, so it is most useful
when the external image shows similar content.

Deps: matplotlib, Pillow, and (only for the one-time bonsai download) remotezip
-- NOT the full nerfstudio stack.

    python app/pick_points.py                          # dataset frames (downloads bonsai images once)
    python app/pick_points.py --extra ~/photo.jpg      # dataset frames + one external image
    python app/pick_points.py --extra-only --extra a.jpg b.png   # only the external images

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


def download_scene_images(scene_dir: Path, scene: str = "bonsai") -> None:
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


def ensure_scene(scene_dir: Path, downscale: int) -> list:
    """Make sure <scene_dir>/images_<downscale>/ exists, downloading + resizing
    as needed, and return the sorted list of its image paths."""
    scene_dir = scene_dir.expanduser().resolve()
    src = scene_dir / "images"
    dst = scene_dir / f"images_{downscale}"

    if not src.is_dir() or not any(src.iterdir()):
        download_scene_images(scene_dir)

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
    """Ordered list of browsable images: dataset frames first, then --extra.

    Each item is (name, path, is_external). `name` (the file's basename) is what
    goes into COORDS -- referencing frames by filename, not position, keeps the
    picker and the Kaggle notebook aligned even if their image sets differ
    slightly."""
    items, seen = [], {}
    if not args.extra_only:
        for p in ensure_scene(args.scene_dir, args.downscale):
            items.append((p.name, p, False))
            seen[p.name] = True
    for e in args.extra or []:
        p = Path(e).expanduser().resolve()
        if not p.is_file():
            raise SystemExit(f"--extra file not found: {p}")
        if p.name in seen:
            raise SystemExit(f"--extra {p.name!r} clashes with a dataset frame name; rename it")
        items.append((p.name, p, True))
        seen[p.name] = True
    if not items:
        raise SystemExit("nothing to show (use --extra, or drop --extra-only)")
    return items


def format_block(coords: list, items: list) -> str:
    if not coords:
        return "COORDS = []  # nothing picked"
    ext = {name for name, _, is_ext in items if is_ext}
    frame_no = {name: n for n, name in
                enumerate(name for name, _, is_ext in items if not is_ext)}
    lines = ["COORDS = ["]
    for name, x, y in coords:
        if name in ext:
            lines.append(f'    ["{name}", {x}, {y}],  # EXTERNAL - attach a Kaggle Dataset containing {name!r}')
        else:
            lines.append(f'    ["{name}", {x}, {y}],  # bonsai frame {frame_no.get(name, "?")}')
    lines.append("]")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--scene-dir", type=Path, default=Path("data/mipnerf360/bonsai"))
    ap.add_argument("--downscale", type=int, default=2)
    ap.add_argument("--max-points", type=int, default=5)
    ap.add_argument("--extra", nargs="+", metavar="IMG",
                    help="local image file(s) to pick on in addition to the dataset frames")
    ap.add_argument("--extra-only", action="store_true",
                    help="browse only --extra images; skip the bonsai download")
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

    items = build_items(args)
    n_ext = sum(is_ext for _, _, is_ext in items)
    print(f"{len(items)} images ({len(items) - n_ext} bonsai frames + {n_ext} external)")

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
        name, path, is_ext = items[state["i"]]
        img = Image.open(path)
        ax.imshow(img)
        r = max(img.size) // 45
        for k, (cname, x, y) in enumerate(coords):
            if cname == name:
                ax.add_patch(Circle((x, y), radius=r, fill=False, color="red", lw=2))
                ax.text(x + r, y - r, str(k), color="red", fontsize=13, weight="bold")
        kind = "EXTERNAL" if is_ext else "bonsai"
        ax.set_title(
            f"[{state['i']}/{len(items) - 1}]  {kind}  {name}   points: {len(coords)}/{args.max_points}"
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
        if len(coords) >= args.max_points:
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
        hit = next((k for k, (name, _, _) in enumerate(items)
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
    ext_names = {name for name, _, is_ext in items if is_ext}
    picked_ext = sorted({name for name, _, _ in coords if name in ext_names})
    if picked_ext:
        print("external images picked -- put these in a Kaggle Dataset and attach it:")
        for name in picked_ext:
            print(f"  {name}")
    if args.out:
        args.out.write_text(block + "\n")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
