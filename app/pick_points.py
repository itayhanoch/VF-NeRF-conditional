#!/usr/bin/env python
"""Local interactive point picker for the conditional-NF probe workflow.

Opens a matplotlib window with the bonsai training images (and any --extra
images you pass). Click points on the objects you want to probe; the script
prints a ready-to-paste `COORDS` block for the Kaggle notebook's cell 6a (and,
equivalently, for `kaggle_explorer.py` / `app/gradio_app.py`).

A `COORDS` entry is either:
  * `[frame_int, x, y]`  -- a dataset frame, index into sorted(images_<downscale>/*)
  * `["name.ext", x, y]` -- any other image; on Kaggle, attach a Dataset that
    contains that file (it is looked up by name under /kaggle/input/**).

Coordinates are in that image's own pixel space. For an external image only its
DINOv2 feature at the clicked pixel is used -- the sampled novel views are still
rendered as bonsai views through the frozen bonsai NeRF, so it is most useful
when the external image shows similar content.

    python app/pick_points.py                          # dataset frames (downloads bonsai ~1GB once)
    python app/pick_points.py --extra ~/photo.jpg      # dataset frames + one external image
    python app/pick_points.py --extra-only --extra a.jpg b.png   # only the external images

Controls: click = add point | n / -> next image | p / <- prev image
          u = undo last point | r = reset all | s = print block | q = quit
"""
import argparse
from pathlib import Path

import matplotlib  # native backend on purpose; there's an 'agg' guard in main()
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from PIL import Image


def ensure_scene(scene_dir: Path, downscale: int) -> list:
    """Make sure <scene_dir>/images_<downscale>/ exists, downloading + resizing
    as needed, and return the sorted list of its image paths."""
    scene_dir = scene_dir.expanduser().resolve()

    if not (scene_dir / "transforms.json").exists():
        # reuse the project's downloader (idempotent); import lazily so users who
        # already have the data don't need a working nerfstudio install.
        from scripts.downloads.download_mipnerf360 import download_scene

        print(f"No scene at {scene_dir}; downloading bonsai (~1GB, one time) ...")
        got = download_scene("bonsai", scene_dir.parent).resolve()
        if got != scene_dir:
            print(f"(using {got})")
            scene_dir = got

    src = scene_dir / "images"
    dst = scene_dir / f"images_{downscale}"
    src_imgs = sorted(src.glob("*"))
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

    Each item is (ref, path): ref is an int (dataset frame index) or the
    external image's basename (str)."""
    items = []
    if not args.extra_only:
        for i, p in enumerate(ensure_scene(args.scene_dir, args.downscale)):
            items.append((i, p))
    for e in args.extra or []:
        p = Path(e).expanduser().resolve()
        if not p.is_file():
            raise SystemExit(f"--extra file not found: {p}")
        items.append((p.name, p))
    if not items:
        raise SystemExit("nothing to show (use --extra, or drop --extra-only)")
    return items


def format_block(coords: list) -> str:
    if not coords:
        return "COORDS = []  # nothing picked"
    lines = ["COORDS = ["]
    for ref, x, y in coords:
        if isinstance(ref, int):
            lines.append(f"    [{ref}, {x}, {y}],")
        else:
            lines.append(f'    ["{ref}", {x}, {y}],  # EXTERNAL - attach a Kaggle Dataset containing {ref!r}')
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
    print(f"{len(items)} images "
          f"({sum(isinstance(r, int) for r, _ in items)} dataset + "
          f"{sum(not isinstance(r, int) for r, _ in items)} external)")

    state = {"i": 0}
    coords: list = []  # [ [ref, x, y], ... ]  ref = int frame index or str basename

    fig, ax = plt.subplots(figsize=(13, 9))
    try:
        fig.canvas.manager.set_window_title("VF-NeRF point picker")
    except Exception:
        pass

    def show():
        ax.clear()
        ref, path = items[state["i"]]
        img = Image.open(path)
        ax.imshow(img)
        r = max(img.size) // 45
        for k, (cref, x, y) in enumerate(coords):
            if cref == ref:
                ax.add_patch(Circle((x, y), radius=r, fill=False, color="red", lw=2))
                ax.text(x + r, y - r, str(k), color="red", fontsize=13, weight="bold")
        tag = f"frame {ref}" if isinstance(ref, int) else f"EXTERNAL {ref}"
        ax.set_title(
            f"[{state['i']}/{len(items) - 1}]  {tag}   "
            f"points: {len(coords)}/{args.max_points}   "
            f"[click add | n/p image | u undo | r reset | s print | q quit]"
        )
        ax.set_xlabel("x (px)")
        ax.set_ylabel("y (px)")
        fig.canvas.draw_idle()

    def on_click(event):
        if event.inaxes is not ax or event.xdata is None or event.button != 1:
            return
        if len(coords) >= args.max_points:
            print(f"already at {args.max_points} points (u = undo, r = reset)")
            return
        ref = items[state["i"]][0]
        x, y = int(round(event.xdata)), int(round(event.ydata))
        coords.append([ref, x, y])
        print(f"point {len(coords) - 1}: [{ref!r}, {x}, {y}]")
        show()

    def on_key(event):
        if event.key in ("n", "right"):
            state["i"] = (state["i"] + 1) % len(items)
            show()
        elif event.key in ("p", "left"):
            state["i"] = (state["i"] - 1) % len(items)
            show()
        elif event.key == "u" and coords:
            print("undo", coords.pop())
            show()
        elif event.key == "r":
            coords.clear()
            print("reset")
            show()
        elif event.key == "s":
            print("\n" + format_block(coords) + "\n")
        elif event.key in ("q", "escape"):
            plt.close(fig)

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    show()
    plt.show()

    block = format_block(coords)
    print("\n" + block + "\n")
    ext = sorted({r for r, _, _ in coords if not isinstance(r, int)})
    if ext:
        print("external images picked -- put these in a Kaggle Dataset and attach it:")
        for name in ext:
            print(f"  {name}")
    if args.out:
        args.out.write_text(block + "\n")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
