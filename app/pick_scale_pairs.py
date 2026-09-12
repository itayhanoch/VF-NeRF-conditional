#!/usr/bin/env python
"""Local interactive picker for scene SCALE references: pairs of points.

Every evaluation number in this repo -- the depth filter, the nearest-sample
distances, the backoff -- is in nerfstudio's normalized scene units, which mean
nothing on their own. This picker gives each scene a physical anchor: browse its
frames, click the two ends of a recognisable object (a plate, a book, a tile),
and the script prints a ready-to-paste `SCALE_PAIRS` block for
`notebooks/measure_scene_scale.ipynb`, which back-projects both pixels through
the frozen NeRF and reports the 3-D distance between them.

Each row is `["name.ext", x1, y1, x2, y2, "TAG", "label"]`: the frame's basename
(found in the notebook's images_<downscale>/ folder), the two clicked pixels in
that image's own pixel space, the TRAIN/TEST split tag (as pick_points.py derives
it) and a free-text label for the object. Both points of a pair must be on the
SAME frame -- the pair is measured through that frame's camera.

Deps: matplotlib, Pillow, and (only for the one-time scene downloads) remotezip
-- NOT the full nerfstudio stack. Shares its scene/download/split code with
app/pick_points.py.

    python app/pick_scale_pairs.py                               # counter, kitchen, room
    python app/pick_scale_pairs.py --scenes counter --out scale_pairs.json

Controls: click = point A, click again on the same frame = point B (pair done).
Type an object name into the "label" box before the second click to attach it.
Buttons along the bottom (Prev / Next / Undo / Reset / Print / Done) and a
"go to" box (index or filename substring). Keys: n/p or arrows = change frame,
u = undo, r = reset, s = print, q = quit. The toolbar's zoom / pan tools work
and are NOT treated as clicks; the zoom is kept while you stay on a frame, so
zoom in to place the points precisely.
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib  # native backend on purpose; there's an 'agg' guard in main()
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.widgets import Button, TextBox
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pick_points import SCENES, SPLIT_COLOR, TRAIN_SPLIT_FRACTION, ensure_scene, ensure_split_tags  # noqa: E402

DEFAULT_SCENES = ("counter", "kitchen", "room")


def build_items(args):
    """[(scene, name, path, split)] for every frame of every --scenes scene, in
    sorted(images_<ds>/*) order per scene. split is "train" / "test" / None."""
    items = []
    for scene in args.scenes:
        scene_dir = args.data_root / scene
        paths = ensure_scene(scene_dir, args.downscale, scene)
        tags = None if args.no_split_tags else ensure_split_tags(scene_dir, scene, args.train_split_fraction)
        for p in paths:
            items.append((scene, p.name, p, (tags or {}).get(p.name)))
        print(f"{scene}: {len(paths)} frames"
              + (f" ({sum(1 for p in paths if (tags or {}).get(p.name) == 'train')} train)" if tags else ""))
    return items


def format_block(pairs: list, items: list) -> str:
    """`SCALE_PAIRS = {scene: [[name, x1, y1, x2, y2, TAG, label], ...]}` block."""
    split = {(s, n): sp for s, n, _, sp in items}
    frame_no = {}
    for s, n, _, _ in items:
        frame_no[(s, n)] = sum(1 for t in items if t[0] == s and t[1] < n)
    scenes = list(dict.fromkeys(s for s, *_ in items))
    if not pairs:
        return "SCALE_PAIRS = {" + ", ".join(f'"{s}": []' for s in scenes) + "}  # nothing picked"
    out = ["SCALE_PAIRS = {"]
    for s in scenes:
        rows = []
        for scene, name, x1, y1, x2, y2, label in pairs:
            if scene != s:
                continue
            tag = {"train": "TRAIN", "test": "TEST"}.get(split.get((s, name)), "UNKNOWN")
            rows.append((f'["{name}", {x1}, {y1}, {x2}, {y2}, "{tag}", {json.dumps(label)}],',
                         f"{s} frame {frame_no[(s, name)]}"))
        if not rows:
            out.append(f'    "{s}": [],')
            continue
        width = max(len(r) for r, _ in rows)
        out.append(f'    "{s}": [')
        out += [f"        {r:<{width}}  # {c}" for r, c in rows]
        out.append("    ],")
    out.append("}")
    return "\n".join(out)


def to_json(pairs: list, items: list) -> dict:
    split = {(s, n): sp for s, n, _, sp in items}
    scenes = list(dict.fromkeys(s for s, *_ in items))
    d = {s: [] for s in scenes}
    for scene, name, x1, y1, x2, y2, label in pairs:
        d[scene].append({"image": name, "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                         "tag": {"train": "TRAIN", "test": "TEST"}.get(split.get((scene, name)), "UNKNOWN"),
                         "label": label})
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenes", nargs="+", choices=SCENES, default=list(DEFAULT_SCENES),
                    help="Mip-NeRF 360 scenes to browse (downloaded once if missing)")
    ap.add_argument("--data-root", type=Path, default=Path("data/mipnerf360"))
    ap.add_argument("--downscale", type=int, default=2, help="images_<N>/ folder to pick on (must match the notebook)")
    ap.add_argument("--max-pairs", type=int, default=None, help="Cap on pairs per scene (default: unlimited)")
    ap.add_argument("--train-split-fraction", type=float, default=TRAIN_SPLIT_FRACTION)
    ap.add_argument("--no-split-tags", action="store_true", help="Skip TRAIN/TEST tagging (and the one-time images.bin fetch)")
    ap.add_argument("--out", type=Path, default=None, help="Also write the pairs as JSON to this file")
    args = ap.parse_args()

    if matplotlib.get_backend().lower() == "agg":
        raise SystemExit(
            "matplotlib has no interactive backend (got 'agg'). Run this from a real "
            "terminal on your machine (not a headless / notebook context), e.g.\n"
            "    python app/pick_scale_pairs.py"
        )
    if not 0.0 < args.train_split_fraction <= 1.0:
        raise SystemExit(f"--train-split-fraction must be in (0, 1], got {args.train_split_fraction}")

    items = build_items(args)
    if not items:
        raise SystemExit("no frames found")

    state = {"i": 0, "pending": None, "label": ""}   # pending = (x, y) of a first click awaiting its B
    pairs: list = []  # [ [scene, name, x1, y1, x2, y2, label], ... ]

    # free up keys we bind below from matplotlib's default toolbar shortcuts
    # (p = pan, r = home/reset view, s = save figure, q = quit)
    for _k in ("keymap.pan", "keymap.back", "keymap.forward", "keymap.home", "keymap.save", "keymap.quit"):
        plt.rcParams[_k] = []

    fig, ax = plt.subplots(figsize=(13, 9))
    fig.subplots_adjust(bottom=0.16)
    try:
        fig.canvas.manager.set_window_title("VF-NeRF scale-pair picker")
    except Exception:
        pass
    view = {"name": None, "xlim": None, "ylim": None}  # zoom kept while the frame stays the same

    def scene_pairs(scene):
        return [p for p in pairs if p[0] == scene]

    def show():
        scene, name, path, split = items[state["i"]]
        same = view["name"] == (scene, name)
        if same:
            view["xlim"], view["ylim"] = ax.get_xlim(), ax.get_ylim()
        ax.clear()
        img = Image.open(path)
        ax.imshow(img)
        r = max(img.size) // 60
        for k, (s, n, x1, y1, x2, y2, label) in enumerate(pairs):
            if (s, n) != (scene, name):
                continue
            ax.plot([x1, x2], [y1, y2], color="red", lw=1.2)
            for x, y, t in ((x1, y1, "A"), (x2, y2, "B")):
                ax.add_patch(Circle((x, y), radius=r, fill=False, color="red", lw=2))
                ax.text(x + r, y - r, t, color="red", fontsize=11, weight="bold")
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            ax.plot([mx], [my], marker="+", color="lime", ms=10, mew=1.5)
            ax.text(mx + r, my + 2 * r, f"{k}{' ' + label if label else ''}", color="red",
                    fontsize=11, weight="bold")
        if state["pending"] is not None:
            x, y = state["pending"]
            ax.add_patch(Circle((x, y), radius=r, fill=False, color="orange", lw=2, ls="--"))
            ax.text(x + r, y - r, "A", color="orange", fontsize=11, weight="bold")
        kind = f"{scene} {split.upper()}" if split else scene
        cap = args.max_pairs if args.max_pairs is not None else "∞"
        ax.set_title(
            f"[{state['i']}/{len(items) - 1}]  {kind}  {name}   pairs in {scene}: {len(scene_pairs(scene))}/{cap}"
            f"   total: {len(pairs)}\n"
            + ("click = point B (same frame)" if state["pending"] is not None else "click = point A")
            + "   (buttons below, or keys n/p u r s q; toolbar zoom/pan are not clicks)",
            color=SPLIT_COLOR.get(split, "black"),
        )
        ax.set_xlabel("x (px)")
        ax.set_ylabel("y (px)")
        if same and view["xlim"] is not None:
            ax.set_xlim(view["xlim"])
            ax.set_ylim(view["ylim"])
        view["name"] = (scene, name)
        fig.canvas.draw_idle()

    def drop_pending(reason):
        if state["pending"] is not None:
            print(f"! discarded dangling point A {state['pending']} ({reason})")
            state["pending"] = None

    def step(d):
        drop_pending("changed frame")
        state["i"] = (state["i"] + d) % len(items)
        show()

    def goto(i):
        if 0 <= i < len(items):
            drop_pending("changed frame")
            state["i"] = i
            show()
        else:
            print(f"index {i} out of range 0..{len(items) - 1}")

    def add_point(x, y):
        scene, name, _, _ = items[state["i"]]
        xi, yi = int(round(x)), int(round(y))
        if state["pending"] is None:
            if args.max_pairs is not None and len(scene_pairs(scene)) >= args.max_pairs:
                print(f"already at {args.max_pairs} pairs for {scene} (undo / reset first)")
                return
            state["pending"] = (xi, yi)
            print(f"A: ({xi}, {yi}) on {name} -- now click B")
        else:
            x1, y1 = state["pending"]
            state["pending"] = None
            label = state["label"]
            pairs.append([scene, name, x1, y1, xi, yi, label])
            print(f"pair {len(pairs) - 1}: [{name!r}, {x1}, {y1}, {xi}, {yi}, {label!r}]  ({scene})")
        show()

    def undo(*_):
        if state["pending"] is not None:
            drop_pending("undo")
        elif pairs:
            print("undo", pairs.pop())
        show()

    def reset(*_):
        pairs.clear()
        state["pending"] = None
        print("reset")
        show()

    def dump(*_):
        print("\n" + format_block(pairs, items) + "\n")

    def toolbar_busy():
        tb = getattr(fig.canvas, "toolbar", None)
        return bool(getattr(tb, "mode", "") or "")

    def on_click(event):
        if toolbar_busy():
            return
        if event.inaxes is ax and event.xdata is not None and event.button == 1:
            add_point(event.xdata, event.ydata)

    text_boxes = []  # typing into these must not trigger the single-letter bindings

    def on_key(event):
        if any(getattr(t, "capturekeystrokes", False) for t in text_boxes):
            return
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
        b = Button(fig.add_axes([0.03 + j * 0.095, 0.04, 0.085, 0.055]), label)
        b.on_clicked(cb)
        widgets.append(b)

    def on_goto(text):
        text = text.strip()
        if not text:
            return
        if text.isdigit():
            goto(int(text))
            return
        hit = next((k for k, (_, name, _, _) in enumerate(items) if text.lower() in name.lower()), None)
        if hit is None:
            print(f"no frame matching {text!r}")
        else:
            goto(hit)

    def on_label(text):
        state["label"] = text.strip()
        print(f"label for the next pair(s): {state['label']!r}")

    tb_goto = TextBox(fig.add_axes([0.66, 0.04, 0.10, 0.055]), "go to ")
    tb_goto.on_submit(on_goto)
    tb_label = TextBox(fig.add_axes([0.83, 0.04, 0.14, 0.055]), "label ")
    tb_label.on_submit(on_label)
    tb_label.on_text_change(lambda t: state.__setitem__("label", t.strip()))
    text_boxes += [tb_goto, tb_label]
    widgets += text_boxes

    show()
    plt.show()

    drop_pending("window closed")
    block = format_block(pairs, items)
    print("\n" + block + "\n")
    for s in dict.fromkeys(t[0] for t in items):
        print(f"{s}: {len(scene_pairs(s))} pair(s)")
    if args.out:
        args.out.write_text(json.dumps(to_json(pairs, items), indent=2) + "\n")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
