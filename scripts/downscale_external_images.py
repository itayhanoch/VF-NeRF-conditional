#!/usr/bin/env python
"""One-time downscale of the external probe images in `external images/`.

The Mip-NeRF 360 training frames the conditional NF sees are a uniform
~1559x1039 (images_2/), but the external images picked on with
app/pick_points.py arrive at whatever resolution they were downloaded at
(4444x3333 down to 612x408). That matters because DINOv2 refuses inputs whose
long side exceeds `MAX_DINO_SIDE` (1568) and silently bilinear-downscales them
for the forward pass -- see nerfstudio/utils/dino_features.py -- so one patch
token covers a different amount of real content per image, and the feature
picked on a 4444px photo is much coarser than any the NF trained on.

This rewrites each image in place at 1/factor size and parks the original in a
sibling `<dir>_originals/` folder. In place, same basename, because that folder
is what gets uploaded as the Kaggle Dataset: keeping one version per name means
the picked coordinates and the notebook's `/kaggle/input/**/<name>` lookup are
in the same pixel space by construction, with no chance of the full-res file
winning the glob.

Idempotent -- a file whose original is already parked is skipped, so re-running
never downscales twice.

    python scripts/downscale_external_images.py --dry-run   # look first
    python scripts/downscale_external_images.py             # do it

Deps: Pillow only.
"""
import argparse
import shutil
from pathlib import Path

from PIL import Image, UnidentifiedImageError

# DINOv2 runs an image at native resolution only up to this long side
# (dino_features.MAX_DINO_SIDE); above it, features are extracted from a shrunk
# copy. Kept in sync by hand -- this script deliberately avoids importing the
# nerfstudio stack.
MAX_DINO_SIDE = 1568
# Below this long side an image is only ~37 patch cells across, so a picked
# pixel's DINO feature smears over a large fraction of the subject. Not an
# error, just worth seeing before you pick points on it.
SMALL_SIDE_WARN = 518


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--dir", type=Path, default=Path("external images"),
                   help="folder of external images to rewrite in place")
    p.add_argument("--factor", type=int, default=4,
                   help="integer downscale factor (w//factor, h//factor)")
    p.add_argument("--originals-dir", type=Path, default=None,
                   help="where to park the originals (default: <dir>_originals, a "
                        "SIBLING of --dir so it never lands in the uploaded folder)")
    p.add_argument("--dry-run", action="store_true",
                   help="print what would happen, touch nothing")
    return p.parse_args()


def main():
    args = parse_args()
    src_dir = args.dir.expanduser()
    if not src_dir.is_dir():
        raise SystemExit(f"not a directory: {src_dir}")
    if args.factor < 1:
        raise SystemExit(f"--factor must be >= 1 (got {args.factor})")
    orig_dir = args.originals_dir or src_dir.with_name(src_dir.name + "_originals")

    done = skipped = 0
    for path in sorted(p for p in src_dir.glob("*") if p.is_file()):
        parked = orig_dir / path.name
        if parked.exists():
            print(f"  skip {path.name}  (already downscaled; original in {orig_dir.name}/)")
            skipped += 1
            continue
        try:
            im = Image.open(path)
            im.load()
        except (UnidentifiedImageError, OSError):
            print(f"  skip {path.name}  (not an image)")
            skipped += 1
            continue

        w, h = im.size
        nw, nh = max(1, w // args.factor), max(1, h // args.factor)
        note = ""
        if max(nw, nh) < SMALL_SIDE_WARN:
            note = f"   ! long side {max(nw, nh)}px -- very coarse DINO features"
        print(f"  {path.name}  {w}x{h} -> {nw}x{nh}{note}")
        if args.dry_run:
            done += 1
            continue

        # Park the original first: if the resize/save then fails, the source
        # still exists somewhere rather than being half-overwritten.
        orig_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(str(path), str(parked))
        try:
            out = Image.open(parked)
            out.load()
            # Same recipe as pick_points.ensure_scene / the notebook's cell 2b.
            resized = out.resize((nw, nh), Image.LANCZOS)
            if resized.mode in ("P", "LA", "RGBA") and path.suffix.lower() in (".jpg", ".jpeg"):
                resized = resized.convert("RGB")
            resized.save(path)
        except Exception:
            shutil.move(str(parked), str(path))  # put it back, then re-raise
            raise
        done += 1

    verb = "would rewrite" if args.dry_run else "rewrote"
    print(f"\n{verb} {done} image(s) at 1/{args.factor}, skipped {skipped}")
    if done and not args.dry_run:
        print(f"originals parked in {orig_dir}/ (not part of the folder you upload)")
        print(f"upload {src_dir}/ as the Kaggle Dataset, and pick points on it with:")
        print(f'  python app/pick_points.py --extra "{src_dir}"/*')
    if args.dry_run:
        print("(dry run -- nothing was touched)")


if __name__ == "__main__":
    main()
