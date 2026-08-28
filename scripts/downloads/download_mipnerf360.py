#!/usr/bin/env python
"""Download one scene from the Mip-NeRF 360 dataset.

Reliably hosted on Google Cloud Storage (https://jonbarron.info/mipnerf360/),
unlike nerfstudio's own Google-Drive-hosted example captures, which currently
throttle/block scripted downloads and in at least one case (poster) return a
plain 404 -- see README. Used here as a stand-in for VF-NeRF's own "table"
scene (a real, casually-captured, object-centric 360 scene), which was never
publicly released.

Only the requested scene's files are fetched via HTTP range requests
(remotezip), not the full ~12.5GB (360_v2.zip) / ~4.5GB (360_extra_scenes.zip)
archives that bundle all scenes together. The release already includes a
COLMAP reconstruction per scene (sparse/0/{cameras,images,points3D}.bin), so
this converts straight to transforms.json via nerfstudio's own colmap_to_json
-- no local COLMAP run needed.

Example:
    python scripts/downloads/download_mipnerf360.py --scene bonsai --save-dir data/mipnerf360
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from remotezip import RemoteZip

from nerfstudio.process_data.colmap_utils import colmap_to_json

NUM_WORKERS = 8  # each file is its own HTTP range request -- parallelize to avoid
# serializing hundreds of network round trips for what's a modest total byte count.

# Indoor, bounded, object-centric scenes (360_v2.zip) -- the closest analogs to
# VF-NeRF's "table" scene. bicycle/garden/stump (also in 360_v2.zip) and
# flowers/treehill (360_extra_scenes.zip) are outdoor/unbounded and omitted
# here since they don't fit this project's use case, though they exist in the
# same archives if ever needed.
ARCHIVE_URL = "http://storage.googleapis.com/gresearch/refraw360/360_v2.zip"
SCENES = ("bonsai", "counter", "kitchen", "room")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scene", default="bonsai", choices=SCENES)
    p.add_argument("--save-dir", type=Path, default=Path("data/mipnerf360"))
    return p.parse_args()


def download_scene(scene: str, save_dir: Path) -> Path:
    scene_dir = save_dir / scene
    images_dir = scene_dir / "images"
    sparse_dir = scene_dir / "sparse" / "0"

    if (scene_dir / "transforms.json").exists():
        print(f"{scene_dir}/transforms.json already exists, skipping download.")
        return scene_dir

    images_dir.mkdir(parents=True, exist_ok=True)
    sparse_dir.mkdir(parents=True, exist_ok=True)

    print(f"Fetching {scene}/ from {ARCHIVE_URL} (partial download via HTTP range requests) ...")
    with RemoteZip(ARCHIVE_URL) as z:
        members = [n for n in z.namelist() if n.startswith(f"{scene}/images/") or n.startswith(f"{scene}/sparse/0/")]
    if not members:
        raise RuntimeError(f"No files found for scene {scene!r} in {ARCHIVE_URL} -- check the scene name.")

    # Each worker opens its own RemoteZip (re-fetches the small central directory
    # once per worker, negligible next to the per-file range-GET latency this
    # avoids serializing) rather than sharing one instance across threads, since
    # RemoteZip's internal read position isn't documented as thread-safe.
    done = 0

    def extract_chunk(chunk):
        with RemoteZip(ARCHIVE_URL) as z:
            for member in chunk:
                z.extract(member, path=save_dir)
        return len(chunk)

    chunks = [members[i::NUM_WORKERS] for i in range(NUM_WORKERS)]
    chunks = [c for c in chunks if c]
    with ThreadPoolExecutor(max_workers=len(chunks)) as pool:
        futures = [pool.submit(extract_chunk, chunk) for chunk in chunks]
        for future in as_completed(futures):
            done += future.result()
            print(f"  {done}/{len(members)} files")

    num_frames = colmap_to_json(recon_dir=sparse_dir, output_dir=scene_dir)
    print(f"Wrote {scene_dir}/transforms.json ({num_frames} registered images).")
    return scene_dir


def main():
    args = parse_args()
    download_scene(args.scene, args.save_dir)


if __name__ == "__main__":
    main()
