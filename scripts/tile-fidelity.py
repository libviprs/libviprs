#!/usr/bin/env python3
"""How much a lossy tile pyramid moved away from the lossless one beside it.

This is the analysis half of `docs/tile-codec-benchmarks.md`. The generation
half is the CLI: one `viprs pyramid --storage directory` run per cell over the
same source, the same plan and the same layout, so every cell's tree holds the
same Deep Zoom coordinates and the trees can be walked against each other.

The PNG tree is the reference. PNG is lossless here, so a PNG tile *is* the
raster the sink was handed, and comparing against it compares against the
input rather than against another codec's opinion of the input.

Two metrics, both inherited from libviprs#1134, spelled out here because that
issue describes them in prose and the script it used is not in the tree:

  PSNR      over RGB after the reference's alpha is dropped. The drop is
            checked rather than assumed: a reference tile with any sample
            below 255 in its alpha channel aborts the run, because then
            dropping it would not be the flatten the JPEG sink performs.

  ink IoU   ink is luma below 128, on the BT.601 luma both the JPEG colour
            transform and PIL's greyscale conversion use. The comparison is
            restricted to pixels within 2 px of a reference intensity gradient
            above 16/255, so flat paper and flat ink, where one flipped level
            moves the metric without moving anything a reader can see, are
            excluded. IoU is then reference ink against candidate ink inside
            that mask.

            #1134's warning about the metric stands: on a recoloured sheet the
            ink mask barely moves while the colour under it is destroyed, so
            IoU is a linework metric and not a fidelity metric.

It also encodes the reference tiles through libjpeg at the four knob settings
#1134 separated (4:4:4 or 4:2:0, standard or optimized Huffman tables). That
is the positive control on the whole comparison rather than a result: if those
four totals land on #1134's four numbers, then that run and this one are
looking at the same pixels, and its JPEG rows and these are comparable.

Usage:

    pip install numpy pillow
    python3 scripts/tile-fidelity.py <out-dir> [<results.json>]

where `<out-dir>` holds `png-dir/`, `jpeg-q85-dir/` and the rest, each one the
output directory of a `viprs pyramid` run. Results print as they land and the
whole document is written to `<out-dir>/../fidelity.json`.

Not a test and not wired into CI. It needs numpy and Pillow, which are not
dependencies of anything else here, and it needs a rendered pyramid, which
needs pdfium, which this repository's CI never loads.
"""

import hashlib
import io
import json
import os
import sys

import numpy as np
import PIL.features
from PIL import Image

# Every cell the document reports, as (output directory prefix, tile suffix).
CANDIDATES = [
    ("jpeg-q75", "jpeg"),
    ("jpeg-q85", "jpeg"),
    ("jpeg-q95", "jpeg"),
    ("webp", "webp"),
]

# The four cells libviprs#1134 separated, as (subsampling, optimize) for PIL.
# 0 is 4:4:4 and 2 is 4:2:0.
KNOBS = {
    "444-std": (0, False),
    "444-opt": (0, True),
    "420-std": (2, False),
    "420-opt": (2, True),
}

LUMA_INK_THRESHOLD = 128.0
GRADIENT_THRESHOLD = 16.0
GRADIENT_RADIUS = 2


def luma(rgb):
    """BT.601 luma, which is what the JPEG colour transform computes."""
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def gradient_neighbourhood(ref_luma):
    """Pixels within GRADIENT_RADIUS of a real intensity edge.

    A forward difference in each direction rather than a Sobel: the question
    is whether a neighbouring sample differs, and a smoothing kernel would
    answer a slightly different one on a tile whose ink is one pixel wide.
    """
    gx = np.zeros_like(ref_luma)
    gy = np.zeros_like(ref_luma)
    gx[:, :-1] = np.abs(np.diff(ref_luma, axis=1))
    gy[:-1, :] = np.abs(np.diff(ref_luma, axis=0))
    edge = np.maximum(gx, gy) > GRADIENT_THRESHOLD
    if not edge.any():
        return edge
    out = np.zeros_like(edge)
    h, w = edge.shape
    for dy in range(-GRADIENT_RADIUS, GRADIENT_RADIUS + 1):
        for dx in range(-GRADIENT_RADIUS, GRADIENT_RADIUS + 1):
            ys = slice(max(0, dy), h + min(0, dy))
            yd = slice(max(0, -dy), h + min(0, -dy))
            xs = slice(max(0, dx), w + min(0, dx))
            xd = slice(max(0, -dx), w + min(0, -dx))
            out[yd, xd] |= edge[ys, xs]
    return out


def tiles(root, suffix):
    """Every tile under `root`, keyed on its path relative to it."""
    found = {}
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if not name.endswith("." + suffix):
                continue
            path = os.path.join(dirpath, name)
            rel = os.path.relpath(path, root)
            found[rel[: -(len(suffix) + 1)]] = path
    return found


def pct(values, q):
    return float(np.percentile(np.array(values), q)) if values else None


def occupancy(root, suffix, archive=None):
    """What a cell's tree costs, and what an archive of the same tiles costs.

    Three byte columns, because they answer three different questions. The
    apparent bytes are what the tiles are. The allocated bytes are what the
    filesystem gave them, `st_blocks * 512`, which is the column the tile-size
    argument lives in: a 260-byte tile still occupies a block. The deduped
    bytes are one copy of each distinct payload, keyed on the sha256 of the
    file, which is what a store that deduplicates would hold and what makes
    the archive column comparable, since PMTiles dedupes by construction.

    `entries` counts every file and every directory including the root, so it
    includes `p.dzi` and the two `.libviprs-job.*` sidecars.
    """
    tile_paths, sidecars, directories = [], [], 0
    for dirpath, dirnames, filenames in os.walk(root):
        directories += len(dirnames)
        for name in filenames:
            path = os.path.join(dirpath, name)
            (tile_paths if name.endswith("." + suffix) else sidecars).append(path)
    apparent, allocated, digests, sizes = 0, 0, {}, []
    levels = {}
    for path in tile_paths:
        stat = os.stat(path)
        apparent += stat.st_size
        allocated += stat.st_blocks * 512
        sizes.append(stat.st_size)
        with open(path, "rb") as handle:
            digest = hashlib.sha256(handle.read()).hexdigest()
        digests.setdefault(digest, stat.st_size)
        # A Deep Zoom tile is <root>/<name>/<level>/<col>_<row>.<suffix>, so
        # the level is the directory the file sits in.
        level = os.path.basename(os.path.dirname(path))
        row = levels.setdefault(level, {"tiles": 0, "bytes": 0, "digests": {}})
        row["tiles"] += 1
        row["bytes"] += stat.st_size
        row["digests"].setdefault(digest, stat.st_size)
    sizes.sort()
    row = {
        "tiles": len(tile_paths),
        "apparent_tile_bytes": apparent,
        "allocated_tile_bytes": allocated,
        "dedup_bytes": sum(digests.values()),
        "distinct_payloads": len(digests),
        "p50_tile_bytes": sizes[len(sizes) // 2] if sizes else 0,
        "tiles_under_one_block": sum(1 for size in sizes if size < 4096),
        "sidecars": sorted(os.path.basename(p) for p in sidecars),
        "sidecar_bytes": sum(os.stat(p).st_size for p in sidecars),
        "files": len(tile_paths) + len(sidecars),
        "directories": directories + 1,
        "entries": len(tile_paths) + len(sidecars) + directories + 1,
        "levels": {
            level: {
                "tiles": row["tiles"],
                "bytes": row["bytes"],
                "dedup_bytes": sum(row["digests"].values()),
                "distinct_payloads": len(row["digests"]),
            }
            for level, row in sorted(
                levels.items(),
                key=lambda kv: int(kv[0]) if kv[0].isdigit() else -1,
            )
        },
    }
    if archive and os.path.exists(archive):
        stat = os.stat(archive)
        row["archive_bytes"] = stat.st_size
        row["archive_allocated_bytes"] = stat.st_blocks * 512
    return row


def load_reference(root):
    """The PNG tree as float arrays, with the alpha assumption checked."""
    paths = tiles(root, "png")
    if not paths:
        sys.exit("no reference tiles under " + root)
    cache, with_alpha = {}, 0
    for key, path in paths.items():
        arr = np.asarray(Image.open(path))
        if arr.ndim == 3 and arr.shape[2] == 4:
            if not (arr[..., 3] == 255).all():
                sys.exit("reference tile %s is not opaque, so dropping its "
                         "alpha is not what the JPEG sink does" % key)
            with_alpha += 1
            arr = arr[..., :3]
        elif arr.ndim == 2:
            arr = np.dstack([arr] * 3)
        cache[key] = arr.astype(np.float64)
    return cache, with_alpha


def measure(cell_root, suffix, reference, masks):
    """One candidate cell against the reference, overall and by level."""
    candidate = tiles(cell_root, suffix)
    missing = set(reference) - set(candidate)
    if missing:
        sys.exit("%s is missing %d tiles the reference has" % (cell_root, len(missing)))
    psnrs, ious, exact, per_level = [], [], 0, {}
    for key, ref in reference.items():
        arr = np.asarray(Image.open(candidate[key]).convert("RGB")).astype(np.float64)
        mse = float(np.mean((arr - ref) ** 2))
        if mse == 0.0:
            exact += 1
            psnr = float("inf")
        else:
            psnr = 10.0 * np.log10(255.0 * 255.0 / mse)
        mask, ref_ink = masks[key]
        candidate_ink = luma(arr) < LUMA_INK_THRESHOLD
        union = np.count_nonzero(mask & (ref_ink | candidate_ink))
        iou = 1.0 if union == 0 else (
            np.count_nonzero(mask & ref_ink & candidate_ink) / union
        )
        psnrs.append(psnr)
        ious.append(iou)
        level = key.split(os.sep)[1] if os.sep in key else "?"
        per_level.setdefault(level, []).append((psnr, iou))
    finite = [p for p in psnrs if np.isfinite(p)]
    return {
        "tiles": len(psnrs),
        "exact_matches": exact,
        "psnr_median": pct(finite, 50),
        "psnr_p10": pct(finite, 10),
        "iou_mean": float(np.mean(ious)),
        "iou_median": pct(ious, 50),
        "iou_p10": pct(ious, 10),
        "levels": {
            level: {
                "tiles": len(rows),
                "psnr_median": pct([p for p, _ in rows if np.isfinite(p)], 50),
                "iou_mean": float(np.mean([i for _, i in rows])),
                "iou_p10": pct([i for _, i in rows], 10),
            }
            for level, rows in sorted(
                per_level.items(),
                key=lambda kv: int(kv[0]) if kv[0].isdigit() else -1,
            )
        },
    }


def libjpeg_control(reference, quality=85):
    """The reference tiles through libjpeg at #1134's four knob settings."""
    control = {}
    for label, (subsampling, optimize) in KNOBS.items():
        total, digests = 0, {}
        for ref in reference.values():
            buf = io.BytesIO()
            Image.fromarray(ref.astype(np.uint8)).save(
                buf, "JPEG", quality=quality,
                subsampling=subsampling, optimize=optimize,
            )
            data = buf.getvalue()
            total += len(data)
            digests.setdefault(hashlib.sha256(data).hexdigest(), len(data))
        control[label] = {
            "total_bytes": total,
            "dedup_bytes": sum(digests.values()),
            "distinct_payloads": len(digests),
        }
        print("libjpeg q%d %s %s" % (quality, label, control[label]), flush=True)
    return control


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "out"
    destination = sys.argv[2] if len(sys.argv) > 2 else os.path.join(out, "..", "fidelity.json")
    reference_root = os.path.join(out, "png-dir")

    reference, with_alpha = load_reference(reference_root)
    masks = {}
    for key, ref in reference.items():
        ref_luma = luma(ref)
        masks[key] = (gradient_neighbourhood(ref_luma), ref_luma < LUMA_INK_THRESHOLD)

    results = {
        "reference": reference_root,
        "reference_tiles": len(reference),
        "reference_tiles_with_alpha": with_alpha,
        "pillow_version": Image.__version__,
        "libjpeg_version": PIL.features.version("jpg"),
        "cells": {},
        "occupancy": {},
    }
    for name, suffix in [("png", "png")] + CANDIDATES:
        root = os.path.join(out, name + "-dir")
        if not os.path.isdir(root):
            print("skipping %s, no tree at %s" % (name, root), flush=True)
            continue
        results["occupancy"][name] = occupancy(
            root, suffix, os.path.join(out, name + ".pmtiles")
        )
        print("occupancy", name, results["occupancy"][name], flush=True)
        if name == "png":
            continue
        results["cells"][name] = measure(root, suffix, reference, masks)
        print(name, json.dumps(results["cells"][name]["levels"], default=str)[:80],
              "iou_mean", round(results["cells"][name]["iou_mean"], 5), flush=True)

    results["libjpeg_reference_q85"] = libjpeg_control(reference)

    with open(destination, "w") as handle:
        json.dump(results, handle, indent=2, default=str)
    print("wrote", destination)


if __name__ == "__main__":
    main()
