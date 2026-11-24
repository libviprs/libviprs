#!/usr/bin/env python3
"""Ink recall and precision of one PDF against another, page by page.

    python3 -I tools/plot_recall.py REFERENCE.pdf PLOTTED.pdf PAGES DPI TOL

PAGES is a comma list of the reference's 1-based page numbers, in the order the plotted PDF has them
(its page 1 is compared with the first of the list). Both PDFs are rendered with `pdftoppm` (poppler)
at DPI to 8-bit grey; a pixel is ink when it is under 128. Recall is the share of the reference's ink
pixels that have plotted ink within TOL pixels, precision the share of the plotted ink that has
reference ink within TOL pixels. The gate of the plot work (#1670) is 150 dpi and 3 px, which is the
same 1.44 pt as 100 dpi and 2 px on a coarser raster.

Prints one JSON object: the medians and every page's [page, recall, precision, reference ink,
plotted ink]. A page whose two renders differ in size is [page, "size"]. Pure standard library, so it
is slow (a minute or two for 64 pages at 150 dpi). Exits 2 when `pdftoppm` is not installed.
"""

import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile


def render(pdf, page, directory, dpi):
    base = os.path.join(directory, "p")
    subprocess.run(
        ["pdftoppm", "-gray", "-r", str(dpi), "-f", str(page), "-l", str(page), pdf, base],
        check=True,
    )
    name = next(n for n in os.listdir(directory) if n.startswith("p-"))
    path = os.path.join(directory, name)
    with open(path, "rb") as f:
        data = f.read()
    os.remove(path)
    parts = data.split(None, 4)
    width, height = int(parts[1]), int(parts[2])
    return width, height, data[len(data) - width * height :]


def ink(width, pixels):
    return {(i % width, i // width) for i, v in enumerate(pixels) if v < 128}


def dilate(points, r):
    out = set()
    for x, y in points:
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                out.add((x + dx, y + dy))
    return out


def main(argv):
    if shutil.which("pdftoppm") is None:
        print("pdftoppm is not installed", file=sys.stderr)
        return 2
    reference, plotted, pages, dpi, tol = argv[1], argv[2], argv[3], int(argv[4]), int(argv[5])
    pages = [int(p) for p in pages.split(",")]
    rows = []
    for n, page in enumerate(pages, 1):
        with tempfile.TemporaryDirectory() as d:
            rw, rh, rp = render(reference, page, d, dpi)
            ow, oh, op = render(plotted, n, d, dpi)
        if (rw, rh) != (ow, oh):
            rows.append([page, "size"])
            continue
        a, b = ink(rw, rp), ink(ow, op)
        near_b, near_a = dilate(b, tol), dilate(a, tol)
        recall = sum(1 for p in a if p in near_b) / len(a) if a else 1.0
        precision = sum(1 for p in b if p in near_a) / len(b) if b else 1.0
        rows.append([page, round(recall, 4), round(precision, 4), len(a), len(b)])
    scored = [r for r in rows if isinstance(r[1], float)]
    print(
        json.dumps(
            {
                "dpi": dpi,
                "tolerance_px": tol,
                "pages": len(rows),
                "size_mismatches": len(rows) - len(scored),
                "median_recall": statistics.median(r[1] for r in scored) if scored else None,
                "median_precision": statistics.median(r[2] for r in scored) if scored else None,
                "per_page": rows,
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
