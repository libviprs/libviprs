#!/usr/bin/env python3
"""Pair a DWG's per-layout PDFs with its paper layouts and write the manifest (#1621, panel D5).

Reads, never writes, the directory holding the DWG and the PDFs (fixtures-rom is read-only), and
writes one JSON manifest somewhere else. The rules are the ones in #1621's test plan:

* One PDF per layout named `<stem>-<layout>.pdf` (a space or underscore before the layout is
  tolerated), matched case-insensitively and anchored, where `<stem>` is the DWG's file name
  without `.dwg`. For Rom's drawing that is
  `^MASTER_FLOOR_PLAN_Rev\\.6\\.0_NOV_08_2026-(SHEET_\\d{3})\\.pdf$`. Never "the name contains the
  layout": SHEET_01 is not SHEET_010, and a stale Rev.5 file must not pair.
* A PDF whose stem is not the DWG's fails (a wrong revision is worse than none), a PDF naming a
  layout the DWG does not have fails, and an extra or duplicate PDF fails.
* Layouts are listed in tab order (the layout table), never name order.
* One multi-page PDF is paired by bookmark title `<stem>-<layout>` first; without bookmarks, by tab
  order only when it has one page per layout and the page sizes match the table in that order.
* Sizes are compared after `/Rotate` (CropBox, which defaults to the MediaBox), 1 pt tolerance; an
  orientation mismatch is reported apart from a size mismatch. Either fails.
* Zero pairs fails. A layout that does not pair fails unless `--unpaired LAYOUT=REASON` lists it
  (a layout Rom did not deliver), and then it is written as unpaired with that reason.

The manifest has one row per layout: layout, tab, expected size in mm and orientation, PDF file,
page, sha256 and the PDF's own size and rotation, plus the DWG's sha256. Exit status 0 when every
rule held, 1 when the manifest carries errors (it is still written, with `"ok": false`), 2 on bad
arguments.

Needs python3 and poppler's `pdfinfo` (and `pdftohtml` for the bookmarks of a single file).

    tools/dwg-layouts-manifest.py --dir /Users/rom/workspace/fixtures-rom \\
        --layouts /Users/rom/workspace/handoff-2026-11-18/dwg-layouts-table.txt \\
        --out "$SCRATCH/dwg-layouts-manifest.json"
"""

import argparse
import datetime
import hashlib
import html
import json
import os
import re
import subprocess
import sys

PT_PER_MM = 72.0 / 25.4
TOLERANCE_PT = 1.0


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_layouts(table):
    """The paper layouts of the table, in tab order: name, tab, effective size in mm."""
    rows = []
    with open(table, encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    head = lines[0].split("|")
    want = ["tab", "name", "media", "paper w x h mm", "rot", "plot_type"]
    if head != want:
        sys.exit(f"{table}: header is {head}, expected {want}")
    for line in lines[1:]:
        tab, name, _media, size, rot, _plot = line.split("|")
        if name.lower() == "model":
            continue
        w, h = (float(v) for v in size.split(" x "))
        if int(rot) in (1, 3):
            w, h = h, w
        rows.append({"layout": name, "tab": int(tab), "size_mm": [w, h]})
    rows.sort(key=lambda r: r["tab"])
    return rows


def orientation(w, h):
    return "landscape" if w > h else "portrait" if h > w else "square"


def pdf_pages(path):
    """Each page's effective size in points after /Rotate, with its rotation and boxes."""
    info = subprocess.run(["pdfinfo", path], capture_output=True, text=True, check=True).stdout
    m = re.search(r"^Pages:\s+(\d+)", info, re.M)
    if not m:
        raise ValueError(f"{path}: pdfinfo reports no page count")
    n = int(m.group(1))
    out = subprocess.run(
        ["pdfinfo", "-box", "-f", "1", "-l", str(n), path],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    pages = {}
    for line in out.splitlines():
        m = re.match(r"Page\s+(\d+)\s+(MediaBox|CropBox|rot):\s+(.*)$", line)
        if not m:
            continue
        page = pages.setdefault(int(m.group(1)), {})
        if m.group(2) == "rot":
            page["rotate"] = int(float(m.group(3))) % 360
        else:
            page[m.group(2)] = [float(v) for v in m.group(3).split()]
    result = []
    for k in range(1, n + 1):
        p = pages.get(k, {})
        box = p.get("CropBox") or p.get("MediaBox")
        if not box:
            raise ValueError(f"{path}: pdfinfo gives no box for page {k}")
        w, h = abs(box[2] - box[0]), abs(box[3] - box[1])
        rot = p.get("rotate", 0)
        if rot in (90, 270):
            w, h = h, w
        result.append(
            {
                "page": k,
                "size_pt": [round(w, 3), round(h, 3)],
                "rotate": rot,
                "media_box": p.get("MediaBox"),
                "crop_box": p.get("CropBox"),
            }
        )
    return result


def bookmarks(path):
    """Top-level bookmark titles with their page numbers, or [] when there are none."""
    try:
        xml = subprocess.run(
            ["pdftohtml", "-xml", "-i", "-q", "-stdout", path],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return []
    return [
        (html.unescape(re.sub(r"<[^>]+>", "", t)).strip(), int(p))
        for p, t in re.findall(r'<item page="(\d+)">(.*?)</item>', xml, re.S)
    ]


def compare(expected_mm, page):
    """None when the page is the layout's size, else what is wrong with it."""
    ew, eh = (v * PT_PER_MM for v in expected_mm)
    pw, ph = page["size_pt"]
    if abs(pw - ew) <= TOLERANCE_PT and abs(ph - eh) <= TOLERANCE_PT:
        return None
    if abs(pw - eh) <= TOLERANCE_PT and abs(ph - ew) <= TOLERANCE_PT:
        return (
            f"orientation mismatch: the page is {pw:.1f} x {ph:.1f} pt after /Rotate "
            f"{page['rotate']}, the layout is {ew:.1f} x {eh:.1f} pt"
        )
    return (
        f"size mismatch: the page is {pw:.1f} x {ph:.1f} pt after /Rotate {page['rotate']}, "
        f"the layout is {ew:.1f} x {eh:.1f} pt"
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", required=True, help="the directory with the DWG and its PDFs (read only)")
    ap.add_argument("--dwg", help="the DWG's file name, when the directory holds more than one")
    ap.add_argument("--layouts", required=True, help="the layout table, in tab order")
    ap.add_argument("--out", required=True, help="the manifest to write, outside --dir")
    ap.add_argument(
        "--unpaired",
        action="append",
        default=[],
        metavar="LAYOUT=REASON",
        help="a layout that was not delivered, written as unpaired instead of failing",
    )
    a = ap.parse_args()

    src = os.path.realpath(a.dir)
    out = os.path.realpath(a.out)
    if os.path.commonpath([src, out]) == src:
        print(f"refusing to write {out}: it is inside {src}, which is read only", file=sys.stderr)
        return 2
    allowed = {}
    for item in a.unpaired:
        name, sep, reason = item.partition("=")
        if not sep or not reason.strip():
            print(f"--unpaired {item!r}: expected LAYOUT=REASON", file=sys.stderr)
            return 2
        allowed[name.strip().upper()] = reason.strip()

    names = sorted(os.listdir(src))
    dwgs = [n for n in names if n.lower().endswith(".dwg")]
    if a.dwg:
        dwgs = [n for n in dwgs if n == a.dwg]
    if len(dwgs) != 1:
        print(f"expected one DWG in {src}, found {dwgs}", file=sys.stderr)
        return 2
    dwg = dwgs[0]
    stem = dwg[: -len(".dwg")]
    layouts = read_layouts(a.layouts)
    by_name = {r["layout"].upper(): r for r in layouts}
    pdfs = [n for n in names if n.lower().endswith(".pdf")]

    errors = []
    pattern = re.compile(r"^" + re.escape(stem) + r"[-_ ](?P<layout>.+)\.pdf$", re.I)
    # Anything that looks like `<some stem>-<layout>.pdf`, to tell a wrong revision from noise.
    foreign = re.compile(r"^(?P<stem>.+?)[-_ ](?P<layout>" + "|".join(map(re.escape, by_name)) + r")\.pdf$", re.I)
    pairs = {}  # layout name (upper) -> (file, page dict)
    single = None
    for n in pdfs:
        m = pattern.match(n)
        if len(pdfs) == 1 and not foreign.match(n) and not (m and m.group("layout").upper() in by_name):
            # The only PDF, named for no layout (`<stem>.pdf`, `<stem>-all.pdf`): a whole set.
            single = n
            continue
        if m:
            key = m.group("layout").upper()
            if key not in by_name:
                errors.append(f"{n}: names layout {m.group('layout')!r}, which the DWG does not have")
            elif key in pairs:
                errors.append(f"{n}: a second PDF for {by_name[key]['layout']} (first {pairs[key][0]})")
            else:
                pairs[key] = (n, None)
            continue
        f = foreign.match(n)
        if f:
            errors.append(f"{n}: its stem {f.group('stem')!r} is not the DWG's {stem!r} (wrong revision?)")
        else:
            errors.append(f"{n}: does not match {pattern.pattern} and is not the only PDF")

    pages_of = {}
    for n in sorted({v[0] for v in pairs.values()} | ({single} if single else set())):
        try:
            pages_of[n] = pdf_pages(os.path.join(src, n))
        except (OSError, ValueError, subprocess.CalledProcessError) as e:
            errors.append(f"{n}: cannot read its pages ({e})")

    mode = "one-per-layout"
    if single and single in pages_of:
        # One multi-page file: bookmarks first, then the page-size fingerprint in tab order.
        pages = pages_of[single]
        marks = bookmarks(os.path.join(src, single))
        title = re.compile(r"^" + re.escape(stem) + r"[-_ ](?P<layout>.+)$", re.I)
        by_mark = {}
        for t, p in marks:
            m = title.match(t)
            if m and m.group("layout").upper() in by_name:
                by_mark.setdefault(m.group("layout").upper(), p)
        if by_mark:
            mode = "single-file-bookmarks"
            for key, p in by_mark.items():
                if 1 <= p <= len(pages):
                    pairs[key] = (single, pages[p - 1])
                else:
                    errors.append(f"{single}: bookmark for {key} points at page {p} of {len(pages)}")
        elif len(pages) == len(layouts) and all(
            compare(r["size_mm"], pg) is None for r, pg in zip(layouts, pages)
        ):
            mode = "single-file-fingerprint"
            for r, pg in zip(layouts, pages):
                pairs[r["layout"].upper()] = (single, pg)
        else:
            errors.append(
                f"{single}: one file of {len(pages)} pages with no usable bookmarks, and its page "
                f"sizes do not match the {len(layouts)} layouts in tab order"
            )
    for key, (n, pg) in list(pairs.items()):
        if pg is None and n in pages_of:
            pp = pages_of[n]
            if len(pp) != 1:
                errors.append(f"{n}: {len(pp)} pages, expected one for {by_name[key]['layout']}")
            pairs[key] = (n, pp[0])

    hashes = {n: sha256(os.path.join(src, n)) for n in pages_of}
    rows = []
    paired = 0
    for r in layouts:
        key = r["layout"].upper()
        row = {
            "layout": r["layout"],
            "tab": r["tab"],
            "size_mm": r["size_mm"],
            "orientation": orientation(*r["size_mm"]),
        }
        hit = pairs.get(key)
        if hit and hit[1] is not None:
            n, pg = hit
            row.update(
                {
                    "file": n,
                    "page": pg["page"],
                    "sha256": hashes[n],
                    "pdf_size_pt": pg["size_pt"],
                    "pdf_rotate": pg["rotate"],
                    "status": "paired",
                }
            )
            wrong = compare(r["size_mm"], pg)
            if wrong:
                errors.append(f"{r['layout']} ({n}, page {pg['page']}): {wrong}")
                row["status"] = "size-mismatch"
                row["reason"] = wrong
            else:
                paired += 1
        elif key in allowed:
            row.update({"file": None, "page": None, "sha256": None, "status": "unpaired", "reason": allowed[key]})
        else:
            row.update({"file": None, "page": None, "sha256": None, "status": "missing"})
            errors.append(f"{r['layout']} (tab {r['tab']}): no PDF pairs with it")
        rows.append(row)
    for key in allowed:
        if key not in by_name:
            errors.append(f"--unpaired names {key}, which the DWG does not have")
    if paired == 0:
        errors.append("zero layouts paired: the naming guess is wrong, or the PDFs are not here")

    manifest = {
        "schema": 1,
        "generated_by": "tools/dwg-layouts-manifest.py",
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "dwg": {"file": dwg, "stem": stem, "sha256": sha256(os.path.join(src, dwg))},
        "rules": {
            "pattern": pattern.pattern,
            "case_insensitive": True,
            "order": "tab",
            "size": "CropBox (MediaBox when absent) after /Rotate",
            "tolerance_pt": TOLERANCE_PT,
            "pairing": mode,
        },
        "summary": f"{paired}/{len(layouts)} paired",
        "ok": not errors,
        "errors": errors,
        "layouts": rows,
    }
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)
        f.write("\n")
    print(f"{manifest['summary']}, {len(errors)} error(s), pairing {mode}: {out}")
    for e in errors:
        print(f"  {e}")
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
