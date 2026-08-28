#!/usr/bin/env python3
"""
Oracle capture for the Analyze area (issue #640, sub-issue of #510).

Runs the real installed vips CLI over hand-built Analyze headers and records
exactly what `analyzeload` does with them. Everything a port needs is here,
because none of it can be derived from the format spec:

  * the 348-byte `struct dsr` is read as BIG-endian regardless of the file,
    and so are the pixels, so on any little-endian host both are swapped
  * `dim[0]` is a rank in 2..=7 and every axis above 2 is FOLDED INTO THE
    HEIGHT, so a 3x2x2 volume is one 3x4 image
  * `vox_offset` is parsed, attached as metadata, and then ignored: the
    pixels always start at byte 0 of the `.img`
  * a negative dimension is NOT refused, it is silently clamped to 1 by
    GObject's property range check
  * nothing validates the `.img` against the header, so the header load
    always succeeds and the failure lands on the first pixel fetch
  * roughly seventy `dsr-*` metadata fields, with the string ones truncated
    one byte short and non-ascii bytes rewritten to `@`

The version discrepancy this capture exists partly to record: `vips
--vips-config` prints `enable Analyze7 load: true` (meson.build:704) while
the operator prints `load an Analyze6 image` (analyzeload.c:120) and the
implementation's own file comment calls it "Old-style header (so called 7.5
format)" (analyze2vips.c:1). Three names, one 348-byte Mayo `dbh.h` header.
Follow the operator, not the config string.

The oracle binary moved under this capture: Homebrew replaced 8.18.4 with
8.18.6 part-way through. `vips__isanalyze` and `read_header` disassembled out
of the 8.18.6 dylib still match the checkout source, so nothing here is known
to have shifted, but `meta.vips_version` names what actually produced these
numbers and the companion foreign-mat capture found a predicate that DID
change.

Writes:
  commands.sh  - every vips CLI command actually executed, in order
  oracle.json  - structured records
  fixtures/    - every `.hdr` / `.img` pair the records refer to

Re-running needs only the vips binary at VIPS; every input is generated from
scratch, deterministically. Nothing outside this script's own directory is
written.
"""
import hashlib
import json
import os
import re
import struct
import subprocess

ROOT = os.path.dirname(os.path.abspath(__file__))
FIX = os.path.join(ROOT, "fixtures")
OUT = os.path.join(ROOT, "outputs")

VIPS = "/opt/homebrew/bin/vips"
VIPSHEADER = "/opt/homebrew/bin/vipsheader"

os.makedirs(FIX, exist_ok=True)
os.makedirs(OUT, exist_ok=True)

COMMANDS = []

# This machine's vips prints a heif module-load failure on stderr for every
# single invocation (libheif wants an x265 dylib that is not installed). It
# has nothing to do with Analyze, so it is stripped from every captured
# stderr rather than being pinned as if it were part of the answer.
NOISE = ("VIPS-WARNING", "Referenced from:", "Reason: tried:",
         "libheif", "x265", "unable to load")


# A GLib runtime message carries the pid and a wall-clock timestamp, and vips
# repeats `<op>: load error` a thread-dependent number of times. Neither is
# part of the answer and both would make this capture differ from run to run,
# so the prefix is stripped and consecutive duplicates are folded.
GLIB_PREFIX = re.compile(
    r"^\((?:process|vips|vipsheader):\d+\): "
    r"(?P<domain>[\w-]+) \*\*: \d\d:\d\d:\d\d\.\d+: ")


def clean(text):
    """stderr with this build's unrelated module warnings removed, the pid
    and timestamp stripped off any GLib message, and consecutive duplicate
    lines folded, so re-running capture.py reproduces oracle.json byte for
    byte."""
    keep = []
    for ln in text.splitlines():
        if not ln.strip() or any(n in ln for n in NOISE):
            continue
        # vips repeats this plumbing trailer a thread-dependent number of
        # times and it carries no information the line above it does not.
        if ln.strip() == "analyzeload: load error":
            continue
        ln = GLIB_PREFIX.sub(lambda m: m.group("domain") + ": ", ln)
        ln = ln.replace(ROOT + "/", "")
        if not keep or keep[-1] != ln:
            keep.append(ln)
    return "\n".join(keep)


def run(args, allow_fail=True):
    """Run a command, logging it (with this directory's absolute path
    reduced to a relative one) for commands.sh."""
    COMMANDS.append(" ".join(a.replace(ROOT + "/", "") for a in args))
    proc = subprocess.run(args, capture_output=True, text=True)
    if proc.returncode != 0 and not allow_fail:
        raise SystemExit(f"failed: {' '.join(args)}\n{proc.stderr}")
    return proc


def vips(*args, allow_fail=True):
    return run([VIPS, *args], allow_fail=allow_fail)


def header(path, all_fields=False):
    """`vipsheader` on a path: the SNIFFED path, where analyzeload has to win
    the is_a race against every other loader."""
    args = [VIPSHEADER] + (["-a"] if all_fields else []) + [path]
    proc = run(args)
    if proc.returncode != 0:
        return {"exit": proc.returncode, "stderr": clean(proc.stderr)}
    if not all_fields:
        return {"exit": 0,
                "summary": proc.stdout.strip().replace(ROOT + "/", "")}
    out = {"exit": 0}
    for line in proc.stdout.splitlines()[1:]:
        if ": " in line:
            name, value = line.split(": ", 1)
            out[name.strip()] = value.strip().replace(ROOT + "/", "")
    return out


def direct(path):
    """`vips analyzeload` called by name, which SKIPS is_a entirely. The
    difference matters: a header analyzeload rejects falls through the
    sniffing chain to magickload, so `vipsheader` reports a TGA error and
    never mentions Analyze at all."""
    dest = os.path.join(OUT, "direct.v")
    proc = vips("analyzeload", path, dest)
    r = {"exit": proc.returncode, "stderr": clean(proc.stderr)}
    if proc.returncode == 0:
        r["header"] = header(dest)["summary"]
    return r


def pixels(path, w, h):
    """Every pixel of a small image, in raster order."""
    out = []
    for y in range(h):
        for x in range(w):
            proc = vips("getpoint", path, str(x), str(y))
            if proc.returncode != 0:
                return {"exit": proc.returncode, "stderr": clean(proc.stderr)}
            out.append([float(v) for v in proc.stdout.split()])
    return out


def measured_size(summary):
    """The width and height vips itself reports, pulled out of a vipsheader
    summary line. Every getpoint sweep here is driven by THIS rather than by
    a size worked out in Python, so no expectation in oracle.json was
    computed by hand."""
    m = re.search(r": (\d+)x(\d+) ", summary)
    return (int(m.group(1)), int(m.group(2))) if m else (None, None)


def stat(path):
    """`vips avg`, which forces the whole `.img` through the pipeline. This
    is the call that fails when the header was fine and the pixels are not."""
    proc = vips("avg", path)
    if proc.returncode != 0:
        return {"exit": proc.returncode, "stderr": clean(proc.stderr)}
    return {"exit": 0, "avg": float(proc.stdout)}


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


# ---------------------------------------------------------------------------
# The 348-byte struct dsr, written big-endian by default because that is the
# only byte order read_header (analyze2vips.c:320-348) will accept.
# ---------------------------------------------------------------------------
def dsr(dims, datatype, bitpix, descrip=b"", vox_offset=0.0, sizeof_hdr=348,
        extents=16384, regular=b"r", glmax=0, glmin=0, patient_id=b"",
        endian=">"):
    e = endian
    hk = (struct.pack(e + "i", sizeof_hdr)
          + b"\0" * 10                       # data_type[10]
          + b"\0" * 18                       # db_name[18]
          + struct.pack(e + "i", extents)
          + struct.pack(e + "h", 0)          # session_error
          + regular                          # regular
          + b"\0")                           # hkey_un0
    assert len(hk) == 40, len(hk)

    d8 = list(dims) + [0] * (8 - len(dims))
    dime = b"".join(struct.pack(e + "h", v) for v in d8)   # dim[8]
    dime += b"\0" * 4                                       # vox_units[4]
    dime += b"\0" * 8                                       # cal_units[8]
    dime += struct.pack(e + "h", 0)                         # unused1
    dime += struct.pack(e + "h", datatype)
    dime += struct.pack(e + "h", bitpix)
    dime += struct.pack(e + "h", 0)                         # dim_un0
    dime += b"".join(struct.pack(e + "f", v)
                     for v in (0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0))
    dime += struct.pack(e + "f", vox_offset)
    dime += struct.pack(e + "f", 0.0) * 3                   # funused1..3
    dime += struct.pack(e + "f", 0.0) * 2                   # cal_max, cal_min
    dime += struct.pack(e + "i", 0) * 2                     # compressed, verified
    dime += struct.pack(e + "i", glmax) + struct.pack(e + "i", glmin)
    assert len(dime) == 108, len(dime)

    hist = (descrip.ljust(80, b"\0")
            + b"\0" * 24                     # aux_file[24]
            + b"\0"                          # orient
            + b"\0" * 10                     # originator[10]
            + b"\0" * 10                     # generated[10]
            + b"\0" * 10                     # scannum[10]
            + patient_id.ljust(10, b"\0")    # patient_id[10]
            + b"\0" * 10                     # exp_date[10]
            + b"\0" * 10                     # exp_time[10]
            + b"\0" * 3                      # hist_un0[3]
            + struct.pack(e + "i", 0) * 8)   # views .. smin
    assert len(hist) == 200, len(hist)

    out = hk + dime + hist
    assert len(out) == 348, len(out)
    return out


def pair(name, dims, datatype, bitpix, img, **kw):
    """Write fixtures/<name>.hdr and fixtures/<name>.img, return the .hdr."""
    hdr_path = os.path.join(FIX, name + ".hdr")
    img_path = os.path.join(FIX, name + ".img")
    with open(hdr_path, "wb") as f:
        f.write(dsr(dims, datatype, bitpix, **kw))
    with open(img_path, "wb") as f:
        f.write(img)
    return hdr_path


records = {}
notes = []

# ---------------------------------------------------------------------------
# 1. The base case, and the .hdr / .img pairing.
# ---------------------------------------------------------------------------
BASE = bytes([10, 20, 30, 40, 50, 60])
base = pair("base_2d_uchar", [2, 3, 2], 2, 8, BASE, descrip=b"libviprs oracle")

pairing = {}
for label, path in (("hdr", base),
                    ("img", base[:-4] + ".img"),
                    ("stem", base[:-4])):
    pairing[label] = {"sniffed": header(path), "direct": direct(path)}

# .hdr present, .img missing.
nohdr = os.path.join(FIX, "no_img.hdr")
with open(nohdr, "wb") as f:
    f.write(dsr([2, 3, 2], 2, 8))
pairing["hdr_without_img"] = {"sniffed": header(nohdr),
                              "avg": stat(nohdr),
                              "direct": direct(nohdr)}

# .img present, .hdr missing.
onlyimg = os.path.join(FIX, "no_hdr.img")
with open(onlyimg, "wb") as f:
    f.write(BASE)
pairing["img_without_hdr"] = {"sniffed": header(onlyimg),
                              "direct": direct(onlyimg)}

records["hdr_img_pairing"] = {
    "what": "generate_filenames (analyze2vips.c:224-231) rewrites whatever "
            "you hand it into BOTH names via vips__change_suffix, stripping "
            "a trailing `.img` or `.hdr` first. So `.hdr`, `.img` and the "
            "bare stem all name the same image, and the pixels always come "
            "from the `.img` while the geometry always comes from the "
            "`.hdr`. The bare stem works only through a DIRECT "
            "`vips analyzeload` call: `vipsheader fred` fails first, in "
            "VipsForeignLoad's own existence check, before is_a ever runs. "
            "The gtkdoc at analyzeload.c:161-163 says you can load `fred` "
            "and that is true of the operator, not of new_from_file.",
    "registered_suffixes": [".img", ".hdr"],
    "cases": pairing,
    "give_up_point": "A missing or short `.img` is invisible to the header "
                     "load: vips__analyze_read_header only reads the `.hdr`. "
                     "The failure lands in vips__analyze_read, at "
                     "vips_image_new_from_file_raw, i.e. on the first pixel "
                     "fetch.",
}
notes.append(
    "analyzeload is registered at priority -50 with is_a = vips__isanalyze, "
    "and vips__isanalyze opens and fully parses the .hdr. When it says no, "
    "the sniffing chain falls through to magickload, which reports a TGA "
    "error. Every refusal here therefore records BOTH the sniffed message "
    "(useless) and the direct-operator message (the real one)."
)

# ---------------------------------------------------------------------------
# 2. The carriers, empirically.
# ---------------------------------------------------------------------------
DT = {
    1: ("DT_BINARY", 1, 1),
    2: ("DT_UNSIGNED_CHAR", 8, 1),
    4: ("DT_SIGNED_SHORT", 16, 2),
    8: ("DT_SIGNED_INT", 32, 4),
    16: ("DT_FLOAT", 32, 4),
    32: ("DT_COMPLEX", 64, 8),
    64: ("DT_DOUBLE", 64, 8),
    128: ("DT_RGB", 24, 3),
    0: ("unnamed (0)", 0, 1),
    256: ("unnamed (256)", 8, 1),
    511: ("unnamed (511)", 8, 1),
}
carriers = {}
for dt in sorted(DT):
    name, bitpix, elsize = DT[dt]
    body = bytes(range(1, 1 + 4 * elsize))
    p = pair(f"dt{dt}", [2, 2, 2], dt, bitpix, body)
    entry = {
        "constant": name,
        "bitpix_written": bitpix,
        "img_bytes": list(body),
        "sniffed": header(p),
        "direct": direct(p),
    }
    if entry["sniffed"]["exit"] == 0:
        entry["pixels"] = pixels(p, 2, 2)
        entry["avg"] = stat(p)
    carriers[str(dt)] = entry
records["datatype_to_carrier"] = {
    "what": "get_vips_properties (analyze2vips.c:385-425) is the whole "
            "carrier table, and it is a closed switch on dime.datatype: "
            "everything else is `datatype %d not supported`. bitpix is "
            "attached as metadata and NEVER consulted, so a header claiming "
            "bitpix 1 for DT_UNSIGNED_CHAR still reads 8-bit pixels. Note "
            "DT_BINARY (1) is in dbh.h and is NOT implemented, and DT_RGB "
            "(128) is the only value that gives more than one band.",
    "measured": carriers,
    "getpoint_on_complex": "DT_COMPLEX gives a 1-band VIPS_FORMAT_COMPLEX "
                           "image and `vips getpoint` prints only the REAL "
                           "part, so the dt32 pixels below are the first "
                           "float of each 8-byte pair and the imaginary "
                           "halves are not pinned here.",
    "interpretation_rule": "bands == 1 -> b-w, otherwise sRGB "
                           "(analyze2vips.c:544-546). There is no 16-bit "
                           "greyscale or RGB16 tag: a DT_SIGNED_SHORT image "
                           "is tagged b-w, not grey16.",
}

# ---------------------------------------------------------------------------
# 3. Rank, and how the higher axes are folded into the height.
# ---------------------------------------------------------------------------
rank = {}
for r, dims in ((0, [0, 2, 2]),
                (1, [1, 4]),
                (2, [2, 3, 2]),
                (3, [3, 3, 2, 2]),
                (4, [4, 3, 2, 2, 2]),
                (7, [7, 2, 2, 2, 1, 1, 1, 1]),
                (8, [8, 2, 2, 1, 1, 1, 1, 1])):
    body = bytes((i * 7) % 256 for i in range(256))
    dim = dims + [0] * (8 - len(dims))
    p = pair(f"rank{r}", dims, 2, 8, body)
    entry = {"dim": dim, "sniffed": header(p), "direct": direct(p)}
    if entry["sniffed"]["exit"] == 0:
        w, h = measured_size(entry["sniffed"]["summary"])
        entry["measured_width"] = w
        entry["measured_height"] = h
        # The C rule, restated in Python and CHECKED against the binary
        # rather than written into the record as an expectation.
        product = dim[2]
        for i in range(3, dim[0] + 1):
            product *= dim[i]
        entry["width_equals_dim1"] = w == dim[1]
        entry["height_equals_product_of_dim2_up"] = h == product
        entry["pixels"] = pixels(p, w, h)
    rank[str(r)] = entry
records["rank_and_flattening"] = {
    "what": "dim[0] is the rank and must be in 2..=7 (analyze2vips.c:368). "
            "width is dim[1]; height starts at dim[2] and is MULTIPLIED by "
            "every dim[i] up to dim[rank] (analyze2vips.c:377-381). So a "
            "3x2x2 volume flattens to one 3x4 image with the slices stacked "
            "vertically, exactly like a vips toilet roll, but with no "
            "page-height tag written: nothing in the loaded image records "
            "that it was ever 3-D, beyond the dsr-image_dimension.dim[] "
            "metadata. Rank 1 is refused even though a 1-D array is a "
            "perfectly good image, and rank 8 is refused before dim[8] can "
            "be read, which is just as well since dim[] only has 8 slots.",
    "measured": rank,
}

# ---------------------------------------------------------------------------
# 4. Byte order, in both the header and the pixels.
# ---------------------------------------------------------------------------
SHORTS = (1, 256, -1, 4660)
be16 = pair("be_short", [2, 2, 2], 4, 16, struct.pack(">4h", *SHORTS))
le_hdr = os.path.join(FIX, "le_header.hdr")
with open(le_hdr, "wb") as f:
    f.write(dsr([2, 3, 2], 2, 8, endian="<"))
with open(os.path.join(FIX, "le_header.img"), "wb") as f:
    f.write(BASE)
le_px = pair("le_short", [2, 2, 2], 4, 16, struct.pack("<4h", *SHORTS))
records["byte_order"] = {
    "what": "Both halves are big-endian, always. read_header "
            "(analyze2vips.c:317-348) byte-swaps every SHORT/INT/FLOAT field "
            "of the dsr when !vips_amiMSBfirst(), and vips__analyze_read "
            "(analyze2vips.c:589) runs vips__byteswap_bool over the PIXELS "
            "under the same condition. There is no flag, no sniff and no "
            "escape hatch: a little-endian `.hdr` is simply rejected, "
            "because its sizeof_hdr reads back as 0x5c010000 rather than "
            "348, and little-endian pixels are silently read as big-endian "
            "garbage. This is the single most likely place for a port on a "
            "little-endian host to get it backwards.",
    "shorts_written": list(SHORTS),
    "big_endian_pixels": {
        "fixture": "fixtures/be_short.hdr",
        "header": header(be16),
        "pixels": pixels(be16, 2, 2),
    },
    "little_endian_pixels_are_read_as_big_endian": {
        "fixture": "fixtures/le_short.hdr",
        "header": header(le_px),
        "pixels": pixels(le_px, 2, 2),
        "note": "same four shorts, written the other way round. Every value "
                "comes back byte-swapped and no error is raised.",
    },
    "little_endian_header_is_refused": {
        "fixture": "fixtures/le_header.hdr",
        "sniffed": header(le_hdr),
        "direct": direct(le_hdr),
        "sizeof_hdr_as_read": struct.unpack(">i", struct.pack("<i", 348))[0],
    },
}

# ---------------------------------------------------------------------------
# 5. vox_offset is parsed and then ignored.
# ---------------------------------------------------------------------------
vox = pair("vox_offset_64", [2, 3, 2], 2, 8, BASE + b"\xff" * 64,
           vox_offset=64.0)
records["vox_offset_is_ignored"] = {
    "what": "dime.vox_offset is the format's own `skip this many bytes of "
            "the .img` field. vips parses it, attaches it as "
            "`dsr-image_dimension.vox_offset`, and then calls "
            "vips_image_new_from_file_raw(image, w, h, sizeof_line, 0) "
            "(analyze2vips.c:582-583) with a hard-coded offset of 0. So the "
            "pixels below are the FIRST six bytes of the .img, not the six "
            "at offset 64. A port that honours vox_offset would disagree "
            "with vips on every file that sets it.",
    "fixture": "fixtures/vox_offset_64.hdr",
    "vox_offset": 64.0,
    "img_bytes": list(BASE) + ["0xff x 64"],
    "header": header(vox),
    "pixels": pixels(vox, 3, 2),
    "avg": stat(vox),
}

# ---------------------------------------------------------------------------
# 6. Refusals, and exactly where the load gives up.
# ---------------------------------------------------------------------------
refusals = {}

# 6a. A .hdr that is not 348 bytes long.
for n, delta in (("hdr_349_bytes", 1), ("hdr_347_bytes", -1),
                 ("hdr_0_bytes", -348)):
    p = os.path.join(FIX, n + ".hdr")
    body = dsr([2, 3, 2], 2, 8)
    with open(p, "wb") as f:
        f.write(body + b"\x00" * delta if delta > 0 else body[:348 + delta])
    with open(os.path.join(FIX, n + ".img"), "wb") as f:
        f.write(BASE)
    refusals[n] = {"bytes": 348 + delta,
                   "sniffed": header(p), "direct": direct(p)}

# 6b. sizeof_hdr disagreeing with the real length.
for n, sz in (("sizeof_hdr_200", 200), ("sizeof_hdr_0", 0),
              ("sizeof_hdr_348", 348)):
    p = pair(n, [2, 3, 2], 2, 8, BASE, sizeof_hdr=sz)
    refusals[n] = {"sizeof_hdr": sz,
                   "sniffed": header(p), "direct": direct(p)}

# 6c. A header whose dimensions the .img cannot possibly satisfy.
huge = pair("dims_32767", [2, 32767, 32767], 2, 8, BASE)
refusals["dims_32767x32767_with_6_byte_img"] = {
    "declared_pixels": 32767 * 32767,
    "img_bytes": len(BASE),
    "sniffed": header(huge),
    "avg": stat(huge),
    "direct": direct(huge),
    "note": "the header load SUCCEEDS and reports a 1.07 gigapixel image. "
            "Nothing compares the .img against the header until the pipeline "
            "actually opens it.",
}

short_img = pair("img_truncated", [2, 3, 2], 2, 8, b"\x01\x02")
refusals["img_two_bytes_short_of_six"] = {
    "sniffed": header(short_img),
    "getpoint_0_0": pixels(short_img, 1, 1),
    "avg": stat(short_img),
    "direct": direct(short_img),
}

extra_img = pair("img_oversize", [2, 3, 2], 2, 8, BASE + b"\xff" * 100)
refusals["img_100_bytes_longer_than_needed"] = {
    "sniffed": header(extra_img),
    "pixels": pixels(extra_img, 3, 2),
    "avg": stat(extra_img),
    "note": "a trailing tail is accepted and ignored; only a SHORT .img is "
            "an error.",
}

# 6d. A negative dimension. This one is the interesting failure.
neg_w = pair("dim1_negative", [2, -3, 2], 2, 8, BASE)
neg_h = pair("dim2_negative", [2, 3, -2], 2, 8, BASE)
zero_w = pair("dim1_zero", [2, 0, 2], 2, 8, BASE)
refusals["negative_and_zero_dimensions_are_clamped_not_refused"] = {
    "what": "dim[] is a signed short and nothing in get_vips_properties "
            "range-checks it. A negative or zero width reaches "
            "vips_image_init_fields, GObject's property range check rejects "
            "the value, prints a GLib-GObject-CRITICAL to stderr, LEAVES THE "
            "PROPERTY AT ITS DEFAULT OF 1, and the load carries on and "
            "SUCCEEDS with a silently wrong geometry. exit is 0. A port "
            "should refuse instead, and should say so as a deliberate "
            "divergence rather than reproducing this.",
    "dim1_minus_3": {"sniffed": header(neg_w), "avg": stat(neg_w),
                     "direct": direct(neg_w)},
    "dim2_minus_2": {"sniffed": header(neg_h), "avg": stat(neg_h),
                     "direct": direct(neg_h)},
    "dim1_zero": {"sniffed": header(zero_w), "avg": stat(zero_w),
                  "direct": direct(zero_w)},
}

# 6e. Garbage where the header should be.
for n, body in (("all_zero_348", b"\x00" * 348),
                ("all_ff_348", b"\xff" * 348),
                ("ascii_348", b"not an analyze header at all " * 12)):
    p = os.path.join(FIX, n + ".hdr")
    with open(p, "wb") as f:
        f.write(body[:348])
    with open(os.path.join(FIX, n + ".img"), "wb") as f:
        f.write(BASE)
    refusals[n] = {"sniffed": header(p), "direct": direct(p)}

records["refusals"] = {
    "what": "analyzeload is registered UNTRUSTED (analyzeload.c:125), so "
            "this is the half of its behaviour that matters most. There are "
            "exactly four ways it says no, and all four live in "
            "analyze2vips.c: `header file size incorrect` (the .hdr is not "
            "sizeof(struct dsr) == 348 bytes, line 306), `header size "
            "incorrect` (the sizeof_hdr FIELD disagrees with the real "
            "length, line 350), `%d-dimensional images not supported` (rank "
            "outside 2..=7, line 368) and `datatype %d not supported` (line "
            "421). Everything else it accepts, including geometries the "
            ".img cannot satisfy and dimensions that are not positive.",
    "cases": refusals,
}

# ---------------------------------------------------------------------------
# 7. The metadata, and what getstr does to it.
# ---------------------------------------------------------------------------
FULL80 = bytes(ord("A") + (i % 26) for i in range(80))
CTRL = b"ok\x01\x02\x7f\xc3\xa9end"
meta = pair("meta_strings", [2, 3, 2], 2, 8, BASE,
            descrip=FULL80, patient_id=CTRL, glmax=255, glmin=0,
            vox_offset=0.0, extents=16384)
records["metadata"] = {
    "what": "attach_meta (analyze2vips.c:437-482) sets the whole 348-byte "
            "header as a blob named `dsr`, then walks the same 70-entry "
            "table again setting one field per struct member, named "
            "`dsr-<section>.<member>`; dsr_field_count below is how many "
            "that turned out to be, counted off vipsheader rather than "
            "off the C table. Two traps in getstr "
            "(analyze2vips.c:237-256): g_strlcpy is given the FIELD length "
            "as its buffer size, so it copies at most len-1 characters and "
            "an 80-byte descrip loses its last byte; and every byte failing "
            "`isascii(c) || c >= 32` is rewritten to `@`, which is lossy and "
            "not reversible. Note what that test does NOT catch: 0x7f DEL is "
            "ascii and is >= 32, so it passes through untouched, while the "
            "two bytes of a UTF-8 `e-acute` both become `@`.",
    "fixture": "fixtures/meta_strings.hdr",
    "descrip_written": FULL80.decode(),
    "descrip_written_len": len(FULL80),
    "patient_id_written": list(CTRL),
    "header": header(meta, all_fields=True),
    "blob_field": "dsr",
    "blob_bytes": 348,
}
records["metadata"]["dsr_field_count"] = sum(
    1 for k in records["metadata"]["header"] if k.startswith("dsr-"))

# ---------------------------------------------------------------------------
# 8. A colour image, so the DT_RGB band layout is pinned rather than guessed.
# ---------------------------------------------------------------------------
RGB = bytes([10, 11, 12, 20, 21, 22, 30, 31, 32,
             40, 41, 42, 50, 51, 52, 60, 61, 62])
rgb = pair("rgb_2d", [2, 3, 2], 128, 24, RGB)
records["dt_rgb_is_interleaved"] = {
    "what": "DT_RGB gives bands = 3 and format uchar, and the pixels are "
            "read straight through vips_image_new_from_file_raw with a line "
            "size of bands * sizeof(fmt), so the .img is INTERLEAVED RGB, "
            "not three planes. The Analyze spec allows either and vips only "
            "reads the interleaved one.",
    "fixture": "fixtures/rgb_2d.hdr",
    "img_bytes": list(RGB),
    "header": header(rgb, all_fields=False),
    "pixels": pixels(rgb, 3, 2),
}

# ---------------------------------------------------------------------------
# 9. The three-way version disagreement, recorded from the binary.
# ---------------------------------------------------------------------------
version = run([VIPS, "--version"]).stdout.strip()
config = run([VIPS, "--vips-config"])
config_line = [ln.strip() for ln in config.stdout.replace(",", "\n").splitlines()
               if "nalyze" in ln]
listing = run([VIPS, "-l"])
op_line = [ln.strip() for ln in listing.stdout.splitlines()
           if "analyzeload" in ln]
records["analyze6_vs_analyze7"] = {
    "what": "libvips names this format three different ways and they do not "
            "agree. The operator is the one to follow, because it is what "
            "`vips -l` and every caller actually see, and because the "
            "implementation is a straight read of the Mayo `dbh.h` struct "
            "that both Analyze 6 and Analyze 7.5 share. Nothing in the "
            "loader distinguishes a 6.0 file from a 7.5 one: there is no "
            "version field in the 348-byte header to distinguish them with. "
            "So the version in the name is decoration, and a port should not "
            "try to act on it.",
    "vips_config_says": config_line,
    "vips_config_source": "meson.build:704, `'enable Analyze7 load'`",
    "both_halves_re_read_on": version,
    "discrepancy_survived_the_8_18_4_to_8_18_6_bump": True,
    "operator_says": op_line,
    "operator_source": "analyzeload.c:120, "
                       "`object_class->description = _(\"load an Analyze6 "
                       "image\")`",
    "gtkdoc_says": "analyzeload.c:161, `Load an Analyze 6.0 file`",
    "implementation_comment_says": "analyze2vips.c:1, `Read a Analyze file. "
                                   "Old-style header (so called 7.5 format)`",
    "dbh_header_says": "dbh.h:15-21, `The previous-generation header for "
                       "Analyze images ... a 348 byte header stored in a "
                       "file with a .hdr suffix`",
    "resolution": "follow the operator: analyzeload, .img/.hdr, "
                  "priority -50, untrusted, load-only.",
}
notes.append(
    "ORACLE BINARY MOVED. Homebrew replaced libvips 8.18.4 with 8.18.6 on "
    "this machine while this capture was being taken, so every number in "
    "this file came from 8.18.6 even though the epic runbook pins the "
    "oracle at 8.18.4. Only 8.18.6 is in the Cellar now, so 8.18.4 cannot "
    "be re-measured here. Nothing in THIS file is known to differ between "
    "the two: vips__isanalyze and read_header were disassembled out of the "
    "8.18.6 dylib and match the checkout source instruction for "
    "instruction in shape. The companion foreign-mat capture found a "
    "predicate that DOES differ, so the warning is worth carrying."
)
notes.append(
    "There is NO analyzesave. The format is load-only in this build and in "
    "libvips generally, so there is no round trip to pin and no save "
    "behaviour to match."
)
notes.append(
    "priority is -50 (analyzeload.c:131), the lowest of any loader in this "
    "build, because is_a has to open and fully parse a second file. Any port "
    "that sniffs Analyze earlier than that will steal files from loaders "
    "that should have won."
)

# ---------------------------------------------------------------------------
fixture_bytes = sum(os.path.getsize(os.path.join(FIX, f))
                    for f in os.listdir(FIX))
oracle = {
    "meta": {
        "area": "foreign-analyze",
        "issue": 640,
        "parent_issue": 510,
        "vips_version": version,
        "vips_binary": VIPS,
        "operation": "analyzeload",
        "operation_listing": op_line[0] if op_line else None,
        "save_operation": None,
        "captured_by": "oracle-captures/foreign-analyze/capture.py",
        "reference_c": "libvips v8.18.0-95-gfe420cf3a for the file and line "
                       "numbers quoted here. The binary every measurement "
                       "came out of is the installed release named in "
                       "vips_version above and is a DIFFERENT ARTEFACT "
                       "from that tree, which is not a formality: see the "
                       "oracle_binary_moved note.",
        "oracle_binary": {
            "version": "measured by `vips --version` at capture time and "
                       "recorded in vips_version above; do not assume the "
                       "8.18.4 the epic runbook names",
            "config_line": "`vips --vips-config` prints booleans only and "
                           "names no library versions, so there is no "
                           "matio or Analyze version to read off it",
            "upgraded_mid_capture": "Homebrew replaced libvips 8.18.4 with "
                                    "8.18.6 on this machine at 08:10:03 on "
                                    "2026-08-26 and deleted the old keg, so "
                                    "only 8.18.6 remains and 8.18.4 cannot "
                                    "be re-measured here",
            "which_side_of_the_upgrade": "EVERY number in this file came "
                                         "from 8.18.6. capture.py was first "
                                         "run at 08:12 and has been re-run "
                                         "since; the only 8.18.4 readings "
                                         "taken in this lane were "
                                         "exploratory probes between 08:06 "
                                         "and 08:09 that reached no record "
                                         "except the explicitly-labelled "
                                         "before/after in foreign-mat's "
                                         "sniff_predicate",
            "not_reconciled": "the pre-existing capture areas "
                              "(convolution, foreign-radiance, "
                              "foreign-webp) and the in-flight FITS, EXR "
                              "and JXL ones record 8.18.4. That difference "
                              "is tracked by the epic orchestrator and is "
                              "not resolved here.",
        },
        "fixture_count": len(os.listdir(FIX)),
        "fixture_bytes": fixture_bytes,
    },
    "notes": notes,
    "records": records,
}

with open(os.path.join(ROOT, "oracle.json"), "w") as f:
    # allow_nan=False so a non-finite measurement stops the capture here
    # rather than writing a file nobody outside Python can parse (#682).
    json.dump(oracle, f, indent=2, sort_keys=False, allow_nan=False)
    f.write("\n")

with open(os.path.join(ROOT, "commands.sh"), "w") as f:
    f.write("#!/bin/sh\n")
    f.write("# Every vips command capture.py ran, in order. Regenerate with\n")
    f.write("# `python3 capture.py` from this directory.\n")
    f.write("#\n")
    f.write("# Many of these are EXPECTED to fail: the refusal records are "
            "the point.\n")
    f.write("set +e\n\n")
    for c in COMMANDS:
        f.write(c + "\n")

print(f"wrote oracle.json ({len(records)} records), commands.sh "
      f"({len(COMMANDS)} commands), {len(os.listdir(FIX))} fixtures "
      f"({fixture_bytes} bytes)")
