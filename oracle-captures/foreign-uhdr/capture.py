#!/usr/bin/env python3
"""
Oracle capture for the UltraHDR area (issue #639).

Runs the real vips CLI over small, deterministic UltraHDR files and records
what `uhdrload`, `uhdrsave` and `uhdr2scRGB` actually do. Two subjects carry
most of the weight, because a port can get everything else right and still be
wrong on them:

  * `uhdr2scRGB` is where the format's semantics live. It is the transform
    that turns the gainmap into pixels, so a container-only port passes every
    header check and produces wrong images. Every arm of it is pinned at full
    `getpoint` precision: the base linearisation, the mono and the RGB gainmap
    paths (which do NOT agree), the gamma, boost and offset terms, the gainmap
    resize, and the degenerate metadata that produces inf and NaN.

  * detection. `uhdrload` declares the ordinary JPEG suffixes, so a UHDR file
    is found by content, through a two-stage gate whose halves disagree: the
    fast `is_a` wants an MPF APP2 marker, the header wants libuhdr's
    `is_uhdr_image()`. A file that satisfies only the second loads as a plain
    JPEG by default and as UltraHDR when you name `uhdrload`.

The oracle moved under this capture: a `brew upgrade` running on the host
replaced vips 8.18.4 with 8.18.6 mid-run. `meta` records what every number was
measured against and `notes` says what changed.

Writes commands.sh, oracle.json and fixtures/. Re-running needs only the vips
binary at VIPS; every input is generated from scratch, deterministically,
except REFERENCE_UHDR, which is recorded by path and sha256 and skipped if
absent. Nothing outside this script's own directory is written.
"""
import base64
import hashlib
import json
import math
import os
import re
import struct
import subprocess

ROOT = os.path.dirname(os.path.abspath(__file__))
FIX = os.path.join(ROOT, "fixtures")
OUT = os.path.join(ROOT, "outputs")

VIPS = "/opt/homebrew/bin/vips"
VIPSHEADER = "/opt/homebrew/bin/vipsheader"
VIPSEDIT = "/opt/homebrew/bin/vipsedit"

# The libvips reference suite's own UltraHDR image, 3840x2160. Recorded by
# path and digest; far too big to check in, and every number taken from it says
# so.
REFERENCE_UHDR = "/Users/rom/workspace/libvips/test/test-suite/images/ultra-hdr.jpg"

os.makedirs(FIX, exist_ok=True)
os.makedirs(OUT, exist_ok=True)

COMMANDS = []

# This build's libheif module cannot be dlopened (a stale x265 dependency), so
# every single vips invocation prints a multi-line VIPS-WARNING to stderr.
# Nothing here loads a HEIC, so it is pure noise and it is filtered out of
# every captured stderr.
NOISE = ("VIPS-WARNING", "unable to load", "x265", "Referenced from", "Reason:")


# A GLib warning carries the pid and the wall clock, neither of which is a
# measurement. Both are dropped so re-running produces the same oracle.json.
STAMP = re.compile(r"\(vips:\d+\): (.*) \*\*: \d\d:\d\d:\d\d\.\d+: ")


def scrub(text):
    return "\n".join(STAMP.sub(r"(vips): \1 **: ", line)
                     for line in text.splitlines()
                     if line.strip() and not any(n in line for n in NOISE))


def run(args, allow_fail=False, stdin=None, log=True):
    """Run a command, logging it (with this directory's absolute path reduced
    to a relative one) for commands.sh."""
    if log:
        COMMANDS.append(" ".join(a.replace(ROOT + "/", "") for a in args))
    proc = subprocess.run(args, capture_output=True, text=True, input=stdin)
    if proc.returncode != 0 and not allow_fail:
        raise SystemExit(f"failed: {' '.join(args)}\n{scrub(proc.stderr)}")
    return proc.returncode, proc.stdout.strip(), scrub(proc.stderr)


def vips(*args, allow_fail=False, log=True):
    return run([VIPS, *args], allow_fail=allow_fail, log=log)


def header(path, all_fields=False):
    args = [VIPSHEADER] + (["-a"] if all_fields else []) + [path]
    rc, out, err = run(args, allow_fail=True)
    if rc != 0:
        return {"error": err}
    out = out.replace(ROOT + "/", "")
    if not all_fields:
        return {"summary": out}
    fields = {}
    for line in out.splitlines()[1:]:
        if ": " in line:
            name, value = line.split(": ", 1)
            fields[name.strip()] = value.strip()
    return fields


def getpoint(path, w, h=1):
    """Every pixel of a small image, in raster order. vips prints 17
    significant digits for a float band, so this pins the exact value.

    Logged to commands.sh as one shell loop rather than w*h separate lines,
    which is the same set of commands and a tenth of the file."""
    rel = path.replace(ROOT + "/", "")
    COMMANDS.append(f"for y in $(seq 0 {h - 1}); do for x in $(seq 0 {w - 1}); "
                    f"do {VIPS} getpoint {rel} $x $y; done; done")
    out = []
    for y in range(h):
        for x in range(w):
            _, txt, _ = vips("getpoint", path, str(x), str(y), log=False)
            out.append([float(v) for v in txt.split()])
    return out


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def raster(name, data, w, h, bands, fmt="uchar", interp="srgb"):
    """A raw buffer loaded into a `.v` and tagged with an interpretation."""
    raw = os.path.join(FIX, name + ".raw")
    with open(raw, "wb") as f:
        f.write(data)
    plain = os.path.join(OUT, name + "-raw.v")
    vips("rawload", raw, plain, str(w), str(h), str(bands), "--format", fmt)
    tagged = os.path.join(OUT, name + ".v")
    vips("copy", plain, tagged, "--interpretation", interp)
    return tagged


def gainmap_jpeg(name, data, w, h, bands):
    """A gainmap, as the JPEG blob `gainmap-data` carries, plus the pixels it
    actually decodes to. Q=100 with subsampling off round-trips these flat
    ramps exactly, but the decoded values are measured rather than assumed:
    they are what uhdr2scRGB sees."""
    src = raster(name + "-src", data, w, h, bands,
                 interp="b-w" if bands == 1 else "srgb")
    jpg = os.path.join(FIX, name + ".jpg")
    vips("jpegsave", src, jpg, "--Q", "100", "--subsample-mode", "off",
         "--keep", "none")
    dec = os.path.join(OUT, name + "-decoded.v")
    vips("jpegload", jpg, dec)
    return jpg, dec


META_KEYS = ("max-content-boost", "min-content-boost", "gamma",
             "offset-sdr", "offset-hdr")


def attach(base_v, out_name, gainmap_jpg, meta):
    """Write a `.v` carrying the gainmap blob and the gainmap metadata
    uhdr2scRGB reads.

    `vipsedit --setext` replaces a `.v` file's XML trailer wholesale, which is
    the only way to set an arbitrary VipsArrayDouble or VipsBlob from the CLI.
    It is how every uhdr2scRGB record below gets metadata that no encoder would
    ever produce, which is the point: the transform has to be pinned over its
    whole domain, not over the one corner libuhdr's tonemapper emits."""
    out = os.path.join(OUT, out_name)
    with open(base_v, "rb") as f:
        blob = f.read()
    with open(out, "wb") as f:
        f.write(blob)

    fields = []
    if gainmap_jpg:
        with open(gainmap_jpg, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        fields.append(f'<field type="VipsBlob" name="gainmap-data">{b64}</field>')
    for key in META_KEYS:
        if key in meta:
            values = " ".join(repr(float(v)) for v in meta[key]) + " "
            fields.append('<field type="VipsArrayDouble" '
                          f'name="gainmap-{key}">{values}</field>')
    xml = ('<?xml version="1.0"?>\n'
           '<root xmlns="http://www.vips.ecs.soton.ac.uk/vips/8.18.4">\n'
           '  <header>\n'
           '    <field type="VipsRefString" name="Hist"></field>\n'
           '  </header>\n  <meta>\n    '
           + "\n    ".join(fields) + '\n  </meta>\n</root>\n')
    COMMANDS.append(f"{VIPSEDIT} --setext outputs/{out_name}   "
                    "# gainmap-data + gainmap-* arrays, from a heredoc")
    proc = subprocess.run([VIPSEDIT, "--setext", out], input=xml.encode(),
                          capture_output=True)
    if proc.returncode:
        raise SystemExit(scrub(proc.stderr.decode()))
    return out


def transform(label, base_v, gainmap_jpg, meta, w):
    """Run uhdr2scRGB over a pinned base and gainmap and return the pixels."""
    src = attach(base_v, f"p-{label}.v", gainmap_jpg, meta)
    dst = os.path.join(OUT, f"h-{label}.v")
    rc, _, err = vips("uhdr2scRGB", src, dst, allow_fail=True)
    if rc:
        return {"exit": rc, "stderr": err}
    return {"exit": 0, "header": header(dst)["summary"].split(": ", 1)[1],
            "scRGB": getpoint(dst, w)}


def segments(path):
    """The JPEG marker directory of a file: [marker, offset, total length,
    first bytes of the payload]. UltraHDR is two concatenated JPEGs, so this
    walks past EOI rather than stopping there."""
    with open(path, "rb") as f:
        d = f.read()
    out, p = [], 0
    while p + 1 < len(d):
        if d[p] != 0xFF:
            p += 1
            continue
        m = d[p + 1]
        if m in (0xD8, 0xD9, 0x01) or 0xD0 <= m <= 0xD7:
            out.append([hex(m), p, 2, ""])
            p += 2
            continue
        if m in (0xFF, 0x00):
            p += 1 if m == 0xFF else 2
            continue
        if p + 4 > len(d):
            break
        ln = int.from_bytes(d[p + 2:p + 4], "big")
        tag = d[p + 4:p + 4 + 28]
        out.append([hex(m), p, 2 + ln,
                    "".join(chr(c) if 32 <= c < 127 else "." for c in tag)])
        if m == 0xDA:
            q = p + 2 + ln
            while q + 1 < len(d):
                if (d[q] == 0xFF and d[q + 1] != 0x00
                        and not 0xD0 <= d[q + 1] <= 0xD7):
                    break
                q += 1
            p = q
            continue
        p += 2 + ln
    return out


def find_segment(path, marker, prefix):
    with open(path, "rb") as f:
        d = f.read()
    for m, off, ln, _ in segments(path):
        if int(m, 16) == marker and d[off + 4:off + 4 + len(prefix)] == prefix:
            return off, ln
    return None, None


records = {}
notes = []

# ---------------------------------------------------------------------------
# 0. What this build actually offers, taken from the binary.
# ---------------------------------------------------------------------------
_, op_list, _ = vips("-l")
_, config, _ = vips("--vips-config")
uhdr_ops = [l.strip() for l in op_list.splitlines() if "uhdr" in l.lower()]
jpeg_ops = [l.strip() for l in op_list.splitlines()
            if "jpegload" in l or "jpegsave_base" in l]
records["operator_surface"] = {
    "what": "The registered UltraHDR operations, verbatim from `vips -l`. "
            "uhdrsave declares NO suffixes (uhdrsave.c:58, "
            "`vips__uhdr_suffs = { NULL }`, commented `we don't want to "
            "trigger directly for jpeg save`), so a save to `.jpg` picks "
            "jpegsave and jpegsave hands over. uhdrload declares the four "
            "ordinary JPEG suffixes, so load is decided by content.",
    "vips_config_line": [l for l in config.splitlines() if "uhdr" in l.lower()],
    "uhdr_operations": uhdr_ops,
    "jpeg_operations_for_comparison": jpeg_ops,
    "uhdrload_priority": 100,
    "jpegload_priority": 50,
    "shared_suffixes": [".jpg", ".jpeg", ".jpe", ".jfif"],
    "uhdrsave_suffixes": [],
}

# ---------------------------------------------------------------------------
# 1. A small UltraHDR file, made by uhdrsave from a deterministic scRGB
#    raster. Everything downstream is derived from this one file.
# ---------------------------------------------------------------------------
W = H = 64
scrgb = bytearray()
for y in range(H):
    for x in range(W):
        if x >= 48 and y < 16:
            # an out-of-range highlight, so the gainmap has real content
            r, g, b = 8.0, 6.0, 4.0
        else:
            r, g, b = x / (W - 1), y / (H - 1), 0.25
        scrgb += struct.pack("<fff", r, g, b)
scrgb_v = raster("hdr64", bytes(scrgb), W, H, 3, fmt="float", interp="scrgb")

uhdr = os.path.join(FIX, "uhdr.jpg")
vips("uhdrsave", scrgb_v, uhdr)

records["uhdrsave_writes_this_container"] = {
    "what": "`vips uhdrsave` on a 64x64 scRGB float raster. The scRGB branch "
            "(uhdrsave.c:474-500) forces the image through vips_colourspace "
            "to scRGB, adds an alpha band if there is none because libuhdr "
            "wants RGBA, converts every sample to a half float with the "
            "round-to-nearest-even helper at uhdrsave.c:104-140, and hands "
            "the raw HDR image to libuhdr to tonemap. libuhdr, not libvips, "
            "chooses the gainmap and its metadata.",
    "fixture": "fixtures/uhdr.jpg",
    "sha256": sha256(uhdr),
    "bytes": os.path.getsize(uhdr),
    "segments": segments(uhdr),
    "structure": "Two concatenated JPEGs. The base runs SOI..EOI and carries "
                 "APP0 JFIF, APP1 Exif, APP2 ICC_PROFILE, APP2 "
                 "`urn:iso:std:iso:ts:21496:-1` and, immediately before SOS, "
                 "APP2 MPF. The gainmap is a second complete JPEG appended "
                 "after that EOI, mono, half the base's size, carrying its own "
                 "ISO 21496-1 APP2 and a COM segment naming the libuhdr build.",
    "header": header(uhdr, all_fields=True),
    "deterministic": "uhdrsave is byte-for-byte reproducible across runs of "
                     "one build. It is NOT reproducible across libuhdr "
                     "versions: 1.5.1 and 2.0.2 differ in exactly the three "
                     "bytes of the version string in the gainmap's COM "
                     "segment, and in nothing else.",
}

with open(uhdr, "rb") as f:
    uhdr_bytes = f.read()
com = uhdr_bytes.find(b"Source: google libuhdr")
libuhdr_com = uhdr_bytes[com:uhdr_bytes.find(b"\xff", com)].decode("ascii", "replace")

# The same raster saved as an ordinary JPEG, for contrast. jpegsave routes to
# uhdrsave for an scRGB image, so this one has to go through a uchar sRGB copy.
sdr_v = os.path.join(OUT, "sdr64.v")
vips("colourspace", scrgb_v, sdr_v, "srgb")
plain = os.path.join(FIX, "plain.jpg")
vips("jpegsave", sdr_v, plain, "--keep", "none")

# ---------------------------------------------------------------------------
# 2. How libvips decides an ordinary JPEG is an UltraHDR file.
# ---------------------------------------------------------------------------
iso = b"urn:iso:std:iso:ts:21496"
mpf_off, mpf_len = find_segment(uhdr, 0xE2, b"MPF\x00")
iso_offs = [(o, l) for m, o, l, _ in segments(uhdr)
            if int(m, 16) == 0xE2 and uhdr_bytes[o + 4:o + 4 + len(iso)] == iso]
base_eoi = uhdr_bytes.find(b"\xff\xd9") + 2

variants = {
    "no-mpf.jpg": uhdr_bytes[:mpf_off] + uhdr_bytes[mpf_off + mpf_len:],
    "no-iso-base.jpg": (uhdr_bytes[:iso_offs[0][0]]
                        + uhdr_bytes[iso_offs[0][0] + iso_offs[0][1]:]),
    "no-iso-gainmap.jpg": (uhdr_bytes[:iso_offs[1][0]]
                           + uhdr_bytes[iso_offs[1][0] + iso_offs[1][1]:]),
    "base-only.jpg": uhdr_bytes[:base_eoi],
    "truncated-gainmap.jpg": uhdr_bytes[:base_eoi + 600],
    "truncated-base.jpg": uhdr_bytes[:len(uhdr_bytes) // 2],
}
with open(plain, "rb") as f:
    plain_bytes = f.read()
# a plain JPEG wearing the UltraHDR file's MPF marker: passes the fast gate,
# fails is_uhdr_image
variants["mpf-graft.jpg"] = (plain_bytes[:2]
                             + uhdr_bytes[mpf_off:mpf_off + mpf_len]
                             + plain_bytes[2:])
for name, blob in variants.items():
    with open(os.path.join(FIX, name), "wb") as f:
        f.write(blob)
COMMANDS.append("# fixtures/no-*.jpg, base-only.jpg, truncated-*.jpg and "
                "mpf-graft.jpg are cut from fixtures/uhdr.jpg and "
                "fixtures/plain.jpg by this script; vips cannot edit JPEG "
                "marker segments")

detection = {}
for name in ["uhdr.jpg", "plain.jpg"] + list(variants):
    path = os.path.join(FIX, name)
    chosen = header(path)
    dst = os.path.join(OUT, "det-" + name + ".v")
    rc, _, err = vips("uhdrload", path, dst, allow_fail=True)
    entry = {
        "bytes": os.path.getsize(path),
        "sha256": sha256(path),
        "chosen_loader": (chosen.get("summary", "").split(", ")[-1]
                          if "summary" in chosen else chosen),
        "uhdrload_exit": rc,
        "uhdrload_stderr": err,
    }
    if rc == 0:
        entry["uhdrload_header"] = header(dst)["summary"].split(": ", 1)[1]
        entry["has_gainmap_data"] = "gainmap-data" in header(dst, True)
    detection[name] = entry

records["detection_two_stage_gate"] = {
    "what": "What makes a JPEG an UltraHDR file, measured by cutting one "
            "marker at a time out of fixtures/uhdr.jpg. The result that "
            "matters: the MPF APP2 marker is ONLY a fast pre-filter for "
            "`is_a`. Remove it and libuhdr still accepts the file, so "
            "`vips uhdrload` succeeds and returns the gainmap, while the "
            "loader chooser falls through to jpegload and silently drops it. "
            "The marker libuhdr genuinely needs is the ISO 21496-1 APP2 on "
            "the GAINMAP image; removing the base image's copy of it changes "
            "nothing, removing the gainmap's makes the file stop being "
            "UltraHDR entirely.",
    "gate": {
        "is_a": "vips__isjpeg_source, then jpeg_read_header with APP2 markers "
                "saved, then `p->data_length > 4 && vips_isprefix(\"MPF\", "
                "p->data)`, and only if that hits, is_uhdr_image() over the "
                "whole file mapped into memory (uhdrload.c:95-160)",
        "header": "is_uhdr_image() alone, with no MPF check "
                  "(uhdrload.c:499-503)",
        "consequence": "the two halves disagree on no-mpf.jpg",
    },
    "files": detection,
}
notes.append(
    "The MPF APP2 marker gates only the fast `is_a`. fixtures/no-mpf.jpg is "
    "accepted by `vips uhdrload` and rejected by the loader chooser, which "
    "hands it to jpegload instead."
)

# ---------------------------------------------------------------------------
# 3. Loader priority: uhdrload and jpegload claim the same four suffixes.
# ---------------------------------------------------------------------------
via_jpeg = os.path.join(OUT, "via-jpegload.v")
vips("jpegload", uhdr, via_jpeg)
via_stdin = os.path.join(OUT, "via-stdin.v")
with open(uhdr, "rb") as f:
    uhdr_stdin = f.read()
COMMANDS.append(f"cat fixtures/uhdr.jpg | {VIPS} copy stdin outputs/via-stdin.v")
stdin_proc = subprocess.run([VIPS, "copy", "stdin", via_stdin],
                            input=uhdr_stdin, capture_output=True)

records["priority_over_jpegload"] = {
    "what": "uhdrload is priority=100 and jpegload is priority=50, and they "
            "declare the same four suffixes, so for a file that passes "
            "uhdrload's is_a the chooser picks uhdrload. Naming jpegload "
            "explicitly still works and gives a visibly different image: no "
            "gainmap fields, the JPEG's own resolution instead of the 1x1 "
            "uhdrload hard-codes at uhdrload.c:541, plus the jpeg-* and "
            "xmp-data fields uhdrload never attaches.",
    "same_file_via_uhdrload": "see records.uhdrsave_writes_this_container"
                              ".header, which is this file's `vipsheader -a`",
    "same_file_via_jpegload": header(via_jpeg, all_fields=True),
    "source_variant": {
        "what": "`cat fixtures/uhdr.jpg | vips copy stdin out.v` goes through "
                "the same chooser over a VipsSource. uhdrload_source wins "
                "there too, so detection is not a filename thing at any "
                "level. This is the only one of the four buffer and target "
                "variants the CLI can reach.",
        "exit": stdin_proc.returncode,
        "header": header(via_stdin)["summary"].split(": ", 1)[1],
    },
}

# ---------------------------------------------------------------------------
# 4. What uhdrload produces: the carriers, and the shrink option.
# ---------------------------------------------------------------------------
gm_data_path = os.path.join(OUT, "gainmap-data.jpg")
_, _, _ = vips("copy", uhdr, os.path.join(OUT, "uhdr-loaded.v"))
# pull the gainmap blob back out of the .v trailer so its own header can be read
with open(os.path.join(OUT, "uhdr-loaded.v"), "rb") as f:
    trailer = f.read()
start = trailer.find(b'name="gainmap-data">') + len(b'name="gainmap-data">')
end = trailer.find(b"</field>", start)
with open(gm_data_path, "wb") as f:
    f.write(base64.b64decode(trailer[start:end]))
COMMANDS.append("# outputs/gainmap-data.jpg is the gainmap-data blob lifted "
                "out of the .v XML trailer, so its own header can be read")
gm_dec = os.path.join(OUT, "gainmap-data-decoded.v")
vips("jpegload", gm_data_path, gm_dec)

shrink = {}
for factor in (1, 2, 3, 4, 8, 9):
    dst = os.path.join(OUT, f"shrink{factor}.v")
    rc, _, err = vips("uhdrload", uhdr, dst, "--shrink", str(factor),
                      allow_fail=True)
    entry = {"exit": rc, "stderr": err}
    if rc == 0:
        # read the fields off the live load: the `gainmap` carrier is a
        # VipsImage and has no representation in a .v trailer, so it does not
        # survive the write to disk.
        fields = header(f"{uhdr}[shrink={factor}]", all_fields=True)
        entry["size"] = f"{fields['width']}x{fields['height']}"
        entry["gainmap_image"] = fields.get("gainmap")
        entry["gainmap_data_still_present"] = "gainmap-data" in fields
        entry["gainmap_scale_factor"] = fields.get("gainmap-scale-factor")
        entry["survives_write_to_v"] = "gainmap" in header(dst, all_fields=True)
    shrink[str(factor)] = entry

records["uhdrload_carriers"] = {
    "what": "uhdrload never returns HDR pixels. It decodes the BASE image "
            "with the ordinary libjpeg path (uhdrload.c:553-577) and hands "
            "back 3-band uchar sRGB, with the gainmap attached as metadata "
            "for uhdr2scRGB or for a re-save to pick up. The gainmap arrives "
            "as `gainmap-data`, a compressed JPEG blob, and is decompressed "
            "into a real image only when shrink != 1.",
    "output_carrier": "3 bands, uchar, VIPS_INTERPRETATION_sRGB, "
                      "VIPS_CODING_NONE, xres = yres = 1.0 "
                      "(vips_image_init_fields at uhdrload.c:541)",
    "get_flags": "VIPS_FOREIGN_PARTIAL, because the whole thing is decoded to "
                 "memory anyway (uhdrload.c:236-241)",
    "metadata_names": [
        "gainmap-data", "gainmap-max-content-boost",
        "gainmap-min-content-boost", "gainmap-gamma", "gainmap-offset-sdr",
        "gainmap-offset-hdr", "gainmap-hdr-capacity-min",
        "gainmap-hdr-capacity-max", "gainmap-use-base-cg",
        "gainmap-scale-factor", "exif-data", "icc-profile-data",
    ],
    "gainmap_data_is_a_jpeg": {
        "bytes": os.path.getsize(gm_data_path),
        "sha256": sha256(gm_data_path),
        "header": header(gm_dec)["summary"].split(": ", 1)[1],
        "com_segment": libuhdr_com,
        "segments": segments(gm_data_path),
    },
    "gainmap_scale_factor": "VIPS_MAX(1, image_width / gainmap_width) with "
                            "INTEGER division (uhdrload.c:473-478), so it is "
                            "a derived field, not something read out of the "
                            "file.",
    "icc_prefix_is_stripped": "uhdrload.c:399-411 drops `ICC_PROFILE` plus "
                              "three more bytes when the blob libuhdr returns "
                              "starts with that string, so icc-profile-data "
                              "is 14 bytes shorter than the APP2 payload.",
    "shrink": shrink,
    "shrink_notes": "shrink is declared 1..8 but goes straight to libjpeg's "
                    "scale denominator, which only accepts 1, 2, 4 and 8; 3 "
                    "fails outright. Out of range, GObject refuses the "
                    "property with a CRITICAL on stderr, the load SUCCEEDS, "
                    "and shrink stays 1. When shrink != 1 the loader also "
                    "decompresses the gainmap at the same shrink and attaches "
                    "it as the IMAGE `gainmap` (uhdrload.c:418-432), so both "
                    "carriers are present at once and vips_image_get_gainmap "
                    "prefers the image. That carrier is a VipsImage with no "
                    "representation in a .v trailer, so it is dropped by a "
                    "write to disk while `gainmap-data` survives: "
                    "`survives_write_to_v` below is false everywhere.",
}

# ---------------------------------------------------------------------------
# 5. The reference image, if it is on this machine.
# ---------------------------------------------------------------------------
if os.path.exists(REFERENCE_UHDR):
    ref_hdr = os.path.join(OUT, "reference-scrgb.v")
    vips("uhdr2scRGB", REFERENCE_UHDR, ref_hdr)
    _, ref_max, _ = vips("max", ref_hdr)
    _, ref_min, _ = vips("min", ref_hdr)
    _, ref_avg, _ = vips("avg", ref_hdr)
    records["reference_image"] = {
        "what": "libvips's own UltraHDR test image, 3840x2160. Recorded by "
                "path and digest rather than checked in. This is the file "
                "libvips's test_foreign.py asserts against, so it is the one "
                "place where this capture and upstream's suite can be "
                "compared directly.",
        "path": REFERENCE_UHDR,
        "sha256": sha256(REFERENCE_UHDR),
        "bytes": os.path.getsize(REFERENCE_UHDR),
        "header": header(REFERENCE_UHDR, all_fields=True),
        "uses_adobe_xmp_not_iso": "This file carries the Adobe "
                                  "`hdrgm` XMP packet in APP1 rather than the "
                                  "ISO 21496-1 APP2 that libuhdr 2.0.2 writes, "
                                  "and libuhdr accepts both. Detection is not "
                                  "tied to either spelling.",
        "uhdr2scRGB": {"max": float(ref_max), "min": float(ref_min),
                       "avg": float(ref_avg)},
    }
else:
    records["reference_image"] = {"skipped": REFERENCE_UHDR + " not present"}

# ---------------------------------------------------------------------------
# 6. uhdr2scRGB: the carrier, and what it refuses.
# ---------------------------------------------------------------------------
BASE = bytes(v for x in range(16)
             for v in (x * 17, 255 - x * 17, (x * 37) % 256))
base16 = raster("base16", BASE, 16, 1, 3)
gm16, gm16_dec = gainmap_jpeg("gainmap-mono16",
                              bytes(x * 17 for x in range(16)), 16, 1, 1)
gm16_values = [int(p[0]) for p in getpoint(gm16_dec, 16)]

FLAT = {"max-content-boost": [8, 8, 8], "min-content-boost": [1, 1, 1],
        "gamma": [1, 1, 1], "offset-sdr": [0, 0, 0], "offset-hdr": [0, 0, 0]}

rejects = {}
one_band = raster("reject-mono", bytes(range(16)), 16, 1, 1, interp="b-w")
four_band = raster("reject-rgba",
                   bytes(v for x in range(16)
                         for v in (x * 17, 255 - x * 17, (x * 37) % 256, 255)),
                   16, 1, 4)
wide = raster("reject-ushort",
              b"".join(struct.pack("<H", (x * 4369) % 65536) for x in range(48)),
              16, 1, 3, fmt="ushort", interp="rgb16")
for label, src in (("one_band", one_band), ("four_band", four_band),
                   ("ushort", wide)):
    got = transform("reject-" + label, src, gm16, FLAT, 16)
    rejects[label] = {"exit": got["exit"], "stderr": got.get("stderr", "")}
no_gainmap = transform("reject-no-gainmap", base16, None, FLAT, 16)
rejects["no_gainmap"] = {"exit": no_gainmap["exit"],
                         "stderr": no_gainmap.get("stderr", "")}
no_meta = transform("reject-no-metadata", base16, gm16, {}, 16)
rejects["no_metadata"] = {"exit": no_meta["exit"],
                          "stderr": no_meta.get("stderr", "")}

records["uhdr2scRGB_carrier"] = {
    "what": "uhdr2scRGB takes exactly one image and reads everything else out "
            "of its metadata. The input must be 3-band uchar; the gainmap it "
            "finds must be 1 or 3 bands. Output is always 3-band float tagged "
            "scRGB (uhdr2scRGB.c:277-285), whatever the input was.",
    "input": "3 bands, uchar. vips_check_bands(nickname, in, 3) and an "
             "explicit BandFmt test at uhdr2scRGB.c:204-209.",
    "output": "3 bands, VIPS_FORMAT_FLOAT, VIPS_INTERPRETATION_scRGB",
    "gainmap_source": "vips_image_get_gainmap (header.c:1077): the attached "
                      "image `gainmap` if there is one, otherwise "
                      "jpegload_buffer over the `gainmap-data` blob. Nothing "
                      "else is looked at.",
    "alpha_is_not_detached": "colour->input_bands = 0 at uhdr2scRGB.c:196 "
                             "turns off VipsColour's automatic alpha "
                             "detach/reattach, because a 1-band gainmap "
                             "against a 3-band image would otherwise be read "
                             "as an alpha band. A 4-band input is refused "
                             "rather than unpacked.",
    "rejections": rejects,
    "no_gainmap_fails_silently": "With the metadata present and no gainmap at "
                                 "all, vips_image_get_gainmap returns NULL "
                                 "without calling vips_error, so the "
                                 "operation exits 1 having printed NOTHING. "
                                 "A port should raise a real error there.",
}

# ---------------------------------------------------------------------------
# 7. uhdr2scRGB: the base linearisation is exactly sRGB2scRGB.
# ---------------------------------------------------------------------------
ramp = raster("ramp256", bytes(v for x in range(256) for v in (x, x, x)),
              256, 1, 3)
lin = os.path.join(OUT, "ramp256-linear.v")
vips("sRGB2scRGB", ramp, lin)
lin_raw = os.path.join(OUT, "ramp256-linear.raw")
vips("rawsave", lin, lin_raw)
with open(lin_raw, "rb") as f:
    lin_bytes = f.read()
lut = struct.unpack("<%df" % (len(lin_bytes) // 4), lin_bytes)[0::3]

gm_ramp, _ = gainmap_jpeg("gainmap-ramp256", bytes(range(256)), 256, 1, 1)
unity = dict(FLAT, **{"max-content-boost": [1, 1, 1]})
ident_src = attach(ramp, "p-identity.v", gm_ramp, unity)
ident = os.path.join(OUT, "h-identity.v")
vips("uhdr2scRGB", ident_src, ident)
ident_raw = os.path.join(OUT, "h-identity.raw")
vips("rawsave", ident, ident_raw)
with open(ident_raw, "rb") as f:
    ident_bytes = f.read()

records["uhdr2scRGB_base_linearisation"] = {
    "what": "The base image is linearised through vips_v2Y_8, the same "
            "256-entry LUT sRGB2scRGB uses (uhdr2scRGB.c:83-85 and "
            "sRGB2scRGB.c:81-83). With min_content_boost == "
            "max_content_boost == 1 and both offsets zero, the whole "
            "transform collapses to that lookup, and the result is identical "
            "to sRGB2scRGB byte for byte over all 256 codes.",
    "lut": "vips_v2Y_8, built by calcul_tables in LabQ2sRGB.c:130-159: "
           "f = i / 255; v2Y[i] = f <= 0.04045 ? f / 12.92f : "
           "powf((f + 0.055f) / 1.055f, 2.4f). LANE_RUNBOOK section 14 "
           "applies: the shipped dylib fuses multiply-adds this source does "
           "not show, so transcribe against these values, not against the "
           "expression.",
    "identical_to_sRGB2scRGB": ident_bytes == lin_bytes,
    "compared_bytes": len(lin_bytes),
    "v2Y_8_sha256": hashlib.sha256(
        b"".join(struct.pack("<f", v) for v in lut)).hexdigest(),
    "v2Y_8_le_f32_hex": " ".join(struct.pack("<f", v).hex() for v in lut),
    "v2Y_8_spot_values": {str(i): repr(lut[i])
                          for i in (0, 1, 10, 11, 12, 63, 128, 254, 255)},
    "knee_index": "12 is the last index on the linear segment: 12/255 = "
                  "0.047058... is above 0.04045, so the knee falls between "
                  "10 and 11 and index 11 already uses the power branch.",
}

# ---------------------------------------------------------------------------
# 8. uhdr2scRGB: the mono gainmap path, and the metadata index it reads.
# ---------------------------------------------------------------------------
mono = {}
mono["canonical"] = transform("mono-canonical", base16, gm16, FLAT, 16)
mono["gamma_2_2"] = transform(
    "mono-gamma", base16, gm16, dict(FLAT, gamma=[1, 2.2, 1]), 16)
mono["min_boost_half"] = transform(
    "mono-minhalf", base16, gm16,
    dict(FLAT, **{"min-content-boost": [1, 0.5, 1]}), 16)
mono["offsets"] = transform(
    "mono-offsets", base16, gm16,
    dict(FLAT, **{"offset-sdr": [0, 0.015625, 0],
                  "offset-hdr": [0, 0.03125, 0]}), 16)
# every array index except [1] set to something absurd: the mono path must not
# notice
mono["red_and_blue_are_ignored"] = transform(
    "mono-green-only", base16, gm16,
    {"max-content-boost": [999, 8, 999], "min-content-boost": [999, 1, 999],
     "gamma": [999, 1, 999], "offset-sdr": [999, 0, 999],
     "offset-hdr": [999, 0, 999]}, 16)

records["uhdr2scRGB_mono_gainmap"] = {
    "what": "The one-band gainmap path (uhdr2scRGB.c:74-105), which is the "
            "common case because that is what libuhdr writes. Per output "
            "pixel, with p1 the base and p2 the gainmap:\n"
            "    r,g,b  = vips_v2Y_8[p1[0..2]]\n"
            "    gg     = p2[0] / 255.0\n"
            "    if gamma[1] != 1: gg = pow(gg, 1.0f / gamma[1])\n"
            "    boost  = log2(min_content_boost[1]) * (1.0f - gg)\n"
            "           + log2(max_content_boost[1]) * gg\n"
            "    gain   = exp2(boost)\n"
            "    q[i]   = ((rgb[i] + offset_sdr[1]) * gain) - offset_hdr[1]\n"
            "log2, exp2 and pow are the DOUBLE forms; gg, boost and gain are "
            "float locals, so each one rounds to float in between.",
    "the_index_trap": "Every metadata term the mono path uses is read at "
                      "index [1], the GREEN entry, and applied to all three "
                      "output channels. Indices [0] and [2] are never read. "
                      "The `red_and_blue_are_ignored` record sets them to 999 "
                      "and gets a result identical to `canonical`.",
    "base_sRGB_bytes": [BASE[i:i + 3].hex() for i in range(0, len(BASE), 3)],
    "gainmap_fixture": "fixtures/gainmap-mono16.jpg",
    "gainmap_decoded_values": gm16_values,
    "gainmap_is_NOT_linearised": "gg is p2[0] / 255.0, a plain scale. The "
                                 "comment at uhdr2scRGB.c:88 is `the gainmap "
                                 "is not gamma corrected in libultrahdr, "
                                 "confusingly`. The three-band path in the "
                                 "next record does linearise it, and the two "
                                 "therefore disagree on identical bytes.",
    "metadata": {"canonical": FLAT,
                 "gamma_2_2": {"gamma": [1, 2.2, 1]},
                 "min_boost_half": {"min-content-boost": [1, 0.5, 1]},
                 "offsets": {"offset-sdr": [0, 0.015625, 0],
                             "offset-hdr": [0, 0.03125, 0]},
                 "red_and_blue_are_ignored": "[0] and [2] set to 999"},
    "results": mono,
    "identical_to_canonical": (
        mono["red_and_blue_are_ignored"]["scRGB"] == mono["canonical"]["scRGB"]),
}
assert mono["red_and_blue_are_ignored"]["scRGB"] == mono["canonical"]["scRGB"]

# ---------------------------------------------------------------------------
# 9. uhdr2scRGB: the three-band gainmap path, which is NOT the mono path
#    applied per channel.
# ---------------------------------------------------------------------------
gm_rgb, gm_rgb_dec = gainmap_jpeg(
    "gainmap-rgb16",
    bytes(x * 17 for x in range(16) for _ in range(3)), 16, 1, 3)
rgb_values = [[int(v) for v in p] for p in getpoint(gm_rgb_dec, 16)]
rgb_result = transform("rgb-canonical", base16, gm_rgb, FLAT, 16)

records["uhdr2scRGB_rgb_gainmap"] = {
    "what": "The three-band gainmap path (uhdr2scRGB.c:109-150), chosen "
            "purely on `uhdr->gainmap->Bands == 1` at uhdr2scRGB.c:160. It "
            "differs from the mono path in one line and that line changes "
            "every value: the gainmap samples go through vips_v2Y_8, the "
            "sRGB-to-linear LUT, instead of a plain divide by 255. It also "
            "uses metadata index [0], [1] and [2] for the three channels "
            "rather than [1] for all of them.",
    "the_divergence": "This fixture's gainmap holds the same byte values as "
                      "fixtures/gainmap-mono16.jpg, replicated into three "
                      "bands, and is run with identical metadata. The results "
                      "below do not match the mono ones anywhere except where "
                      "the LUT is exact (0 and 255). A port that implements "
                      "one path and reuses it per channel for the other is "
                      "wrong, and no header check will show it.",
    "gainmap_fixture": "fixtures/gainmap-rgb16.jpg",
    "gainmap_decoded_values": rgb_values,
    "metadata": FLAT,
    "results": rgb_result,
    "differs_from_mono_at": [i for i, (a, b) in enumerate(
        zip(rgb_result["scRGB"], mono["canonical"]["scRGB"])) if a != b],
}

# ---------------------------------------------------------------------------
# 10. uhdr2scRGB: the gainmap is resized to 1:1 before anything else.
# ---------------------------------------------------------------------------
gm8, gm8_dec = gainmap_jpeg(
    "gainmap-mono8", bytes(min(x * 36, 255) for x in range(8)), 8, 1, 1)
gm8_values = [int(p[0]) for p in getpoint(gm8_dec, 8)]
resized = os.path.join(OUT, "gainmap-mono8-resized.v")
vips("resize", gm8_dec, resized, "2.0", "--vscale", "1.0",
     "--kernel", "linear")
half = transform("mono-half-size", base16, gm8, FLAT, 16)

records["uhdr2scRGB_gainmap_resize"] = {
    "what": "A gainmap is almost always smaller than its base image, and "
            "uhdr2scRGB scales it to 1:1 with vips_resize BEFORE the "
            "per-pixel transform (uhdr2scRGB.c:233-240): separate hscale and "
            "vscale, both a plain ratio of the two sizes, and "
            "VIPS_KERNEL_LINEAR. Anything else, nearest included, gives "
            "different pixels everywhere the gainmap is not flat.",
    "call": "vips_resize(gainmap, &out, (double) in->Xsize / gainmap->Xsize, "
            "\"vscale\", (double) in->Ysize / gainmap->Ysize, \"kernel\", "
            "VIPS_KERNEL_LINEAR, NULL)",
    "gainmap_fixture": "fixtures/gainmap-mono8.jpg",
    "gainmap_decoded_values": gm8_values,
    "resized_to_16_wide": [int(p[0]) for p in getpoint(resized, 16)],
    "metadata": FLAT,
    "results": half,
    "scale_1_is_the_identity": "The other records all use a gainmap the same "
                               "size as the base, where the scale is exactly "
                               "1.0 and the resize is a no-op. Their pinned "
                               "values therefore isolate the transform from "
                               "the resample.",
}

# ---------------------------------------------------------------------------
# 11. uhdr2scRGB: metadata that produces inf and NaN.
# ---------------------------------------------------------------------------
degenerate = {
    "min_boost_zero": transform(
        "deg-minzero", base16, gm16,
        dict(FLAT, **{"min-content-boost": [1, 0, 1]}), 16),
    "max_boost_zero": transform(
        "deg-maxzero", base16, gm16,
        dict(FLAT, **{"max-content-boost": [8, 0, 8]}), 16),
    "min_above_max": transform(
        "deg-inverted", base16, gm16,
        dict(FLAT, **{"min-content-boost": [1, 4, 1],
                      "max-content-boost": [8, 2, 8]}), 16),
}
records["uhdr2scRGB_degenerate_metadata"] = {
    "what": "Nothing validates the gainmap metadata, so log2(0) reaches the "
            "boost expression as -inf and the C spelling decides what "
            "happens. `log2(min) * (1.0f - gg)` with min == 0 and gg == 1 is "
            "-inf * 0, which is NaN; at gg == 0 it is -inf, so exp2 gives 0 "
            "and the pixel goes black. An inverted pair (min > max) is "
            "accepted and simply runs the interpolation backwards. All three "
            "exit 0.",
    "results": degenerate,
    "why_it_matters": "A port that reorders the boost expression, or that "
                      "special-cases min == max before taking the logs, will "
                      "not reproduce the NaN, and a differential test over "
                      "random metadata will disagree without saying why.",
}

# ---------------------------------------------------------------------------
# 12. uhdrsave: what it accepts, and how jpegsave hands over to it.
# ---------------------------------------------------------------------------
save = {}
rc, _, err = vips("uhdrsave", sdr_v, os.path.join(OUT, "no-metadata.jpg"),
                  allow_fail=True)
save["sdr_without_gainmap_metadata"] = {
    "exit": rc, "stderr": err,
    "note": "saveable is VIPS_FOREIGN_SAVEABLE_ANY (uhdrsave.c:539), so "
            "uhdrsave accepts any image the type system can offer, and then "
            "fails inside vips_foreign_save_uhdr_set_compressed_gainmap "
            "because the metadata is not there. The failure is a "
            "vips_image_get miss, not a saveable check.",
}

loaded = os.path.join(OUT, "uhdr-reloaded.v")
vips("copy", uhdr, loaded)
resaved = os.path.join(OUT, "resaved.jpg")
vips("uhdrsave", loaded, resaved)
save["sdr_with_gainmap_metadata"] = {
    "exit": 0,
    "bytes": os.path.getsize(resaved),
    "sha256": sha256(resaved),
    "header": header(resaved, all_fields=True),
    "note": "the SDR branch (uhdrsave.c:377-393) re-compresses the base with "
            "jpegsave_target at Q and passes the ORIGINAL compressed gainmap "
            "through untouched, so a load/save cycle re-encodes the base and "
            "not the gainmap.",
}

quality = {}
for q in (1, 50, 75, 100):
    dst = os.path.join(OUT, f"q{q}.jpg")
    vips("uhdrsave", scrgb_v, dst, "--Q", str(q))
    quality[str(q)] = {"bytes": os.path.getsize(dst), "sha256": sha256(dst)}
save["Q"] = {"default": 75, "range": [1, 100], "results": quality,
             "note": "Q is set on BOTH the base and the gainmap "
                     "(uhdrsave.c:459-471)."}

scale = {}
for factor in (1, 2, 4, 8, 128, 129):
    dst = os.path.join(OUT, f"scale{factor}.jpg")
    rc, _, err = vips("uhdrsave", scrgb_v, dst, "--gainmap-scale-factor",
                      str(factor), allow_fail=True)
    entry = {"exit": rc, "stderr": err}
    if rc == 0:
        entry["bytes"] = os.path.getsize(dst)
        entry["gainmap_scale_factor_read_back"] = header(
            dst, all_fields=True).get("gainmap-scale-factor")
        entry["gainmap_size"] = header(
            dst + "[shrink=1]", all_fields=True).get("gainmap")
    scale[str(factor)] = entry
save["gainmap_scale_factor"] = {
    "default": 2, "range": [1, 128], "results": scale,
    "note": "only consulted on the scRGB branch (uhdrsave.c:361), because the "
            "SDR branch reuses the gainmap it was handed.",
}

routed = os.path.join(OUT, "routed-by-jpegsave.jpg")
vips("jpegsave", scrgb_v, routed)
routed_sdr = os.path.join(OUT, "routed-sdr.jpg")
vips("jpegsave", loaded, routed_sdr)
save["jpegsave_routes_here"] = {
    "what": "jpegsave_build (jpegsave.c:151-158) checks for a `gainmap-data` "
            "field OR an scRGB interpretation and, on either, calls "
            "vips_uhdrsave_target and returns. That is why uhdrsave declares "
            "no suffixes: saving an scRGB image to `.jpg` produces UltraHDR "
            "without anybody asking for it.",
    "scRGB_via_jpegsave_sha256": sha256(routed),
    "scRGB_via_uhdrsave_sha256": sha256(uhdr),
    "byte_identical": sha256(routed) == sha256(uhdr),
    "sdr_with_gainmap_via_jpegsave": header(routed_sdr)["summary"].split(
        ": ", 1)[1],
    "Q_is_forwarded_but_nothing_else": "only Q crosses the handover; every "
                                       "other jpegsave option (interlace, "
                                       "subsample_mode, quant_table, the "
                                       "profile) is dropped.",
}
records["uhdrsave"] = save
assert sha256(routed) == sha256(uhdr)

# ---------------------------------------------------------------------------
# 13. Round trips, in both directions.
# ---------------------------------------------------------------------------
def stats(a, b, label):
    diff = os.path.join(OUT, label + "-diff.v")
    absd = os.path.join(OUT, label + "-absdiff.v")
    vips("subtract", a, b, diff)
    vips("abs", diff, absd)
    _, avg, _ = vips("avg", absd)
    _, mx, _ = vips("max", absd)
    return {"mean_abs_error": float(avg), "max_abs_error": float(mx)}


hdr1 = os.path.join(OUT, "rt-hdr1.v")
vips("uhdr2scRGB", uhdr, hdr1)
rt_jpg = os.path.join(OUT, "rt.jpg")
vips("uhdrsave", hdr1, rt_jpg)
hdr2 = os.path.join(OUT, "rt-hdr2.v")
vips("uhdr2scRGB", rt_jpg, hdr2)

sdr1 = os.path.join(OUT, "rt-sdr1.v")
vips("copy", uhdr, sdr1)
sdr2 = os.path.join(OUT, "rt-sdr2.v")
vips("copy", resaved, sdr2)

records["round_trip"] = {
    "what": "Neither direction is lossless, and the numbers are worth having "
            "because they set the tolerance any differential test can use.",
    "hdr_path": {
        "steps": "uhdr2scRGB -> uhdrsave -> uhdr2scRGB",
        "source_max": float(vips("max", hdr1)[1]),
        "round_tripped_max": float(vips("max", hdr2)[1]),
        **stats(hdr1, hdr2, "hdr"),
        "note": "libuhdr re-tonemaps from scratch on every save, so the "
                "gainmap that comes back is not the one that went in and the "
                "highlight ceiling moves. libvips's own test_uhdrsave_"
                "roundtrip_hdr allows a mean absolute error of 0.05.",
    },
    "sdr_path": {
        "steps": "uhdrload -> uhdrsave -> uhdrload, base pixels only",
        **stats(sdr1, sdr2, "sdr"),
        "note": "the base is re-encoded as a Q=75 JPEG, so this is ordinary "
                "JPEG generation loss on uchar samples. The gainmap blob "
                "itself is passed through and does not degrade.",
    },
}

# ---------------------------------------------------------------------------
# 14. Malformed input.
# ---------------------------------------------------------------------------
malformed = {}
for name in ("truncated-base.jpg", "truncated-gainmap.jpg", "base-only.jpg"):
    path = os.path.join(FIX, name)
    rc_l, _, err_l = vips("uhdrload", path, os.path.join(OUT, "m-" + name + ".v"),
                          allow_fail=True)
    rc_t, _, err_t = vips("uhdr2scRGB", path,
                          os.path.join(OUT, "t-" + name + ".v"), allow_fail=True)
    malformed[name] = {
        "bytes": os.path.getsize(path),
        "uhdrload": {"exit": rc_l, "stderr": err_l},
        "uhdr2scRGB": {"exit": rc_t, "stderr": err_t},
        "vipsheader": header(path),
    }

# a gainmap-data blob that is a truncated JPEG: the metadata is fine, the
# gainmap is not
with open(gm16, "rb") as f:
    short = f.read()[:120]
short_path = os.path.join(FIX, "gainmap-truncated.jpg")
with open(short_path, "wb") as f:
    f.write(short)
COMMANDS.append("# fixtures/gainmap-truncated.jpg is the first 120 bytes of "
                "fixtures/gainmap-mono16.jpg")
bad = transform("malformed-gainmap", base16, short_path, FLAT, 16)
malformed["gainmap-truncated.jpg"] = {
    "bytes": len(short),
    "uhdr2scRGB": {"exit": bad["exit"], "stderr": bad.get("stderr", "")},
    "note": "the failure surfaces from libjpeg through jpegload_buffer, not "
            "from uhdr2scRGB, so the message names VipsJpeg.",
}
records["malformed"] = {
    "what": "Truncated and dismembered files, and what each entry point says "
            "about them. Nothing here crashes and nothing silently produces "
            "pixels; the interesting part is which layer reports the error.",
    "files": malformed,
}

# ---------------------------------------------------------------------------
# meta and write-out.
# ---------------------------------------------------------------------------
_, version, _ = vips("--version")
_, otool, _ = run(["/usr/bin/otool", "-L", "/opt/homebrew/lib/libvips.42.dylib"],
                  allow_fail=True)
libuhdr_link = [l.strip() for l in otool.splitlines() if "uhdr" in l.lower()]

notes.append(
    "vips 8.18.4 is what every other directory under oracle-captures/ was "
    "measured against and it is NOT what this one used. A brew upgrade "
    "running on the host mid-capture replaced vips 8.18.4 with 8.18.6 and "
    "libultrahdr 1.5.1 with 2.0.2, and deleted both older kegs, so 8.18.4 "
    "cannot be re-run here. The only difference I could measure across the "
    "two libuhdr majors is the version string in the gainmap's JPEG COM "
    "segment; the uhdr2scRGB values and the container layout were identical "
    "before and after."
)
notes.append(
    "This build's libheif module fails to dlopen (it wants a libx265 that is "
    "not installed), so every vips invocation prints a multi-line "
    "VIPS-WARNING. It is unrelated to UltraHDR and is filtered out of every "
    "captured stderr here."
)
notes.append(
    "`vips <op> --help` prints generic driver usage and exits 0 for these "
    "operations, so it is not an existence test. Everything in "
    "operator_surface came from `vips -l`."
)
notes.append(
    "uhdrsave_buffer, uhdrsave_target, uhdrload_buffer and uhdrload_source "
    "are registered but are not reachable from the vips CLI: the target and "
    "buffer arguments cannot be spelled as a filename. uhdrload_source IS "
    "reachable, as `vips copy stdin out.v`, and that is captured in "
    "operator_surface via the priority record."
)

oracle = {
    "meta": {
        "area": "foreign-uhdr",
        "issue": 639,
        "vips_version": version,
        "vips_binary": VIPS,
        "libuhdr": {
            "linked": libuhdr_link,
            "com_segment": libuhdr_com,
            "note": "the codec is a third-party library and the container it "
                    "writes is its output, not libvips's, so its version is "
                    "part of this capture's provenance in a way no other "
                    "area's is",
        },
        "captured_by": "oracle-captures/foreign-uhdr/capture.py",
        "reference_c": "libvips v8.18.0-95-gfe420cf3a for the file and line "
                       "numbers quoted here (libvips/foreign/uhdrload.c, "
                       "libvips/foreign/uhdrsave.c, "
                       "libvips/colour/uhdr2scRGB.c, "
                       "libvips/iofuncs/header.c, "
                       "libvips/foreign/jpegsave.c); the binary every number "
                       "came out of is the 8.18.6 release and is not the same "
                       "artefact",
    },
    "notes": notes,
    "records": records,
}

SCALAR = r'(?:-?[\d.]+(?:[eE][-+]?\d+)?|true|false|null|"NaN"|"-?Infinity")'
INLINE = re.compile(r"\[\s*(" + SCALAR + r"(?:,\s*" + SCALAR + r")*)\s*\]")


def json_safe(value):
    """Quote the floats JSON has no literal for (issue #674). json.dump
    writes a bare NaN, Infinity or -Infinity by default; Python reads those
    back and no other language does, so the degenerate-metadata NaNs would
    make the whole file unreadable to serde_json, jq and JSON.parse. Quoting
    keeps the three apart the way a null would not. The spelling is pinned for
    every reader in tests/oracle_capture_json.rs; foreign-nifti is NOT the
    precedent, it carries both this spelling and a lowercase str(v) one.
    SCALAR above matches the quoted spelling so those rows still reflow onto
    one line."""
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return "NaN"
        return "Infinity" if value > 0 else "-Infinity"
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def inline_numeric_arrays(text):
    """Put leaf arrays of numbers back on one line. json.dump(indent=2) gives
    every sample its own line, which turns a 16-pixel sweep into 48 of them
    and the file into something nobody reads."""
    previous = None
    while previous != text:
        previous = text
        text = INLINE.sub(
            lambda m: "[" + ", ".join(re.split(r",\s*", m.group(1))) + "]",
            text)
    return text


with open(os.path.join(ROOT, "oracle.json"), "w") as f:
    # allow_nan=False so a value json_safe missed stops the capture here
    # rather than writing a file nobody outside Python can parse.
    f.write(inline_numeric_arrays(
        json.dumps(json_safe(oracle), indent=2, allow_nan=False)))
    f.write("\n")

with open(os.path.join(ROOT, "commands.sh"), "w") as f:
    f.write("#!/bin/sh\n")
    f.write("# Every command capture.py ran, in order. Regenerate with\n")
    f.write("# `python3 capture.py` from this directory.\n")
    f.write("set -e\n\n")
    for c in COMMANDS:
        f.write(c + "\n")

print(f"{len(records)} records, {len(COMMANDS)} commands, "
      f"{len(notes)} notes")
