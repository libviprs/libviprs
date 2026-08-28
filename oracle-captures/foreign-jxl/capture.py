#!/usr/bin/env python3
"""
Oracle capture for the still-image JPEG XL area (issues #619, #620, #622).

Runs the real vips 8.18.4 CLI over small, deterministic rasters and records
what `jxlsave` writes and what `jxlload` reads back. `vips --vips-config`
says `JXL load/save with libjxl: true` and `vips -l | grep -i jxl` lists all
seven operations, so vips is a usable oracle here and nothing below is
hand-derived.

What has to be measured rather than reasoned about:

  * that the lossless modular path is a true identity, and that it is one for
    every carrier the format has: 8-bit, 16-bit and float
  * that the VarDCT path is close but NOT exact, and by exactly how much, so
    the pin in `src/jxl.rs` carries the right tolerance and no more
  * that a one-band image stays one band, which is where JPEG XL and WebP
    part company: `webpsave` promotes `b-w` to three bands because the
    format has no greyscale, `jxlsave` does not because it does
  * that `bits-per-sample` reads 8, 16 and 32 for the three carriers, and
    the interpretation that comes with each
  * that a default load of a three-frame file reads frame 0 at one frame's
    height and reports `n-pages`, leaving the toilet roll to `n=-1`
  * exactly what vips makes of an `Exif` box, which is not the box payload:
    JPEG XL stores the TIFF block behind a big-endian 4-byte offset and
    without the `Exif\\0\\0` prefix, and `jxlload.c:630-664` puts the prefix
    back and skips the offset
  * that vips FAILS the load outright when that offset is out of range,
    where libviprs drops the blob and keeps the pixels
  * the smallest image `jxlsave` accepts, which is smaller than the smallest
    `zune-jpegxl` accepts
  * that vips reads back what libviprs writes, with the same pixels, the
    same band count and the same bit depth

The libvips C line numbers quoted in `src/jxl.rs` come from the v8.18.4
source tree recorded in `meta.reference_c`. Every *number* here came out of
the Homebrew 8.18.4 binary, which is a different artefact from that tree.

Writes:
  commands.sh  - every vips CLI command actually executed, in order
  oracle.json  - structured records
  fixtures/    - the `.jxl` files `src/jxl.rs` embeds verbatim

Re-running needs the vips binary at VIPS and nothing else; every input is
generated from scratch, deterministically. Nothing outside this directory is
written.

The `viprs_*` records are the other direction and are captured by
`python3 capture.py --viprs`, which expects the files libviprs wrote to be
in `outputs/` already (the `jxl::tests::write_oracle_inputs` test puts them
there when `JXL_ORACLE_OUT` names this directory's `outputs/`).
"""

import hashlib
import json
import os
import struct
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
FIX = os.path.join(ROOT, "fixtures")
OUT = os.path.join(ROOT, "outputs")

VIPS = "/opt/homebrew/bin/vips"
VIPSHEADER = "/opt/homebrew/bin/vipsheader"

os.makedirs(FIX, exist_ok=True)
os.makedirs(OUT, exist_ok=True)

COMMANDS = []


def run(args, allow_fail=False):
    """Run a command, logging it (with this directory's absolute path
    reduced to a relative one) for commands.sh."""
    COMMANDS.append(" ".join(a.replace(ROOT + "/", "") for a in args))
    proc = subprocess.run(args, capture_output=True, text=True)
    if proc.returncode != 0 and not allow_fail:
        raise SystemExit(f"failed: {' '.join(args)}\n{proc.stderr}")
    return proc


def vips(*args, allow_fail=False):
    return run([VIPS, *args], allow_fail=allow_fail)


def header(path, all_fields=False, allow_fail=False):
    args = [VIPSHEADER] + (["-a"] if all_fields else []) + [path]
    return run(args, allow_fail=allow_fail)


def field(path, name):
    """One header field, as vips prints it (base64 for a blob)."""
    return run([VIPSHEADER, "-f", name, path]).stdout.strip()


def getpoint(path, w, h, ints=True):
    """Every pixel of a small image, in raster order."""
    out = []
    for y in range(h):
        for x in range(w):
            txt = vips("getpoint", path, str(x), str(y)).stdout.split()
            out.append([int(float(v)) if ints else float(v) for v in txt])
    return out


def ramp8(w, h, bands):
    """The deterministic 8-bit ramp. Band 0 steps by 61 across and 13 down,
    so no two pixels in a 4x3 tile repeat. Same generator the WebP capture
    uses, so the two areas are comparable pixel for pixel."""
    data = bytearray()
    steps = [(61, 13), (97, 151), (29, 211), (85, 40)]
    for y in range(h):
        for x in range(w):
            for b in range(bands):
                dx, dy = steps[b]
                data.append((x * dx + y * dy) % 256)
    return bytes(data)


def ramp16(w, h, bands):
    """The 16-bit ramp, native-endian `u16` samples. The steps are primes so
    the top byte moves as well as the bottom one, which is what makes a
    narrowing bug visible."""
    data = bytearray()
    steps = [1013, 4099, 7919, 5011]
    for y in range(h):
        for x in range(w):
            for b in range(bands):
                m = steps[b]
                data += struct.pack("<H", (x * m + y * (m * 3)) % 65536)
    return bytes(data)


def rampf32(w, h, bands):
    """The float ramp. Every value is a dyadic rational, so it is exact in
    `f32` and a round trip that changes anything at all is visible."""
    data = bytearray()
    for y in range(h):
        for x in range(w):
            for b in range(bands):
                data += struct.pack("<f", x * 0.25 + y * 0.0625 + b * 0.03125)
    return bytes(data)


def rawload(name, data, w, h, bands, fmt="uchar", interp="srgb"):
    """Write a raw buffer and load it into a `.v` with an interpretation."""
    raw = os.path.join(FIX, f"{name}.raw")
    with open(raw, "wb") as f:
        f.write(data)
    v = os.path.join(OUT, f"{name}.v")
    vips("rawload", raw, v, str(w), str(h), str(bands), "--format", fmt)
    tagged = os.path.join(OUT, f"{name}-{interp}.v")
    vips("copy", v, tagged, "--interpretation", interp)
    return tagged


def sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def size(path):
    return os.path.getsize(path)


def boxes(path):
    """The ISOBMFF box directory of a JPEG XL file, or None when the file is
    a bare codestream. A bare codestream starts `FF 0A`; the container form
    starts with the 12-byte signature box."""
    with open(path, "rb") as f:
        d = f.read()
    if d[:2] == b"\xff\x0a":
        return None
    assert d[:12] == b"\x00\x00\x00\x0cJXL \x0d\x0a\x87\x0a", path
    out, p = [], 0
    while p + 8 <= len(d):
        n = struct.unpack(">I", d[p:p + 4])[0]
        tag = d[p + 4:p + 8].decode("latin-1")
        if n == 0:  # runs to the end of the file
            out.append([tag, len(d) - p - 8])
            break
        out.append([tag, n - 8])
        p += n
    return out


def container(codestream, *extra):
    """A JPEG XL ISOBMFF container around `codestream`, with `extra` boxes
    between `ftyp` and `jxlc`. Hand-built on purpose: it is the only way to
    put a box with a chosen payload in front of vips and ask what it makes
    of it, which is what the EXIF records below need."""
    def box(tag, payload):
        return struct.pack(">I", 8 + len(payload)) + tag + payload

    sig = b"\x00\x00\x00\x0cJXL \x0d\x0a\x87\x0a"
    ftyp = box(b"ftyp", b"jxl \x00\x00\x00\x00jxl ")
    return sig + ftyp + b"".join(box(t, p) for t, p in extra) + box(b"jxlc", codestream)


# The minimal little-endian TIFF block every EXIF record below is built from:
# a header pointing at an IFD with zero entries and no next IFD.
TIFF = b"II*\x00\x08\x00\x00\x00\x00\x00"
XMP = b'<x:xmpmeta xmlns:x="adobe:ns:meta/"/>'

RECORDS = {}
NOTES = []


# ---------------------------------------------------------------------------
# The lossless carriers. Three sample types, one claim: the round trip is the
# identity for all of them.
# ---------------------------------------------------------------------------

def lossless(name, data, w, h, bands, fmt, interp, ints=True):
    src = rawload(name, data, w, h, bands, fmt=fmt, interp=interp)
    fixture = os.path.join(FIX, f"{name}.jxl")
    vips("jxlsave", src, fixture, "--lossless", "--keep", "none")
    px = getpoint(fixture, w, h, ints=ints)
    src_px = getpoint(src, w, h, ints=ints)
    RECORDS[name] = {
        "what": f"`vips jxlsave --lossless --keep none` on a {w}x{h} "
                f"{bands}-band {fmt} ramp, then every pixel read back. The "
                f"round trip is the identity.",
        "fixture": f"fixtures/{name}.jxl",
        "sha256": sha(fixture),
        "bytes": size(fixture),
        "boxes": boxes(fixture),
        "header": header(fixture, all_fields=True).stdout.strip(),
        "source_getpoint": src_px,
        "getpoint": px,
        "identity": px == src_px,
    }
    return fixture


ll_rgb = lossless("ll_rgb", ramp8(4, 3, 3), 4, 3, 3, "uchar", "srgb")
lossless("ll_rgba", ramp8(4, 3, 4), 4, 3, 4, "uchar", "srgb")
lossless("ll_grey", ramp8(4, 3, 1), 4, 3, 1, "uchar", "b-w")
lossless("ll_rgb16", ramp16(4, 3, 3), 4, 3, 3, "ushort", "rgb16")
lossless("ll_f32", rampf32(4, 3, 3), 4, 3, 3, "float", "scrgb", ints=False)

NOTES.append(
    "A one-band image stays one band through jxlsave/jxlload, where webpsave "
    "promotes it to three. JPEG XL stores greyscale and WebP does not."
)


# ---------------------------------------------------------------------------
# The VarDCT path, which is where the tolerance comes from.
# ---------------------------------------------------------------------------

src = rawload("lossy_rgb", ramp8(4, 3, 3), 4, 3, 3)
lossy = os.path.join(FIX, "lossy_rgb.jxl")
vips("jxlsave", src, lossy, "--keep", "none")
lossy_px = getpoint(lossy, 4, 3)
RECORDS["lossy_rgb"] = {
    "what": "`vips jxlsave` at the default distance on the same 4x3 ramp: "
            "the VarDCT float path, which libviprs decodes but cannot write. "
            "The pixels differ from the source because the encode was lossy.",
    "fixture": "fixtures/lossy_rgb.jxl",
    "sha256": sha(lossy),
    "bytes": size(lossy),
    "boxes": boxes(lossy),
    "header": header(lossy, all_fields=True).stdout.strip(),
    "getpoint": lossy_px,
    "source_getpoint": getpoint(src, 4, 3),
}


# ---------------------------------------------------------------------------
# Multi-frame. The default load reads frame 0 and says how many there were.
# ---------------------------------------------------------------------------

src = rawload("anim3", ramp8(4, 9, 3), 4, 9, 3)
anim = os.path.join(FIX, "anim3.jxl")
vips("jxlsave", src, anim, "--lossless", "--keep", "none", "--page-height", "3")
RECORDS["anim3"] = {
    "what": "`vips jxlsave --page-height 3` on a 4x9 toilet roll: three 4x3 "
            "frames whose frame 0 is the `ll_rgb` image. A DEFAULT load "
            "reads 4x3 and reports `n-pages: 3`; `[n=-1]` reads the roll.",
    "fixture": "fixtures/anim3.jxl",
    "sha256": sha(anim),
    "bytes": size(anim),
    "boxes": boxes(anim),
    "header_default": header(anim, all_fields=True).stdout.strip(),
    "header_n_all": header(f"{anim}[n=-1]", all_fields=True).stdout.strip(),
    "getpoint_default": getpoint(anim, 4, 3),
    "source_getpoint": getpoint(src, 4, 9),
}


# ---------------------------------------------------------------------------
# Metadata. What vips makes of an `Exif` box is not the box payload.
# ---------------------------------------------------------------------------

# The realistic case first: a JPEG in, so the EXIF is the one libjxl wrote
# into a Brotli-compressed `brob` box rather than one this script invented.
jpg = os.path.join(OUT, "src.jpg")
vips("jpegsave", os.path.join(OUT, "ll_rgb-srgb.v"), jpg, "-Q", "100")
meta = os.path.join(FIX, "meta.jxl")
vips("jxlsave", jpg, meta, "--lossless", "--keep", "all")
RECORDS["meta"] = {
    "what": "A JPEG with EXIF re-saved as lossless JPEG XL with `--keep "
            "all`. libjxl writes the EXIF into a Brotli-compressed `brob` "
            "box, and jxlload reads back the SAME blob the JPEG loader "
            "attached, byte for byte.",
    "fixture": "fixtures/meta.jxl",
    "sha256": sha(meta),
    "bytes": size(meta),
    "boxes": boxes(meta),
    "header": header(meta, all_fields=True).stdout.strip(),
    "exif_data_b64": field(meta, "exif-data"),
    "jpeg_exif_data_b64": field(jpg, "exif-data"),
    "exif_survives_the_transcode": field(meta, "exif-data") == field(jpg, "exif-data"),
    "getpoint": getpoint(meta, 4, 3),
}

with open(ll_rgb, "rb") as f:
    CODESTREAM = f.read()

# Then the discriminating cases, hand-built so the box payload is chosen
# rather than whatever libjxl felt like writing.
for name, offset, pad, with_xmp, what in [
    ("meta_off0", 0, b"", True,
     "A hand-built container with an `Exif` box at tiff_header_offset 0 and "
     "an `xml ` box. vips reports exif-data as `Exif\\0\\0` glued onto the "
     "TIFF block (16 bytes for a 10-byte block) and xmp-data as the `xml ` "
     "payload verbatim."),
    ("meta_off6", 6, b"PADPAD", False,
     "The same TIFF block behind a tiff_header_offset of 6, with six bytes "
     "of padding in front of it. vips reports the IDENTICAL 16-byte blob, "
     "so the offset really is skipped rather than ignored."),
]:
    extra = [(b"Exif", struct.pack(">I", offset) + pad + TIFF)]
    if with_xmp:
        extra.append((b"xml ", XMP))
    path = os.path.join(FIX, f"{name}.jxl")
    with open(path, "wb") as f:
        f.write(container(CODESTREAM, *extra))
    COMMANDS.append(f"# hand-built: {name}.jxl = JXL container + Exif(offset={offset})"
                    + (" + xml " if with_xmp else "") + " + jxlc")
    rec = {
        "what": what,
        "fixture": f"fixtures/{name}.jxl",
        "sha256": sha(path),
        "bytes": size(path),
        "boxes": boxes(path),
        "header": header(path, all_fields=True).stdout.strip(),
        "exif_data_b64": field(path, "exif-data"),
        "getpoint": getpoint(path, 4, 3),
    }
    if with_xmp:
        rec["xmp_data_b64"] = field(path, "xmp-data")
    RECORDS[name] = rec

# And the two malformed shapes, where vips and libviprs part company.
bad = os.path.join(FIX, "meta_badoffset.jxl")
with open(bad, "wb") as f:
    f.write(container(CODESTREAM, (b"Exif", struct.pack(">I", 999) + TIFF)))
COMMANDS.append("# hand-built: meta_badoffset.jxl = JXL container + "
                "Exif(offset=999, payload 10 bytes) + jxlc")
proc = header(bad, all_fields=True, allow_fail=True)
RECORDS["meta_badoffset"] = {
    "what": "An `Exif` box whose tiff_header_offset runs past the payload. "
            "`jxlload.c:646-649` warns `invalid data in EXIF box` and FAILS "
            "THE LOAD; the pixels are never reached. libviprs deliberately "
            "diverges: it drops the blob and keeps the image.",
    "fixture": "fixtures/meta_badoffset.jxl",
    "sha256": sha(bad),
    "bytes": size(bad),
    "boxes": boxes(bad),
    "vips_exit_code": proc.returncode,
    "vips_stdout": proc.stdout.strip(),
    "vips_stderr": proc.stderr.strip(),
}

pref = os.path.join(OUT, "meta_prefixed.jxl")
with open(pref, "wb") as f:
    f.write(container(CODESTREAM, (b"Exif", b"Exif\x00\x00" + TIFF)))
COMMANDS.append("# hand-built: outputs/meta_prefixed.jxl = JXL container + "
                "Exif(payload already carrying the Exif\\0\\0 prefix) + jxlc")
RECORDS["meta_prefixed"] = {
    "what": "A nonconforming `Exif` box whose payload already carries the "
            "`Exif\\0\\0` prefix instead of a 4-byte offset. "
            "`jxlload.c:635-637` special-cases it and attaches it verbatim. "
            "jxl-oxide reads the first four bytes as the offset regardless, "
            "so libviprs drops this blob where vips keeps it.",
    "fixture": "outputs/meta_prefixed.jxl (not pinned; no test embeds it)",
    "sha256": sha(pref),
    "bytes": size(pref),
    "exif_data_b64": field(pref, "exif-data"),
}


# ---------------------------------------------------------------------------
# The smallest image each side will encode.
# ---------------------------------------------------------------------------

tiny = {}
for w, h in [(1, 1), (2, 1), (1, 2), (2, 2), (4, 1)]:
    name = f"tiny_{w}x{h}"
    src = rawload(name, ramp8(w, h, 3), w, h, 3)
    path = os.path.join(OUT, f"{name}.jxl")
    proc = vips("jxlsave", src, path, "--lossless", "--keep", "none", allow_fail=True)
    tiny[f"{w}x{h}"] = {
        "vips_exit_code": proc.returncode,
        "bytes": size(path) if proc.returncode == 0 else None,
        "header": header(path).stdout.strip() if proc.returncode == 0 else None,
    }
RECORDS["minimum_dimensions"] = {
    "what": "`vips jxlsave` accepts every geometry down to 1x1. "
            "`zune-jpegxl`'s `encoder.rs:1059-1064` refuses width <= 1 or "
            "height <= 1 outright, so libviprs has a floor vips does not "
            "and it is 2 pixels on each axis.",
    "vips": tiny,
    "zune_jpegxl_floor": {"width": 2, "height": 2},
}


# ---------------------------------------------------------------------------
# The other direction: vips reading what libviprs wrote.
# ---------------------------------------------------------------------------

if "--viprs" in sys.argv:
    for name, w, h, ints in [
        ("viprs_rgb", 4, 3, True),
        ("viprs_rgba", 4, 3, True),
        ("viprs_grey", 4, 3, True),
        ("viprs_rgb16", 4, 3, True),
    ]:
        path = os.path.join(OUT, f"{name}.jxl")
        if not os.path.exists(path):
            raise SystemExit(
                f"{path} is missing. Run the libviprs test that writes it:\n"
                f"  JXL_ORACLE_OUT={OUT} cargo test --lib "
                f"jxl::tests::write_oracle_inputs -- --ignored"
            )
        RECORDS[name] = {
            "what": f"`{name}.jxl` written by `Raster::encode_jxl`, read by "
                    f"vips. The point of the record is that vips accepts the "
                    f"file at all, and reads the same pixels back.",
            "fixture": f"outputs/{name}.jxl",
            "sha256": sha(path),
            "bytes": size(path),
            "boxes": boxes(path),
            "header": header(path, all_fields=True).stdout.strip(),
            "getpoint": getpoint(path, w, h, ints=ints),
        }


# ---------------------------------------------------------------------------

with open(os.path.join(ROOT, "oracle.json"), "w") as f:
    json.dump(
        {
            "meta": {
                "area": "foreign-jxl",
                "issues": [619, 620, 621, 622],
                "vips_version": run([VIPS, "--version"]).stdout.strip(),
                "vips_binary": VIPS,
                "libjxl": "JXL load/save with libjxl: true (dynamic module: true)",
                "captured_by": "oracle-captures/foreign-jxl/capture.py",
                "reference_c": "libvips v8.18.4 for the file and line numbers "
                               "quoted in src/jxl.rs; the binary every number "
                               "here came out of is the Homebrew 8.18.4 build "
                               "and is a different artefact",
                "decoder_under_test": "jxl-oxide 0.12.6",
                "encoder_under_test": "zune-jpegxl 0.5.2",
            },
            "notes": NOTES,
            "records": RECORDS,
        },
        f,
        indent=2,
        # allow_nan=False so a non-finite measurement stops the
        # capture here rather than writing a file nobody outside
        # Python can parse (#682).
        allow_nan=False,
    )
    f.write("\n")

with open(os.path.join(ROOT, "commands.sh"), "w") as f:
    f.write("#!/bin/sh\n")
    f.write("# Every command capture.py ran, in order. Regenerate with\n")
    f.write("# `python3 capture.py` from this directory.\n")
    f.write("set -e\n\n")
    f.write("\n".join(COMMANDS))
    f.write("\n")

print(f"wrote oracle.json ({len(RECORDS)} records) and commands.sh "
      f"({len(COMMANDS)} commands)")
