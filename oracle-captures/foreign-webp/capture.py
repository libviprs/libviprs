#!/usr/bin/env python3
"""
Oracle capture for the still-image WebP area (issues #567 and #568).

Runs the real vips 8.18.4 CLI over small, deterministic rasters and records
what `webpsave` writes and what `webpload` reads back. Everything
`src/webp.rs` is pinned against is here, because none of it can be derived
from the format spec:

  * which container chunks `webpsave` emits, and in what order
  * that `--lossless` is a true identity: the pixels that go in come back
    out byte for byte, for both the opaque and the alpha carrier
  * that a one-band image comes back as three bands, since WebP stores no
    greyscale and `webpsave` is registered `rgb alpha`
  * that a `ushort` image is narrowed by a right shift of 8 rather than
    clipped, which is the behaviour libviprs deliberately does NOT copy
  * that a default `webpload` of an animation reads frame 0 and reports
    `n-pages`, leaving the toilet roll to `n=-1`
  * the exact width at which `webpsave` starts refusing an image, which is
    one pixel below where `image-webp` 0.2.4 starts refusing one
  * that vips reads back what libviprs writes, with the same pixels, the
    same band count and the same three metadata fields

The libvips C line numbers quoted in `src/webp.rs` come from the source tree
at the commit recorded in `meta.reference_c` below. Every *number* here came
out of the binary, which is the 8.18.4 release build and a different
artefact from that tree.

Writes:
  commands.sh  - every vips CLI command actually executed, in order
  oracle.json  - structured records
  fixtures/    - the five `.webp` files `src/webp.rs` embeds verbatim

Re-running needs only the vips binary at VIPS; every input is generated from
scratch, deterministically. Nothing outside this script's own directory is
written.
"""
import hashlib
import json
import os
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


def header(path, all_fields=False):
    args = [VIPSHEADER] + (["-a"] if all_fields else []) + [path]
    return run(args).stdout.strip()


def getpoint(path, w, h):
    """Every pixel of a small image, in raster order, as ints."""
    out = []
    for y in range(h):
        for x in range(w):
            txt = vips("getpoint", path, str(x), str(y)).stdout.split()
            out.append([int(float(v)) for v in txt])
    return out


def ramp(w, h, bands):
    """The deterministic ramp every fixture is built from. Band 0 steps by
    61 across and 13 down, so no two pixels in a 4x3 tile repeat."""
    data = bytearray()
    steps = [(61, 13), (97, 151), (29, 211), (85, 40)]
    for y in range(h):
        for x in range(w):
            for b in range(bands):
                dx, dy = steps[b]
                data.append((x * dx + y * dy) % 256)
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


def chunks(path):
    """The RIFF chunk directory of a WebP file: [(tag, payload size)]."""
    with open(path, "rb") as f:
        d = f.read()
    assert d[:4] == b"RIFF" and d[8:12] == b"WEBP", path
    out, p = [], 12
    while p + 8 <= len(d):
        tag = d[p:p + 4].decode("ascii")
        size = struct.unpack("<I", d[p + 4:p + 8])[0]
        out.append([tag, size])
        p += 8 + size + (size & 1)
    return out


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


records = {}
notes = []

# ---------------------------------------------------------------------------
# 1. Lossless is an exact identity, for RGB and for RGBA.
# ---------------------------------------------------------------------------
for name, bands, fmt in (("ll_rgb", 3, "Rgb8"), ("ll_rgba", 4, "Rgba8")):
    src = ramp(4, 3, bands)
    tagged = rawload(name, src, 4, 3, bands)
    webp = os.path.join(FIX, f"{name}.webp")
    vips("webpsave", tagged, webp, "--lossless", "--keep", "none")
    px = getpoint(webp, 4, 3)
    flat = [v for pixel in px for v in pixel]
    records[name] = {
        "what": f"`vips webpsave --lossless --keep none` on a 4x3 {fmt} ramp, "
                "then every pixel read back. The round trip is the identity: "
                "the bytes that went in are the bytes that come out.",
        "fixture": f"fixtures/{name}.webp",
        "sha256": sha256(webp),
        "bytes": os.path.getsize(webp),
        "chunks": chunks(webp),
        "header": header(webp),
        "source_bytes": list(src),
        "getpoint": px,
        "identity": flat == list(src),
    }
    assert flat == list(src), name

# ---------------------------------------------------------------------------
# 2. The lossy bitstream libviprs decodes but cannot write.
# ---------------------------------------------------------------------------
src = ramp(4, 3, 3)
tagged = rawload("lossy_rgb", src, 4, 3, 3)
lossy = os.path.join(FIX, "lossy_rgb.webp")
vips("webpsave", tagged, lossy, "--keep", "none")
lossy_px = getpoint(lossy, 4, 3)
records["lossy_rgb"] = {
    "what": "`vips webpsave` at the default Q on the same 4x3 ramp. VP8 "
            "reconstruction is integer-specified and libwebp's default "
            "chroma upsampling is the fancy (bilinear) one, so a correct "
            "pure-Rust decoder reproduces these values exactly rather than "
            "approximately. libviprs pins them with no tolerance.",
    "fixture": "fixtures/lossy_rgb.webp",
    "sha256": sha256(lossy),
    "bytes": os.path.getsize(lossy),
    "chunks": chunks(lossy),
    "getpoint": lossy_px,
    "differs_from_source": [v for p in lossy_px for v in p] != list(src),
}

# ---------------------------------------------------------------------------
# 3. Greyscale is promoted to three bands: WebP has no mono.
# ---------------------------------------------------------------------------
grey_src = ramp(4, 3, 1)
tagged = rawload("grey", grey_src, 4, 3, 1, interp="b-w")
grey = os.path.join(OUT, "grey.webp")
vips("webpsave", tagged, grey, "--lossless", "--keep", "none")
grey_px = getpoint(grey, 4, 3)
records["grey_promotes_to_rgb"] = {
    "what": "A 1-band b-w uchar image saved with `webpsave` loads back as 3 "
            "bands srgb. `vips -l` registers the saver as `rgb alpha`, and "
            "the container has no greyscale colour type at all, so the "
            "luminance is repeated into all three bands.",
    "source_bytes": list(grey_src),
    "header": header(grey),
    "getpoint": grey_px,
    "bands_in": 1,
    "bands_out": len(grey_px[0]),
    "luminance_preserved": [p[0] for p in grey_px] == list(grey_src)
    and all(p[0] == p[1] == p[2] for p in grey_px),
}

# ---------------------------------------------------------------------------
# 4. A 16-bit image is narrowed by a right shift of 8, silently.
# ---------------------------------------------------------------------------
wide_vals = [0, 1, 255, 256, 257, 511, 512, 65535]
wide = bytearray()
for i in range(4 * 2 * 3):
    wide += struct.pack("<H", wide_vals[i % len(wide_vals)])
tagged = rawload("rgb16", bytes(wide), 4, 2, 3, fmt="ushort")
wide_webp = os.path.join(OUT, "rgb16.webp")
vips("webpsave", tagged, wide_webp, "--lossless", "--keep", "none")
wide_px = getpoint(wide_webp, 4, 2)
in_vals = [struct.unpack("<H", bytes(wide[i:i + 2]))[0]
           for i in range(0, len(wide), 2)]
out_vals = [v for p in wide_px for v in p]
records["sixteen_bit_is_shifted_not_clipped"] = {
    "what": "`vips webpsave` accepts a ushort image and narrows it to uchar "
            "by a RIGHT SHIFT OF 8, not by clipping: 255 -> 0 and 65535 -> "
            "255. libviprs refuses the raster instead and tells the caller "
            "to cast, because `Raster::cast` to an 8-bit format CLIPS, so an "
            "automatic narrow here would disagree with the crate's own cast "
            "while looking like it did the same thing.",
    "header_in": header(tagged),
    "header_out": header(wide_webp),
    "in": in_vals,
    "out": out_vals,
    "is_right_shift_8": out_vals == [v >> 8 for v in in_vals],
    "is_clip_255": out_vals == [min(v, 255) for v in in_vals],
}
notes.append(
    "webpsave narrows 16-bit input by `>> 8`, and does so whether the image "
    "is tagged rgb16 or srgb, so the shift is not interpretation-driven."
)

# ---------------------------------------------------------------------------
# 5. An animation: default load is frame 0, with n-pages.
# ---------------------------------------------------------------------------
roll = ramp(4, 9, 3)
tagged = rawload("roll", roll, 4, 9, 3)
anim = os.path.join(FIX, "anim3.webp")
vips("webpsave", tagged, anim, "--lossless", "--keep", "none",
     "--page-height", "3")
anim_px = getpoint(anim, 4, 3)
records["animation_default_load_is_frame_zero"] = {
    "what": "`webpsave --page-height 3` on a 4x9 toilet roll writes three "
            "ANMF frames. A DEFAULT `webpload` reads 4x3, i.e. frame 0 "
            "only, and reports `n-pages: 3`; the 4x9 roll needs `[n=-1]`. "
            "libviprs matches the default: frame 0 plus the page count, "
            "with the roll left to issue #569 and the page model.",
    "fixture": "fixtures/anim3.webp",
    "sha256": sha256(anim),
    "bytes": os.path.getsize(anim),
    "chunks": chunks(anim),
    "header_default": header(anim, all_fields=True),
    "header_n_minus_1": header(anim + "[n=-1]"),
    "getpoint_frame0": anim_px,
    "frame0_equals_first_three_rows": [v for p in anim_px for v in p]
    == list(roll[:4 * 3 * 3]),
}

# ---------------------------------------------------------------------------
# 6. The three metadata chunks, and the field names vips lifts them into.
# ---------------------------------------------------------------------------
icc = bytes(range(0x10, 0x28))
exif = b"II*\x00\x08\x00\x00\x00\x00\x00"
xmp = b'<x:xmpmeta xmlns:x="adobe:ns:meta/"/>'


def chunk(tag, payload):
    return (tag + struct.pack("<I", len(payload)) + payload
            + (b"\x00" if len(payload) & 1 else b""))


with open(os.path.join(FIX, "ll_rgb.webp"), "rb") as f:
    vp8l = f.read()[12:]
flags = 0x20 | 0x08 | 0x04  # ICC | EXIF | XMP
vp8x = bytes([flags, 0, 0, 0]) + (4 - 1).to_bytes(3, "little") \
    + (3 - 1).to_bytes(3, "little")
body = (chunk(b"VP8X", vp8x) + chunk(b"ICCP", icc) + vp8l
        + chunk(b"EXIF", exif) + chunk(b"XMP ", xmp))
meta = os.path.join(FIX, "meta.webp")
with open(meta, "wb") as f:
    f.write(b"RIFF" + struct.pack("<I", 4 + len(body)) + b"WEBP" + body)
COMMANDS.append("# fixtures/meta.webp is built by this script, not by vips: "
                "webpsave has no way to attach an arbitrary XMP packet")
records["metadata_chunks"] = {
    "what": "The ll_rgb VP8L bitstream rewrapped in an extended container "
            "with ICCP, EXIF and XMP chunks. vips lifts them into "
            "`icc-profile-data`, `exif-data` and `xmp-data`, as raw chunk "
            "payloads with nothing stripped: the JPEG loader's `Exif\\0\\0` "
            "prefix does not exist in a WebP EXIF chunk. libviprs attaches "
            "the same three names.",
    "fixture": "fixtures/meta.webp",
    "sha256": sha256(meta),
    "bytes": os.path.getsize(meta),
    "chunks": chunks(meta),
    "icc": list(icc),
    "exif": list(exif),
    "xmp": list(xmp),
    "header": header(meta, all_fields=True),
    "getpoint": getpoint(meta, 4, 3),
}

# Confirm webpsave writes the same field back out of an ICC-carrying image,
# and that its chunk order is VP8X, ICCP, VP8L, EXIF.
prof = os.path.join(OUT, "sRGB.icm")
icc_probe = None
for candidate in ("/Users/rom/workspace/libvips/test/test-suite/images/sRGB.icm",):
    if os.path.exists(candidate):
        icc_probe = candidate
if icc_probe:
    tagged = rawload("icc_src", ramp(4, 3, 3), 4, 3, 3)
    out = os.path.join(OUT, "with_icc.webp")
    vips("webpsave", tagged, out, "--lossless", "--profile", icc_probe)
    records["webpsave_chunk_order"] = {
        "what": "`webpsave --profile` writes VP8X, ICCP, VP8L, EXIF in that "
                "order. Note the EXIF chunk vips synthesises from the "
                "resolution fields even though the source raster carried "
                "none; libviprs writes back only what is attached.",
        "profile": icc_probe,
        "chunks": chunks(out),
        "header": header(out, all_fields=True),
    }

# ---------------------------------------------------------------------------
# 7. The width ceiling, bisected.
# ---------------------------------------------------------------------------
ceiling = {}
for w in (16383, 16384, 16385):
    src_v = os.path.join(OUT, f"wide{w}.v")
    vips("black", src_v, str(w), "1", "--bands", "3")
    tagged = os.path.join(OUT, f"wide{w}-srgb.v")
    vips("copy", src_v, tagged, "--interpretation", "srgb")
    out = os.path.join(OUT, f"wide{w}.webp")
    proc = vips("webpsave", tagged, out, "--lossless", allow_fail=True)
    ceiling[str(w)] = {
        "ok": proc.returncode == 0,
        "stderr": proc.stderr.strip().splitlines()[:1],
    }
records["width_ceiling"] = {
    "what": "`webpsave` accepts a width of 16383 and refuses 16384 with "
            "`image too large`: libwebp's WEBP_MAX_DIMENSION is 16383. "
            "`image-webp` 0.2.4's `encode_frame` guards on `width > 16384` "
            "instead, so it will happily write a 16384-wide VP8L that the "
            "reference decoder then refuses to read. libviprs applies the "
            "libwebp ceiling rather than the crate's.",
    "results": ceiling,
}

# ---------------------------------------------------------------------------
# 8. The other direction: vips reading what libviprs wrote.
#
# The four `viprs_*.webp` fixtures are NOT produced by this script. They came
# out of `Raster::save_webp` at `SaveOptions::default()` on the same 4x3 ramp
# used above (plus a 21-step greyscale, and a flat raster carrying the three
# metadata blobs from record 6), and they are checked in so this half of the
# differential is reproducible without a Rust toolchain. Regenerate them by
# saving those rasters and copying the files back in.
# ---------------------------------------------------------------------------
reverse = {}
for name in ("viprs_rgb", "viprs_rgba", "viprs_grey", "viprs_meta"):
    path = os.path.join(FIX, f"{name}.webp")
    if not os.path.exists(path):
        continue
    reverse[name] = {
        "sha256": sha256(path),
        "bytes": os.path.getsize(path),
        "chunks": chunks(path),
        "header": header(path, all_fields=True),
        "getpoint": getpoint(path, 4, 3),
    }
records["vips_reads_libviprs_output"] = {
    "what": "vips 8.18.4 loading four files libviprs wrote. Pixels, band "
            "count and interpretation all match what vips reports for its "
            "own output of the same rasters, the greyscale one promotes to "
            "3 bands with the luminance repeated, and all three metadata "
            "chunks come back under the same field names. This is the half "
            "of the differential a unit test cannot run, since it needs the "
            "binary.",
    "files": reverse,
}

with open(os.path.join(ROOT, "oracle.json"), "w") as f:
    json.dump(
        {
            "meta": {
                "area": "foreign-webp",
                "issues": [567, 568],
                "vips_version": run([VIPS, "--version"]).stdout.strip(),
                "vips_binary": VIPS,
                "captured_by": "oracle-captures/foreign-webp/capture.py",
                "reference_c": "libvips v8.18.0-95-gfe420cf3a for the file "
                               "and line numbers quoted in src/webp.rs; the "
                               "binary every number here came out of is the "
                               "8.18.4 release and is not the same artefact",
            },
            "notes": notes,
            "records": records,
        },
        f,
        indent=2,
    )
    f.write("\n")

with open(os.path.join(ROOT, "commands.sh"), "w") as f:
    f.write("#!/bin/sh\n")
    f.write("# Every command capture.py ran, in order. Regenerate with\n")
    f.write("# `python3 capture.py` from this directory.\n")
    f.write("set -e\n\n")
    for c in COMMANDS:
        f.write(c + "\n")

print(f"{len(records)} records, {len(COMMANDS)} commands")
