#!/usr/bin/env python3
"""
Oracle capture for the Radiance HDR area (issue #506).

Runs the real vips 8.18.4 CLI over hand-authored `.hdr` files and over the
reference suite's two `sample.hdr` images, and records exactly what it does.
The records here are what `src/radiance.rs` is pinned against, because the
decode and encode constants libvips uses are a matched pair that cannot be
derived by reading the format spec:

  * decode is `(mantissa + 0.5) * 2^(e - 136)` with a hard zero at
    exponent 0 (`libvips/colour/rad2float.c`, `colr_color`)
  * encode is `frexp(max) * 255.9999 / max` with a `1e-32` floor
    (`libvips/colour/float2rad.c`, `setcolr`)

Two traps this capture deliberately avoids, both recorded in `notes`:

 1. `vips getpoint` on a `.hdr` prints FOUR values. The fourth is an
    out-of-bounds read: `getpoint.c:105` sizes its output array from the
    CODED band count (4) while `vips_image_decode` has already reduced the
    buffer to 3 bands. Every getpoint record here goes through `rad2float`
    first and keeps three values.
 2. Two different files named `sample.hdr` live on this machine, 141x980 in
    the libviprs reference-test checkout and 1655x1764 in the libvips source
    tree. Every record says which one it came from.

Writes:
  commands.sh  - every vips CLI command actually executed, in order
  oracle.json  - structured records

Re-running needs the vips binary at VIPS and the two `sample.hdr` files at
SAMPLE_SMALL / SAMPLE_LARGE; every other input is generated from scratch,
deterministically. Nothing outside this script's own directory is written.
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
SAMPLE_SMALL = ("/Users/rom/workspace/libviprs/libviprs-tests/tmp/"
                "libvips-reference-tests/test-suite/images/sample.hdr")
SAMPLE_LARGE = "/Users/rom/workspace/libvips/test/test-suite/images/sample.hdr"

os.makedirs(FIX, exist_ok=True)
os.makedirs(OUT, exist_ok=True)

COMMANDS = []


def run(args, allow_fail=False):
    """Run a vips command, logging it (with this directory's absolute path
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
    proc = run(args, allow_fail=True)
    if proc.returncode != 0:
        return {"error": proc.stderr.strip()}
    if not all_fields:
        # Strip this directory's absolute path so the record is portable.
        return {"summary": proc.stdout.strip().replace(ROOT + "/", "")}
    out = {}
    for line in proc.stdout.splitlines()[1:]:
        if ": " in line:
            name, value = line.split(": ", 1)
            out[name.strip()] = value.strip().replace(ROOT + "/", "")
    return out


def getpoint_via_float(path, points):
    """Three-band getpoint values, taken after rad2float so the coded band
    count cannot leak an out-of-bounds fourth sample."""
    unpacked = os.path.join(OUT, "getpoint.v")
    vips("rad2float", path, unpacked)
    out = []
    for x, y in points:
        proc = vips("getpoint", unpacked, str(x), str(y))
        values = [float(v) for v in proc.stdout.split()]
        out.append({"x": x, "y": y, "values": values[:3]})
    return out


def write_hdr(name, width, height, quads, fmt="32-bit_rle_rgbe", resolu=None,
              magic=b"#?RADIANCE\n", extra=b""):
    body = bytearray(magic)
    if fmt is not None:
        body += b"FORMAT=" + fmt.encode() + b"\n"
    body += extra
    body += b"\n"
    body += resolu if resolu is not None else b"-Y %d +X %d\n" % (height, width)
    for q in quads:
        body += bytes(q)
    path = os.path.join(FIX, name)
    with open(path, "wb") as f:
        f.write(bytes(body))
    return path


def write_floats(name, width, height, triples):
    """A float32 RGB raster loaded as scRGB, the carrier float2rad wants."""
    raw = os.path.join(FIX, name + ".raw")
    with open(raw, "wb") as f:
        for t in triples:
            for c in t:
                f.write(struct.pack("<f", c))
    loaded = os.path.join(OUT, name + ".v")
    scrgb = os.path.join(OUT, name + "-scrgb.v")
    vips("rawload", raw, loaded, str(width), str(height), "3", "--format", "float")
    vips("copy", loaded, scrgb, "--interpretation", "scrgb")
    return scrgb


def payload(path):
    """The pixel bytes of a `.hdr`: everything after the blank line that
    ends the header and the resolution line that follows it."""
    data = open(path, "rb").read()
    i = data.index(b"\n\n")
    rest = data[i + 2:]
    j = rest.index(b"\n")
    return rest[:j + 1], rest[j + 1:]


def sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


records = {}

# --- 1. the half-bit decode constant --------------------------------------
FLAT6 = [(255, 255, 255, 128), (128, 128, 128, 128), (64, 32, 16, 129),
         (255, 0, 0, 140), (0, 0, 0, 0), (1, 2, 3, 0)]
flat6 = write_hdr("flat6.hdr", 6, 1, FLAT6)
records["decode_half_bit_constant"] = {
    "what": "vips decodes RGBE as (mantissa + 0.5) * 2^(e - 136), with a hard "
            "zero when the exponent byte is 0. The `image` crate's plain "
            "mantissa * 2^(e - 136) gives 4080 0 0 for the fourth pixel here, "
            "a 100% error on the two zero mantissas.",
    "file": "fixtures/flat6.hdr",
    "width_below_MINELEN": True,
    "rgbe": [list(q) for q in FLAT6],
    "header": header(flat6, all_fields=True),
    "getpoint": getpoint_via_float(flat6, [(x, 0) for x in range(6)]),
}

# --- 2. the matched encode constant ---------------------------------------
ENCODE_INPUTS = [
    (1.0, 0.5, 0.25), (0.0, 0.0, 0.0), (1e-33, 1e-33, 1e-33),
    (1e-31, 0.0, 0.0), (4088.0, 8.0, 8.0), (-1.0, 2.0, -3.0),
    (65504.0, 1.0, 0.001), (0.998046875, 0.501953125, 0.12890625),
    (float("inf"), 1.0, 1.0), (2.0, 2.0, 2.0),
    (0.998046875, 0.998046875, 0.998046875), (3.0517578125e-05, 1.0, 1.0),
]
enc_src = write_floats("setcolr", 6, 2, ENCODE_INPUTS)
enc_rad = os.path.join(OUT, "setcolr-rad.v")
enc_raw = os.path.join(OUT, "setcolr-rad.raw")
vips("float2rad", enc_src, enc_rad)
vips("rawsave", enc_rad, enc_raw)
records["encode_setcolr"] = {
    "what": "float2rad's setcolr: frexp(max) * 255.9999 / max, a 1e-32 floor, "
            "negatives clamped to zero, and a TRUNCATING conversion to uchar. "
            "Width 6 is below MINELEN so radsave writes these bytes flat.",
    "inputs": [list(t) for t in ENCODE_INPUTS],
    "rgbe": list(open(enc_raw, "rb").read()),
    "notes": "The infinity row is undefined behaviour in C (the product is "
             "NaN and the conversion to uchar is unspecified); this build "
             "produces 0 0 0 128, which is also what Rust's saturating cast "
             "gives.",
}

# --- 3. the run-length encoder --------------------------------------------
RLE_INPUTS = [(1.0, 0.5 if i < 6 else (i - 5) / 16.0, 0.25) for i in range(16)]
rle_src = write_floats("rle16", 16, 1, RLE_INPUTS)
rle_rad = os.path.join(OUT, "rle16-rad.v")
rle_hdr = os.path.join(OUT, "rle16.hdr")
vips("float2rad", rle_src, rle_rad)
vips("radsave", rle_rad, rle_hdr)
resolu, body = payload(rle_hdr)
records["encode_rle_scanline"] = {
    "what": "rle_scanline_write: four separate component planes, MINRUN 4, "
            "run code 128 + count, literal blocks of up to 128.",
    "inputs": [list(t) for t in RLE_INPUTS],
    "resolution_line": resolu.decode(),
    "payload": list(body),
    "decoded": "planes are: red run of 16 x 127, green run of 6 x 63 then a "
               "10-byte literal, blue run of 16 x 31, exponent run of 16 x 129",
}

# --- 4. MINELEN/MAXELEN choose an encoding, they do not gate --------------
size_gate = {}
for width in (4, 16, 40000):
    src = write_floats(f"size{width}", width, 2, [(1.0, 0.5, 0.25)] * (width * 2))
    rad = os.path.join(OUT, f"size{width}-rad.v")
    out = os.path.join(OUT, f"size{width}.hdr")
    vips("float2rad", src, rad)
    proc = vips("radsave", rad, out, allow_fail=True)
    resolu, body = payload(out)
    size_gate[str(width)] = {
        "exit": proc.returncode,
        "resolution_line": resolu.decode(),
        "payload_bytes": len(body),
        "flat_payload_bytes": width * 4 * 2,
        "run_length_encoded": len(body) != width * 4 * 2,
    }
records["size_range_selects_an_encoding"] = {
    "what": "MINELEN 8 and MAXELEN 0x7fff are NOT a dimension gate. "
            "scanline_write (radiance.c:955-978) falls back to a flat, "
            "unencoded write outside 8..=32767 and run-length encodes "
            "inside it. Both widths below save successfully.",
    "widths": size_gate,
}

# --- 5. resolution line orientation ---------------------------------------
orient = {}
for resolu in (b"-Y 1 +X 6\n", b"+X 6 +Y 1\n", b"-Y 1 -X 6\n"):
    name = resolu.decode().strip().replace(" ", "_")
    path = write_hdr(f"orient_{name}.hdr", 6, 1, FLAT6, resolu=resolu)
    orient[resolu.decode().strip()] = header(path)
records["resolution_orientation"] = {
    "what": "The axis written SECOND is the scanline length (scanlen/numscans, "
            "radiance.c:250-251), so `+X 6 +Y 1` is a 1x6 image. The -/+ "
            "direction flags are parsed and then ignored: libvips 'will not "
            "rotate/flip as the FORMAT string asks' (radiance.c:70).",
    "lines": orient,
}

# --- 6. the involution, on both reference images --------------------------
involution = {}
for label, path in (("sample.hdr 141x980 (libviprs reference-test checkout)", SAMPLE_SMALL),
                    ("sample.hdr 1655x1764 (libvips source tree)", SAMPLE_LARGE)):
    if not os.path.exists(path):
        involution[label] = {"skipped": "file not present"}
        continue
    base = os.path.join(OUT, "inv")
    vips("rad2float", path, base + "1.v")
    vips("rawsave", base + "1.v", base + "1.raw")
    vips("float2rad", base + "1.v", base + "r.v")
    vips("radsave", base + "r.v", base + "rt.hdr")
    vips("rad2float", base + "rt.hdr", base + "2.v")
    vips("rawsave", base + "2.v", base + "2.raw")
    involution[label] = {
        "source_sha256": sha256(path),
        "header": header(path),
        "float_payload_sha256_original": sha256(base + "1.raw"),
        "float_payload_sha256_round_tripped": sha256(base + "2.raw"),
        "encoded_payload_sha256": hashlib.sha256(payload(base + "rt.hdr")[1]).hexdigest(),
    }
records["involution"] = {
    "what": "rad2float after float2rad after rad2float reproduces the float "
            "payload byte for byte. The sharper statement, verified over "
            "298,240 combinations in src/radiance.rs, is that float2rad after "
            "rad2float is the IDENTITY on an RGBE quadruple whose largest "
            "mantissa is at least 128 and whose exponent byte is in 23..=255 "
            "-- exactly the normalised form any encoder emits. Below exponent "
            "23 the 1e-32 floor collapses the pixel to all-zero, and a "
            "largest mantissa under 128 is rescaled: (127,63,31,129) becomes "
            "(254,126,62,128).",
    "images": involution,
}

# --- 7. the save trap ------------------------------------------------------
trap = {}
if os.path.exists(SAMPLE_SMALL):
    f = os.path.join(OUT, "trap-f.v")
    vips("rad2float", SAMPLE_SMALL, f)
    direct = os.path.join(OUT, "trap-direct.hdr")
    vips("radsave", f, direct)
    paired_rad = os.path.join(OUT, "trap-pair.v")
    paired = os.path.join(OUT, "trap-pair.hdr")
    vips("float2rad", f, paired_rad)
    vips("radsave", paired_rad, paired)
    def stat(path):
        return {"max": float(vips("max", path).stdout),
                "avg": float(vips("avg", path).stdout)}
    trap = {
        "original": stat(SAMPLE_SMALL),
        "radsave_alone": stat(direct),
        "float2rad_then_radsave": stat(paired),
    }
records["save_trap"] = {
    "what": "`vips radsave` on an image that is not already VIPS_CODING_RAD "
            "routes through vips_colourspace(-> sRGB) and clips to uchar, so "
            "the only HDR-preserving save path in vips is float2rad THEN "
            "radsave. libviprs's save_radiance is equivalent to the pair, "
            "which means libviprs round-trips high dynamic range where a bare "
            "`vips radsave` does not. That is a deliberate divergence.",
    "image": "sample.hdr 141x980",
    "values": trap,
}

# --- 8. magic, FORMAT=, and mono ------------------------------------------
rgbe_magic = write_hdr("rgbe_magic.hdr", 6, 1, FLAT6, magic=b"#?RGBE\n")
xyze = write_hdr("xyze6.hdr", 6, 1, FLAT6, fmt="32-bit_rle_xyze")
bogus = write_hdr("bogusfmt6.hdr", 6, 1, FLAT6, fmt="not_a_radiance_format")
mono_raw = os.path.join(FIX, "mono.raw")
with open(mono_raw, "wb") as fh:
    for i in range(16):
        fh.write(struct.pack("<f", float(i)))
mono = os.path.join(OUT, "mono.v")
vips("rawload", mono_raw, mono, "8", "2", "1", "--format", "float")
mono_out = os.path.join(OUT, "mono.hdr")
mono_proc = vips("radsave", mono, mono_out, allow_fail=True)

records["scope_and_defects"] = {
    "registered_suffixes": [".hdr"],
    "dot_pic_is_not_registered": True,
    "magic_is_the_whole_first_line": {
        "what": "vips__rad_israd (radiance.c:568-577) reads the first line and "
                "compares it to `#?RADIANCE` in full, so `#?RGBE` is not a "
                "Radiance file. Measured, such a file falls through to "
                "magickload.",
        "rgbe_magic_header": header(rgbe_magic),
    },
    "format_line_is_never_read": {
        "what": "rad2vips_process_line calls formatval(line, read->format) "
                "with the arguments the wrong way round for the declaration "
                "at radiance.c:314, so the parsed FORMAT= value never reaches "
                "read->format and the XYZ branch at radiance.c:695-696 is "
                "unreachable in this build. libviprs implements the intent "
                "and tags XYZ; that is a written divergence.",
        "xyze_header": header(xyze, all_fields=True),
        "bogus_format_header": header(bogus, all_fields=True),
    },
    "radsave_cannot_save_mono": {
        "what": "radsave advertises saveable = MONO | RGB (radsave.c:96-97) "
                "but a one-band image fails outright, so there is no mono "
                "behaviour for libviprs to match.",
        "exit": mono_proc.returncode,
        "stderr": mono_proc.stderr.strip(),
    },
}

version = subprocess.run([VIPS, "--version"], capture_output=True, text=True).stdout.strip()
oracle = {
    "meta": {
        "area": "foreign-radiance",
        "issue": 506,
        "vips_version": version,
        "vips_binary": VIPS,
        "captured_by": "oracle-captures/foreign-radiance/capture.py",
        "reference_c": "libvips v8.18.0-95-gfe420cf3a for line numbers; the "
                       "binary is the 8.18.4 release and is not the same "
                       "artefact, so every number here comes from the binary",
    },
    "notes": [
        "`vips getpoint` on a .hdr prints FOUR values; the fourth is an "
        "out-of-bounds read (getpoint.c:105 sizes the array from the coded "
        "band count while vips_image_decode has already reduced the buffer to "
        "3 bands). Every getpoint record here goes through rad2float first.",
        "Two different files named sample.hdr exist on this machine: 141x980 "
        "in the libviprs reference-test checkout and 1655x1764 in the libvips "
        "source tree. Records name which one they used.",
        "The existing oracle-captures/foreign/oracle.json record for "
        "loads['sample.hdr'] has the four-value getpoint bug baked in.",
    ],
    "records": records,
}

with open(os.path.join(ROOT, "oracle.json"), "w") as f:
    json.dump(oracle, f, indent=2, sort_keys=False)
    f.write("\n")

with open(os.path.join(ROOT, "commands.sh"), "w") as f:
    f.write("#!/bin/sh\n")
    f.write("# Every vips command capture.py ran, in order. Regenerate with\n")
    f.write("# `python3 capture.py` from this directory.\n")
    f.write("set -e\n\n")
    for c in COMMANDS:
        f.write(c + "\n")

print(f"wrote oracle.json ({len(records)} records) and commands.sh "
      f"({len(COMMANDS)} commands)")
