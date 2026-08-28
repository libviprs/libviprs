#!/usr/bin/env python3
"""
Oracle capture for the JPEG 2000 area (issue #637, sub-issue of #501).

Runs the real vips CLI over small, deterministic rasters and small
hand-built codestreams, and records what `jp2ksave` writes and what
`jp2kload` reads back. The version is not hardcoded anywhere: `meta` in
oracle.json records whatever `vips --version` said on the run that produced
it, which matters, because Homebrew moved this box from 8.18.4 to 8.18.6
while this capture was being written.

Everything a JPEG 2000 port would have to agree with is here, because almost
none of it can be derived from the format spec:

  * `jp2ksave` ALWAYS writes an RFC 3745 JP2 container, whatever the
    filename says, so `.j2k`, `.j2c`, `.jpc` and `.jpt` are decoration on
    the save side while `jp2kload` really does read both carriers
  * which libvips band format and interpretation each component precision
    and colour space lands on, including the three that are surprising: a
    31-bit `uint` file, a one-band file tagged sRGB, and a three-band file
    tagged greyscale
  * that a sub-8-bit or 9-to-15-bit file is LEFT-JUSTIFIED into the libvips
    element, not returned in its own range, with the true depth left in
    `bits-per-sample`
  * that `page` on the loader is a RESOLUTION LEVEL, not a frame, and that
    `n-pages` is the resolution count, so `[page=1]` is a half-size image
  * that chroma subsampling is a real, pixel-visible transform which is
    invisible on a grey fixture, that `--lossless` silently forces it off,
    and that the loader upsamples by pixel replication rather than
    interpolating
  * that `jp2ksave` writes NO metadata at all: `--profile` and `--keep` are
    inherited no-ops, while `jp2kload` does lift an ICC profile out of a
    METH=2 `colr` box
  * how a truncated or malformed file fails, and that `fail-on` changes
    nothing about it

Fixtures that `jp2ksave` cannot produce (bit depths other than 8/16/31, a
file-level 4:2:0 subsampling, more than one resolution on a small image, a
non-zero image origin) come from `opj_compress`, which is openjpeg's own
CLI and therefore the format's reference implementation: it is the exact
library libvips links, so it is the oracle libvips itself defers to. Every
NUMBER recorded here still comes out of `vips`; opj_compress only makes
input.

Writes:
  commands.sh  - every command actually executed, in order
  oracle.json  - structured records
  fixtures/    - the committed .jp2 / .j2k inputs

Re-running needs the vips binary at VIPS and opj_compress at OPJ; every
input is generated from scratch, deterministically, and both encoders were
checked to produce byte-identical output across repeated runs. Nothing
outside this script's own directory is written.
"""
import hashlib
import json
import os
import re
import struct
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
FIX = os.path.join(ROOT, "fixtures")
OUT = os.path.join(ROOT, "outputs")

VIPS = "/opt/homebrew/bin/vips"
VIPSHEADER = "/opt/homebrew/bin/vipsheader"
OPJ = "/opt/homebrew/bin/opj_compress"

AREA = "foreign-jp2k"

# The oracle is pinned: oracle-captures/ORACLE_PIN.json names the libvips
# build this area is measured against, and check() exits before anything is
# written when the binary on the machine disagrees, so a wrong-oracle run
# leaves no half-updated capture behind. #650 is what happened without it,
# #796 is why every area carries it now, and tests/oracle_capture_pins.rs is
# the half of the guard that runs in CI.
sys.path.insert(0, os.path.abspath(os.path.join(ROOT, os.pardir)))
import oracle_pin  # noqa: E402  (needs the path above)

VIPS_VERSION, ORACLE_PIN = oracle_pin.check(AREA, VIPS)
# `vips --vips-config` names libopenjp2 but not its version, and the version
# is the thing a future disagreement will turn on, so read it off the dylib
# vips actually links rather than off whatever else is installed.
VIPSLIB = "/opt/homebrew/lib/libvips.42.dylib"

os.makedirs(FIX, exist_ok=True)
os.makedirs(OUT, exist_ok=True)

COMMANDS = []

# Two bits of noise have to come off stderr before it can be recorded.
#
# A libvips that cannot dlopen one of its own modules prints a multi-line
# VIPS-WARNING about it in front of every single invocation. The 8.18.4
# build on this box did that for its heif module, because libheif wanted an
# x265 dylib that is not installed; 8.18.6 does not. It has nothing to do
# with jp2k either way, so it goes.
#
# And every warning that does survive carries a pid and a wall-clock stamp,
# which would make oracle.json differ on every run. Normalise those away, so
# re-running the capture reproduces the file byte for byte.
WARN_PREFIX = re.compile(r"^\(\w+:\d+\): (\S+) \*\*: [\d:.]+: ")


def clean_stderr(text):
    lines = []
    in_heif = False
    for line in text.splitlines():
        if "vips-heif" in line:
            in_heif = True
            continue
        if in_heif and (line.startswith((" ", "\t")) or not line.strip()):
            continue
        in_heif = False
        line = line.strip().replace(ROOT + "/", "")
        if line:
            lines.append(WARN_PREFIX.sub(r"\1: ", line))
    return lines


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


def opj(*args, allow_fail=False):
    return run([OPJ, *args], allow_fail=allow_fail)


def header(path, all_fields=False):
    """`vipsheader` on a path, as a dict. Never raises: a load that fails is
    a result too, and several records here are exactly that."""
    args = [VIPSHEADER] + (["-a"] if all_fields else []) + [path]
    proc = run(args, allow_fail=True)
    if proc.returncode != 0:
        return {"error": clean_stderr(proc.stderr)}
    if not all_fields:
        return {"summary": proc.stdout.strip().replace(ROOT + "/", "")}
    out = {}
    for line in proc.stdout.splitlines()[1:]:
        if ": " in line:
            name, value = line.split(": ", 1)
            out[name.strip()] = value.strip().replace(ROOT + "/", "")
    return out


BRIEF_FIELDS = ("format", "bits-per-sample", "n-pages", "tile-width",
                "tile-height", "icc-profile-data", "xmp-data", "exif-data")


def header_brief(path):
    """The one-line vipsheader summary plus the handful of extra fields
    jp2kload can set. The sweeps use this; the committed fixtures still get
    every field."""
    out = header(path)
    if "error" in out:
        return out
    full = header(path, all_fields=True)
    for name in BRIEF_FIELDS:
        if name in full:
            out[name] = full[name]
    return out

def getpoint(path, x, y):
    proc = vips("getpoint", path, str(x), str(y), allow_fail=True)
    if proc.returncode != 0:
        return None
    return [int(float(v)) for v in proc.stdout.split()]


def getpoint_all(path, w, h):
    """Every pixel of a small image, in raster order."""
    return [getpoint(path, x, y) for y in range(h) for x in range(w)]


def decoded(path, tag):
    """sha256 of the FULL decoded raster, taken through `vips rawsave`.
    getpoint only touches the pixels you name; this pins every one of them,
    which is what catches a tile-edge bug."""
    raw = os.path.join(OUT, f"{tag}.raw")
    proc = vips("rawsave", path, raw, allow_fail=True)
    if proc.returncode != 0:
        return {"error": clean_stderr(proc.stderr)}
    with open(raw, "rb") as f:
        data = f.read()
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def magic(path, n=12):
    with open(path, "rb") as f:
        return f.read(n).hex()



def encode(obj, indent=0):
    """json.dumps with indent=2 puts every integer of a pixel dump on its
    own line, which triples the size of this file for no gain. Same JSON,
    but a list that fits on one line stays on one line.

    Every json.dumps below passes allow_nan=False. This is where the leaf
    values are serialised, so guarding one arm and not the others would
    leave most of the document able to carry a bare NaN (#682)."""
    pad = " " * indent
    if isinstance(obj, dict):
        if not obj:
            return "{}"
        flat = json.dumps(obj, allow_nan=False)
        if (len(flat) + indent <= 78
                and not any(isinstance(v, (dict, list))
                            for v in obj.values())):
            return flat
        body = ",\n".join(
            f"{pad}  {json.dumps(k, allow_nan=False)}: "
            f"{encode(v, indent + 2)}"
            for k, v in obj.items())
        return "{\n" + body + "\n" + pad + "}"
    if isinstance(obj, list):
        if not obj:
            return "[]"
        flat = json.dumps(obj, allow_nan=False)
        if (len(flat) + indent <= 78
                or all(not isinstance(i, (dict, list)) for i in obj)):
            return flat
        body = ",\n".join(f"{pad}  {encode(v, indent + 2)}" for v in obj)
        return "[\n" + body + "\n" + pad + "]"
    return json.dumps(obj, allow_nan=False)


# ---------------------------------------------------------------------------
# Container readers. These parse the file's own bytes, the way the WebP
# capture's chunks() reads a RIFF directory. They report structure, never an
# expected pixel: every value compared against vips comes from vips.
# ---------------------------------------------------------------------------
def boxes(path):
    """The ISO box directory of a JP2 file: [(type, length, offset)]."""
    with open(path, "rb") as f:
        d = f.read()
    out, p = [], 0
    while p + 8 <= len(d):
        length = struct.unpack(">I", d[p:p + 4])[0]
        kind = d[p + 4:p + 8].decode("latin1")
        hdr = 8
        if length == 1:
            length = struct.unpack(">Q", d[p + 8:p + 16])[0]
            hdr = 16
        elif length == 0:
            length = len(d) - p
        out.append([kind, length, p])
        if kind == "jp2h":
            # Superbox: step into it rather than over it.
            p += hdr
            continue
        if length < hdr:
            break
        p += length
    return out


def codestream_offset(path):
    """Where the raw JPEG 2000 codestream starts: 0 for a bare .j2k, the
    payload of the jp2c box for a JP2."""
    with open(path, "rb") as f:
        head = f.read(4)
    if head == b"\xff\x4f\xff\x51":
        return 0
    for kind, length, off in boxes(path):
        if kind == "jp2c":
            return off + 8
    return None


def siz(path):
    """The SIZ marker segment, which is where every geometric fact about a
    JPEG 2000 file lives: image and tile grid, and per component the
    precision, the signedness and the subsampling factors."""
    with open(path, "rb") as f:
        d = f.read()
    start = codestream_offset(path)
    if start is None or d[start:start + 2] != b"\xff\x4f":
        return None
    p = start + 2
    if d[p:p + 2] != b"\xff\x51":
        return None
    p += 4  # marker + Lsiz
    fields = struct.unpack(">HIIIIIIIIH", d[p:p + 36])
    p += 36
    comps = []
    for _ in range(fields[9]):
        ssiz, xr, yr = d[p], d[p + 1], d[p + 2]
        comps.append({
            "prec": (ssiz & 0x7F) + 1,
            "sgnd": bool(ssiz & 0x80),
            "dx": xr,
            "dy": yr,
        })
        p += 3
    return {
        "Rsiz": fields[0],
        "Xsiz": fields[1], "Ysiz": fields[2],
        "XOsiz": fields[3], "YOsiz": fields[4],
        "XTsiz": fields[5], "YTsiz": fields[6],
        "XTOsiz": fields[7], "YTOsiz": fields[8],
        "Csiz": fields[9],
        "components": comps,
    }



def colr(path):
    """The JP2 colr box: how the file declares its colour space. METH=1 puts
    an enumerated value there, METH=2 an ICC profile. A bare codestream has
    no boxes at all, so this is None for a .j2k."""
    with open(path, "rb") as f:
        d = f.read()
    for kind, length, off in boxes(path):
        if kind != "colr":
            continue
        body = d[off + 8:off + length]
        out = {"meth": body[0], "prec": body[1], "approx": body[2]}
        if body[0] == 1:
            out["enumcs"] = struct.unpack(">I", body[3:7])[0]
        else:
            out["profile_bytes"] = length - 11
        return out
    return None

def retag_colr(src, dst, enumcs):
    """Rewrite a JP2's colr box to a different enumerated colour space.
    `jp2ksave` derives the enum from the image's interpretation and offers
    no way to override it, so this is the only route to a file whose
    declared colour space disagrees with its component count. The METH=1
    payload is a fixed 7 bytes, so the box length never moves."""
    with open(src, "rb") as f:
        d = f.read()
    for kind, length, off in boxes(src):
        if kind != "colr":
            continue
        body = d[off + 8:off + length]
        assert body[0] == 1, "expected a METH=1 colr box"
        box = (struct.pack(">I", length) + b"colr" + body[:3]
               + struct.pack(">I", enumcs))
        with open(dst, "wb") as f:
            f.write(d[:off] + box + d[off + length:])
        return dst
    return None


# ---------------------------------------------------------------------------
# Deterministic sources.
# ---------------------------------------------------------------------------
def ramp(w, h, bands):
    """Band 0 steps by 61 across and 13 down, so no two pixels of a small
    tile repeat. The other bands use coprime steps of their own."""
    steps = [(61, 13), (97, 151), (29, 211), (85, 40)]
    data = bytearray()
    for y in range(h):
        for x in range(w):
            for b in range(bands):
                dx, dy = steps[b % len(steps)]
                data.append((x * dx + y * dy) % 256)
    return bytes(data)


def gradient(size):
    """A colour gradient. Chroma varies along x and luma along y, so
    throwing the chroma away is visible in the pixels: on a grey ramp,
    subsampling is undetectable."""
    data = bytearray()
    for y in range(size):
        for x in range(size):
            data += bytes([
                (x * 255) // (size - 1),
                (y * 255) // (size - 1),
                255 - ((x * 255) // (size - 1)),
            ])
    return bytes(data)


def load_raw(name, data, w, h, bands, fmt="uchar", interp="srgb"):
    """Write a raw buffer and load it into a `.v` with an interpretation."""
    raw = os.path.join(OUT, f"{name}.raw")
    with open(raw, "wb") as f:
        f.write(data)
    v = os.path.join(OUT, f"{name}.v")
    vips("rawload", raw, v, str(w), str(h), str(bands), "--format", fmt)
    tagged = os.path.join(OUT, f"{name}-{interp}.v")
    vips("copy", v, tagged, "--interpretation", interp)
    return tagged


def opj_raw(name, samples, w, h, bands, depth, signed, subsample=None,
            resolutions=1, offset=None, ext="j2k"):
    """A codestream built by openjpeg's own CLI from a planar raw buffer.
    openjpeg's .raw reader is big-endian and PLANAR: all of component 0,
    then all of component 1, and so on."""
    width = 1 if depth <= 8 else 2
    pack = {(1, False): ">B", (1, True): ">b",
            (2, False): ">H", (2, True): ">h"}[(width, signed)]
    raw = os.path.join(OUT, f"{name}.raw")
    with open(raw, "wb") as f:
        f.write(b"".join(struct.pack(pack, v) for v in samples))
    spec = f"{w},{h},{bands},{depth},{'s' if signed else 'u'}"
    if subsample:
        spec += "@" + ":".join(f"{dx}x{dy}" for dx, dy in subsample)
    path = os.path.join(FIX, f"{name}.{ext}")
    if os.path.exists(path):
        os.remove(path)
    args = ["-i", raw, "-o", path, "-F", spec, "-n", str(resolutions)]
    if bands >= 3:
        # Leave the components alone: an MCT would make the file's samples
        # something other than what was handed in, and these fixtures exist
        # to pin the mapping from file samples to vips pixels.
        args += ["-mct", "0"]
    if offset:
        args += ["-d", f"{offset[0]},{offset[1]}"]
    opj(*args)
    return path


def fixture_record(path, what, w=None, h=None, points=None, tag=None,
                   all_fields=True):
    """The block every fixture gets: what it is, what it hashes to, what its
    codestream says about itself, what vipsheader says, what the decoded
    raster hashes to, and some pinned samples."""
    rel = os.path.relpath(path, ROOT)
    tag = tag or os.path.basename(path).replace(".", "_")
    rec = {
        "what": what,
        "fixture": rel,
        "bytes": os.path.getsize(path),
        "sha256": sha256(path),
        "magic": magic(path),
        "siz": siz(path),
        "colr": colr(path),
        "header": header(path, all_fields=all_fields),
        "decoded_raster": decoded(path, tag),
    }
    if w is not None and h is not None and w * h <= 64:
        rec["getpoint_all"] = getpoint_all(path, w, h)
    if points:
        rec["getpoint"] = {f"{x},{y}": getpoint(path, x, y)
                           for x, y in points}
    return rec


records = {}
notes = []

# ---------------------------------------------------------------------------
# 1. The saver ignores the suffix: every one of the five it advertises
#    produces the same JP2 container.
# ---------------------------------------------------------------------------
rgb_src = ramp(4, 3, 3)
rgb_v = load_raw("rgb", rgb_src, 4, 3, 3)

suffixes = {}
for ext in ("jp2", "j2k", "jpt", "j2c", "jpc"):
    out = os.path.join(OUT, f"suffix.{ext}")
    vips("jp2ksave", rgb_v, out, "--lossless")
    suffixes[ext] = {
        "bytes": os.path.getsize(out),
        "sha256": sha256(out),
        "magic": magic(out),
        "boxes": boxes(out),
    }
records["save_container_is_always_jp2"] = {
    "what": "`jp2ksave` registers five suffixes (.j2k, .jp2, .jpt, .j2c, "
            ".jpc) and writes the SAME RFC 3745 JP2 container for all of "
            "them, byte for byte: it hardcodes OPJ_CODEC_JP2. So a port "
            "must not pick a carrier from the extension on save. The "
            "loader is the opposite: it sniffs the first 12 bytes and "
            "reads a bare codestream or a JP2, ignoring the name entirely.",
    "suffixes": suffixes,
    "all_identical": len({s["sha256"] for s in suffixes.values()}) == 1,
    "load_ignores_the_name": None,   # filled in once the .j2k fixtures exist
}
notes.append(
    "Provenance: the vips version in meta is whatever `vips --version` said "
    "on the run that wrote this file, never a hardcoded string, because "
    "Homebrew moved this box from 8.18.4 to 8.18.6 mid-capture and deleted "
    "the old keg. Every number here came from the version recorded above. "
    "Both halves of the capture go through the SAME openjpeg build: vips "
    "links libopenjp2 2.5.4 and opj_compress was compiled against 2.5.4, "
    "so the fixtures opj_compress wrote and the decodes vips did agree by "
    "construction rather than by luck."
)
notes.append(
    "jp2ksave always writes a JP2 container; the .j2k/.j2c/.jpc/.jpt "
    "suffixes it advertises change nothing about the bytes. Every bare "
    "codestream fixture here therefore had to come from opj_compress."
)

# ---------------------------------------------------------------------------
# 2. Lossless is an exact identity, for RGB, RGBA and CMYK.
# ---------------------------------------------------------------------------
for name, bands, interp, w, h in (
    ("rgb_lossless", 3, "srgb", 4, 3),
    ("rgba_lossless", 4, "srgb", 4, 3),
    ("cmyk_lossless", 4, "cmyk", 4, 3),
):
    src = ramp(w, h, bands)
    tagged = load_raw(name + "_src", src, w, h, bands, interp=interp)
    path = os.path.join(FIX, f"{name}.jp2")
    vips("jp2ksave", tagged, path, "--lossless")
    rec = fixture_record(
        path,
        f"`vips jp2ksave --lossless` on an {w}x{h} {bands}-band {interp} "
        "ramp, then every pixel read back. The round trip is the identity: "
        "the bytes that went in are the bytes that come out, and the band "
        "count and interpretation survive. The reversible 5/3 wavelet is "
        "integer-specified, so a correct port reproduces this exactly "
        "rather than approximately.",
        w, h, tag=name)
    flat = [v for pixel in rec["getpoint_all"] for v in pixel]
    rec["source_bytes"] = list(src)
    rec["identity"] = flat == list(src)
    records[name] = rec
    assert rec["identity"], name

# ---------------------------------------------------------------------------
# 3. The carrier sweep: every band format jp2ksave accepts, and what comes
#    back. Two of these are the ones a port gets wrong.
# ---------------------------------------------------------------------------
carrier = {}
for fmt, width, signed in (
    ("uchar", 1, False), ("char", 1, True),
    ("ushort", 2, False), ("short", 2, True),
    ("uint", 4, False), ("int", 4, True),
    ("float", 4, None), ("double", 8, None),
):
    n = 12
    if signed is None:
        pack = "<f" if width == 4 else "<d"
        vals = [i * 1.5 for i in range(n)]
    else:
        pack = {(1, False): "<B", (1, True): "<b",
                (2, False): "<H", (2, True): "<h",
                (4, False): "<I", (4, True): "<i"}[(width, signed)]
        span = (1 << (8 * width - (1 if signed else 0))) - 1
        base = -(span + 1) // 2 if signed else 0
        vals = [base + (i * (span // n)) for i in range(n)]
    data = b"".join(struct.pack(pack, v) for v in vals)
    tagged = load_raw(f"carrier_{fmt}", data, n, 1, 1,
                      fmt=fmt, interp="b-w")
    out = os.path.join(OUT, f"carrier_{fmt}.jp2")
    if os.path.exists(out):
        os.remove(out)
    proc = vips("jp2ksave", tagged, out, "--lossless", allow_fail=True)
    entry = {"in": vals, "saved": proc.returncode == 0,
             "stderr": clean_stderr(proc.stderr)}
    if proc.returncode == 0:
        entry["siz"] = siz(out)
        entry["header"] = header_brief(out)
        entry["out"] = [p[0] for p in getpoint_all(out, n, 1)]
        entry["identity"] = entry["out"] == vals
    carrier[fmt] = entry
records["carrier_sweep"] = {
    "what": "Every libvips band format handed to `jp2ksave --lossless` as a "
            "1-band b-w 12x1 ramp, and what `jp2kload` gives back. uchar, "
            "char, ushort and short round-trip exactly. float and double "
            "are refused outright with `not an integer format`, so a port "
            "must reject rather than cast. uint and int are the trap: "
            "jp2ksave maps a 4-byte format to a precision of 31, not 32, "
            "because openjpeg cannot do more, and the result is NOT a round "
            "trip. See the thirty_one_bit record for what it does instead.",
    "formats": carrier,
}

# ---------------------------------------------------------------------------
# 4. 31-bit, characterised. This is what a port must know NOT to copy.
# ---------------------------------------------------------------------------
wide = {}
for fmt, vals in (
    ("uint", [0, 1, 1000, 65535, 1 << 20]),
    ("int", [-1000, -1, 0, 1, 1000]),
):
    pack = "<I" if fmt == "uint" else "<i"
    data = b"".join(struct.pack(pack, v) for v in vals)
    tagged = load_raw(f"wide_{fmt}", data, len(vals), 1, 1,
                      fmt=fmt, interp="grey16")
    path = os.path.join(FIX, f"{fmt}31.jp2")
    vips("jp2ksave", tagged, path, "--lossless")
    out = [getpoint(path, x, 0)[0] for x in range(len(vals))]
    wide[fmt] = {
        "fixture": os.path.relpath(path, ROOT),
        "sha256": sha256(path),
        "bytes": os.path.getsize(path),
        "siz": siz(path),
        "header": header(path, all_fields=True),
        "in": vals,
        "out": out,
        "identity": out == vals,
        "is_left_shift_1": out == [(v << 1) for v in vals],
        "is_left_shift_1_plus_dc": out == [((v + (1 << 30)) << 1) % (1 << 32)
                                           for v in vals],
    }
records["thirty_one_bit_round_trip"] = {
    "what": "`jp2ksave` gives a 4-byte libvips format a component precision "
            "of 31 (jp2ksave.c, vips_foreign_save_jp2k_new_image: "
            "'OpenJPEG only supports up to 31'), and `jp2kload` then "
            "left-justifies a 31-bit component into a 32-bit element by "
            "shifting left 1. For a SIGNED int that is all that happens, so "
            "the values come back doubled. For an UNSIGNED uint the DC "
            "level shift does not cancel and 2^30 is added first. Either "
            "way a 32-bit image does not survive a JPEG 2000 round trip "
            "through vips, and the booleans below say which of the two "
            "shapes each one has. Measured, not derived: `in` went into the "
            "binary and `out` came out of it.",
    "formats": wide,
}
notes.append(
    "A 4-byte libvips format goes to a 31-bit component, and 31 bits do not "
    "round-trip: signed comes back doubled, unsigned comes back doubled "
    "with 2^30 added. 8-bit and 16-bit are exact."
)

# ---------------------------------------------------------------------------
# 5. The colour space enum in the colr box, and the interpretation it maps
#    to. This is where the one-band sRGB file comes from.
# ---------------------------------------------------------------------------
one_src = ramp(4, 3, 1)
one_v = load_raw("onecomp", one_src, 4, 3, 1, interp="b-w")
one_base = os.path.join(OUT, "onecomp.jp2")
vips("jp2ksave", one_v, one_base, "--lossless")
three_base = os.path.join(FIX, "rgb_lossless.jp2")
wide_rgb = b"".join(struct.pack("<H", (i * 4369) % 65536)
                    for i in range(4 * 3 * 3))
wide_v = load_raw("threecomp16", wide_rgb, 4, 3, 3, fmt="ushort",
                  interp="rgb16")
wide_base = os.path.join(OUT, "threecomp16.jp2")
vips("jp2ksave", wide_v, wide_base, "--lossless")

spaces = {}
for comps, base in (("1_component_8bit", one_base),
                    ("3_components_8bit", three_base),
                    ("3_components_16bit", wide_base)):
    for enumcs, label in ((12, "CMYK"), (14, "CIELab"), (16, "sRGB"),
                          (17, "greyscale"), (18, "sYCC"), (24, "e-YCC"),
                          (99, "not a defined value")):
        out = os.path.join(OUT, f"cs_{comps}_{enumcs}.jp2")
        retag_colr(base, out, enumcs)
        proc = vips("getpoint", out, "0", "0", allow_fail=True)
        spaces[f"{comps}_enumcs{enumcs}"] = {
            "label": label,
            "header": header_brief(out),
            "pixel_0_0": ([int(float(v)) for v in proc.stdout.split()]
                          if proc.returncode == 0 else None),
            "pixel_error": (None if proc.returncode == 0
                            else clean_stderr(proc.stderr)),
        }
records["colour_space_to_interpretation"] = {
    "what": "The same two codestreams re-tagged with every enumerated "
            "colour space openjpeg recognises. The mapping is not from the "
            "band count: a ONE-BAND file tagged sRGB (EnumCS 16) comes back "
            "as `1 band, srgb`, which is what libvips' own issue412.jp2 "
            "regression fixture is, and a THREE-BAND file tagged greyscale "
            "comes back as `3 bands, b-w`. The ELEMENT WIDTH picks between "
            "the two flavours of each: the same enum that gives b-w and "
            "srgb on an 8-bit file gives grey16 and rgb16 on a 16-bit one, "
            "so the sweep runs over both. sYCC and e-YCC also turn the "
            "inverse YCC transform on, so they change the pixels as well as "
            "the label. Anything "
            "openjpeg does not recognise falls through to UNSPECIFIED, "
            "where vips guesses from the band count instead "
            "(jp2kload.c, vips_foreign_load_jp2k_get_interpretation). A "
            "port that derives the interpretation from the component count "
            "alone gets the first two of these wrong. And one combination "
            "is outright BROKEN in this build, which is worth knowing "
            "before a port copies it: a ONE-component file tagged sRGB, "
            "sYCC or e-YCC has its header expanded to 3 bands by openjpeg "
            "while the tile decode still yields 1, so `vipsheader` cheerfully "
            "reports `3 bands, srgb` and any attempt to read a pixel fails "
            "with `decoded image does not match container`. The header is "
            "not a promise the pixels will arrive.",
    "spaces": spaces,
}
notes.append(
    "The colr box's enumerated colour space, not the component count, "
    "decides the interpretation: 1 component + EnumCS 17 is b-w, but 1 "
    "component + EnumCS 12 is cmyk and 3 components + EnumCS 17 is b-w. A "
    "1-component file tagged sRGB/sYCC/e-YCC reports 3 bands in the header "
    "and then fails on any pixel read with `decoded image does not match "
    "container`."
)

# ---------------------------------------------------------------------------
# 6. Bit depths openjpeg can carry but jp2ksave cannot write. The loader
#    LEFT-JUSTIFIES them, which is the single easiest thing to get wrong.
# ---------------------------------------------------------------------------
depths = {}
for depth, signed in ((2, False), (4, False), (8, False), (10, False),
                      (12, False), (12, True), (14, False), (16, False)):
    if signed:
        vals = [-(1 << (depth - 1)), -1, 0, 1, (1 << (depth - 1)) - 1]
    else:
        vals = [0, 1, (1 << (depth - 2)), (1 << depth) - 2, (1 << depth) - 1]
    name = f"depth{depth}{'s' if signed else 'u'}"
    path = opj_raw(name, vals, len(vals), 1, 1, depth, signed)
    out = [getpoint(path, x, 0)[0] for x in range(len(vals))]
    head = header_brief(path)
    element_bits = {"uchar": 8, "char": 8, "ushort": 16, "short": 16}[
        head["format"]]
    shift = element_bits - depth
    depths[name] = {
        "fixture": os.path.relpath(path, ROOT),
        "sha256": sha256(path),
        "bytes": os.path.getsize(path),
        "siz": siz(path),
        "header": head,
        "in": vals,
        "out": out,
        "identity": out == vals,
        "is_left_justified": out == [(v << shift) for v in vals],
    }
records["bit_depth_is_left_justified"] = {
    "what": "Component precisions from 2 to 16 bits, written by openjpeg's "
            "own opj_compress because jp2ksave can only emit 8, 16 and 31. "
            "`jp2kload` picks uchar for prec <= 8 and ushort for prec <= "
            "16, signed variants for a signed component, and then SHIFTS "
            "THE SAMPLE LEFT to fill the element "
            "(jp2kload.c, vips_foreign_load_jp2k_ljust: shift = "
            "sizeof(element) * 8 - prec). So a 12-bit sample of 4095 comes "
            "back as 65520, not 4095 and not 65535, and the true depth "
            "survives only in `bits-per-sample`. Depths 8 and 16 shift by "
            "zero and are the identity, which is exactly why a port that "
            "forgets the shift still passes an 8-bit test.",
    "depths": depths,
}
notes.append(
    "jp2kload left-justifies: a prec-N component is shifted left by "
    "(element_bits - N). bits-per-sample carries the real depth. Only 8 and "
    "16 are unshifted, so an 8-bit-only test cannot catch this."
)

# ---------------------------------------------------------------------------
# 7. Chroma subsampling on save. Invisible on grey, so this uses colour.
# ---------------------------------------------------------------------------
grad_v = load_raw("grad16", gradient(16), 16, 16, 3)
sub_modes = {}
for mode in ("off", "on", "auto"):
    for Q in (48, 90):
        out = os.path.join(OUT, f"sub_{mode}_{Q}.jp2")
        vips("jp2ksave", grad_v, out, "--subsample-mode", mode, "--Q", str(Q))
        row = [getpoint(out, x, 0) for x in range(6)]
        sub_modes[f"{mode}_Q{Q}"] = {
            "bytes": os.path.getsize(out),
            "sha256": sha256(out),
            "siz": siz(out),
            "colr": colr(out),
            "row0": row,
            "chroma_shared_in_pairs": all(
                row[i][1:] == row[i + 1][1:] for i in (0, 2, 4)),
        }
ll_sub = os.path.join(OUT, "sub_lossless_on.jp2")
vips("jp2ksave", grad_v, ll_sub, "--lossless", "--subsample-mode", "on")
records["chroma_subsample_on_save"] = {
    "what": "`--subsample-mode` across `off`, `on` and `auto` at Q 48 and "
            "Q 90, on a 16x16 image whose chroma varies along x. `off` "
            "leaves all three components at dx=dy=1. `on` halves components "
            "1 and 2 and moves the colr box from EnumCS 16 (sRGB) to "
            "EnumCS 18 (sYCC), which is what makes the loader run the "
            "inverse YCC on the way back. `auto` subsamples only when "
            "Q < 90 AND the image is 3-band sRGB or RGB16, so auto at Q 90 "
            "is byte-identical to off at Q 90 and auto at Q 48 to on at "
            "Q 48; auto_gate_by_image_shape below measures the other half "
            "of that condition. A grey or 4-band fixture would show none of "
            "this, which is the point: the mode is only observable on "
            "3-band colour.",
    "modes": sub_modes,
    "auto_at_Q90_equals_off": (sub_modes["auto_Q90"]["sha256"]
                               == sub_modes["off_Q90"]["sha256"]),
    "auto_at_Q48_equals_on": (sub_modes["auto_Q48"]["sha256"]
                              == sub_modes["on_Q48"]["sha256"]),
    "lossless_forces_subsample_off": {
        "what": "`--lossless --subsample-mode on` does NOT subsample: the "
                "build step overwrites the mode with OFF before it is "
                "read. So the flag combination is silently ignored rather "
                "than refused, and the file is identical to a plain "
                "lossless save.",
        "siz": siz(ll_sub),
        "colr": colr(ll_sub),
        "sha256": sha256(ll_sub),
        "equals_plain_lossless": None,  # filled in below
    },
}
plain_ll = os.path.join(OUT, "sub_lossless_off.jp2")
vips("jp2ksave", grad_v, plain_ll, "--lossless")
records["chroma_subsample_on_save"]["lossless_forces_subsample_off"][
    "equals_plain_lossless"] = sha256(ll_sub) == sha256(plain_ll)

# The other half of auto's gate is the SHAPE of the image, not just Q, so
# measure that too rather than taking the C's word for it.
auto_gate = {}
for label, bands, interp, data in (
    ("1_band_bw", 1, "b-w", ramp(16, 16, 1)),
    ("3_band_srgb", 3, "srgb", gradient(16)),
    ("4_band_srgb", 4, "srgb", ramp(16, 16, 4)),
    ("3_band_multiband", 3, "multiband", gradient(16)),
    ("4_band_cmyk", 4, "cmyk", ramp(16, 16, 4)),
):
    tagged = load_raw(f"auto_{label}", data, 16, 16, bands, interp=interp)
    out = os.path.join(OUT, f"auto_{label}.jp2")
    vips("jp2ksave", tagged, out, "--Q", "48", "--subsample-mode", "auto")
    auto_gate[label] = {
        "siz": siz(out),
        "colr": colr(out),
        "subsampled": any(c["dx"] != 1 or c["dy"] != 1
                          for c in siz(out)["components"]),
    }
records["chroma_subsample_on_save"]["auto_gate_by_image_shape"] = {
    "what": "`auto` at Q 48 across five image shapes. Only 3-band sRGB or "
            "RGB16 subsamples: a 1-band b-w, a 4-band sRGB, a 4-band CMYK "
            "and even a 3-band multiband image all come out at dx=dy=1. So "
            "a port cannot implement auto as `Q < 90` alone.",
    "shapes": auto_gate,
}

for name, args in (("chroma_sub_off", ["--subsample-mode", "off"]),
                   ("chroma_sub_on", ["--subsample-mode", "on"])):
    path = os.path.join(FIX, f"{name}.jp2")
    vips("jp2ksave", grad_v, path, "--Q", "90", *args)
    records[name] = fixture_record(
        path,
        "The committed 16x16 pair the subsample records above are measured "
        "from, saved at Q 90 with the mode forced. Same source raster, "
        "different chroma geometry, so a port that ignores dx/dy decodes "
        "one of the two into visibly wrong colour.",
        points=[(0, 0), (1, 0), (2, 0), (3, 0), (0, 1), (15, 15)],
        tag=name)

# A degenerate size, where subsampling collapses the chroma entirely.
tiny_v = load_raw("tiny", bytes([
    255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 0,
    0, 255, 255, 255, 0, 255, 0, 0, 0, 255, 255, 255,
]), 4, 2, 3)
tiny = os.path.join(FIX, "chroma_tiny_sub_on.jp2")
vips("jp2ksave", tiny_v, tiny, "--Q", "90", "--subsample-mode", "on")
rec = fixture_record(
    tiny,
    "The same subsampling on a 4x2 image of saturated primaries. At this "
    "size the halved chroma components are 2x1 and the encode throws them "
    "away entirely, so every pixel comes back NEUTRAL GREY: red, green, "
    "blue and yellow all decode to r == g == b. The luma survives. This is "
    "not a decoder bug to reproduce so much as a boundary a port has to "
    "know exists, because a test that subsamples a tiny fixture and "
    "compares against the source will fail for a reason that has nothing "
    "to do with the code under test.",
    4, 2, tag="chroma_tiny")
rec["all_pixels_neutral"] = all(p[0] == p[1] == p[2]
                                for p in rec["getpoint_all"])
records["chroma_tiny_sub_on"] = rec

# ---------------------------------------------------------------------------
# 8. Subsampling in the FILE, which is the loader's side of the same coin.
# ---------------------------------------------------------------------------
# LUMA IS DELIBERATELY FLAT. With a varying luma every decoded pixel
# differs and nothing about the upsampler is provable from the numbers. With
# a flat luma, nearest-neighbour replication makes each 2x2 block one solid
# colour and any interpolating upsampler makes it a gradient, so the
# distinction is right there in the pixels.
luma = [128] * (8 * 4)
cb = [16 + i * 30 for i in range(8)]
cr = [240 - i * 30 for i in range(8)]
sub420 = opj_raw("sub420", luma + cb + cr, 8, 4, 3, 8, False,
                 subsample=[(1, 1), (2, 2), (2, 2)])
rec = fixture_record(
    sub420,
    "An 8x4 bare codestream whose components 1 and 2 are halved in both "
    "axes, written by opj_compress with no MCT and no declared colour "
    "space. Two behaviours in one fixture. jp2kload treats UNSPECIFIED + 3 "
    "components + subsampling on bands 1 and 2 as YCC and runs an inverse "
    "YCC->RGB (jp2kload.c, vips_foreign_load_jp2k_get_ycc), so the samples "
    "in the file are not the pixels that come out. And it upsamples the "
    "halved components by PIXEL REPLICATION, planes[i][x / dx], rather than "
    "interpolating. The luma plane is flat at 128 precisely so the second "
    "one is provable: every 2x2 block decodes to a single solid colour, "
    "which an interpolating upsampler (libwebp's fancy upsampler, say) "
    "would turn into a gradient.",
    8, 4, tag="sub420")
rec["source_planes"] = {"Y": luma, "Cb": cb, "Cr": cr}
rec["blocks_are_flat"] = None
records["file_level_subsampling"] = rec

# ---------------------------------------------------------------------------
# 9. The loader's half of record 1: it sniffs, it does not look at the name.
# ---------------------------------------------------------------------------
carriers = {}
bare = os.path.join(FIX, "depth8u.j2k")
with open(bare, "rb") as f:
    bare_bytes = f.read()
for name in ("copy.j2k", "copy.jp2", "copy.jpt", "copy.png", "copy"):
    out = os.path.join(OUT, name)
    with open(out, "wb") as f:
        f.write(bare_bytes)
    carriers[name] = {"header": header(out),
                      "decoded_raster": decoded(out, f"carrier_{name}")}
records["save_container_is_always_jp2"]["load_ignores_the_name"] = {
    "what": "One bare codestream (fixtures/depth8u.j2k) copied under five "
            "names, including a .png and one with no extension at all. "
            "jp2kload takes every one of them and decodes to the same "
            "raster, because vips picks the loader by sniffing the first 12 "
            "bytes (jp2kload.c, vips_foreign_load_jp2k_get_codec_format: "
            "the RFC 3745 signature, the short JP2 magic, or the SOC+SIZ "
            "pair of a raw codestream) and never consults the filename.",
    "names": carriers,
    "all_decode_identically": len(
        {c["decoded_raster"]["sha256"] for c in carriers.values()}) == 1,
}

# ---------------------------------------------------------------------------
# 10. Tiling, including a size that is not a multiple of the tile.
# ---------------------------------------------------------------------------
W, H = 37, 21
grey = bytes(((x * 7 + y * 13) % 251) for y in range(H) for x in range(W))
grey_v = load_raw("grey37", grey, W, H, 1, interp="b-w")

tiles = {}
for tw, th in ((512, 512), (16, 16), (8, 8), (16, 7)):
    out = os.path.join(OUT, f"tile_{tw}x{th}.jp2")
    vips("jp2ksave", grey_v, out, "--lossless",
         "--tile-width", str(tw), "--tile-height", str(th))
    head = header(out, all_fields=True)
    tiles[f"{tw}x{th}"] = {
        "bytes": os.path.getsize(out),
        "sha256": sha256(out),
        "siz": siz(out),
        "tile_grid": [-(-W // tw), -(-H // th)],
        "tile_width_field": head.get("tile-width"),
        "tile_height_field": head.get("tile-height"),
        "decoded_raster": decoded(out, f"tile_{tw}x{th}"),
    }
records["tiling"] = {
    "what": "A 37x21 image, whose dimensions are a multiple of no tile size "
            "here, saved at four tile geometries. Two things to pin. First, "
            "`tile-width` and `tile-height` appear in the header ONLY when "
            "the image is more than one tile: at the default 512 the whole "
            "image is one tile and the fields are ABSENT, so a port that "
            "always sets them disagrees with vips on every small image. "
            "Second, the decoded raster is byte-identical at every tile "
            "geometry, so the partial tiles down the right and bottom edges "
            "carry no seam. Non-square tiles work too.",
    "image": [W, H],
    "geometries": tiles,
    "all_decode_identically": len(
        {t["decoded_raster"]["sha256"] for t in tiles.values()}) == 1,
}

tiled = os.path.join(FIX, "grey_tile8.jp2")
vips("jp2ksave", grey_v, tiled, "--lossless",
     "--tile-width", "8", "--tile-height", "8")
rec = fixture_record(
    tiled,
    "The committed 37x21 fixture on an 8x8 tile grid: 5 by 3 tiles, with "
    "the right column 5 wide and the bottom row 5 tall. The pinned points "
    "sit on both sides of the first vertical tile boundary, on the last "
    "full column and in the partial corner.",
    points=[(7, 0), (8, 0), (0, 7), (0, 8), (31, 20), (36, 20), (36, 0)],
    tag="grey_tile8")
rec["source_bytes_sha256"] = hashlib.sha256(grey).hexdigest()
rec["decoded_matches_source"] = (
    rec["decoded_raster"].get("sha256") == rec["source_bytes_sha256"])
records["grey_tile8"] = rec

# ---------------------------------------------------------------------------
# 11. Lossy: a Q sweep, and where it stops moving.
# ---------------------------------------------------------------------------
quality = {}
for Q in (1, 25, 48, 75, 90, 100):
    out = os.path.join(OUT, f"q{Q}.jp2")
    vips("jp2ksave", grad_v, out, "--Q", str(Q))
    quality[str(Q)] = {
        "bytes": os.path.getsize(out),
        "sha256": sha256(out),
        "getpoint": {f"{x},{y}": getpoint(out, x, y)
                     for x, y in ((0, 0), (8, 8), (15, 15))},
    }
lossy = os.path.join(FIX, "rgb_lossy_q48.jp2")
vips("jp2ksave", rgb_v, lossy, "--Q", "48")
rec = fixture_record(
    lossy,
    "The default save: no --lossless, so Q defaults to 48 and the "
    "irreversible 9/7 wavelet is used. Same 4x3 ramp as rgb_lossless.jp2, "
    "and the pixels differ from it. The 9/7 path is float-specified, so "
    "this is the one place a port may legitimately need a tolerance rather "
    "than exact equality, and pinning it is how that tolerance gets chosen "
    "from evidence instead of taste.",
    4, 3, tag="rgb_lossy_q48")
rec["source_bytes"] = list(rgb_src)
rec["differs_from_source"] = [
    v for p in rec["getpoint_all"] for v in p] != list(rgb_src)
records["rgb_lossy_q48"] = rec
records["quality_sweep"] = {
    "what": "Q from 1 to 100 on the 16x16 gradient. Q sets a distortion "
            "ratio per resolution (Q + 10 * i over 7 layers), not a "
            "quantiser, and on an image this small it saturates: 75, 90 and "
            "100 are byte-identical. The default is 48, chosen upstream to "
            "land near JPEG Q 75 in size.",
    "levels": quality,
    "top_three_identical": len({quality[q]["sha256"]
                                for q in ("75", "90", "100")}) == 1,
}

# ---------------------------------------------------------------------------
# 12. `page` is a resolution level. `n-pages` is the resolution count.
# ---------------------------------------------------------------------------
res_src = [((x * 7 + y * 11) % 256) for y in range(24) for x in range(32)]
res3 = opj_raw("res3", res_src, 32, 24, 1, 8, False, resolutions=3)
pages = {}
for page in (0, 1, 2, 3):
    ref = f"{res3}[page={page}]"
    entry = {"header": header(ref)}
    if "error" not in entry["header"]:
        entry["header_all"] = header(ref, all_fields=True)
        entry["decoded_raster"] = decoded(ref, f"res3_page{page}")
        entry["getpoint_0_0"] = getpoint(ref, 0, 0)
    pages[str(page)] = entry
rec = fixture_record(
    res3,
    "A 32x24 codestream with three resolution levels. `n-pages` is 3, but "
    "there is only ONE image in this file: jp2kload maps numresolutions to "
    "n-pages (jp2kload.c, set_header) and `page` to opj's cp_reduce, so "
    "[page=1] is the same picture at half size and [page=2] at a quarter. "
    "A port that reads n-pages as a frame count, the way it means in GIF or "
    "WebP or a multi-page TIFF, is wrong here in a way that only shows up "
    "on a file with more than one resolution, which no default jp2ksave of "
    "a small image produces. Asking for a page past the end is an error, "
    "not a clamp.",
    points=[(0, 0), (31, 23)],
    tag="res3")
rec["pages"] = pages
records["page_is_a_resolution_level"] = rec
notes.append(
    "jp2kload's `page` is a resolution level and `n-pages` is the "
    "resolution count, not a frame count. jp2ksave's numresolution is "
    "max(1, log2(min(width, height)) - 5), so every fixture small enough to "
    "commit has exactly one resolution and hides this."
)

# ---------------------------------------------------------------------------
# 13. A non-zero image origin becomes a negative Xoffset/Yoffset.
# ---------------------------------------------------------------------------
origin = opj_raw("origin57", res_src, 32, 24, 1, 8, False, offset=(5, 7))
records["image_origin_offset"] = fixture_record(
    origin,
    "The same raster with the codestream's image origin at (5, 7). The JPEG "
    "2000 grid lets the image start away from the origin; openjpeg reports "
    "x0/y0 and clips, so vips sees a 27x17 image and records the origin as "
    "a NEGATIVE Xoffset and Yoffset (jp2kload.c, set_header: Xoffset = "
    "-round(x0 / shrink)). The offset is in unshrunk coordinates and is "
    "divided by the page shrink, so it interacts with [page=N]. Nothing "
    "jp2ksave writes ever exercises this, since it always starts at 0.",
    points=[(0, 0), (26, 16)],
    tag="origin57")

# ---------------------------------------------------------------------------
# 14. Metadata. The saver writes none of it; the loader reads only ICC.
# ---------------------------------------------------------------------------
ICC_PROBE = ("/Users/rom/workspace/libvips/test/test-suite/images/sRGB.icm")
save_meta = {}
if os.path.exists(ICC_PROBE):
    out = os.path.join(OUT, "profile.jp2")
    vips("jp2ksave", grad_v, out, "--lossless", "--profile", ICC_PROBE)
    plain = os.path.join(OUT, "profile_none.jp2")
    vips("jp2ksave", grad_v, plain, "--lossless")
    save_meta["profile"] = {
        "profile_used": ICC_PROBE,
        "profile_bytes": os.path.getsize(ICC_PROBE),
        "boxes": boxes(out),
        "header": header(out, all_fields=True),
        "identical_to_save_without_profile": sha256(out) == sha256(plain),
    }
for keep in ("all", "none", "icc"):
    out = os.path.join(OUT, f"keep_{keep}.jp2")
    vips("jp2ksave", grad_v, out, "--lossless", "--keep", keep)
    save_meta[f"keep_{keep}"] = {"sha256": sha256(out),
                                 "bytes": os.path.getsize(out)}

# A JP2 that really does carry a profile, built by rewriting the colr box of
# one jp2ksave wrote. jp2ksave cannot attach one, and neither can
# opj_compress from a raw input, so this is the only way to reach the
# loader's ICC path. The profile payload is a 24-byte marker rather than a
# real profile: vips copies the colr payload out verbatim without parsing
# it, which is itself worth pinning.
base = os.path.join(OUT, "iccbase.jp2")
vips("jp2ksave", rgb_v, base, "--lossless")
with open(base, "rb") as f:
    raw = f.read()
colr_off = colr_len = jp2h_off = jp2h_len = None
for kind, length, off in boxes(base):
    if kind == "jp2h":
        jp2h_off, jp2h_len = off, length
    if kind == "colr":
        colr_off, colr_len = off, length
profile = bytes(range(0x10, 0x28))
new_colr = struct.pack(">I", 8 + 3 + len(profile)) + b"colr" \
    + bytes([2, 0, 0]) + profile
xmp = b'<x:xmpmeta xmlns:x="adobe:ns:meta/"/>'
uuid_xmp = struct.pack(">I", 8 + 16 + len(xmp)) + b"uuid" \
    + bytes.fromhex("be7acfcb97a942e89c71999491e3afac") + xmp
inner = raw[jp2h_off + 8:colr_off] + new_colr \
    + raw[colr_off + colr_len:jp2h_off + jp2h_len]
icc_path = os.path.join(FIX, "icc_colr.jp2")
with open(icc_path, "wb") as f:
    f.write(raw[:jp2h_off]
            + struct.pack(">I", 8 + len(inner)) + b"jp2h" + inner
            + uuid_xmp
            + raw[jp2h_off + jp2h_len:])
COMMANDS.append("# fixtures/icc_colr.jp2 is assembled by this script, not "
                "by vips: jp2ksave cannot attach a profile at all, so the "
                "colr box is rewritten to METH=2 and a uuid XMP box is "
                "appended by hand")
records["metadata"] = {
    "what": "What survives a JPEG 2000 round trip through vips: nothing on "
            "the way out, and an ICC profile on the way in. `jp2ksave` "
            "inherits `--profile` and `--keep` from VipsForeignSave and "
            "implements NEITHER: there is no icc, exif, xmp or iptc code in "
            "jp2ksave.c at all, and saving with a profile produces a file "
            "byte-identical to saving without one. So `--keep all` and "
            "`--keep none` are the same file too. `jp2kload` does read an "
            "ICC profile, out of a METH=2 colr box, and copies the payload "
            "verbatim into `icc-profile-data` without validating it. A uuid "
            "XMP box beside it is ignored.",
    "save": save_meta,
    "load": fixture_record(
        icc_path,
        "A jp2ksave output whose colr box has been rewritten to METH=2 with "
        "a 24-byte payload, plus an appended uuid XMP box. vips reports "
        "icc-profile-data of exactly 24 bytes and no xmp-data.",
        4, 3, tag="icc_colr"),
    "injected_profile": list(profile),
    "injected_xmp": list(xmp),
}
records["metadata"]["load"]["icc_field"] = \
    records["metadata"]["load"]["header"].get("icc-profile-data")
records["metadata"]["load"]["xmp_field"] = \
    records["metadata"]["load"]["header"].get("xmp-data")
notes.append(
    "jp2ksave writes no metadata whatsoever: --profile and --keep are "
    "inherited no-ops and change no byte of the output. jp2kload reads an "
    "ICC profile and nothing else."
)

# ---------------------------------------------------------------------------
# 15. Malformed and truncated input, cut at structural boundaries so the
#     four failure modes are reproducible rather than accidental.
# ---------------------------------------------------------------------------
with open(os.path.join(FIX, "rgb_lossless.jp2"), "rb") as f:
    good = f.read()
jp2c_off = None
for kind, length, off in boxes(os.path.join(FIX, "rgb_lossless.jp2")):
    if kind == "jp2c":
        jp2c_off, jp2c_len = off, length

broken = {
    "truncated_at_codestream": good[:jp2c_off + 8],
    "truncated_in_siz": good[:jp2c_off + 8 + 6],
    "truncated_in_tile": good[:jp2c_off + 8 + (jp2c_len - 8) * 9 // 10],
    "truncated_in_boxes": good[:jp2c_off - 4],
    "zeroed_body": good[:12] + bytes(len(good) - 12),
    "not_jp2k": b"NOT A JPEG 2000 FILE, NOT AT ALL",
}
malformed = {}
for name, data in broken.items():
    path = os.path.join(FIX, f"{name}.jp2" if name != "not_jp2k"
                        else "not_jp2k.bin")
    with open(path, "wb") as f:
        f.write(data)
    entry = {
        "fixture": os.path.relpath(path, ROOT),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "header": header_brief(path),
        "fail_on": {},
        "decode": None,
    }
    # `vips avg` forces the whole raster through the decoder, which
    # vipsheader does not. The default run is recorded in full; the four
    # explicit fail-on levels only say whether they matched it, because
    # writing the same three-line message out five times is not evidence,
    # it is repetition.
    proc = vips("avg", path, allow_fail=True)
    base = {"rc": proc.returncode, "stdout": proc.stdout.strip(),
            "stderr": clean_stderr(proc.stderr)}
    entry["decode"] = base
    for fail_on in ("none", "truncated", "error", "warning"):
        proc = vips("avg", f"{path}[fail-on={fail_on}]", allow_fail=True)
        entry["fail_on"][fail_on] = (
            proc.returncode == base["rc"]
            and clean_stderr(proc.stderr) == base["stderr"])
    entry["fail_on_changes_nothing"] = all(entry["fail_on"].values())
    malformed[name] = entry
COMMANDS.append("# the malformed fixtures are cut from "
                "fixtures/rgb_lossless.jp2 at box and marker boundaries by "
                "this script, so each one reaches a different check")
records["malformed_and_truncated"] = {
    "what": "Six broken inputs, each cut at a structural boundary so the "
            "failure it reaches is reproducible. Three things to pin. "
            "First, the failures are DISTINCT and named: a missing "
            "codestream gives `Expected a SOC marker`, a cut inside SIZ "
            "gives `Stream too short`, a cut inside a box gives an `Invalid "
            "box size` message quoting both sizes, and total garbage never "
            "reaches jp2kload at all because the sniffer rejects it as `not "
            "a known file format`. Second, a file cut inside the tile data "
            "has a PERFECTLY READABLE HEADER and fails only when the pixels "
            "are pulled, so header success is not decode success. Third, "
            "and this is the trap: `fail-on` makes no difference to any of "
            "them. jp2kload never produces a partial image, so `fail-on=none` "
            "and `fail-on=warning` behave identically and a port must not "
            "wire that option up to anything.",
    "cases": malformed,
    "fail_on_is_inert_everywhere": all(
        c["fail_on_changes_nothing"] for c in malformed.values()),
}
notes.append(
    "fail-on has no observable effect on jp2kload: every level, from none "
    "to warning, gives the same return code and the same message on every "
    "malformed fixture here."
)

# ---------------------------------------------------------------------------
# Finish: the block-flatness boolean for record 8 reads the pixels record 8
# already pinned, so it costs no extra vips calls.
# ---------------------------------------------------------------------------
px = records["file_level_subsampling"]["getpoint_all"]
records["file_level_subsampling"]["blocks_are_flat"] = all(
    px[y * 8 + x] == px[y * 8 + x + 1] == px[(y + 1) * 8 + x]
    == px[(y + 1) * 8 + x + 1]
    for y in (0, 2) for x in (0, 2, 4, 6))

version = run([VIPS, "--version"]).stdout.strip()
config = run([VIPS, "--vips-config"]).stdout
jp2k_config = [part.strip() for part in config.replace("\n", ",").split(",")
               if "JPEG2000" in part]
opj_version = [line for line in
               run([OPJ, "-h"], allow_fail=True).stdout.splitlines()
               if "openjp2 library" in line]
linked = run(["otool", "-L", VIPSLIB], allow_fail=True)
libopenjp2 = [line.strip() for line in linked.stdout.splitlines()
              if "openjp" in line]

oracle = {
    "meta": {
        "area": "foreign-jp2k",
        "issue": 637,
        "vips_version": version,
        "vips_binary": VIPS,
        "vips_config_jp2k": jp2k_config,
        "opj_compress_binary": OPJ,
        "opj_compress_version": opj_version,
        "libopenjp2_linked_by_vips": libopenjp2,
        "captured_by": "oracle-captures/foreign-jp2k/capture.py",
        "reference_c": "libvips v8.18.0-95-gfe420cf3a "
                       "(/Users/rom/workspace/libvips at fe420cf3a) for the "
                       "jp2kload.c and jp2ksave.c function names quoted in "
                       "the records; that tree is AHEAD of the installed "
                       "release recorded in vips_version above, so it "
                       "is cited for names only and every number here came "
                       "out of the binary",
    },
    "notes": notes,
    "records": records,
}
with open(os.path.join(ROOT, "oracle.json"), "w") as f:
    f.write(encode(oracle) + "\n")

with open(os.path.join(ROOT, "commands.sh"), "w") as f:
    f.write("#!/bin/sh\n")
    f.write("# Every command capture.py ran, in order. Regenerate with\n")
    f.write("# `python3 capture.py` from this directory.\n")
    f.write("set -e\n\n")
    for c in COMMANDS:
        f.write(c + "\n")

print(f"{len(records)} records, {len(COMMANDS)} commands, "
      f"{len(os.listdir(FIX))} fixtures")
