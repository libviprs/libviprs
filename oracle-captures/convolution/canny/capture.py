#!/usr/bin/env python3
"""
Oracle capture for `vips canny` (issues #511, #559, #560).

Runs the real vips CLI over synthetic fixtures and records exact
input -> output behaviour for the whole operation and for each of its four
internal stages, on both the vectorised and the portable-C (VIPS_NOVECTOR=1)
libvips paths.

Everything the script needs it generates itself, so a re-run on the same
vips build reproduces oracle.json byte for byte. The pseudo-random fixtures
use a hand-rolled LCG rather than Python's `random`, so the bytes do not
depend on the CPython version. Which vips build "the same" means is no longer
a matter of remembering: oracle-captures/ORACLE_PIN.json names it and this
script refuses to run against anything else unless passed --repin (#650).

Writes, next to this file:
  commands.sh  - every vips CLI command actually executed, in order
  oracle.json  - meta + facts + records
  fixtures/    - the generated inputs (committed) and outputs/ (regenerated)

Run it from anywhere; every path it uses is relative to this file.
"""
import hashlib
import json
import os
import struct
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
FIX = os.path.join(ROOT, "fixtures")
MASKS = os.path.join(FIX, "masks")
OUT = os.path.join(FIX, "outputs")
VIPS = "/opt/homebrew/bin/vips"
VIPSHEADER = "/opt/homebrew/bin/vipsheader"

AREA = "convolution/canny"

# Issue #650. The pin lives one level up, next to the other capture areas.
sys.path.insert(0, os.path.abspath(os.path.join(ROOT, os.pardir, os.pardir)))
import oracle_pin  # noqa: E402  (needs the path above)

VIPS_VERSION, ORACLE_PIN = oracle_pin.check(AREA, VIPS)

for d in (FIX, MASKS, OUT):
    os.makedirs(d, exist_ok=True)

COMMANDS = []


# ---------------------------------------------------------------------------
# plumbing
# ---------------------------------------------------------------------------
def sh_quote(s):
    if s and all(c.isalnum() or c in "._/-=" for c in s):
        return s
    return "'" + s.replace("'", "'\\''") + "'"


def rel(p):
    return os.path.relpath(p, ROOT)


def strip_glib_prefix(text):
    """Drop GLib's "(vips:PID): DOMAIN-LEVEL **: HH:MM:SS.mmm: " prefix.

    The pid and the clock are the only non-reproducible bytes in this capture.
    """
    line = text.strip().splitlines()[-1] if text.strip() else ""
    marker = "**: "
    if line.startswith("(vips:") and marker in line:
        line = line.split(marker, 1)[1]
        if line[:2].isdigit() and ": " in line:
            line = line.split(": ", 1)[1]
    return line[:160]


def run(cmd, novector=False, log=True):
    """Run a vips command. cmd paths are absolute; the log records them relative."""
    env = dict(os.environ)
    env.pop("VIPS_NOVECTOR", None)
    if novector:
        env["VIPS_NOVECTOR"] = "1"
    if log:
        pretty = " ".join(sh_quote(rel(c) if os.path.isabs(c) else c) for c in cmd)
        COMMANDS.append(("VIPS_NOVECTOR=1 " if novector else "") + pretty)
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT, env=env)
    if res.returncode != 0:
        raise RuntimeError("failed: %s\n%s" % (" ".join(cmd), res.stderr))
    return res.stdout.strip()


def header(path):
    out = run([VIPSHEADER, "-a", path], log=False)
    d = {}
    for line in out.splitlines()[1:]:
        if ":" in line:
            k, _, v = line.partition(":")
            d[k.strip()] = v.strip()
    return {
        "width": int(d.get("width", -1)),
        "height": int(d.get("height", -1)),
        "bands": int(d.get("bands", -1)),
        "format": d.get("format", "?"),
        "interpretation": d.get("interpretation", "?"),
    }


FMT = {
    "uchar": ("B", 1),
    "char": ("b", 1),
    "ushort": ("H", 2),
    "short": ("h", 2),
    "uint": ("I", 4),
    "int": ("i", 4),
    "float": ("f", 4),
    "double": ("d", 8),
}


def pixels(path):
    """Every sample of `path`, in raster order, exact. Returns (hdr, flat list)."""
    h = header(path)
    raw_path = path + ".raw"
    if os.path.exists(raw_path):
        os.remove(raw_path)
    run([VIPS, "rawsave", path, raw_path], log=False)
    with open(raw_path, "rb") as f:
        raw = f.read()
    code, size = FMT[h["format"]]
    vals = list(struct.unpack("<%d%s" % (len(raw) // size, code), raw))
    return h, vals, hashlib.sha256(raw).hexdigest()


def rows(h, vals):
    """[y][x] -> sample (1 band) or [samples] (n bands)."""
    w, ht, b = h["width"], h["height"], h["bands"]
    out = []
    for y in range(ht):
        row = []
        for x in range(w):
            px = vals[(y * w + x) * b:(y * w + x) * b + b]
            row.append(px[0] if b == 1 else px)
        out.append(row)
    return out


def extreme(h, vals, want_max):
    """First occurrence, in raster order, of the extreme sample.

    `vips min --x --y` reports A position, not the first one, and which tie it
    lands on moves between runs, so the position is recomputed here to keep
    oracle.json reproducible. The VALUE still comes from the binary.
    """
    target = max(vals) if want_max else min(vals)
    i = vals.index(target)
    w, b = h["width"], h["bands"]
    return {"value": target, "x": (i // b) % w, "y": (i // b) // w, "band": i % b}


def stats(path, h, vals):
    return {
        "avg": float(run([VIPS, "avg", path], log=False)),
        "vips_min": float(run([VIPS, "min", path], log=False)),
        "vips_max": float(run([VIPS, "max", path], log=False)),
        "min": extreme(h, vals, False),
        "max": extreme(h, vals, True),
    }


def describe(path, dump):
    """Header + stats + raw sha + (optionally) the full pixel grid."""
    h, vals, sha = pixels(path)
    d = {"path": rel(path), **h, "raw_sha256": sha, **stats(path, h, vals)}
    if dump:
        d["pixels"] = rows(h, vals)
    return d, h, vals


# ---------------------------------------------------------------------------
# fixtures. Everything below is generated, nothing is downloaded.
# ---------------------------------------------------------------------------
def lcg_bytes(n, seed):
    """Deterministic pseudo-random bytes, independent of the Python version."""
    x = seed & 0x7FFFFFFF
    out = bytearray(n)
    for i in range(n):
        x = (1103515245 * x + 12345) & 0x7FFFFFFF
        out[i] = (x >> 16) & 0xFF
    return bytes(out)


FIXTURE_DOCS = {}


def write_pnm(name, w, h, bands, data, doc):
    magic = b"P5" if bands == 1 else b"P6"
    ext = ".pgm" if bands == 1 else ".ppm"
    path = os.path.join(FIX, name + ext)
    with open(path, "wb") as f:
        f.write(b"%s\n%d %d\n255\n" % (magic, w, h))
        f.write(bytes(bytearray(data)))
    with open(path, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    FIXTURE_DOCS[name] = {
        "path": rel(path), "width": w, "height": h, "bands": bands,
        "format": "uchar", "sha256": sha, "doc": doc,
    }
    COMMANDS.append("# fixture %s: %s" % (rel(path), doc))
    return path


def write_mat(name, w, h, scale, offset, rows_):
    path = os.path.join(MASKS, name + ".mat")
    lines = ["%d %d %g %g" % (w, h, scale, offset)]
    lines += [" ".join(str(v) for v in r) for r in rows_]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    COMMANDS.append("cat > %s << 'EOF'\n%s\nEOF" % (rel(path), "\n".join(lines)))
    return path


# The two gradient masks canny builds internally (canny.c:75-85). The 2x2
# is -1/+1 in x; Gy is vips_rot90 of it, which is [[-1,-1],[1,1]] (measured,
# `vips rot m.mat out.mat d90`). rot90 carries the "offset" metadata across.
MASK_GX_OFF = write_mat("gx_offset128", 2, 2, 1, 128, [[-1, 1], [-1, 1]])
MASK_GY_OFF = write_mat("gy_offset128", 2, 2, 1, 128, [[-1, -1], [1, 1]])
MASK_GX = write_mat("gx", 2, 2, 1, 0, [[-1, 1], [-1, 1]])
MASK_GY = write_mat("gy", 2, 2, 1, 0, [[-1, -1], [1, 1]])

# 1. hard vertical black/white step
STEP = write_pnm(
    "step9", 9, 9, 1,
    [0 if x < 4 else 255 for y in range(9) for x in range(9)],
    "9x9 uchar, columns 0-3 = 0 and 4-8 = 255. A pure Gx edge, the simplest "
    "non-trivial case, and the reference for the border behaviour.")

# 2. white block in the top-left corner: engineered so gx and gy both clip to
#    -128 at (4,4), which is the ONLY way to reach the uchar G ceiling of 64.
SQUARE = write_pnm(
    "square9", 9, 9, 1,
    [255 if (x < 4 and y < 4) else 0 for y in range(9) for x in range(9)],
    "9x9 uchar, 4x4 white block at the top-left. Its bottom-right corner "
    "(4,4) drives both gradient convs into their negative clip, so the uchar "
    "polar stage reaches its maximum G of 64 there.")

# 3/4/5/6. gradient plateaus: two adjacent pixels with IDENTICAL G, which is
#    what makes the `G <= low || G < high` asymmetry observable.
RAMP = [0, 0, 0, 0, 128, 255, 255, 255, 255]
PLATEAU_H = write_pnm(
    "plateau_h", 9, 5, 1, [RAMP[x] for y in range(5) for x in range(9)],
    "9x5 uchar, every row 0 0 0 0 128 255 255 255 255. The half-step at x=4 "
    "gives x=4 and x=5 the same G (32) and the same theta (64), so exactly "
    "one of the two survives suppression.")
PLATEAU_H_REV = write_pnm(
    "plateau_h_rev", 9, 5, 1, [RAMP[::-1][x] for y in range(5) for x in range(9)],
    "plateau_h mirrored left-right: same plateau, theta 192 instead of 64. "
    "The survivor moves to the other side, which is the asymmetry.")
PLATEAU_V = write_pnm(
    "plateau_v", 5, 9, 1, [RAMP[y] for y in range(9) for x in range(5)],
    "plateau_h transposed: theta 0, plateau on rows 4 and 5.")
PLATEAU_V_REV = write_pnm(
    "plateau_v_rev", 5, 9, 1, [RAMP[::-1][y] for y in range(9) for x in range(5)],
    "plateau_v mirrored top-bottom: theta 128.")

# 7. white disc, the fixture the canny.c:225 comment talks about
_N, _C, _R = 33, 16, 12
DISC = write_pnm(
    "disc33", _N, _N, 1,
    [255 if (x - _C) ** 2 + (y - _C) ** 2 <= _R * _R else 0
     for y in range(_N) for x in range(_N)],
    "33x33 uchar, white disc of radius 12 centred on (16,16). This is the "
    "'white disc on a black background' the source comment describes.")

# 8. engineered (gx, gy) grid. For a 2x2 window a b / c d the two convolution
#    sums are sx = b + d - a - c and sy = c + d - a - b, so with a = 128,
#    b = 128 + (sx-sy)/4, c = 128 - (sx-sy)/4 and d = 128 + (sx+sy)/2 any
#    wanted (gx, gy) drops out at the window's bottom-right pixel.
OCTANT_TARGETS = [
    (0, 0), (64, 0), (0, 64), (-64, 0), (0, -64),
    (64, 64), (-64, 64), (64, -64), (-64, -64), (96, 32),
    (32, 96), (-96, 32), (32, -96), (96, -32), (-32, 96),
    (-96, -32), (8, 0), (0, 8), (-8, -8), (120, 120),
]
_W = 26
_img = [[128] * _W for _ in range(_W)]
OCTANT_PROBES = []
for _n, (_sx, _sy) in enumerate(OCTANT_TARGETS):
    _bx, _by = 2 + (_n % 5) * 5, 2 + (_n // 5) * 5
    _k = (_sx - _sy) // 4
    _img[_by][_bx] = 128
    _img[_by][_bx + 1] = 128 + _k
    _img[_by + 1][_bx] = 128 - _k
    _img[_by + 1][_bx + 1] = 128 + (_sx + _sy) // 2
    OCTANT_PROBES.append({"gx": _sx, "gy": _sy, "x": _bx + 1, "y": _by + 1})
OCTANTS = write_pnm(
    "octants26", _W, _W, 1,
    [_img[y][x] for y in range(_W) for x in range(_W)],
    "26x26 uchar on a flat 128 background with twenty 2x2 perturbations, each "
    "engineered to produce one exact (gx, gy) pair at its bottom-right pixel. "
    "Covers all eight octants, the four axis directions, the four diagonals, "
    "gx == gy == 0, and sub-LUT-resolution gradients.")

# 9. pseudo-random uchar. At sigma 0.01 the blur is a copy, so this drives the
#    polar stage directly and hits all 256 atan2 LUT indices.
LUT64 = write_pnm(
    "noise64", 64, 64, 1, lcg_bytes(64 * 64, 20260825),
    "64x64 uchar LCG noise. Every one of the 256 atan2 LUT indices is reached "
    "at sigma 0.01, and G spans the full uchar range 0..64.")

# 10. pseudo-random RGB, for the bands round-trip and the vector/scalar sweep
NOISE_RGB = write_pnm(
    "noise16rgb", 16, 16, 3, lcg_bytes(16 * 16 * 3, 4242),
    "16x16x3 uchar LCG noise. Pins the (w, h, b) round-trip and the "
    "per-band independence of the whole operation.")

# 11. edges hard against the frame, for the Extend::Copy border question
_B = 7
BORDER = write_pnm(
    "border7", _B, _B, 1,
    [255 if (x == 0 or y == _B - 1) else 0 for y in range(_B) for x in range(_B)],
    "7x7 uchar with a white column on the left frame edge and a white row on "
    "the bottom frame edge. Both edges sit in the outer ring, where the "
    "Extend::Copy embed duplicates neighbours instead of supplying zeros.")


def to_v(path, fmt=None):
    """PNM -> .v, optionally cast. Canny's arm selection keys off the format."""
    base = os.path.splitext(os.path.basename(path))[0]
    out = os.path.join(FIX, base + ("" if fmt is None else "_" + fmt) + ".v")
    if fmt is None:
        run([VIPS, "copy", path, out])
    else:
        run([VIPS, "cast", path, out, fmt])
    return out


# ---------------------------------------------------------------------------
# records
# ---------------------------------------------------------------------------
RECORDS = []


def canny_record(record_id, src, sigma, precision, dump=True, points=None,
                 note=None):
    """`vips canny` on both libvips paths, with the two results compared."""
    cmd_tail = ["--sigma", repr(sigma), "--precision", precision]
    arms = {}
    for arm, novector in (("vector", False), ("novector", True)):
        out = os.path.join(OUT, "%s_%s.v" % (record_id, arm))
        cmd = [VIPS, "canny", src, out] + cmd_tail
        run(cmd, novector=novector)
        d, h, vals = describe(out, dump)
        if points:
            d["points"] = {"%d,%d" % (x, y): vals[(y * h["width"] + x) * h["bands"]:
                                                  (y * h["width"] + x) * h["bands"] + h["bands"]]
                           for (x, y) in points}
        arms[arm] = d
        arms[arm + "_values"] = vals
    va, sa = arms.pop("vector_values"), arms.pop("novector_values")
    diffs = [{"index": i, "vector": va[i], "novector": sa[i],
              "delta": va[i] - sa[i]}
             for i in range(len(va)) if va[i] != sa[i]]
    rec = {
        "record_id": record_id,
        "op": "canny",
        "input": rel(src),
        "sigma": sigma,
        "precision": precision,
        "command": "vips canny %s <out> --sigma %s --precision %s"
                   % (rel(src), repr(sigma), precision),
        "vector": arms["vector"],
        "novector": arms["novector"],
        "paths_agree": not diffs,
        "path_diff_count": len(diffs),
        "path_diff_sample": diffs[:24],
    }
    if note:
        rec["note"] = note
    RECORDS.append(rec)
    return rec


def stage_record(record_id, src, sigma, precision, dump=True, note=None):
    """The same pipeline, stage by stage, through the public CLI ops.

    canny.c:377-425 is gaussblur -> two 2x2 convs -> polar -> embed -> thin.
    The first two stages are reachable from the CLI verbatim, so they are
    captured from the binary rather than derived. Which conv arm runs is
    decided by the format of the BLURRED image, not of the input.
    """
    arms = {}
    for arm, novector in (("vector", False), ("novector", True)):
        blur = os.path.join(OUT, "%s_blur_%s.v" % (record_id, arm))
        run([VIPS, "gaussblur", src, blur, repr(sigma),
             "--precision", precision], novector=novector)
        bh = header(blur)
        if bh["format"] == "uchar":
            gxm, gym, gprec = MASK_GX_OFF, MASK_GY_OFF, "integer"
        else:
            gxm, gym, gprec = MASK_GX, MASK_GY, "float"
        gx = os.path.join(OUT, "%s_gx_%s.v" % (record_id, arm))
        gy = os.path.join(OUT, "%s_gy_%s.v" % (record_id, arm))
        run([VIPS, "conv", blur, gx, gxm, "--precision", gprec], novector=novector)
        run([VIPS, "conv", blur, gy, gym, "--precision", gprec], novector=novector)
        arms[arm] = {
            "gaussblur": describe(blur, dump)[0],
            "gradient_precision": gprec,
            "gradient_mask_offset": 128 if gprec == "integer" else 0,
            "Gx": describe(gx, dump)[0],
            "Gy": describe(gy, dump)[0],
        }
    rec = {
        "record_id": record_id,
        "op": "canny_stages",
        "input": rel(src),
        "sigma": sigma,
        "precision": precision,
        "commands": [
            "vips gaussblur %s <blur> %s --precision %s" % (rel(src), repr(sigma), precision),
            "vips conv <blur> <Gx> fixtures/masks/{gx_offset128|gx}.mat --precision {integer|float}",
            "vips conv <blur> <Gy> fixtures/masks/{gy_offset128|gy}.mat --precision {integer|float}",
        ],
        "vector": arms["vector"],
        "novector": arms["novector"],
    }
    if note:
        rec["note"] = note
    RECORDS.append(rec)
    return rec


# ---------------------------------------------------------------------------
# the sweep
# ---------------------------------------------------------------------------
COMMANDS.append("#!/bin/sh")
COMMANDS.append("# Reproducible vips CLI commands for the canny oracle capture.")
COMMANDS.append("# Run from oracle-captures/convolution/canny/ (paths are relative).")
COMMANDS.append("set -e")
COMMANDS.append("")

STEP_V = to_v(STEP)
SQUARE_V = to_v(SQUARE)
PLATEAU_H_V = to_v(PLATEAU_H)
PLATEAU_H_REV_V = to_v(PLATEAU_H_REV)
PLATEAU_V_V = to_v(PLATEAU_V)
PLATEAU_V_REV_V = to_v(PLATEAU_V_REV)
DISC_V = to_v(DISC)
OCTANTS_V = to_v(OCTANTS)
LUT64_V = to_v(LUT64)
NOISE_RGB_V = to_v(NOISE_RGB)
BORDER_V = to_v(BORDER)

STEP_F = to_v(STEP, "float")
STEP_D = to_v(STEP, "double")
STEP_US = to_v(STEP, "ushort")
OCTANTS_F = to_v(OCTANTS, "float")
SQUARE_F = to_v(SQUARE, "float")
NOISE_RGB_F = to_v(NOISE_RGB, "float")

# 1. the documented default, on every carrier
for prec in ("float", "integer", "approximate"):
    canny_record("default_step9_%s" % prec, STEP_V, 1.4, prec)
canny_record("default_square9_float", SQUARE_V, 1.4, "float")
canny_record("default_square9_integer", SQUARE_V, 1.4, "integer")
canny_record("default_disc33_float", DISC_V, 1.4, "float")
canny_record("default_disc33_integer", DISC_V, 1.4, "integer")
canny_record("default_noise16rgb_float", NOISE_RGB_V, 1.4, "float", dump=True)
canny_record("default_noise16rgb_integer", NOISE_RGB_V, 1.4, "integer", dump=True)

# format table: the arm and the OUTPUT FORMAT are decided by the blurred image
FORMAT_TABLE = []
for fmt in ("uchar", "char", "ushort", "short", "uint", "int", "float", "double"):
    src = STEP_V if fmt == "uchar" else to_v(STEP, fmt)
    for prec in ("integer", "float", "approximate"):
        for sigma in (1.4, 0.5, 0.2, 0.19, 0.1):
            blur = os.path.join(OUT, "fmtblur_%s_%s_%s.v" % (fmt, prec, sigma))
            run([VIPS, "gaussblur", src, blur, repr(sigma), "--precision", prec])
            out = os.path.join(OUT, "fmt_%s_%s_%s.v" % (fmt, prec, sigma))
            run([VIPS, "canny", src, out, "--sigma", repr(sigma), "--precision", prec])
            h = header(out)
            FORMAT_TABLE.append({
                "input_format": fmt,
                "precision": prec,
                "sigma": sigma,
                "blurred_format": header(blur)["format"],
                "gradient_arm": ("integer, mask offset 128"
                                 if header(blur)["format"] == "uchar"
                                 else "float, mask offset 0"),
                "output_format": h["format"],
                "output_bands": h["bands"],
                "output_size": [h["width"], h["height"]],
            })

# 2. the uchar precision override, isolated. Same pixels, same sigma, same
#    requested precision; only the declared input format differs.
canny_record("override_uchar_s0.1_float", STEP_V, 0.1, "float",
             note="uchar input, sigma below gaussblur's 0.2 copy threshold, "
                  "caller asked for float")
canny_record("override_uchar_s0.1_integer", STEP_V, 0.1, "integer")
canny_record("override_float_s0.1_float", STEP_F, 0.1, "float",
             note="identical pixels declared float: the override cannot fire")
canny_record("override_float_s0.1_integer", STEP_F, 0.1, "integer")
canny_record("override_uchar_s1.4_float", STEP_V, 1.4, "float")
canny_record("override_uchar_s1.4_integer", STEP_V, 1.4, "integer")
canny_record("override_float_s1.4_integer", STEP_F, 1.4, "integer")
stage_record("override_stages_uchar_s0.1_float", STEP_V, 0.1, "float")
stage_record("override_stages_float_s0.1_float", STEP_F, 0.1, "float")

# 3. G ceiling on the uchar path, and the absence of one on the float path
canny_record("gmax_square9_uchar", SQUARE_V, 0.01, "integer")
canny_record("gmax_square9_float", SQUARE_F, 0.01, "float")
stage_record("gmax_stages_square9_uchar", SQUARE_V, 0.01, "integer")
stage_record("gmax_stages_square9_float", SQUARE_F, 0.01, "float")
canny_record("gmax_noise64_uchar", LUT64_V, 0.01, "integer", dump=False,
             note="the wide sweep behind the 0..64 claim")

# 4. orientation, on the fixture the source comment names
stage_record("orientation_disc33_uchar", DISC_V, 1.4, "integer", dump=False)
stage_record("orientation_disc33_float", DISC_V, 1.4, "float", dump=False)

# 5. octants
canny_record("octants_uchar", OCTANTS_V, 0.01, "integer")
canny_record("octants_float", OCTANTS_F, 0.01, "float")
stage_record("octants_stages_uchar", OCTANTS_V, 0.01, "integer")
stage_record("octants_stages_float", OCTANTS_F, 0.01, "float")

# 6. the suppression asymmetry
for name, src in (("plateau_h", PLATEAU_H_V), ("plateau_h_rev", PLATEAU_H_REV_V),
                  ("plateau_v", PLATEAU_V_V), ("plateau_v_rev", PLATEAU_V_REV_V)):
    canny_record("suppress_%s_uchar" % name, src, 0.01, "integer")
    canny_record("suppress_%s_float" % name, src, 0.01, "float")
    stage_record("suppress_stages_%s_uchar" % name, src, 0.01, "integer")

# 7. border
canny_record("border7_uchar", BORDER_V, 0.01, "integer")
canny_record("border7_float", BORDER_V, 1.4, "float")
canny_record("border_step9_float", STEP_V, 1.4, "float")

# 8. sigma edge cases
SIGMA_TABLE = []
for sigma in (0.01, 0.1, 0.19, 0.2, 0.5, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0):
    for prec in ("integer", "float"):
        gm = os.path.join(OUT, "gaussmat_%s_%s.mat" % (sigma, prec))
        run([VIPS, "gaussmat", gm, repr(sigma), "0.2",
             "--separable", "--precision", prec])
        with open(gm) as f:
            head, body = f.readline().split(), f.readline().split()
        blur = os.path.join(OUT, "sigblur_%s_%s.v" % (sigma, prec))
        run([VIPS, "gaussblur", STEP_V, blur, repr(sigma), "--precision", prec])
        bh, bvals, _ = pixels(blur)
        SIGMA_TABLE.append({
            "sigma": sigma,
            "precision": prec,
            "mask_width": int(head[0]),
            "mask_scale": float(head[2]) if len(head) > 2 else 1.0,
            "mask": [float(v) for v in body],
            "blurred_format": bh["format"],
            "blur_is_identity": rows(bh, bvals)[4] == [0, 0, 0, 0, 255, 255, 255, 255, 255],
        })
for sigma in (0.01, 0.1, 0.19, 0.2, 1.4):
    canny_record("sigma_step9_%s_integer" % sigma, STEP_V, sigma, "integer")
    canny_record("sigma_step9_%s_float" % sigma, STEP_V, sigma, "float")

# out-of-range sigma: GObject refuses the property and the default stands
SIGMA_REJECT = []
for text in ("0", "0.009", "-1", "1000.1", "1000", "1.4"):
    out = os.path.join(OUT, "sigreject_%s.v" % text)
    res = subprocess.run([VIPS, "canny", STEP_V, out, "--sigma", text],
                         capture_output=True, text=True, cwd=ROOT)
    COMMANDS.append("vips canny %s <out> --sigma %s   # exit %d"
                    % (rel(STEP_V), text, res.returncode))
    _, _, sha = pixels(out)
    SIGMA_REJECT.append({
        "sigma_text": text, "exit_code": res.returncode,
        "stderr_head": strip_glib_prefix(res.stderr),
        "output_raw_sha256": sha, "output_format": header(out)["format"],
    })

# 9. dimensions and bands round-trip, plus interpretation
ROUNDTRIP = []
run([VIPS, "copy", NOISE_RGB_V, os.path.join(FIX, "noise16rgb_srgb.v"),
     "--interpretation", "srgb"])
run([VIPS, "copy", LUT64_V, os.path.join(FIX, "noise64_bw.v"),
     "--interpretation", "b-w"])
for src in (NOISE_RGB_V, os.path.join(FIX, "noise16rgb_srgb.v"),
            LUT64_V, os.path.join(FIX, "noise64_bw.v"), NOISE_RGB_F):
    ih = header(src)
    for prec in ("integer", "float"):
        out = os.path.join(OUT, "rt_%s_%s.v" % (os.path.basename(src)[:-2], prec))
        run([VIPS, "canny", src, out, "--precision", prec])
        oh = header(out)
        ROUNDTRIP.append({"input": rel(src), "precision": prec,
                          "in": ih, "out": oh,
                          "size_preserved": (ih["width"], ih["height"]) == (oh["width"], oh["height"]),
                          "bands_preserved": ih["bands"] == oh["bands"],
                          "interpretation_preserved": ih["interpretation"] == oh["interpretation"]})

# 10. the vector/scalar sweep. This is the reason both arms are captured.
VECTOR_SWEEP = []
for src in (LUT64_V, NOISE_RGB_V, STEP_V, DISC_V):
    for prec in ("integer", "float", "approximate"):
        for sigma in (0.01, 0.5, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0, 4.0):
            a = os.path.join(OUT, "vs_vec.v")
            b = os.path.join(OUT, "vs_nov.v")
            run([VIPS, "canny", src, a, "--sigma", repr(sigma), "--precision", prec], log=False)
            run([VIPS, "canny", src, b, "--sigma", repr(sigma), "--precision", prec],
                novector=True, log=False)
            _, va, sha_a = pixels(a)
            _, sa, sha_b = pixels(b)
            deltas = sorted({va[i] - sa[i] for i in range(len(va)) if va[i] != sa[i]})
            VECTOR_SWEEP.append({
                "input": rel(src), "precision": prec, "sigma": sigma,
                "samples": len(va),
                "differing": sum(1 for i in range(len(va)) if va[i] != sa[i]),
                "deltas": deltas[:8],
                "max_abs_delta": max((abs(d) for d in deltas), default=0),
                "vector_raw_sha256": sha_a, "novector_raw_sha256": sha_b,
            })
COMMANDS.append("# vector/scalar sweep: for every (input, precision, sigma) below, both")
COMMANDS.append("#   vips canny <in> <out> --sigma S --precision P")
COMMANDS.append("#   VIPS_NOVECTOR=1 vips canny <in> <out> --sigma S --precision P")
COMMANDS.append("# see oracle.json -> vector_scalar_sweep for the pairs and the diff counts.")


# ---------------------------------------------------------------------------
# derived: the atan2 LUT and the polar values behind the orientation claim
# ---------------------------------------------------------------------------
import math  # noqa: E402  (kept next to the one block that uses it)

LUT = []
for i in range(256):
    _x = i & 0xF
    if _x & 0x8:
        _x -= 0x10
    _y = (i >> 4) & 0xF
    if _y & 0x8:
        _y -= 0x10
    LUT.append(int(256 * (math.degrees(math.atan2(_x, _y)) + 360) / 360) & 0xFF)


def polar_uchar(gx, gy):
    return ((gx * gx + gy * gy + 256) >> 9,
            LUT[((gx >> 4) & 0xF) | (gy & 0xF0)])


def polar_float(gx, gy, narrow=None):
    """canny.c:131-152. vips does the arithmetic in double and stores the
    result in the pixel type, so a float image keeps only f32 of it."""
    narrow = narrow or (lambda v: v)
    return (narrow((gx * gx + gy * gy + 256.0) / 512.0),
            narrow(256.0 * math.fmod(math.degrees(math.atan2(gx, gy)) + 360.0, 360.0) / 360.0))


def octant_polar():
    out = []
    for probe in OCTANT_PROBES:
        gx, gy = probe["gx"], probe["gy"]
        gu, tu = polar_uchar(gx, gy)
        gf, tf = polar_float(float(gx), float(gy), f32)
        out.append({**probe, "uchar_G": gu, "uchar_theta": tu,
                    "float_G": gf, "float_theta": tf})
    return out


def disc_polar():
    """theta at the four cardinal edges of the disc, read off Gx/Gy the binary
    produced. Only the arithmetic is ours; the gradients are measured."""
    res = {}
    for arm, prec in (("uchar", "integer"), ("float", "float")):
        rid = "orientation_disc33_%s" % arm
        rec = [r for r in RECORDS if r["record_id"] == rid][0]
        gxp = os.path.join(ROOT, rec["vector"]["Gx"]["path"])
        gyp = os.path.join(ROOT, rec["vector"]["Gy"]["path"])
        hx, gxv, _ = pixels(gxp)
        _, gyv, _ = pixels(gyp)
        w = hx["width"]
        uchar = hx["format"] == "uchar"
        pol = {}
        for y in range(hx["height"]):
            for x in range(w):
                gx = gxv[y * w + x] - (128 if uchar else 0)
                gy = gyv[y * w + x] - (128 if uchar else 0)
                pol[(x, y)] = (polar_uchar(gx, gy) if uchar
                               else polar_float(gx, gy, f32))
        cardinals = {}
        for name, cells in (
                ("top", [(_C, y) for y in range(_C)]),
                ("bottom", [(_C, y) for y in range(_C, _N)]),
                ("left", [(x, _C) for x in range(_C)]),
                ("right", [(x, _C) for x in range(_C, _N)])):
            bx, by = max(cells, key=lambda c: pol[c][0])
            cardinals[name] = {"x": bx, "y": by, "G": pol[(bx, by)][0],
                               "theta": pol[(bx, by)][1]}
        res[arm] = cardinals
    return res



# ---------------------------------------------------------------------------
# a reference model, so the capture validates itself
#
# Stages 1 and 2 come from the binary (gaussblur + two convs through the CLI);
# stages 3 and 4 are canny.c:110-152 and 251-289 transcribed. If the model
# reproduces every measured output byte for byte then the derived G/theta
# values above are trustworthy, and any future divergence points at the stage
# that moved. Plain Python on purpose: no third-party import, so a re-run
# needs nothing but CPython and vips.
# ---------------------------------------------------------------------------
def f32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]


def model_polar(gxv, gyv, h, uchar):
    """canny.c POLAR_UCHAR / POLAR(TYPE): interleave (G, theta), 2x the bands."""
    out = []
    for i in range(len(gxv)):
        if uchar:
            gx, gy = gxv[i] - 128, gyv[i] - 128
            out.append((gx * gx + gy * gy + 256) >> 9)
            out.append(LUT[((gx >> 4) & 0xF) | (gy & 0xF0)])
        else:
            gx, gy = float(gxv[i]), float(gyv[i])
            g = (gx * gx + gy * gy + 256.0) / 512.0
            t = 256.0 * math.fmod(math.degrees(math.atan2(gx, gy)) + 360.0, 360.0) / 360.0
            narrow = f32 if h["format"] == "float" else float
            out.append(narrow(g))
            out.append(narrow(t))
    return out


def model_embed_copy(vals, w, h, b):
    """vips_embed(1, 1, w+2, h+2, VIPS_EXTEND_COPY): duplicate the edge pixels."""
    def px(x, y):
        x = 0 if x < 0 else (w - 1 if x >= w else x)
        y = 0 if y < 0 else (h - 1 if y >= h else y)
        return vals[(y * w + x) * b:(y * w + x) * b + b]
    out = []
    for y in range(-1, h + 1):
        for x in range(-1, w + 1):
            out += px(x, y)
    return out


def model_thin(vals, w, h, b2, uchar, narrow):
    """canny.c THIN(TYPE). b2 is the doubled band count of the polar image."""
    b = b2 // 2
    out = []
    # counter-clockwise from top-middle, as (dy, dx) from the 3x3 top-left
    off = ((0, 1), (0, 0), (1, 0), (2, 0), (2, 1), (2, 2), (1, 2), (0, 2))
    for y in range(h - 2):
        for x in range(w - 2):
            for band in range(b):
                def at(k):
                    dy, dx = off[k]
                    return vals[((y + dy) * w + (x + dx)) * b2 + 2 * band]
                c = ((y + 1) * w + (x + 1)) * b2 + 2 * band
                g, theta = vals[c], vals[c + 1]
                low_theta = int(theta / 32) & 0x7
                high_theta = (low_theta + 1) & 0x7
                if uchar:
                    residual = (theta - low_theta * 32) & 0xFF
                    low = ((at(low_theta) * (32 - residual)
                            + at(high_theta) * residual) // 32) & 0xFF
                    high = ((at((low_theta + 4) & 0x7) * (32 - residual)
                             + at((high_theta + 4) & 0x7) * residual) // 32) & 0xFF
                else:
                    residual = narrow(theta - low_theta * 32)
                    low = narrow(narrow(narrow(at(low_theta) * narrow(32 - residual))
                                        + narrow(at(high_theta) * residual)) / 32)
                    high = narrow(narrow(narrow(at((low_theta + 4) & 0x7) * narrow(32 - residual))
                                         + narrow(at((high_theta + 4) & 0x7) * residual)) / 32)
                out.append(0 if (g <= low or g < high) else g)
    return out


def model_canny(src, sigma, precision, tag, novector):
    blur = os.path.join(OUT, "model_%s_blur.v" % tag)
    run([VIPS, "gaussblur", src, blur, repr(sigma), "--precision", precision],
        novector=novector, log=False)
    bh = header(blur)
    uchar = bh["format"] == "uchar"
    gxm, gym, gprec = ((MASK_GX_OFF, MASK_GY_OFF, "integer") if uchar
                       else (MASK_GX, MASK_GY, "float"))
    gxp = os.path.join(OUT, "model_%s_gx.v" % tag)
    gyp = os.path.join(OUT, "model_%s_gy.v" % tag)
    run([VIPS, "conv", blur, gxp, gxm, "--precision", gprec], novector=novector, log=False)
    run([VIPS, "conv", blur, gyp, gym, "--precision", gprec], novector=novector, log=False)
    gh, gxv, _ = pixels(gxp)
    _, gyv, _ = pixels(gyp)
    w, h, b = gh["width"], gh["height"], gh["bands"]
    narrow = f32 if gh["format"] == "float" else float
    pol = model_polar(gxv, gyv, gh, uchar)
    emb = model_embed_copy(pol, w, h, 2 * b)
    return model_thin(emb, w + 2, h + 2, 2 * b, uchar, narrow)


MODEL_CHECK = []
for _rec in [r for r in RECORDS if r["op"] == "canny"]:
    _src = os.path.join(ROOT, _rec["input"])
    if header(_src)["format"] not in ("uchar", "float"):
        continue
    _row = {"record_id": _rec["record_id"]}
    for _arm, _nv in (("vector", False), ("novector", True)):
        _got = model_canny(_src, _rec["sigma"], _rec["precision"],
                           _rec["record_id"] + "_" + _arm, _nv)
        _, _want, _ = pixels(os.path.join(ROOT, _rec[_arm]["path"]))
        _row[_arm] = (len(_got) == len(_want)
                      and all(a == b for a, b in zip(_got, _want)))
    MODEL_CHECK.append(_row)
    _rec["model_reproduces_binary"] = _row["novector"]

# ---------------------------------------------------------------------------
# write out
# ---------------------------------------------------------------------------
DOC = {
    "meta": {
        "vips_version": VIPS_VERSION,
        "vips_binary": VIPS,
        "area": AREA,
        "oracle_pin": {
            "file": "oracle-captures/ORACLE_PIN.json",
            "pinned_vips_version": ORACLE_PIN["pinned_vips_version"],
            "checked_how": (
                "capture.py compares `vips --version` against the pin before "
                "it writes anything; tests/oracle_capture_pins.rs compares "
                "the version recorded here against the same file. Issue "
                "#650, after a brew upgrade redefined the oracle mid-session "
                "and nothing was looking."),
        },
        "homebrew_kegs": oracle_pin.homebrew_kegs(VIPS),
        "why_homebrew_kegs": (
            "Every Homebrew keg the vips binary reaches transitively, with "
            "its version, walked from `otool -L`. `vips --vips-config` names "
            "the codec libraries with no version for any of them, and the "
            "codec version is what a future disagreement over these numbers "
            "turns on. Provenance, not a pin: only the vips version is "
            "enforced."),
        "issues": ["#511", "#559", "#560"],
        "source": "libvips/libvips/convolution/canny.c",
        "captured_by": "oracle-captures/convolution/canny/capture.py",
        "scope_note": (
            "This capture adds a new directory only. It does not read or write "
            "anything under oracle-captures/convolution/ itself, which PR #557 owns."),
        "both_libvips_paths": (
            "Every canny record is captured twice: once with the default "
            "vectorised libvips (HWY) and once with VIPS_NOVECTOR=1, which is "
            "the portable C that libviprs ports. `paths_agree` says whether "
            "they matched; see vector_scalar_sweep for the full picture."),
        "hashes": "raw_sha256 is sha256 of `vips rawsave` output, i.e. of the pixels alone.",
    },
    "facts": {
        "signature": (
            "vips canny in out [--sigma S] [--precision integer|float|approximate]. "
            "sigma defaults to 1.4 with min 0.01 and max 1000, precision defaults "
            "to float. There are no hysteresis threshold arguments in 8.18.6."),
        "precision_never_reaches_the_gradient": (
            "The caller's `precision` is handed to ONE stage, the gaussblur. The "
            "gradient stage picks its own: canny.c:80-85 tests the format of the "
            "BLURRED image, not of the input, and takes INTEGER + mask offset 128 "
            "for uchar and FLOAT + offset 0 for everything else. So `approximate` "
            "never reaches vips_conva at all inside canny, and see "
            "`uchar_precision_override` for what that means for uchar input."),
        "uchar_precision_override": (
            "Real, but only reachable through sigma. gaussblur.c:71 returns a plain "
            "vips_copy when sigma < 0.2, so the blurred image is still uchar and the "
            "gradient stage forces INTEGER even though the caller asked for float. "
            "At sigma >= 0.2 with precision float the blur has already promoted the "
            "image to float, so the uchar branch cannot fire and the whole operation "
            "runs in float. Measured on step9: sigma 0.19 precision float gives uchar "
            "output with a peak of 32, sigma 0.2 precision float gives float output. "
            "With precision approximate the override always fires, because convasep "
            "keeps uchar. See format_table and the override_* records."),
        "output_format": (
            "uchar in -> uchar out when the blur stays uchar (precision integer or "
            "approximate, or any precision with sigma < 0.2), float out otherwise. "
            "double in -> double out. Every other input format -> float out, on every "
            "precision. Size, bands and interpretation always round-trip."),
        "G_range_uchar": (
            "0..64, confirmed. The uchar conv clips gx and gy to -128..127, so "
            "(gx*gx + gy*gy + 256) >> 9 tops out at (16384 + 16384 + 256) >> 9 = 64. "
            "square9 at sigma 0.01 hits exactly 64 at (4,4), where both convs clip "
            "negative, and noise64 at sigma 0.01 also reaches 64. #511 is right."),
        "G_range_float": (
            "Not bounded to a byte and not bounded to 64. The float polar stage has "
            "no clip in front of it, so a hard 0/255 step gives G = 508.5078125 and "
            "vips writes that straight out. A flat region gives G = 0.5, not 0, "
            "because of the +256.0 in the numerator, and suppression then zeroes it "
            "since every neighbour is 0.5 too."),
        "orientation": (
            "The canny.c:225-229 comment is WRONG. Measured on disc33, theta reads "
            "0 at the TOP, 64 on the LEFT, 128 at the BOTTOM and 192 on the RIGHT. "
            "The comment says '128 on the right and 192 on the right edge', which "
            "names the right twice and drops the bottom. The uchar path lands on "
            "exactly 0/64/128/192 at the four cardinal points; the float path reads "
            "2.65/61.35/125.35/194.65 there, because the 2x2 mask measures the "
            "gradient half a pixel off centre and the LUT's 4-bit quantisation hides "
            "that on the uchar path."),
        "suppression_asymmetry": (
            "Observable, and the plateau fixtures pin it. plateau_h gives x=4 and "
            "x=5 the same G (32) and the same theta (64); vips keeps x=4 and zeroes "
            "x=5. plateau_h_rev has the same plateau at theta 192 and vips keeps x=5 "
            "instead. The survivor is always the one on the strict `<` side. Both "
            "comparisons written `<=` erases the edge completely, both written `<` "
            "keeps a 2-pixel-wide edge, and swapping them keeps the wrong pixel. "
            "plateau_v / plateau_v_rev repeat it on the theta 0 / 128 axis."),
        "border": (
            "The outer ring is NOT zeroed. vips_embed with VIPS_EXTEND_COPY "
            "duplicates the edge pixels, so an edge lying on the frame compares "
            "against copies of itself and survives. On step9 at sigma 1.4 precision "
            "float, column 4 carries 47.99231719970703 in EVERY row including y=0 "
            "and y=8, and on square9 row 4 survives from x=0 to x=3. border7 puts "
            "real edges on the frame on purpose: its last row comes out "
            "0 64 32 32 32 32 32, live data right on the boundary."),
        "sigma": (
            "gaussblur short-circuits to vips_copy below 0.2 (gaussblur.c:71), so "
            "sigma 0.01 makes the blur an exact no-op and canny reduces to gradient "
            "+ polar + thin. libviprs's own `try_gaussblur` copy-below-0.2 rule "
            "matches vips exactly. The documented min of 0.01 is NOT a clamp: "
            "GObject refuses any value outside 0.01..1000 with a GLib-GObject-"
            "CRITICAL, leaves sigma at its 1.4 default and still exits 0, so "
            "`--sigma 0`, `--sigma -1` and `--sigma 1000.1` all produce the same "
            "bytes as `--sigma 1.4`. From 0.2 up to 0.55 the integer gaussmat is "
            "still a 1x1 identity (0.6 is the first sigma with a 3x1 mask), so the "
            "blur remains a value-level no-op there, but it now runs through "
            "convsep and therefore CHANGES THE FORMAT on the float precision arm."),
        "vector_vs_scalar": (
            "`vips canny --precision integer` disagrees with itself between the "
            "vectorised libvips and VIPS_NOVECTOR=1 at almost every sigma, by as "
            "much as 28 on a byte. The default sigma of 1.4 is the lucky exception: "
            "its integer gaussmat has scale 64, a power of two, so the vector path's "
            "shift and C's divide agree. Every other sigma tried (0.8, 1.0, 1.2, "
            "1.6, 1.8, 2.0, 2.5, 3.0, 4.0) has a non-power-of-two scale and "
            "diverges. The float and approximate arms agree on every fixture and "
            "every sigma. The divergence lives entirely in stage 1: the two 2x2 "
            "gradient convs have scale 1, so they cannot round differently. "
            "See vector_scalar_sweep."),
        "flat_region": (
            "gx == gy == 0 gives LUT index 0 and theta 0 on the uchar path, and G 0 "
            "(the +256 does not survive the >> 9). On the float path the same pixel "
            "gives theta 0.0 and G 0.5. Both are suppressed, because every neighbour "
            "holds the same value and the `<=` against low fires."),
        "lut_quantisation": (
            "The uchar LUT throws away the bottom 4 bits of each axis, so gradients "
            "smaller than 16 collapse into the wrong bucket. (gx, gy) = (8, 0) reads "
            "theta 0 on the uchar path, pointing straight up, where the float path "
            "reads the correct 64. That is not a porting bug to fix, it is what the "
            "binary does, and octants26 pins it."),
    },
    "fixtures": FIXTURE_DOCS,
    "atan2_lut": {
        "doc": (
            "canny.c:200-222, recomputed here in f64 and pinned. Not directly "
            "observable from the CLI: it is validated indirectly, because a "
            "model of the whole operation built on this table reproduces every "
            "`vips canny` output in this file byte for byte, and noise64 at "
            "sigma 0.01 reaches all 256 indices."),
        "index": "((gx >> 4) & 0xF) | (gy & 0xF0), with gx, gy in -128..127",
        "table": LUT,
    },
    "format_table": FORMAT_TABLE,
    "sigma_table": SIGMA_TABLE,
    "sigma_out_of_range": SIGMA_REJECT,
    "roundtrip": ROUNDTRIP,
    "vector_scalar_sweep": VECTOR_SWEEP,
    "derived_polar": {
        "doc": (
            "G and theta are internal to vips_canny_polar and never surface on "
            "the CLI. These come from the Gx/Gy the binary actually produced "
            "(captured in the canny_stages records) put through the canny.c "
            "formulas. Every one of them is consistent with a full model that "
            "reproduces the measured canny output exactly."),
        "octants": octant_polar(),
        "disc_cardinals": disc_polar(),
    },
    "model_check": {
        "doc": (
            "The reference model at the bottom of capture.py, run against every "
            "canny record above. `novector` true means the portable-C model "
            "reproduces the binary sample for sample, which is what libviprs "
            "must match; `vector` true means the HWY path agreed too."),
        "results": MODEL_CHECK,
    },
    "records": RECORDS,
}

with open(os.path.join(ROOT, "oracle.json"), "w") as f:
    # allow_nan=False so a non-finite measurement stops the capture here
    # rather than writing a file nobody outside Python can parse (#682).
    json.dump(DOC, f, indent=1, sort_keys=False, allow_nan=False)
    f.write("\n")
with open(os.path.join(ROOT, "commands.sh"), "w") as f:
    f.write("\n".join(COMMANDS) + "\n")
os.chmod(os.path.join(ROOT, "commands.sh"), 0o755)

print("records: %d" % len(RECORDS))
print("vector/scalar pairs: %d, divergent: %d"
      % (len(VECTOR_SWEEP), sum(1 for v in VECTOR_SWEEP if v["differing"])))
print("oracle.json: %d bytes" % os.path.getsize(os.path.join(ROOT, "oracle.json")))
