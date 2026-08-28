#!/usr/bin/env python3
"""
Oracle capture for the "convolution" area of libviprs-tests.

Runs the real vips CLI over synthetic and reference-suite inputs and
records exact input -> output behaviour for conv, compass, convsep, fastcor,
spcor, gaussblur, sharpen, plus supplementary convasep/sobel/canny records.

Every record that can reach `vips_convi` is captured TWICE, once on each of
libvips's two integer-convolution implementations (issue #558):

  * `VIPS_NOVECTOR=1` runs `vips_convi_gen`, the portable C loop that
    libvips's own docs name as the specification (`convi.c:1271-1284`) and
    that libviprs implements. **That is the arm this file pins.**
  * the default run takes the HWY vector path on uchar images, a fixed-point
    approximation that convolves with requantised coefficients. libvips
    does not bound the gap between the two.

`paths_agree` on each record says whether the two matched, and the
KNOWN_DIVERGENT table below states the expectation up front: a record that
starts agreeing, or stops agreeing, fails the capture loudly instead of
drifting inside somebody's tolerance. That is the whole point of the dual
capture -- `vips_convi_intize`'s accept/reject predicate is the part upstream
actually edits (`223d66a0b` added a shift guard inside the 8.18 patch line),
and moving a mask across it changes which kernel runs.

Writes:
  commands.sh  - every vips CLI command actually executed, in order
  oracle.json  - structured records (dims, avg, min/max + a deterministic
                 position, getpoint samples, sha256 of the PIXELS, precision
                 used, and the vector/novector comparison)

Re-running needs two things this directory does not carry: the vips binary at
VIPS below, and the libvips reference suite's `sample.jpg` at REF_IMAGES (a
22 KB JPEG, not vendored here). Every other input is generated from scratch by
this script -- the noise fixtures use `random.Random` with a fixed seed and the
kernels come from `vips gaussmat`.

## Two things that used to make a re-run diff against the committed file (#649)

The inputs were always deterministic and the committed `.pgm` / `.ppm`
fixtures are byte-identical after a re-run. About 932 of the 13,794 pinned
leaves were not, for two reasons that had nothing to do with libvips being
wrong, and both are fixed here.

**The `sha256` pins hashed the `.v` container, not the pixels.** A `.v` is a
64-byte header, the pixel data, and an XML metadata trailer, and `build_xml`
(libvips/iofuncs/vips.c:857-859) writes VIPS_MICRO_VERSION into that trailer's
namespace URI. So a vips patch release moves every whole-file hash at once
without a single pixel moving: measured 8.18.4 against 8.18.6, all 452 hashes
changed and every one of them is reproduced exactly by flipping that one byte
back, while all 4294 pinned samples, all 452 `avg` values and every geometry
field were identical. As a regression detector that is exactly backwards, so
the pin is now `raw_sha256`, the sha256 of `vips rawsave` output, which is the
pixels alone. That is the same thing canny/capture.py already pins.

**The min/max coordinates came from `vips min --x --y`,** which reports WHERE a
worker thread found the extreme. 614 of the 904 extremes in this capture sit on
a tie (312 min, 302 max), so which coordinate comes back is a race: two runs of
the identical binary on the identical fixtures differ at 478 leaves, all of
them `x` or `y`. The VALUE still comes from the binary; the POSITION is now
recomputed as the first occurrence in raster order, and `ties` records how many
positions hold the extreme so a reader can tell an unambiguous pin from an
arbitrary-but-reproducible one. Again the shape canny/capture.py already uses.

## The oracle is pinned (#650)

`oracle-captures/ORACLE_PIN.json` names the libvips build this area is
measured against. This script refuses to run against anything else unless it
is passed `--repin`, and `tests/oracle_capture_pins.rs` fails if the committed
`oracle.json` records a version that file does not declare for this area. That
is the half that matters: the previous arrangement recorded the version in a
comment and in a meta key nothing ever read, so a `brew upgrade` nobody ran
deliberately redefined the reference implementation and the only reason anyone
noticed was that a lane happened to run the same command twice.

Does not touch any git working tree; everything is written under this
script's own directory (oracle-captures/convolution/).
"""
import hashlib
import json
import os
import struct
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
FIX = os.path.join(ROOT, "fixtures")
OUT = os.path.join(FIX, "outputs")
KER = os.path.join(FIX, "kernels")
REF_IMAGES = "/Users/rom/workspace/libviprs/libviprs-tests/tmp/libvips-reference-tests/test-suite/images"

VIPS = "/opt/homebrew/bin/vips"
VIPSHEADER = "/opt/homebrew/bin/vipsheader"

AREA = "convolution"

# The oracle is pinned: oracle-captures/ORACLE_PIN.json names the libvips
# build this area is measured against, and check() exits before anything is
# written if the binary on the machine disagrees. See #650, and see
# tests/oracle_capture_pins.rs for the half of the guard that runs in CI.
sys.path.insert(0, os.path.abspath(os.path.join(ROOT, os.pardir)))
import oracle_pin  # noqa: E402  (needs the path above)

VIPS_VERSION, ORACLE_PIN = oracle_pin.check(AREA, VIPS)

os.makedirs(OUT, exist_ok=True)
os.makedirs(KER, exist_ok=True)

COMMANDS_LOG = []
RECORDS = []

# The records where libvips's two integer-convolution paths are KNOWN to
# disagree, with the sample counts measured against vips 8.18.4 on the
# targets recorded in oracle.json -> meta.provenance.
#
# Everything NOT listed here must agree byte for byte; everything listed must
# still disagree. Both directions are asserted, because both are interesting:
# a record that starts agreeing means upstream's accuracy gate now rejects a
# mask it used to accept (or the reverse), and either way the pinned bytes
# stopped meaning what the note next to them says.
KNOWN_DIVERGENT = {
    # 2D gaussmat(sigma=2, min_ampl=0.1), scale 494 -- the big one.
    "convsep_mono_conv2d_integer": (118357, 2),
    "convsep_colour_conv2d_integer": (360671, 2),
    # the separable arm of the same mask, scale 22, applied twice
    "convsep_mono_convsep1d_integer": (4875, 1),
    "convsep_colour_convsep1d_integer": (14629, 1),
    "convsep_synthmono_conv2d_integer": (10, 1),
    # gaussmat scales 58 (sigma 1.2) and 70 (sigma 1.6); 1.4 gives 64 and
    # 216 -- a power of two only for the SEPARABLE mask, which is why the
    # 2D sigma-1.4 records are lucky here and the canny sweep's are not.
    "gaussblur_colour_integer_s1.2_conv": (23, 1),
    "gaussblur_mono_integer_s1.6_direct": (19, 1),
    "gaussblur_colour_integer_s1.6_direct": (31, 1),
    # the uchar edge arm: scale 2, offset 128, and `2 * (p - 128)` doubles
    # each inner gap, on both gradients at once
    "sobel_sample_mono": (44177, 4),
    "sobel_sample_colour": (134545, 4),
    # deliberately-divergent fixtures added by #558 (see section 11)
    "discrim_boxsum1147_blur_integer": (9, 1),
    "discrim_gaussblur_noise_s0.8_integer": (1550, 2),
    "discrim_hostile_mask_noise_integer": (155, 73),
    "discrim_sobel_noise_mono": (412, 4),
}


def sh_quote(s):
    if all(c.isalnum() or c in "._/-" for c in s):
        return s
    return "'" + s.replace("'", "'\\''") + "'"


def rel(p):
    """Path as it should appear in a COMMITTED record: relative to this file.

    Absolute paths from whatever checkout happened to run the capture must not
    reach oracle.json or commands.sh -- they are worktree-specific noise that
    makes two identical captures diff. The one external input, the libvips
    reference suite's sample.jpg, becomes $REF_IMAGES/sample.jpg.
    """
    if not isinstance(p, str) or not os.path.isabs(p):
        return p
    if p.startswith(REF_IMAGES + os.sep):
        return "$REF_IMAGES/" + os.path.relpath(p, REF_IMAGES)
    r = os.path.relpath(p, ROOT)
    return p if r.startswith("..") else r


def pretty(cmd):
    """A command line, shell-quoted, with every path relative to this file."""
    return " ".join(sh_quote(rel(c)) for c in cmd)


def run(cmd, novector=False, log=True, info=False):
    """Run a command (list of str), return stripped stdout text.

    `novector=True` sets VIPS_NOVECTOR=1 for this one call, which is how
    libvips is told to run `vips_convi_gen` -- the portable C loop -- instead
    of the HWY vector path (`convi.c:1271-1284` documents the switch). The
    variable is read at library init, so it has to be in the environment of
    the process, which is why this is a per-call env rather than a global.

    `info=True` additionally returns VIPS_INFO=1's diagnostic text, which is
    where libvips prints which of the two kernels it picked. GLib's default
    log writer sends INFO to **stdout**, not stderr, so both streams are
    returned joined -- reading only stderr silently finds nothing.
    """
    env = dict(os.environ)
    env.pop("VIPS_NOVECTOR", None)
    env.pop("VIPS_INFO", None)
    if novector:
        env["VIPS_NOVECTOR"] = "1"
    if info:
        env["VIPS_INFO"] = "1"
    if log:
        COMMANDS_LOG.append(
            ("VIPS_NOVECTOR=1 " if novector else "") + pretty(cmd)
        )
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT, env=env)
    if res.returncode != 0:
        raise RuntimeError(
            f"command failed: {' '.join(cmd)}\nstdout={res.stdout}\nstderr={res.stderr}"
        )
    if info:
        return res.stdout.strip(), res.stdout + "\n" + res.stderr
    return res.stdout.strip()


def convi_decision(cmd):
    """Which integer-conv kernel libvips picks for `cmd`, and why.

    Returns (kernel, intize_complaint) where kernel is "vector", "C" or
    "none" (the op never reached `vips_convi`). `convi.c` prints exactly one
    of "convi: using vector path" / "convi: using C path" under VIPS_INFO=1,
    so this reads the decision OFF THE BINARY rather than re-deriving
    `vips_convi_intize`'s predicate here and hoping the two stay in step.

    The caller must hand in a command whose output path is a throwaway: this
    runs the op with the vector path live, and must not be allowed to
    clobber the pinned portable-C artefact.
    """
    _, err = run(cmd, log=False, info=True)
    kernel = "none"
    if "convi: using vector path" in err:
        kernel = "vector"
    elif "convi: using C path" in err:
        kernel = "C"
    reason = None
    for line in err.splitlines():
        if "vips_convi_intize:" in line:
            reason = line.split("vips_convi_intize:", 1)[1].strip()
    return kernel, reason


FMT_STRUCT = {
    "uchar": ("B", 1),
    "char": ("b", 1),
    "ushort": ("H", 2),
    "short": ("h", 2),
    "uint": ("I", 4),
    "int": ("i", 4),
    "float": ("f", 4),
    "double": ("d", 8),
}


def raster(path):
    """(header, every sample in raster order, sha256 of the pixel bytes).

    `vips rawsave` writes the samples with no header and no trailer, so this
    is the pixels and nothing else. Everything a record pins about an image
    comes from here rather than from the file on disk, because the file on
    disk is a container: a 64-byte header, the pixels, and an XML trailer
    whose namespace carries VIPS_MICRO_VERSION (vips.c:857-859). Hashing the
    container makes a patch release move all 452 pins at once with no pixel
    moving, which is what #649 is about.

    The `.raw` dumps are large and land in fixtures/outputs/, which
    .gitignore already excludes. They are cached per path: several callers
    want the same image and rawsave is not free at 384540 samples.
    """
    st = os.stat(path)
    key = (path, st.st_size, st.st_mtime_ns)
    cached = RASTER_CACHE.get(key)
    if cached is not None:
        return cached
    hdr = vipsheader(path)
    raw = path + ".raw"
    if os.path.exists(raw):
        os.remove(raw)
    run(["vips", "rawsave", path, raw], log=False)
    with open(raw, "rb") as f:
        blob = f.read()
    code, size = FMT_STRUCT[hdr["format"]]
    vals = list(struct.unpack("<%d%s" % (len(blob) // size, code), blob))
    out = (hdr, vals, hashlib.sha256(blob).hexdigest())
    RASTER_CACHE[key] = out
    return out


RASTER_CACHE = {}


def pixels(path):
    """Every sample of `path`, in raster order, exact."""
    return raster(path)[1]


def vipsheader(path):
    """Return dict of width/height/bands/format/interpretation for path."""
    out = run(["vipsheader", "-a", path], log=False)
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


def avg(path):
    return float(run(["vips", "avg", path], log=False))


def extreme(hdr, vals, want_max, reported):
    """The extreme sample, with a position that survives a re-run.

    `vips min --x --y` reports where a worker thread happened to find the
    extreme. 614 of the 904 extremes in this capture sit on a tie, so which
    coordinate comes back depends on which thread got there first, and two
    runs of the identical binary on the identical fixtures differ at 478
    leaves purely from that (#649). A pin that cannot survive a re-run of the
    same binary on the same input is worse than no pin, because it trains
    everyone to ignore a red diff.

    So the VALUE still comes from the binary and the POSITION is recomputed
    here as the first occurrence in raster order, which is the shape
    canny/capture.py already uses. `ties` says how many positions hold the
    extreme: 1 means this position is the only answer vips could have given,
    anything more means it is one of several and only the raster-order rule
    makes it reproducible.

    `reported` is what `vips min`/`vips max` actually answered, and it is
    asserted rather than recorded: the value must agree, and when the extreme
    is unique the position must agree too. That keeps the binary in the loop
    instead of quietly replacing it with arithmetic done here.
    """
    target = max(vals) if want_max else min(vals)
    w, b = hdr["width"], hdr["bands"]
    first = vals.index(target)
    ties = len({i // b for i, v in enumerate(vals) if v == target})
    pos = {
        "value": target,
        "x": (first // b) % w,
        "y": (first // b) // w,
        "band": first % b,
        "ties": ties,
    }
    op = "max" if want_max else "min"
    got_value, got_x, got_y = reported
    # The CLI prints the extreme with six decimals, so it can be half a unit
    # of the sixth decimal away from the sample and no further. `value` is the
    # SAMPLE, not the print: the old pin recorded 7.0 where the float32 sample
    # is 6.999999523162842, and a parity test comparing exactly against the
    # print would have been comparing against a rounding.
    if abs(got_value - target) > 1e-6 + 1e-9 * abs(target):
        raise AssertionError(
            "%s: vips %s printed %r, the raster says %r, and that is further "
            "apart than six-decimal printing can explain"
            % (op, op, got_value, target)
        )
    if ties == 1 and (got_x, got_y) != (pos["x"], pos["y"]):
        raise AssertionError(
            "%s: vips %s --x --y says (%d,%d) but only (%d,%d) holds %r"
            % (op, op, got_x, got_y, pos["x"], pos["y"], target)
        )
    return pos


def minmax(path, hdr, vals):
    """The min and max blocks for one image, positions deterministic."""
    mn = run(["vips", "min", path, "--x", "--y"], log=False).splitlines()
    mx = run(["vips", "max", path, "--x", "--y"], log=False).splitlines()
    return {
        "min": extreme(hdr, vals, False,
                       (float(mn[2]), int(mn[0]), int(mn[1]))),
        "max": extreme(hdr, vals, True,
                       (float(mx[2]), int(mx[0]), int(mx[1]))),
    }


def getpoint(path, x, y):
    out = run(["vips", "getpoint", path, str(x), str(y)], log=False)
    return [float(v) for v in out.split()]


def out_path(name):
    return os.path.join(OUT, name + ".v")


VECTOR_ARM = "__vector"


def describe(path, points):
    """The summary block a record carries for one output image."""
    hdr, vals, sha = raster(path)
    return {
        "path": os.path.relpath(path, ROOT),
        **hdr,
        "raw_sha256": sha,
        "avg": avg(path),
        **minmax(path, hdr, vals),
        "points": {f"{x},{y}": getpoint(path, x, y) for (x, y) in points},
    }


def run_op(cmd, name, dual):
    """Execute one capture command, on one or both of libvips's conv paths.

    With `dual`, the same command runs twice and the PINNED artefact at
    `out_path(name)` is always the `VIPS_NOVECTOR=1` one: `vips_convi_gen`,
    the portable C that libvips's docs call the specification and that
    libviprs implements (#558). The vectorised run lands next to it as
    `<name>__vector.v` and is compared sample by sample.

    Without `dual` the command runs once, unwrapped, exactly as before --
    that is for the ops that cannot reach `vips_convi` at all.
    """
    if not dual:
        run(cmd)
        return None
    pinned = out_path(name)
    vec_out = out_path(name + VECTOR_ARM)
    run([vec_out if c == pinned else c for c in cmd], novector=False)
    run(cmd, novector=True)
    return vec_out


def compare_paths(name, vec_out, points):
    """The vector-vs-novector block, measured on the samples themselves."""
    pinned = out_path(name)
    a = pixels(vec_out)
    b = pixels(pinned)
    diffs = [
        {"index": i, "vector": a[i], "novector": b[i], "delta": a[i] - b[i]}
        for i in range(min(len(a), len(b)))
        if a[i] != b[i]
    ]
    block = {
        "pinned_arm": "novector",
        "vector": describe(vec_out, points),
        "novector": describe(pinned, points),
        "paths_agree": not diffs and len(a) == len(b),
        "path_diff_count": len(diffs),
        "path_diff_max_abs": max((abs(d["delta"]) for d in diffs), default=0),
        "path_diff_sample": diffs[:24],
        "samples": len(b),
    }
    return block


def record_output(op, name, cmd, precision, points, note="", vec_out=None):
    """Common post-processing: header, avg, minmax, sha256, requested points.

    The top-level `output`/`avg`/`min`/`max`/`points` are the PINNED values,
    i.e. the portable-C path whenever this record was captured on both. When
    `vec_out` is given the record also carries the full vector/novector
    comparison, and the measured `paths_agree` is checked against
    KNOWN_DIVERGENT -- a record that changes regime fails the capture here
    rather than drifting quietly inside somebody's tolerance.
    """
    p = out_path(name)
    hdr, vals, sha = raster(p)
    rec = {
        "op": op,
        "record_id": name,
        "command": pretty(cmd),
        "precision": precision,
        "output": {
            "path": os.path.relpath(p, ROOT),
            **hdr,
            "raw_sha256": sha,
        },
        "avg": avg(p),
        **minmax(p, hdr, vals),
        "points": {f"{x},{y}": getpoint(p, x, y) for (x, y) in points},
    }
    if vec_out is not None:
        block = compare_paths(name, vec_out, points)
        rec.update(block)
        expected_agree = name not in KNOWN_DIVERGENT
        rec["expects_paths_agree"] = expected_agree
        if expected_agree != rec["paths_agree"]:
            raise AssertionError(
                "%s: paths_agree is %s, KNOWN_DIVERGENT says it should be %s "
                "(%d of %d samples differ, max abs %d). libvips moved this "
                "mask across vips_convi_intize's accept/reject boundary; the "
                "pinned bytes and the note next to them need re-deriving, not "
                "a wider tolerance."
                % (name, rec["paths_agree"], expected_agree,
                   rec["path_diff_count"], rec["samples"],
                   rec["path_diff_max_abs"])
            )
        if not expected_agree:
            want = KNOWN_DIVERGENT[name]
            got = (rec["path_diff_count"], rec["path_diff_max_abs"])
            if want != got:
                raise AssertionError(
                    "%s: measured divergence %r, KNOWN_DIVERGENT records %r"
                    % (name, got, want)
                )
    if note:
        rec["note"] = note
    RECORDS.append(rec)
    return rec


COMMANDS_LOG.append("#!/bin/sh")
COMMANDS_LOG.append("# Reproducible vips CLI commands for the convolution oracle capture.")
COMMANDS_LOG.append("# Run from the 'convolution' directory (paths are relative to it).")
COMMANDS_LOG.append("set -e")
COMMANDS_LOG.append("")

print("Setup done, ROOT =", ROOT)

# ---------------------------------------------------------------------------
# 1. Fixtures
# ---------------------------------------------------------------------------

KERNELS = {
    # name: (width, height, scale, offset, rows)
    "sharp": (3, 3, 8, 0, [[-1, -1, -1], [-1, 16, -1], [-1, -1, -1]]),
    "blur": (3, 3, 9, 0, [[1, 1, 1], [1, 1, 1], [1, 1, 1]]),
    "line": (3, 3, 1, 0, [[1, 1, 1], [-2, -2, -2], [1, 1, 1]]),
    "sobelk": (3, 3, 1, 0, [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]),
    "box5": (5, 5, 25, 0, [[1] * 5 for _ in range(5)]),
    # deliberately asymmetric (non-square, nonzero offset, no rotational/
    # reflective symmetry) so orientation and offset handling are pinned.
    "asym": (3, 2, 4, 10, [[1, 2, 3], [-1, 0, 1]]),
    # true 1D (separable) kernel: convasep requires width==1 or height==1
    # (a real 1xN/Nx1 matrix), it does not decompose a 2D mask itself.
    "box5_1d": (5, 1, 5, 0, [[1, 1, 1, 1, 1]]),
}


def write_mat(name, w, h, scale, offset, rows):
    path = os.path.join(KER, name + ".mat")
    lines = [f"{w} {h} {scale} {offset}"]
    for row in rows:
        lines.append(" ".join(str(v) for v in row))
    text = "\n".join(lines) + "\n"
    with open(path, "w") as f:
        f.write(text)
    heredoc = "\n".join(
        [f"cat > {os.path.relpath(path, ROOT)} << 'EOF'"] + lines + ["EOF"]
    )
    COMMANDS_LOG.append(f"# kernel: {name}.mat")
    COMMANDS_LOG.append(heredoc)
    COMMANDS_LOG.append("")
    return path


for name, (w, h, scale, offset, rows) in KERNELS.items():
    write_mat(name, w, h, scale, offset, rows)

# The exact synthetic colour fixture from ported_convolution.rs::make_test_colour:
# mask_ideal(100,100, 0.5, reject, optical)*[1,2,3] + [2,3,4]. r = sqrt(dx^2+dy^2),
# dx=x/w, dy=y/h; m=1 if r>0.5 else 0. Band values are (m+2, 2m+3, 3m+4), i.e.
# either (2,3,4) [m=0] or (3,5,7) [m=1]. Probe points (25,50)/(50,50) (and the
# point_conv anchors (24,49)/(49,49) for up to a 9x9 kernel support) lie
# entirely in the flat m=1 region, so every kernel/precision combination over
# this fixture has an exact, non-clipping expected output at those points.
COMMANDS_LOG.append("# fixture: colour_test.ppm (synthetic mask_ideal pattern, generated by capture.py)")
w, h = 100, 100
data = bytearray(w * h * 3)
for y in range(h):
    for x in range(w):
        dx = x / w
        dy = y / h
        r = (dx * dx + dy * dy) ** 0.5
        m = 1 if r > 0.5 else 0
        off = (y * w + x) * 3
        data[off] = m + 2
        data[off + 1] = m * 2 + 3
        data[off + 2] = m * 3 + 4
ppm_path = os.path.join(FIX, "colour_test.ppm")
with open(ppm_path, "wb") as f:
    f.write(f"P6\n{w} {h}\n255\n".encode())
    f.write(data)

colour_v = os.path.join(FIX, "colour.v")
run(["vips", "copy", ppm_path, colour_v])
mono_v = os.path.join(FIX, "mono.v")
run(["vips", "extract_band", colour_v, mono_v, "1"])

# Impulse-response canvas: 21x21 black image with a single white (255) pixel
# at the centre (10,10). Convolving this with a kernel K reveals K itself
# (scaled/offset per the op's precision rules) directly in the output
# neighbourhood around (10,10); this is the exact-oracle technique for
# pinning kernel semantics (flip/no-flip, edge extend, offset handling).
impulse_black = os.path.join(FIX, "impulse_black.v")
run(["vips", "black", impulse_black, "21", "21"])
impulse_mono = os.path.join(FIX, "impulse_mono.v")
run(["vips", "copy", impulse_black, impulse_mono])
# draw_rect's "image" argument is a MODIFY arg: the CLI loads it, draws in
# place, and rewrites the same file, so we draw onto the copy above.
run(["vips", "draw_rect", impulse_mono, "255", "10", "10", "1", "1"])

sample_jpg = os.path.join(REF_IMAGES, "sample.jpg")
sample_colour = os.path.join(FIX, "sample_colour.v")
run(["vips", "copy", sample_jpg, sample_colour])
sample_mono = os.path.join(FIX, "sample_mono.v")
run(["vips", "extract_band", sample_colour, sample_mono, "1"])

# Deterministic high-entropy noise fixtures (lossless) for fastcor/spcor,
# where a repeated-pattern flat fixture would give a non-unique best match.
import random

rng = random.Random(42)
nw, nh = 64, 64
noise_mono = bytes(rng.randrange(256) for _ in range(nw * nh))
noise_pgm = os.path.join(FIX, "noise_mono.pgm")
with open(noise_pgm, "wb") as f:
    f.write(f"P5\n{nw} {nh}\n255\n".encode())
    f.write(noise_mono)
COMMANDS_LOG.append("# fixture: noise_mono.pgm (deterministic PRNG seed=42, generated by capture.py)")
noise_mono_v = os.path.join(FIX, "noise_mono.v")
run(["vips", "copy", noise_pgm, noise_mono_v])

rng2 = random.Random(43)
noise_rgb = bytes(rng2.randrange(256) for _ in range(nw * nh * 3))
noise_ppm = os.path.join(FIX, "noise_colour.ppm")
with open(noise_ppm, "wb") as f:
    f.write(f"P6\n{nw} {nh}\n255\n".encode())
    f.write(noise_rgb)
COMMANDS_LOG.append("# fixture: noise_colour.ppm (deterministic PRNG seed=43, generated by capture.py)")
noise_colour_v = os.path.join(FIX, "noise_colour.v")
run(["vips", "copy", noise_ppm, noise_colour_v])

print("Fixtures written.")

# ---------------------------------------------------------------------------
# 2. conv: impulse response per kernel (exact kernel reveal)
# ---------------------------------------------------------------------------
COMMANDS_LOG.append("")
COMMANDS_LOG.append("# --- conv: impulse response (reveals each kernel exactly) ---")

impulse_pts = [(x, y) for x in range(8, 13) for y in range(8, 13)] + [(0, 0), (20, 20)]

for kname in KERNELS:
    for precision in ["integer", "float"]:
        rec_name = f"conv_impulse_{kname}_{precision}"
        cmd = [
            "vips", "conv", impulse_mono, out_path(rec_name),
            os.path.join(KER, kname + ".mat"), "--precision", precision,
        ]
        vec = run_op(cmd, rec_name, dual=True)
        record_output(
            "conv", rec_name, cmd, precision, impulse_pts, vec_out=vec,
            note=f"impulse response of kernel '{kname}' (see fixtures/kernels/{kname}.mat); "
                 "white=255 impulse at (10,10) on a 21x21 black canvas",
        )

# ---------------------------------------------------------------------------
# 3. conv: on the exact Rust make_test_colour fixture (mono + colour),
#    matching test_conv's probe points (25,50) and (50,50).
# ---------------------------------------------------------------------------
COMMANDS_LOG.append("")
COMMANDS_LOG.append("# --- conv: on make_test_colour fixture (matches ported test_conv probes) ---")

conv_kernels = ["sharp", "blur", "line", "sobelk"]
conv_probe_pts = [(25, 50), (50, 50), (24, 49), (49, 49)]

for img_name, img_path in [("mono", mono_v), ("colour", colour_v)]:
    for kname in conv_kernels:
        for precision in ["integer", "float"]:
            rec_name = f"conv_{img_name}_{kname}_{precision}"
            cmd = [
                "vips", "conv", img_path, out_path(rec_name),
                os.path.join(KER, kname + ".mat"), "--precision", precision,
            ]
            vec = run_op(cmd, rec_name, dual=True)
            record_output(
                "conv", rec_name, cmd, precision, conv_probe_pts, vec_out=vec,
                note="probes (25,50)/(50,50) are the ported test_conv assertion points; "
                     "both lie in the fixture's flat mask=1 region so the kernel never clips",
            )

print("conv done:", len(RECORDS))

# ---------------------------------------------------------------------------
# 4. compass: rotate-and-combine convolution
# ---------------------------------------------------------------------------
COMMANDS_LOG.append("")
COMMANDS_LOG.append("# --- compass: impulse response (exact kernel-rotation reveal) ---")

compass_pts = [(x, y) for x in range(7, 14) for y in range(7, 14)]

for times in [1, 2, 3]:
    for combine in ["max", "sum"]:
        for precision in ["integer", "float"]:
            rec_name = f"compass_impulse_sharp_t{times}_{combine}_{precision}"
            cmd = [
                "vips", "compass", impulse_mono, out_path(rec_name),
                os.path.join(KER, "sharp.mat"),
                "--times", str(times), "--angle", "d45", "--combine", combine,
                "--precision", precision,
            ]
            vec = run_op(cmd, rec_name, dual=True)
            record_output(
                "compass", rec_name, cmd, precision, compass_pts, vec_out=vec,
                note=f"sharp kernel rotated d45 x{times}, combine={combine}; "
                     "impulse at (10,10) on 21x21 black canvas",
            )

COMMANDS_LOG.append("")
COMMANDS_LOG.append("# --- compass: on ported test_compass inputs (sample.jpg mono/colour) ---")
COMMANDS_LOG.append("# NB: test_compass only asserts output dims == input dims (no numeric probe);")
COMMANDS_LOG.append("# we still capture avg + a probe point as a behavioral spot-check.")

for img_name, img_path in [("mono", sample_mono), ("colour", sample_colour)]:
    in_hdr = vipsheader(img_path)
    for times in [1, 2, 3]:
        for combine in ["max", "sum"]:
            for precision in ["integer", "float"]:
                rec_name = f"compass_sample_{img_name}_t{times}_{combine}_{precision}"
                cmd = [
                    "vips", "compass", img_path, out_path(rec_name),
                    os.path.join(KER, "sharp.mat"),
                    "--times", str(times), "--angle", "d45", "--combine", combine,
                    "--precision", precision,
                ]
                vec = run_op(cmd, rec_name, dual=True)
                rec = record_output(
                    "compass", rec_name, cmd, precision, [(25, 50)], vec_out=vec,
                    note="JPEG-derived input (sample.jpg): dims/avg/probe are exact for this "
                         "real-vips run, but treat as tolerance-band oracle if compared against "
                         "a different JPEG decoder",
                )
                assert rec["output"]["width"] == in_hdr["width"]
                assert rec["output"]["height"] == in_hdr["height"]

print("compass done:", len(RECORDS))

# ---------------------------------------------------------------------------
# 5. convsep: separable convolution vs conv with the 2D kernel
# ---------------------------------------------------------------------------
COMMANDS_LOG.append("")
COMMANDS_LOG.append("# --- convsep: gaussmat(sigma=2, min_ampl=0.1) 2D vs separable ---")


def gaussmat(sigma, min_ampl, separable, precision, tag):
    name = f"gaussmat_{tag}_{'sep' if separable else '2d'}_{precision}"
    path = os.path.join(KER, name + ".mat")
    cmd = ["vips", "gaussmat", path, str(sigma), str(min_ampl), "--precision", precision]
    if separable:
        cmd.append("--separable")
    run(cmd)
    hdr = vipsheader(path)
    matrix_text = run(["vips", "matrixprint", path], log=False)
    return path, hdr, matrix_text


for img_name, img_path in [("mono", sample_mono), ("colour", sample_colour)]:
    for precision in ["integer", "float"]:
        g2d_path, g2d_hdr, g2d_txt = gaussmat(2.0, 0.1, False, precision, "sample")
        gsep_path, gsep_hdr, gsep_txt = gaussmat(2.0, 0.1, True, precision, "sample")

        assert g2d_hdr["width"] == g2d_hdr["height"], "2D gaussmat must be square"
        assert gsep_hdr["width"] == g2d_hdr["width"]
        assert gsep_hdr["height"] == 1, "separable gaussmat must be 1 row"

        rec_a = f"convsep_{img_name}_conv2d_{precision}"
        cmd_a = ["vips", "conv", img_path, out_path(rec_a), g2d_path, "--precision", precision]
        vec_a = run_op(cmd_a, rec_a, dual=True)
        ra = record_output("conv", rec_a, cmd_a, precision, [(25, 50)], vec_out=vec_a,
                            note="2D gaussmat via conv(); compare against convsep below")
        ra["kernel"] = {"path": rel(g2d_path), "matrix": g2d_txt}

        rec_b = f"convsep_{img_name}_convsep1d_{precision}"
        cmd_b = ["vips", "convsep", img_path, out_path(rec_b), gsep_path, "--precision", precision]
        vec_b = run_op(cmd_b, rec_b, dual=True)
        rb = record_output("convsep", rec_b, cmd_b, precision, [(25, 50)], vec_out=vec_b,
                            note="separable 1D gaussmat via convsep(); compare against conv2d above")
        rb["kernel"] = {"path": rel(gsep_path), "matrix": gsep_txt}

# Lossless bonus on our exact synthetic fixture (not exercised by the ported
# test, which uses sample.jpg, but gives a byte-exact comparison with no
# JPEG-decode ambiguity).
for precision in ["integer", "float"]:
    g2d_path, g2d_hdr, g2d_txt = gaussmat(2.0, 0.1, False, precision, "synth")
    gsep_path, gsep_hdr, gsep_txt = gaussmat(2.0, 0.1, True, precision, "synth")

    rec_a = f"convsep_synthmono_conv2d_{precision}"
    cmd_a = ["vips", "conv", mono_v, out_path(rec_a), g2d_path, "--precision", precision]
    vec_a = run_op(cmd_a, rec_a, dual=True)
    ra = record_output("conv", rec_a, cmd_a, precision, [(25, 50)], vec_out=vec_a,
                        note="lossless bonus: synthetic make_test_colour mono fixture, 2D gaussmat")
    ra["kernel"] = {"path": rel(g2d_path), "matrix": g2d_txt}

    rec_b = f"convsep_synthmono_convsep1d_{precision}"
    cmd_b = ["vips", "convsep", mono_v, out_path(rec_b), gsep_path, "--precision", precision]
    vec_b = run_op(cmd_b, rec_b, dual=True)
    rb = record_output("convsep", rec_b, cmd_b, precision, [(25, 50)], vec_out=vec_b,
                        note="lossless bonus: synthetic make_test_colour mono fixture, separable gaussmat")
    rb["kernel"] = {"path": rel(gsep_path), "matrix": gsep_txt}

print("convsep done:", len(RECORDS))

# ---------------------------------------------------------------------------
# 6. gaussblur: conv(gaussmat) vs the gaussblur convenience op, on the exact
#    Rust make_test_colour fixture (matches ported test_gaussblur exactly:
#    same fixture, same sigma sweep, same probe point).
# ---------------------------------------------------------------------------
COMMANDS_LOG.append("")
COMMANDS_LOG.append("# --- gaussblur: conv(gaussmat) vs gaussblur() on make_test_colour fixture ---")

for img_name, img_path in [("mono", mono_v), ("colour", colour_v)]:
    for precision in ["integer", "float"]:
        for i in range(5, 10):
            sigma = i / 5.0
            tag = f"{img_name}_{precision}_s{sigma}"
            g_path, g_hdr, g_txt = gaussmat(sigma, 0.2, False, precision, tag)

            rec_a = f"gaussblur_{tag}_conv"
            cmd_a = ["vips", "conv", img_path, out_path(rec_a), g_path, "--precision", precision]
            vec_a = run_op(cmd_a, rec_a, dual=True)
            ra = record_output("conv", rec_a, cmd_a, precision, [(25, 50)], vec_out=vec_a,
                                note=f"conv(gaussmat(sigma={sigma}, min_ampl=0.2)); "
                                     "probe (25,50) is in the fixture's flat mask=1 region")
            ra["kernel"] = {"path": rel(g_path), "matrix": g_txt, "sigma": sigma}

            rec_b = f"gaussblur_{tag}_direct"
            cmd_b = ["vips", "gaussblur", img_path, out_path(rec_b), str(sigma),
                     "--min-ampl", "0.2", "--precision", precision]
            vec_b = run_op(cmd_b, rec_b, dual=True)
            rb = record_output("gaussblur", rec_b, cmd_b, precision, [(25, 50)], vec_out=vec_b,
                                note=f"gaussblur(sigma={sigma}, min_ampl=0.2) direct convenience op; "
                                     "compare against conv(gaussmat) above")

            av = ra["points"]["25,50"]
            bv = rb["points"]["25,50"]
            max_delta = max(abs(x - y) for x, y in zip(av, bv))
            rb["conv_vs_gaussblur_delta"] = max_delta
            # Ported test_gaussblur uses a < 1.0 tolerance; real vips itself
            # is not bit-identical between the two code paths (tiny float
            # rounding, e.g. 5.000000476837158 vs 5.0 at float precision),
            # so we assert the same tolerance the port checks rather than
            # exact equality, and record the observed delta for the record.
            assert max_delta < 1.0, f"gaussblur/conv mismatch at sigma={sigma}: {av} vs {bv}"

print("gaussblur done:", len(RECORDS))

# ---------------------------------------------------------------------------
# 7. sharpen: unsharp mask, on the ported test_sharpen input (sample.jpg),
#    including the m1=0,m2=0 identity check the ported test asserts.
# ---------------------------------------------------------------------------
COMMANDS_LOG.append("")
COMMANDS_LOG.append("# --- sharpen: unsharp mask on sample.jpg (matches ported test_sharpen) ---")

for img_name, img_path in [("mono", sample_mono), ("colour", sample_colour)]:
    in_bands = vipsheader(img_path)["bands"]
    band_note = ""
    if img_name == "mono":
        # IMPORTANT real-vips finding: sharpen is colourspace-aware (it runs
        # the unsharp mask in LabS). sample_mono.v is `extract_band(1)` of an
        # srgb-interpretation JPEG, so it is 1 band of PIXEL data but still
        # tagged interpretation=srgb. vips_colourspace's srgb route assumes 3
        # bands are present, so sharpen's output on this input is promoted
        # to 3 bands (each band equal to the original L value when m1=m2=0;
        # confirmed via vips subtract's auto band-broadcast, diff=0). If the
        # source is instead tagged interpretation=b-w (a "real" mono image),
        # sharpen stays 1-band; if tagged interpretation=multiband (e.g. our
        # impulse fixture from `vips black`), sharpen hard-errors with
        # "no known route from 'labs' to 'multiband'". The ported Rust test
        # only asserts width/height equality (not band count), so this
        # 1-band -> 3-band promotion would NOT be caught by test_sharpen as
        # currently written; a libviprs mono Raster without an equivalent
        # sRGB-interpretation tag will very likely NOT reproduce this
        # promotion, so exact numeric/shape parity with this record should
        # not be expected unless libviprs's Raster model tracks the same
        # interpretation metadata as real vips.
        band_note = (
            " NOTE: real vips promotes this srgb-tagged 1-band input to a "
            "3-band output (see oracle.json meta.sharpen_mono_band_promotion "
            "for the full explanation); dims (w,h) still match but bands do not."
        )

    for sigma in [0.5, 1.0, 1.5, 2.0]:
        rec_s = f"sharpen_{img_name}_s{sigma}_m1_2"
        cmd_s = ["vips", "sharpen", img_path, out_path(rec_s),
                 "--sigma", str(sigma), "--m1", "1", "--m2", "2"]
        vec_s = run_op(cmd_s, rec_s, dual=True)
        record_output("sharpen", rec_s, cmd_s, "n/a", [(25, 50), (50, 50)], vec_out=vec_s,
                       note="sharpen(sigma, m1=1, m2=2); dims must equal input dims." + band_note)

        rec_n = f"sharpen_{img_name}_s{sigma}_noop"
        cmd_n = ["vips", "sharpen", img_path, out_path(rec_n),
                 "--sigma", str(sigma), "--m1", "0", "--m2", "0"]
        vec_n = run_op(cmd_n, rec_n, dual=True)
        rn = record_output("sharpen", rec_n, cmd_n, "n/a", [(25, 50), (50, 50)], vec_out=vec_n,
                            note="sharpen(sigma, m1=0, m2=0) should be an identity transform." + band_note)
        rn["input_bands"] = in_bands
        rn["output_bands_differs_from_input"] = rn["output"]["bands"] != in_bands

        # Identity check: max |orig - noop| must be 0, matching the ported
        # test's assertion (computed the same way: subtract, abs, max).
        diff_name = f"sharpen_{img_name}_s{sigma}_diff"
        diff_path = out_path(diff_name)
        run(["vips", "subtract", img_path, out_path(rec_n), diff_path + ".sub.v"])
        run(["vips", "abs", diff_path + ".sub.v", diff_path])
        max_diff = float(run(["vips", "max", diff_path], log=False))
        rn["identity_check"] = {
            "command": f"vips subtract {os.path.relpath(img_path, ROOT)} "
                       f"{os.path.relpath(out_path(rec_n), ROOT)} <tmp>.v && "
                       f"vips abs <tmp>.v <diff>.v && vips max <diff>.v",
            "max_abs_diff": max_diff,
        }
        assert max_diff == 0.0, f"sharpen noop not identity at sigma={sigma}: max_diff={max_diff}"

print("sharpen done:", len(RECORDS))

# ---------------------------------------------------------------------------
# 8. fastcor / spcor: template matching (matches ported test_fastcor/spcor,
#    plus a lossless high-entropy-noise bonus for a byte-exact record).
# ---------------------------------------------------------------------------
COMMANDS_LOG.append("")
COMMANDS_LOG.append("# --- fastcor / spcor: template matching ---")


def corr_case(tag, img_path, left, top, tw, th, exp_x, exp_y):
    template = out_path(f"{tag}_template")
    cmd_t = ["vips", "extract_area", img_path, template, str(left), str(top), str(tw), str(th)]
    run(cmd_t)

    rec_f = f"{tag}_fastcor"
    cmd_f = ["vips", "fastcor", img_path, template, out_path(rec_f)]
    run(cmd_f)
    rf = record_output("fastcor", rec_f, cmd_f, "n/a", [(exp_x, exp_y)],
                        note=f"template = extract_area({left},{top},{tw},{th}); "
                             f"expect global min 0 at ({exp_x},{exp_y})")
    rf["template"] = {"path": os.path.relpath(template, ROOT), "left": left, "top": top,
                       "width": tw, "height": th}
    assert rf["min"]["value"] == 0.0
    assert rf["min"]["x"] == exp_x and rf["min"]["y"] == exp_y, \
        f"fastcor {tag}: min at ({rf['min']['x']},{rf['min']['y']}), expected ({exp_x},{exp_y})"

    rec_s = f"{tag}_spcor"
    cmd_s = ["vips", "spcor", img_path, template, out_path(rec_s)]
    run(cmd_s)
    rs = record_output("spcor", rec_s, cmd_s, "n/a", [(exp_x, exp_y)],
                        note=f"template = extract_area({left},{top},{tw},{th}); "
                             f"expect global max ~1.0 at ({exp_x},{exp_y})")
    rs["template"] = {"path": os.path.relpath(template, ROOT), "left": left, "top": top,
                       "width": tw, "height": th}
    assert abs(rs["max"]["value"] - 1.0) < 0.001
    assert rs["max"]["x"] == exp_x and rs["max"]["y"] == exp_y, \
        f"spcor {tag}: max at ({rs['max']['x']},{rs['max']['y']}), expected ({exp_x},{exp_y})"


# Ported-test-matching cases: sample.jpg, patch (20,45,10,10) -> match at (25,50)
corr_case("sample_mono", sample_mono, 20, 45, 10, 10, 25, 50)
corr_case("sample_colour", sample_colour, 20, 45, 10, 10, 25, 50)

# Lossless bonus: deterministic noise fixtures, byte-exact, unambiguous match
corr_case("noise_mono", noise_mono_v, 20, 20, 10, 10, 25, 25)
corr_case("noise_colour", noise_colour_v, 20, 20, 10, 10, 25, 25)

print("fastcor/spcor done:", len(RECORDS))

# ---------------------------------------------------------------------------
# 9. Supplementary ops named in the task brief (convasep/sobel/canny) that do
#    NOT appear as distinct ops in ported_convolution.rs today (sobel there
#    is only a raw 3x3 kernel matrix used inside test_conv, not the vips
#    Sobel-operator convenience op). Captured as forward-looking oracle
#    records in case the port grows dedicated bindings for them.
# ---------------------------------------------------------------------------
COMMANDS_LOG.append("")
COMMANDS_LOG.append("# --- supplementary: convasep / sobel / canny (not exercised by the current port) ---")

# NB: convasep requires a literal 1xN/Nx1 matrix ("separable matrix images
# must have width or height 1"); it does not decompose an arbitrary 2D mask
# itself the way conv's --precision approximate mode does. sharp.mat (3x3,
# not separable) and box5.mat (5x5, stored 2D even though separable in
# value) both fail with that error; box5_1d.mat (literally 1x5) works.
for kname in ["box5_1d"]:
    rec_i = f"convasep_impulse_{kname}"
    cmd_i = ["vips", "convasep", impulse_mono, out_path(rec_i), os.path.join(KER, kname + ".mat")]
    vec_i = run_op(cmd_i, rec_i, dual=True)
    record_output("convasep", rec_i, cmd_i, "integer-only (no precision option)", impulse_pts,
                   vec_out=vec_i,
                   note=f"approximate separable integer convolution; kernel={kname}; "
                        "convasep has no --precision flag (always integer/approximate); "
                        "requires a literal 1xN/Nx1 mask, unlike conv/convsep")

for img_name, img_path in [("mono", sample_mono), ("colour", sample_colour)]:
    rec = f"convasep_sample_{img_name}_box5_1d"
    cmd = ["vips", "convasep", img_path, out_path(rec), os.path.join(KER, "box5_1d.mat")]
    vec = run_op(cmd, rec, dual=True)
    record_output("convasep", rec, cmd, "integer-only (no precision option)", [(25, 50)],
                   vec_out=vec,
                   note="JPEG-derived input; approximate separable 1x5 box blur")

for tag, img_path in [("impulse", impulse_mono), ("sample_mono", sample_mono),
                       ("sample_colour", sample_colour)]:
    rec = f"sobel_{tag}"
    cmd = ["vips", "sobel", img_path, out_path(rec)]
    vec = run_op(cmd, rec, dual=True)
    pts = impulse_pts if tag == "impulse" else [(25, 50)]
    record_output("sobel", rec, cmd, "n/a", pts, vec_out=vec,
                   note="VipsSobel edge-detector convenience op (distinct from the raw 3x3 "
                        "sobel kernel matrix used inside conv); no --precision option")

for img_name, img_path in [("mono", sample_mono), ("colour", sample_colour)]:
    for sigma in [1.4]:
        for precision in ["integer", "float"]:
            rec = f"canny_{img_name}_s{sigma}_{precision}"
            cmd = ["vips", "canny", img_path, out_path(rec), "--sigma", str(sigma),
                   "--precision", precision]
            vec = run_op(cmd, rec, dual=True)
            record_output("canny", rec, cmd, precision, [(25, 50)], vec_out=vec,
                           note="Canny edge detector (gaussian blur + gradient + non-max "
                                "suppression + hysteresis); JPEG-derived input")

print("supplementary done:", len(RECORDS))

# ---------------------------------------------------------------------------
# 10. Discriminating fixtures (#558).
#
# Every record above where the two libvips paths disagree does so by
# accident: nobody picked those masks to separate the paths, they just did.
# The four below are chosen for it, so the suite keeps a witness even if the
# accidental ones stop witnessing.
# ---------------------------------------------------------------------------
COMMANDS_LOG.append("")
COMMANDS_LOG.append("# --- #558 discriminating fixtures: masks/sigmas chosen to SEPARATE ---")
COMMANDS_LOG.append("# --- libvips's two integer-convolution paths, not to agree with them ---")

# (a) the scale-9 box blur, on a window that sums to 1147.
#
# The C path computes (1147 + 9/2) / 9 = 127. Floor gives 127 too, so this
# is NOT a rounding-mode difference. The vector path filters with
# 57/512 = 0.111328 instead of 1/9 = 0.111111 and returns
# (57 * 1147 + 256) >> 9 = 128. Three pixels of headroom in a 3x3 image:
# every sample here is the same window, so the whole output moves.
COMMANDS_LOG.append("# fixture: boxsum1147.pgm (3x3, eight 127s around a 131; window sum 1147)")
box_vals = [127, 127, 127, 127, 131, 127, 127, 127, 127]
assert sum(box_vals) == 1147
box_pgm = os.path.join(FIX, "boxsum1147.pgm")
with open(box_pgm, "wb") as f:
    f.write(b"P5\n3 3\n255\n")
    f.write(bytes(box_vals))
box_v = os.path.join(FIX, "boxsum1147.v")
run(["vips", "copy", box_pgm, box_v])

box_pts = [(x, y) for y in range(3) for x in range(3)]
rec_box = "discrim_boxsum1147_blur_integer"
cmd_box = ["vips", "conv", box_v, out_path(rec_box),
           os.path.join(KER, "blur.mat"), "--precision", "integer"]
vec_box = run_op(cmd_box, rec_box, dual=True)
r_box = record_output(
    "conv", rec_box, cmd_box, "integer", box_pts, vec_out=vec_box,
    note="#558 discriminator. 3x3 box mask, scale 9, on a window summing to "
         "1147. C: (1147 + 4) / 9 = 127. Floor: 127 as well, so the rounding "
         "mode is NOT the mechanism. The vector path convolves with "
         "57/512 rather than 1/9 and returns (57 * 1147 + 256) >> 9 = 128. "
         "Every sample of this 3x3 sees the same replicated window, so the "
         "whole image moves by one.")

# (b) gaussblur at sigma 0.8. sigma 1.4, the default, has a separable
# gaussmat of scale 64 -- a power of two, so the two paths agree and a suite
# pinned only at the default sees none of this. 0.8 gives scale 38.
rec_g08 = "discrim_gaussblur_noise_s0.8_integer"
cmd_g08 = ["vips", "gaussblur", noise_mono_v, out_path(rec_g08), "0.8",
           "--min-ampl", "0.2", "--precision", "integer"]
vec_g08 = run_op(cmd_g08, rec_g08, dual=True)
record_output(
    "gaussblur", rec_g08, cmd_g08, "integer", [(20, 20), (32, 32), (63, 63)],
    vec_out=vec_g08,
    note="#558 discriminator. Separable gaussmat at sigma 0.8 has scale 38; "
         "the default sigma 1.4 has scale 64, a power of two, which is why "
         "a suite pinned at the default passes on both paths and sees "
         "nothing. Note the 2D gaussmat at 1.4 has scale 216 and is NOT "
         "lucky -- the luck is a property of the separable mask alone.")

# (c) a hostile mask that libvips's own accuracy gate accepts.
#
# vips_convi_intize's check (convi.c:1096-1113) compares the requantised
# mask against EXACT REAL ARITHMETIC at one grey level on a flat field. It
# is a DC-gain check: it constrains sum(w_hat - w) and says nothing at all
# about per-pixel error, which is sum((w_hat - w) * p). This mask sails
# through it and still moves samples by dozens.
COMMANDS_LOG.append("# kernel: hostile.mat (3x3 scale 3; passes vips_convi_intize, breaks any tolerance)")
hostile_rows = [[45, -17, -25], [-33, -15, -34], [55, 53, -26]]
hostile_path = write_mat("hostile", 3, 3, 3, 0, hostile_rows)

rec_h = "discrim_hostile_mask_noise_integer"
cmd_h = ["vips", "conv", noise_mono_v, out_path(rec_h), hostile_path,
         "--precision", "integer"]
vec_h = run_op(cmd_h, rec_h, dual=True)
r_h = record_output(
    "conv", rec_h, cmd_h, "integer", [(20, 20), (32, 32), (63, 63)],
    vec_out=vec_h,
    note="#558 bound-breaking regression. vips_convi_intize ACCEPTS this "
         "mask (VIPS_INFO reports 'convi: using vector path'), and the two "
         "paths still differ by far more than any tolerance the suite has "
         "ever written down. The gate is a DC-gain check against exact real "
         "arithmetic on a flat field (convi.c:1096-1113); it constrains "
         "sum(w_hat - w) and says nothing about sum((w_hat - w) * p). "
         "Nobody has ever bounded the gap between libvips's two paths.")

# (d) sobel on a lossless fixture. sobel_sample_{mono,colour} above show the
# same thing, but on a JPEG decode, so their bytes carry a decoder caveat.
rec_sn = "discrim_sobel_noise_mono"
cmd_sn = ["vips", "sobel", noise_mono_v, out_path(rec_sn)]
vec_sn = run_op(cmd_sn, rec_sn, dual=True)
record_output(
    "sobel", rec_sn, cmd_sn, "n/a", [(20, 20), (32, 32), (63, 63)],
    vec_out=vec_sn,
    note="#558 discriminator, lossless. The uchar edge arm stamps the mask "
         "scale 2 / offset 128 and recovers each gradient as 2 * (p - 128), "
         "so a one-unit inner gap comes out doubled, and Gx and Gy can both "
         "be off at once. sobel_sample_mono shows the same on sample.jpg; "
         "this one has no decoder caveat on its bytes.")

print("discriminators done:", len(RECORDS))

# ---------------------------------------------------------------------------
# 11. Regime boundaries (#558).
#
# The suite leans on all three of these implicitly and tests none of them.
# They are captured as assertions about the binary rather than as records,
# because what is being pinned is WHICH KERNEL RUNS, and libvips will tell
# you that directly under VIPS_INFO=1.
# ---------------------------------------------------------------------------
COMMANDS_LOG.append("")
COMMANDS_LOG.append("# --- #558 regime boundaries: which of the two kernels libvips picks ---")
COMMANDS_LOG.append("# --- (re-run any of these with VIPS_INFO=1 to see the decision) ---")

REGIMES = {}


def regime(tag, cmd, name, expect_agree, expect_kernel=None, note=""):
    """Run `cmd` on both paths and record which kernel libvips chose."""
    vec = run_op(cmd, name, dual=True)
    block = compare_paths(name, vec, [])
    # A THIRD run, into a throwaway output, purely to read the VIPS_INFO
    # decision. It must not write out_path(name): that file is the pinned
    # portable-C artefact and this run has the vector path live.
    probe = out_path(name + "__probe")
    kernel, reason = convi_decision(
        [probe if c == out_path(name) else c for c in cmd])
    entry = {
        "command": pretty(cmd),
        "convi_kernel": kernel,
        "intize_says": reason,
        "paths_agree": block["paths_agree"],
        "path_diff_count": block["path_diff_count"],
        "path_diff_max_abs": block["path_diff_max_abs"],
        "samples": block["samples"],
    }
    if note:
        entry["note"] = note
    REGIMES.setdefault(tag, {})[name] = entry
    assert entry["paths_agree"] == expect_agree, \
        "%s/%s: paths_agree=%s, expected %s (%d samples differ)" % (
            tag, name, entry["paths_agree"], expect_agree,
            entry["path_diff_count"])
    if expect_kernel is not None:
        assert kernel == expect_kernel, \
            "%s/%s: libvips ran the %s kernel, expected %s" % (
                tag, name, kernel, expect_kernel)
    return entry


# (i) ushort never diverges. convi.c:1151 gates the vector path on
#     BandFmt == VIPS_FORMAT_UCHAR, so a 16-bit image takes the C path on
#     both builds and libviprs cannot be wrong about it.
ushort_v = os.path.join(FIX, "noise_mono_ushort.v")
run(["vips", "cast", noise_mono_v, ushort_v, "ushort"])
regime("ushort_never_diverges",
       ["vips", "conv", ushort_v, out_path("regime_ushort_conv_blur"),
        os.path.join(KER, "blur.mat"), "--precision", "integer"],
       "regime_ushort_conv_blur", expect_agree=True, expect_kernel="C",
       note="convi.c:1151 gates the vector path on BandFmt == UCHAR")
regime("ushort_never_diverges",
       ["vips", "gaussblur", ushort_v, out_path("regime_ushort_gaussblur_s1.6"),
        "1.6", "--precision", "integer"],
       "regime_ushort_gaussblur_s1.6", expect_agree=True, expect_kernel="C",
       note="sigma 1.6 diverges on uchar (scale 70); on ushort it cannot")

# (ii) the low-sigma fallback, which is NOT the rule this epic recorded.
#
#      "sigma <= 0.6 falls back to the C path on both builds, because intize
#      bails too inaccurate" is false in both halves, and the assertions below
#      are what caught it:
#
#        sigma 0.5, separable 1x1 scale 20 -> VECTOR path, agrees anyway
#        sigma 0.5, 2D      1x1 scale 20   -> VECTOR path, agrees anyway
#        sigma 0.6, separable 3x1 scale 30 -> C path, "too inaccurate"   <- the
#                                             only case that actually falls back
#        sigma 0.6, 2D      3x3 scale 44   -> VECTOR path, 4065/4096 differ,
#                                             max 4
#
#      So there is no sigma threshold. There is a per-mask predicate, and the
#      2D and separable masks for the same sigma can land on opposite sides of
#      it. A test author cannot tell from the API which one they are in.
for sig, sep_scale, kernel in (("0.5", 20, "vector"), ("0.6", 30, "C")):
    regime("low_sigma",
           ["vips", "gaussblur", noise_mono_v,
            out_path("regime_gaussblur_s%s" % sig), sig,
            "--min-ampl", "0.2", "--precision", "integer"],
           "regime_gaussblur_s%s" % sig, expect_agree=True,
           expect_kernel=kernel,
           note="separable gaussmat scale %d; %s" % (
               sep_scale,
               "the vector path RUNS and agrees anyway (a 1x1 mask "
               "requantises exactly)" if kernel == "vector" else
               "vips_convi_intize declines this mask, so libvips runs "
               "vips_convi_gen itself"))
for sig, agree, divergence in (("0.5", True, None), ("0.6", False, (4065, 4))):
    gm = os.path.join(KER, "gaussmat_regime_s%s_2d_integer.mat" % sig)
    run(["vips", "gaussmat", gm, sig, "0.2", "--precision", "integer"])
    entry = regime("low_sigma",
                   ["vips", "conv", noise_mono_v,
                    out_path("regime_conv_gaussmat2d_s%s" % sig), gm,
                    "--precision", "integer"],
                   "regime_conv_gaussmat2d_s%s" % sig, expect_agree=agree,
                   expect_kernel="vector",
                   note="2D gaussmat, header %r. The 2D and the SEPARABLE "
                        "mask for this sigma do not agree with each other "
                        "about which regime they are in."
                        % open(gm).read().splitlines()[0])
    if divergence is not None:
        got = (entry["path_diff_count"], entry["path_diff_max_abs"])
        assert got == divergence, \
            "regime_conv_gaussmat2d_s%s: measured %r, recorded %r" % (
                sig, got, divergence)

# (iii) scale-1 masks are exact. vips_convi_intize's requantisation is the
#       identity when the mask already divides by 1, so the vector path runs
#       and still answers what the C path answers.
regime("scale_1_is_exact",
       ["vips", "conv", noise_mono_v, out_path("regime_scale1_sobelk"),
        os.path.join(KER, "sobelk.mat"), "--precision", "integer"],
       "regime_scale1_sobelk", expect_agree=True, expect_kernel="vector",
       note="the vector path RUNS here and still agrees: intize is exact "
            "for a scale-1 mask, so both kernels convolve with the same "
            "coefficients")

# (iv) float precision never diverges, on any mask. vips_convf has one
#      implementation.
regime("float_never_diverges",
       ["vips", "conv", noise_mono_v, out_path("regime_float_hostile"),
        hostile_path, "--precision", "float"],
       "regime_float_hostile", expect_agree=True, expect_kernel="none",
       note="the same hostile mask that moves 73 at integer precision")

# (v) sharpen is NOT in #558's blast radius, whatever the shared tolerance in
#     libviprs-tests currently implies. It convolves the L of LabS, which is
#     16-bit, so the uchar gate at convi.c:1151 never opens.
regime("sharpen_is_not_558",
       ["vips", "sharpen", noise_mono_v, out_path("regime_sharpen_s1.6"),
        "--sigma", "1.6", "--m1", "1", "--m2", "2"],
       "regime_sharpen_s1.6", expect_agree=True, expect_kernel="C",
       note="LabS is 16-bit, so sharpen takes the C path on both builds. "
            "Any sharpen deviation libviprs shows is a libviprs bug (#581), "
            "not this one.")

print("regimes done:", sum(len(v) for v in REGIMES.values()))

# ---------------------------------------------------------------------------
# 12. Write commands.sh and oracle.json
# ---------------------------------------------------------------------------
commands_sh_path = os.path.join(ROOT, "commands.sh")
with open(commands_sh_path, "w") as f:
    f.write("\n".join(COMMANDS_LOG) + "\n")
os.chmod(commands_sh_path, 0o755)

oracle = {
    "meta": {
        "vips_version": VIPS_VERSION,
        "vips_binary": VIPS,
        "area": AREA,
        "oracle_pin": {
            "file": "oracle-captures/ORACLE_PIN.json",
            "pinned_vips_version": ORACLE_PIN["pinned_vips_version"],
            "checked_how": (
                "capture.py compares `vips --version` against the pin before "
                "it writes anything and exits non-zero on a mismatch unless "
                "it is passed --repin; tests/oracle_capture_pins.rs compares "
                "the version recorded HERE against the same file, so a "
                "--repin capture stays red until the pin moves too. Issue "
                "#650: the previous arrangement wrote 8.18.4 into a comment "
                "and into a meta key nothing read, and a brew upgrade "
                "redefined the oracle mid-session without anyone noticing."
            ),
        },
        "provenance": {
            "vips_version": VIPS_VERSION,
            "homebrew_kegs": oracle_pin.homebrew_kegs(VIPS),
            "why_homebrew_kegs": (
                "Every Homebrew keg the vips binary reaches transitively, "
                "with its version, walked from `otool -L`. `vips "
                "--vips-config` names libopenjp2, libheif, matio and the "
                "rest with no version for any of them, and the codec version "
                "is what a future disagreement over these numbers turns on. "
                "The stack moves independently of vips: the upgrade that "
                "took vips 8.18.4 to 8.18.6 also took libheif 1.23.1 to "
                "1.23.2, x265 4.2 to 4.3 and libultrahdr 1.5.1 to 2.0.2. "
                "This is provenance, not a pin: only the vips version is "
                "enforced. Issue #650."
            ),
            "vips_targets": "; ".join(
                " ".join(line.split())
                for line in run(["vips", "--targets"], log=False).splitlines()
            ),
            "vips_binary": VIPS,
            "vips_on_path": subprocess.run(
                ["/usr/bin/which", "vips"], capture_output=True, text=True
            ).stdout.strip(),
            "captured_by": "oracle-captures/convolution/capture.py",
            "why_targets": (
                "`vips --targets` is recorded because the vector path's "
                "availability is a build-and-CPU property, not a version "
                "property. It is NOT recorded because the target set changes "
                "the arithmetic: it does not. Dropping NEON_BF16 and leaving "
                "NEON gives byte-identical output at nine sigmas, while the "
                "same comparison against VIPS_NOVECTOR=1 diverges at eight of "
                "the nine. The kernel is exact int32, so lane count cannot "
                "move a value. What varies is whether the vector path runs at "
                "all -- build config, CPU, VIPS_NOVECTOR, and the mask."
            ),
        },
        "which_libvips_path_is_pinned": (
            "VIPS_NOVECTOR=1, i.e. `vips_convi_gen`, the portable C loop. "
            "libvips's own documentation names it as the specification: "
            "'@mask is converted to an integer mask with rint() of each "
            "element ... For UCHAR images, vips_convi uses a fast vector "
            "path based on half-float arithmetic. This can produce slightly "
            "different results. Disable the vector path with --vips-novector "
            "or VIPS_NOVECTOR or vips_vector_set_enabled()' "
            "(convi.c:1271-1284). It is also the path libvips itself falls "
            "back to whenever vips_convi_intize declines a mask, so it is "
            "the floor rather than a competing choice. libviprs implements "
            "it, and this capture pins it. Issue #558."
        ),
        "both_libvips_paths": (
            "Every record that can reach vips_convi is captured twice. The "
            "top-level output/avg/min/max/points are the PINNED (novector) "
            "values; `vector` and `novector` carry the two arms in full, and "
            "`paths_agree` / `path_diff_count` / `path_diff_max_abs` / "
            "`path_diff_sample` carry the comparison. `expects_paths_agree` "
            "is checked at capture time against the KNOWN_DIVERGENT table in "
            "capture.py, so a record that changes regime aborts the capture "
            "instead of quietly re-pinning."
        ),
        "what_is_hashed": (
            "`raw_sha256` on every output block is the sha256 of `vips "
            "rawsave` output, i.e. of the pixel samples alone, in raster "
            "order, little-endian, no header and no trailer. It is NOT a "
            "hash of the `.v` file. It used to be, and that was backwards: "
            "`build_xml` (libvips/iofuncs/vips.c:857-859) writes "
            "VIPS_MICRO_VERSION into the `.v` trailer's XML namespace, so "
            "every whole-file hash in this capture moved when the machine "
            "went from vips 8.18.4 to 8.18.6, and all 452 of them are "
            "reproduced exactly by flipping that one byte back. Zero pixels "
            "had changed. A pin that is blind to the thing it is guarding "
            "and loud about the thing it is not trains everyone to ignore a "
            "red diff. Issue #649."
        ),
        "min_max_positions_are_recomputed": (
            "`min`/`max` carry `value` from the binary and `x`/`y`/`band` "
            "recomputed here as the FIRST OCCURRENCE IN RASTER ORDER, plus "
            "`ties`, the number of (x,y) positions holding that extreme. "
            "`vips min --x --y` reports where a worker thread found the "
            "extreme, and 614 of the 904 extremes here sit on a tie (312 "
            "min, 302 max), so the coordinate it returns is a race: two runs "
            "of the identical binary on the identical fixtures differed at "
            "478 leaves, every one of them an x or a y. Where `ties` is 1 "
            "the recorded position is the only answer vips could have given "
            "and the capture asserts that it did give it. Where `ties` is "
            "more than 1, matching this exact coordinate is not a parity "
            "requirement on libviprs, only a reproducibility rule for this "
            "file. Issue #649."
        ),
        "no_tolerance_is_derivable": (
            "vips_convi_intize's accuracy gate (convi.c:1096-1113) is often "
            "read as bounding the gap between libvips's two paths at 2. It "
            "does not, and the misreading is load-bearing wherever a suite "
            "wrote tol=1 or tol=2 for integer conv. The gate compares the "
            "requantised mask against EXACT REAL ARITHMETIC, at one grey "
            "level, on a flat field; vips_convi_gen appears nowhere in it. "
            "It is a DC-gain check on sum(w_hat - w) and says nothing about "
            "per-pixel error, which is sum((w_hat - w) * p). See the "
            "discrim_hostile_mask_noise_integer record: the gate accepts the "
            "mask and the paths still differ by dozens."
        ),
        "regime_boundaries": (
            "See the top-level `regime_boundaries` block. Three regimes, not "
            "two: masks where the vector path runs and disagrees, masks "
            "where it runs and agrees (scale 1, or any requantisation that "
            "happens to be exact -- including every power-of-two scale), and "
            "masks where libvips declines it and runs the C path itself. "
            "Nothing on the API surface tells a test author which one a mask "
            "is in."
        ),
        "ported_test_file": (
            "tests/ported_convolution.rs (conv, compass, convsep, fastcor, "
            "spcor, gaussblur, sharpen)"
        ),
        "fixtures": {
            "colour.v": "exact port of make_test_colour() in ported_convolution.rs "
                        "(100x100 RGB, mask_ideal radial pattern, values (2,3,4)/(3,5,7))",
            "mono.v": "band 1 of colour.v",
            "impulse_mono.v": "21x21 uchar black image, single 255 impulse at (10,10); "
                               "used for exact kernel/rotation reveal via convolution",
            "sample_colour.v": "vips copy of the reference-suite sample.jpg (290x442 RGB); "
                                "JPEG-derived, values below carry a decoder-tolerance note",
            "sample_mono.v": "band 1 of sample_colour.v",
            "noise_mono.v": "64x64 uchar, PRNG(seed=42) noise, lossless PGM; used for an "
                             "unambiguous fastcor/spcor match position",
            "noise_colour.v": "64x64x3 uchar, PRNG(seed=43) noise, lossless PPM",
            "kernels/*.mat": "matrixload text kernels: sharp/blur/line/sobelk (3x3, from the "
                              "ported test_conv kernels), box5 (5x5 box), asym (3x2, offset=10, "
                              "no symmetry, pins flip/no-flip + offset semantics), box5_1d "
                              "(1x5, the only shape convasep accepts)",
        },
        "sharpen_mono_band_promotion": (
            "Real-vips finding, not a capture bug: vips sharpen runs the unsharp mask in "
            "LabS colourspace. sample_mono.v is extract_band(1) of an srgb-interpretation "
            "JPEG, so it carries 1 band of pixel data but is still tagged "
            "interpretation=srgb. vips_colourspace's srgb route assumes 3 bands, so "
            "sharpen's output on this input is silently promoted to 3 bands (verified: "
            "with m1=m2=0 each output band equals the original single band, via vips "
            "subtract's auto band-broadcast giving an exact 0 diff). Feeding sharpen a "
            "1-band image tagged interpretation=b-w instead keeps the output at 1 band; "
            "tagging it interpretation=multiband (e.g. a `vips black` fixture) makes "
            "sharpen hard-error with 'no known route from labs to multiband'. "
            "ported_convolution.rs::test_sharpen only asserts width/height equality (not "
            "band count), so this 1-band -> 3-band promotion would not be caught by the "
            "port as currently written; if libviprs's Raster type has no equivalent "
            "interpretation tag, its mono sharpen output will very likely stay 1-band and "
            "will NOT numerically match these sharpen_*_mono_* records shape-for-shape, "
            "only dimension-for-dimension."
        ),
    },
    "known_divergent": {
        "what": (
            "The records where libvips's two integer-convolution paths are "
            "known NOT to agree, asserted in both directions at capture time."
        ),
        "table": {k: {"path_diff_count": v[0], "path_diff_max_abs": v[1]}
                  for k, v in KNOWN_DIVERGENT.items()},
    },
    "regime_boundaries": REGIMES,
    "records": RECORDS,
}

oracle_json_path = os.path.join(ROOT, "oracle.json")
with open(oracle_json_path, "w") as f:
    json.dump(oracle, f, indent=2, sort_keys=False)

dual = [r for r in RECORDS if "paths_agree" in r]
divergent = [r for r in dual if not r["paths_agree"]]
print(f"\nWrote {len(RECORDS)} records to {oracle_json_path}")
print(f"Wrote {len(COMMANDS_LOG)} log lines to {commands_sh_path}")
print(f"dual-path records: {len(dual)}, divergent: {len(divergent)}")
for r in divergent:
    print("  %-42s %6d/%-7d max %d" % (
        r["record_id"], r["path_diff_count"], r["samples"],
        r["path_diff_max_abs"]))






