#!/usr/bin/env python3
"""
Oracle capture for the NIfTI area (issue #641, sub-issue of #510).

NIfTI is the one format in #510 that libvips cannot act as oracle for. This
build reports `NIfTI load/save with libnifti: false`, `vips -l` registers
neither `niftiload` nor `niftisave`, and pointing `vipsheader` at a real
`.nii` written by the reference implementation falls through the whole
sniffing chain to magickload, which guesses TGA and fails. All three of those
are re-measured by record `libvips_is_not_the_oracle` on every run, so if a
future build ever gains libnifti the capture says so out loud instead of
quietly continuing to be right for the wrong reason.

So the oracle here is nifti_clib from NIH, the format's own reference
implementation and the library libvips itself would have linked. Runbook §8
puts a reference implementation first in the preference order, ahead of a
widely-deployed C library, which makes this the best available oracle rather
than a fallback.

Two tools drive it:

  * `nifti_tool`, the reference CLI, for the header displays, the collapsed
    image values and the diffs
  * `probe.c` in this directory, compiled against `libnifti2.a`, for the
    library-level answers no CLI exposes: `nifti_datatype_sizes` over every
    code, `nifti_header_version` at every buffer length, `offsetof` for all
    three header structs, and the in-memory bytes `nifti_image_load` leaves
    behind once it has swapped them

Everything a port has to get right and cannot read off the spec is here:

  * the datatype table, measured code by code, including the four codes that
    look valid and are not and the one whose C type is a different width on
    this platform than the name promises
  * `sizeof_hdr` and the magic as a combined version-and-endianness sentinel,
    and the fact that 348 needs only 348 bytes to identify a 540-byte header
  * `scl_slope == 0` meaning "no scaling", and the asymmetry between the
    NIfTI-1 and NIfTI-2 write paths around it
  * the NIfTI-1 header sitting on top of the Analyze 7.5 one field for field,
    and the exact reason libvips's `analyzeload` does not pick a NIfTI pair up
  * where the loader gives up, and what it says when it does

Writes:
  commands.sh  - every command actually executed, in order
  oracle.json  - structured records
  fixtures/    - every `.nii` / `.hdr` / `.img` the records refer to
  outputs/     - the compiled probe and scratch, not pinned

Re-running needs nifti_clib built and installed under NIFTI_PREFIX (see the
`oracle_build` record for exactly how this one was obtained) and a C compiler.
Nothing outside this script's own directory is written.
"""
import hashlib
import json
import os
import re
import shutil
import struct
import subprocess

ROOT = os.path.dirname(os.path.abspath(__file__))
FIX = os.path.join(ROOT, "fixtures")
OUT = os.path.join(ROOT, "outputs")

# Where nifti_clib was built and installed. Overridable so a re-run on
# another machine does not need this exact path.
NIFTI_PREFIX = os.environ.get(
    "NIFTI_PREFIX", "/Users/rom/workspace/nifti-oracle-641/install")
NIFTI_SRC = os.environ.get(
    "NIFTI_SRC", "/Users/rom/workspace/nifti-oracle-641/nifti_clib")

NT = os.path.join(NIFTI_PREFIX, "bin", "nifti_tool")
PROBE = os.path.join(OUT, "probe")
CC = os.environ.get("CC", "cc")

# libvips is NOT the oracle here. These two are used only to record, on every
# run, that this build has no NIfTI support at all.
VIPS = "/opt/homebrew/bin/vips"
VIPSHEADER = "/opt/homebrew/bin/vipsheader"

os.makedirs(FIX, exist_ok=True)
os.makedirs(OUT, exist_ok=True)

COMMANDS = []

# This machine's vips prints a heif module-load failure on stderr for every
# invocation (libheif wants an x265 dylib that is not installed), and the
# user's LIBRARY_PATH points the linker at an openssl directory that does not
# exist. Neither has anything to do with NIfTI, so both are stripped rather
# than pinned as if they were part of the answer.
NOISE = ("VIPS-WARNING", "Referenced from:", "Reason: tried:", "libheif",
         "x265", "unable to load", "/usr/local/opt/openssl/lib/")

# A GLib runtime message carries a pid and a wall-clock timestamp; neither is
# part of the answer and both would make this capture differ run to run.
GLIB_PREFIX = re.compile(
    r"^\((?:process|vips|vipsheader):\d+\): "
    r"(?P<domain>[\w-]+) \*\*: \d\d:\d\d:\d\d\.\d+: ")


def clean(text):
    """stderr with this machine's unrelated warnings removed, GLib pids and
    timestamps stripped, absolute paths made relative, and consecutive
    duplicates folded, so re-running reproduces oracle.json byte for byte."""
    keep = []
    for ln in text.splitlines():
        if not ln.strip() or any(n in ln for n in NOISE):
            continue
        ln = GLIB_PREFIX.sub(lambda m: m.group("domain") + ": ", ln)
        ln = ln.replace(ROOT + "/", "").replace(NIFTI_PREFIX + "/", "")
        if not keep or keep[-1] != ln:
            keep.append(ln)
    return "\n".join(keep)


def run(args, allow_fail=True, log=True):
    """Run a command, logging it (with this directory's absolute path reduced
    to a relative one) for commands.sh."""
    if log:
        COMMANDS.append(" ".join(
            a.replace(ROOT + "/", "").replace(NIFTI_PREFIX, "$NIFTI_PREFIX")
            for a in args))
    proc = subprocess.run(args, capture_output=True, text=True)
    if proc.returncode != 0 and not allow_fail:
        raise SystemExit(f"failed: {' '.join(args)}\n{proc.stderr}")
    return proc


def strip_root(value):
    """Absolute paths make the capture machine-specific; several probe
    subcommands echo the path they were given, so they are made relative."""
    if isinstance(value, str):
        return value.replace(ROOT + "/", "").replace(
            NIFTI_PREFIX + "/", "$NIFTI_PREFIX/")
    if isinstance(value, dict):
        return {k: strip_root(v) for k, v in value.items()}
    if isinstance(value, list):
        return [strip_root(v) for v in value]
    return value


def probe(*args):
    """A probe.c subcommand, returning parsed JSON."""
    proc = run([PROBE, *[str(a) for a in args]])
    if not proc.stdout.strip():
        return {"probe_failed": clean(proc.stderr)}
    return strip_root(json.loads(proc.stdout))


def probe_full(*args):
    """A probe.c subcommand, keeping stderr too: the debug chatter IS the
    answer for the refusal records."""
    proc = run([PROBE, *[str(a) for a in args]])
    out = strip_root(json.loads(proc.stdout)) if proc.stdout.strip() else None
    return {"json": out, "stderr": clean(proc.stderr), "exit": proc.returncode}


def ntool(*args, quiet=True):
    argv = [NT] + (["-quiet"] if quiet else []) + [str(a) for a in args]
    proc = run(argv)
    return {"exit": proc.returncode,
            "stdout": clean(proc.stdout),
            "stderr": clean(proc.stderr)}


def sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def fixture_bytes(name):
    return list(open(os.path.join(FIX, name), "rb").read())


def make(prefix, ni_ver, ftype, datatype, dims, slope=1.0, inter=0.0):
    """Write a dataset through nifti_image_write, filled with a ramp of
    consecutive bytes, so the datatype's reinterpretation of those bytes is
    visible in the readback and no port can fake it."""
    d = list(dims) + [0] * (8 - len(dims))
    return probe("make", prefix, ni_ver, ftype, datatype,
                 d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7], slope, inter)


def edit(src, dst, patches, truncate=None):
    """Copy a fixture and poke raw bytes into it. Malformed INPUTS have to be
    hand-built (the reference implementation will not write a broken file);
    every ANSWER about them still comes from the oracle."""
    data = bytearray(open(src, "rb").read())
    for off, raw in patches:
        data[off:off + len(raw)] = raw
    if truncate is not None:
        data = data[:truncate]
    with open(dst, "wb") as f:
        f.write(bytes(data))
    return dst


def fx(name):
    return os.path.join(FIX, name)


records = {}

# ======================================================================
# 1. libvips is not, and cannot be, the oracle for this area.
# ======================================================================
vips_version = run([VIPS, "--version"]).stdout.strip()
vips_config = run([VIPS, "--vips-config"])
config_lines = [ln.strip() for ln in vips_config.stdout.splitlines() if ln.strip()]
nifti_config_line = next(
    (ln for ln in config_lines if "NIfTI" in ln or "nifti" in ln), None)
vips_list = run([VIPS, "-l"])
nifti_ops = [ln.strip() for ln in vips_list.stdout.splitlines()
             if "nifti" in ln.lower()]
niftiload_probe = run([VIPS, "niftiload", "--help"])
niftisave_probe = run([VIPS, "niftisave", "--help"])

# ======================================================================
# 2. The oracle itself: what it is and where it came from.
# ======================================================================
git_head = run(["git", "-C", NIFTI_SRC, "rev-parse", "HEAD"]).stdout.strip()
git_describe = run(["git", "-C", NIFTI_SRC, "describe", "--tags"]).stdout.strip()
git_date = run(["git", "-C", NIFTI_SRC, "log", "-1", "--format=%ci"]).stdout.strip()
git_remote = run(["git", "-C", NIFTI_SRC, "config", "--get",
                  "remote.origin.url"]).stdout.strip()
cc_version = run([CC, "--version"]).stdout.splitlines()[0].strip()

CFLAGS = ["-O2", "-std=c99", "-Wall", "-Wextra",
          "-I", os.path.join(NIFTI_PREFIX, "include", "nifti")]
LIBS = [os.path.join(NIFTI_PREFIX, "lib", "libnifti2.a"),
        os.path.join(NIFTI_PREFIX, "lib", "libznz.a"), "-lm", "-lz"]
compile_cmd = [CC, *CFLAGS, "-o", PROBE, os.path.join(ROOT, "probe.c"), *LIBS]
compile_proc = run(compile_cmd, allow_fail=False)

# Only once the probe is known to build: clear the derived trees so a re-run
# cannot inherit a stale fixture from an older version of this script, and so
# nifti_tool's refusal to overwrite an existing -prefix target cannot make the
# second run differ from the first.
for name in sorted(os.listdir(FIX)):
    os.unlink(fx(name))
for name in sorted(os.listdir(OUT)):
    if name != "probe":
        path = os.path.join(OUT, name)
        shutil.rmtree(path) if os.path.isdir(path) else os.unlink(path)

tool_ver = ntool("-ver")["stdout"]
lib_ver = ntool("-nifti_ver")["stdout"]
with_zlib = ntool("-with_zlib")["stdout"]
env = probe("env")

# ======================================================================
# 3. Every datatype code the library will answer for.
# ======================================================================
dt_table = probe("datatypes")
# The codes worth writing a file for: the ones the library calls valid for
# NIfTI and gives a non-zero width. Derived from the sweep, not typed in.
writable = [r for r in dt_table if r["valid_for_nifti"] == 1 and r["nbyper"] > 0]

DIMS = [3, 2, 3, 1]          # 2x3x1: six voxels, the smallest useful volume
dt_roundtrip = {}
for r in writable:
    code = r["code"]
    short = r["nifti_datatype_to_string"].replace("NIFTI_TYPE_", "").lower()
    name = f"dt{code}_{short}.nii"
    made = make(fx(name[:-4]), 1, 1, code, DIMS)
    if not made.get("made"):
        dt_roundtrip[str(code)] = {"nifti_make_new_nim_refused": True}
        continue
    read = probe("read", fx(name))
    ci = ntool("-disp_ci", -1, -1, -1, -1, -1, -1, -1, "-infiles", fx(name))
    hdr = probe("readhdr", fx(name), 1)
    dt_roundtrip[str(code)] = {
        "fixture": f"fixtures/{name}",
        "name": r["nifti_datatype_to_string"],
        "nbyper": r["nbyper"],
        "swapsize": r["swapsize"],
        "file_bytes": os.path.getsize(fx(name)),
        "payload_written_hex": made["payload_written_hex"],
        "header_bitpix": hdr.get("bitpix"),
        "header_datatype": hdr.get("datatype"),
        "nvox": read["nim"]["nvox"],
        "data_hex_after_load": read["nim"]["data_hex_after_load"],
        "disp_ci": ci,
    }

# ======================================================================
# 4. NIfTI-1 against NIfTI-2, on disk.
# ======================================================================
offsets = probe("offsets")
n1 = make(fx("ver_n1_single"), 1, 1, 4, DIMS)
n2 = make(fx("ver_n2_single"), 2, 4, 4, DIMS)
n1_hdrver = probe("hdrver", fx("ver_n1_single.nii"))
n2_hdrver = probe("hdrver", fx("ver_n2_single.nii"))
n1_head = open(fx("ver_n1_single.nii"), "rb").read(60)
n2_head = open(fx("ver_n2_single.nii"), "rb").read(60)

# The 8-byte NIfTI-2 magic carries a four-byte tail whose whole job is to
# catch a file mangled by an ftp or a text-mode copy. Break just that tail.
edit(fx("ver_n2_single.nii"), fx("n2_magic_tail_mangled.nii"),
     [(8, b"\n\n\n\n")])
# And the same idea one step further: a NIfTI-2 header whose first four magic
# bytes are right but whose fifth is not.
edit(fx("ver_n2_single.nii"), fx("n2_magic_tail_partial.nii"),
     [(8, b"\r\n\x1a\x00")])

# ======================================================================
# 5. NIfTI-1 sits exactly on top of the Analyze 7.5 header.
# ======================================================================
# Join the two offset tables the oracle produced: for every NIfTI-1 field,
# which Analyze field occupies the same bytes. This is a join over measured
# offsets, not a hand-written table.
ana = offsets["nifti_analyze75"]
overlay = []
for f in offsets["nifti_1_header"]:
    lo, hi = f["offset"], f["offset"] + f["size"]
    covered = [a["field"] for a in ana
               if a["offset"] < hi and a["offset"] + a["size"] > lo]
    overlay.append({
        "nifti1_field": f["field"], "offset": f["offset"], "size": f["size"],
        "analyze75_fields_at_same_bytes": covered,
        "same_name": covered == [f["field"]],
    })

# A NIfTI-1 header whose magic is all zeros is, by the sentinel, an ANALYZE
# file. Measure that rather than assert it.
edit(fx("ver_n1_single.nii"), fx("magic_zero_analyze.nii"),
     [(344, b"\x00\x00\x00\x00")])
# The same header as a PAIR, because the answer is different and the reason
# is the filename rather than the bytes.
make(fx("magic_zero_pair_src"), 1, 2, 4, DIMS)
edit(fx("magic_zero_pair_src.hdr"), fx("magic_zero_pair.hdr"),
     [(344, b"\x00\x00\x00\x00")])
shutil.copyfile(fx("magic_zero_pair_src.img"), fx("magic_zero_pair.img"))

# The libvips contrast, and it is a sharp one. A big-endian NIfTI-1 PAIR is
# byte-compatible with what libvips's analyzeload wants, and the ONLY thing
# stopping it loading is the four-byte extender NIfTI appends, which makes
# the .hdr 352 bytes where analyze2vips demands exactly 348. Strip those four
# bytes and libvips loads a NIfTI file as Analyze, silently, ignoring every
# NIfTI-only field.
make(fx("pair_be_src"), 1, 2, 4, DIMS)
probe("swapfile", fx("pair_be_src.hdr"), fx("pair_be.hdr"), 1, 352, 2)
probe("swapfile", fx("pair_be_src.img"), fx("pair_be.img"), 1, 0, 2)
edit(fx("pair_be.hdr"), fx("pair_be348.hdr"), [], truncate=348)
shutil.copyfile(fx("pair_be.img"), fx("pair_be348.img"))
vips_on_be_pair_352 = run([VIPSHEADER, fx("pair_be.hdr")])
vips_on_be_pair_348 = run([VIPSHEADER, "-a", fx("pair_be348.hdr")])
vips_on_le_pair = run([VIPSHEADER, fx("pair_be_src.hdr")])
vips_on_nii = run([VIPSHEADER, fx("ver_n1_single.nii")])

# ======================================================================
# 6. Single file against the .hdr/.img pair.
# ======================================================================
make(fx("pair_n1"), 1, 2, 4, DIMS)
make(fx("pair_n2"), 2, 5, 4, DIMS)
pair_records = {}
for label, base, ext_hdr in (("nifti1_pair", "pair_n1", ".hdr"),
                             ("nifti2_pair", "pair_n2", ".hdr"),
                             ("nifti1_single", "ver_n1_single", ".nii"),
                             ("nifti2_single", "ver_n2_single", ".nii")):
    path = fx(base + ext_hdr)
    read = probe("read", path)
    pair_records[label] = {
        "fixture": f"fixtures/{base}{ext_hdr}",
        "header_bytes": os.path.getsize(path),
        "image_bytes": (os.path.getsize(fx(base + ".img"))
                        if os.path.exists(fx(base + ".img")) else None),
        "magic": fixture_bytes(base + ext_hdr)[344:348] if ext_hdr == ".hdr" or True else None,
        "nifti_type": read["nim"]["nifti_type"],
        "iname_offset": read["nim"]["iname_offset"],
        "fname": read["nim"]["fname"],
        "iname": read["nim"]["iname"],
        "names": probe("names", path),
    }
# NIfTI-2's magic is eight bytes, so slice it from the right place for those.
for label in ("nifti2_pair", "nifti2_single"):
    base = "pair_n2" if label == "nifti2_pair" else "ver_n2_single"
    ext = ".hdr" if label == "nifti2_pair" else ".nii"
    pair_records[label]["magic"] = fixture_bytes(base + ext)[4:12]

# What happens when half a pair is missing, in both directions.
shutil.copyfile(fx("pair_n1.hdr"), fx("lonely_hdr.hdr"))
shutil.copyfile(fx("pair_n1.img"), fx("lonely_img.img"))
missing = {
    "hdr_without_img": {
        "fixture": "fixtures/lonely_hdr.hdr",
        "read": probe_full("read", fx("lonely_hdr.hdr")),
        "readhdr": probe_full("readhdr", fx("lonely_hdr.hdr"), 1),
        "names": probe("names", fx("lonely_hdr.hdr")),
        "nifti_tool_disp_hdr": ntool("-disp_hdr", "-infiles", fx("lonely_hdr.hdr")),
    },
    "img_without_hdr": {
        "fixture": "fixtures/lonely_img.img",
        "read": probe_full("read", fx("lonely_img.img")),
        "names": probe("names", fx("lonely_img.img")),
    },
}
# A single-file .nii whose magic claims the paired form, and the reverse.
edit(fx("ver_n1_single.nii"), fx("nii_with_ni1_magic.nii"), [(344, b"ni1\x00")])
edit(fx("pair_n1.hdr"), fx("hdr_with_np1_magic.hdr"), [(344, b"n+1\x00")])
shutil.copyfile(fx("pair_n1.img"), fx("hdr_with_np1_magic.img"))
crossed = {
    "single_file_claiming_pair_magic": {
        "fixture": "fixtures/nii_with_ni1_magic.nii",
        "read": probe_full("read", fx("nii_with_ni1_magic.nii")),
        "names": probe("names", fx("nii_with_ni1_magic.nii")),
    },
    "pair_header_claiming_single_magic": {
        "fixture": "fixtures/hdr_with_np1_magic.hdr",
        "read": probe_full("read", fx("hdr_with_np1_magic.hdr")),
    },
}

# ======================================================================
# 7. scl_slope / scl_inter, and the zero-slope rule.
# ======================================================================
scaling = {}
for label, ver, ftype, slope, inter in (
        ("identity_slope_1_inter_0", 1, 1, 1.0, 0.0),
        ("slope_2_inter_minus_3", 1, 1, 2.0, -3.0),
        ("slope_0_inter_7_nifti1", 1, 1, 0.0, 7.0),
        ("slope_0_inter_7_nifti2", 2, 4, 0.0, 7.0),
        ("negative_slope", 1, 1, -0.5, 100.0),
        ("slope_2_inter_minus_3_nifti2", 2, 4, 2.0, -3.0)):
    made = make(fx("scl_" + label), ver, ftype, 4, DIMS, slope, inter)
    path = fx("scl_" + label + ".nii")
    read = probe("read", path)
    hdr = probe("readhdr", path, 1)
    scaling[label] = {
        "fixture": f"fixtures/scl_{label}.nii",
        "asked_for": {"scl_slope": slope, "scl_inter": inter},
        "header_scl_slope": hdr.get("scl_slope"),
        "header_scl_inter": hdr.get("scl_inter"),
        "nim_scl_slope": read["nim"]["scl_slope"],
        "nim_scl_inter": read["nim"]["scl_inter"],
        "data_hex_after_load": read["nim"]["data_hex_after_load"],
        "disp_ci_values": ntool("-disp_ci", -1, -1, -1, -1, -1, -1, -1,
                                "-infiles", path)["stdout"],
        "file_bytes": os.path.getsize(path),
    }

# A non-finite slope. nifti2_io.h wraps every float the header hands over in
# FIXED_FLOAT, which is isfinite(x) ? x : 0, so an infinity or a NaN in the
# file becomes a plain 0.0 in nifti_image, and 0.0 is exactly the value that
# means "do not scale".
for label, raw in (("slope_inf", struct.pack("<f", float("inf"))),
                   ("slope_nan", b"\x00\x00\xc0\x7f")):
    src = fx("scl_identity_slope_1_inter_0.nii")
    dst = fx(f"scl_{label}.nii")
    edit(src, dst, [(112, raw), (116, struct.pack("<f", 5.0))])
    read = probe("read", dst)
    hdr = probe("readhdr", dst, 1)
    scaling[label] = {
        "fixture": f"fixtures/scl_{label}.nii",
        "bytes_at_offset_112": list(raw),
        "header_scl_slope": hdr.get("scl_slope"),
        "header_scl_inter": hdr.get("scl_inter"),
        "nim_scl_slope": read["nim"]["scl_slope"],
        "nim_scl_inter": read["nim"]["scl_inter"],
    }

# ======================================================================
# 8. Byte order, both ways.
# ======================================================================
byte_order = {"host": {"nifti_short_order": env["nifti_short_order"],
                       "meaning": env["nifti_short_order_meaning"]}}
for label, ver, dtype, swapsize, off in (
        ("nifti1_int16", 1, 4, 2, 352),
        ("nifti1_float32", 1, 16, 4, 352),
        ("nifti1_float64", 1, 64, 8, 352),
        ("nifti2_int16", 2, 4, 2, 544)):
    ftype = 1 if ver == 1 else 4
    le = fx(f"endian_{label}_le")
    make(le, ver, ftype, dtype, DIMS)
    be = fx(f"endian_{label}_be.nii")
    swap = probe("swapfile", le + ".nii", be, ver, off, swapsize)
    le_read = probe("read", le + ".nii")
    be_read = probe("read", be)
    byte_order[label] = {
        "le_fixture": f"fixtures/endian_{label}_le.nii",
        "be_fixture": f"fixtures/endian_{label}_be.nii",
        "le_first_16_bytes": fixture_bytes(f"endian_{label}_le.nii")[:16],
        "be_first_16_bytes": fixture_bytes(f"endian_{label}_be.nii")[:16],
        "le_byteorder": le_read["nim"]["byteorder_meaning"],
        "be_byteorder": be_read["nim"]["byteorder_meaning"],
        "le_data_hex_after_load": le_read["nim"]["data_hex_after_load"],
        "be_data_hex_after_load": be_read["nim"]["data_hex_after_load"],
        "loads_to_identical_memory":
            le_read["nim"]["data_hex_after_load"] ==
            be_read["nim"]["data_hex_after_load"],
        "le_disp_ci": ntool("-disp_ci", -1, -1, -1, -1, -1, -1, -1,
                            "-infiles", le + ".nii")["stdout"],
        "be_disp_ci": ntool("-disp_ci", -1, -1, -1, -1, -1, -1, -1,
                            "-infiles", be)["stdout"],
        "be_hdrver": probe("hdrver", be),
        "swap_header_after": swap.get("header_first_16_after_swap"),
    }
# The three swap entry points, run over one header, so a port can see they
# are not interchangeable.
byte_order["swap_routines_on_one_nifti1_header"] = probe(
    "swap", fx("endian_nifti1_int16_le.nii"))

# ======================================================================
# 9. Dimensionality.
# ======================================================================
dims = {}
for label, d in (("rank1_6", [1, 6]),
                 ("rank2_2x3", [2, 2, 3]),
                 ("rank3_2x3x2", [3, 2, 3, 2]),
                 ("rank4_2x3x2x2", [4, 2, 3, 2, 2]),
                 ("rank5_2x2x2x2x2", [5, 2, 2, 2, 2, 2]),
                 ("rank7_all2", [7, 2, 2, 2, 2, 2, 2, 2])):
    made = make(fx("dim_" + label), 1, 1, 2, d)
    path = fx("dim_" + label + ".nii")
    read = probe("read", path)
    dims[label] = {
        "fixture": f"fixtures/dim_{label}.nii",
        "asked_for_dim": d + [0] * (8 - len(d)),
        "header_dim": probe("readhdr", path, 1)["dim"],
        "nim_dim": read["nim"]["dim"],
        "ndim": read["nim"]["ndim"],
        "nvox": read["nim"]["nvox"],
        "nx_ny_nz_nt_nu_nv_nw": [read["nim"][k] for k in
                                 ("nx", "ny", "nz", "nt", "nu", "nv", "nw")],
        "file_bytes": os.path.getsize(path),
    }

# What a 3D or a 4D volume does when the consumer wants one 2D plane:
# nifti_read_collapsed_image, which nifti_tool exposes as -cci / -disp_ci.
vol4 = fx("dim_rank4_2x3x2x2.nii")
collapse = {
    "whole_volume": ntool("-disp_ci", -1, -1, -1, -1, -1, -1, -1,
                          "-infiles", vol4),
    "one_3d_volume_t0": ntool("-disp_ci", -1, -1, -1, 0, -1, -1, -1,
                              "-infiles", vol4),
    "one_3d_volume_t1": ntool("-disp_ci", -1, -1, -1, 1, -1, -1, -1,
                              "-infiles", vol4),
    "one_2d_plane_z0_t0": ntool("-disp_ci", -1, -1, 0, 0, -1, -1, -1,
                                "-infiles", vol4),
    "one_2d_plane_z1_t1": ntool("-disp_ci", -1, -1, 1, 1, -1, -1, -1,
                                "-infiles", vol4),
    "one_voxel_time_series": ntool("-disp_ts", 1, 2, 1, "-infiles", vol4),
    "single_voxel": ntool("-disp_ci", 1, 2, 1, 1, -1, -1, -1, "-infiles", vol4),
    "index_out_of_range": ntool("-disp_ci", 9, 0, 0, 0, -1, -1, -1,
                                "-infiles", vol4),
}
# -cbl / -cci also WRITE a smaller dataset; pin what the extracted plane's
# own header says, because that is the shape a 2D consumer ends up with.
extract = ntool("-cci", -1, -1, 0, 0, -1, -1, -1,
                "-prefix", os.path.join(OUT, "plane_z0_t0.nii"),
                "-infiles", vol4)
extract_read = probe("read", os.path.join(OUT, "plane_z0_t0.nii"))
collapse["extracted_plane_header"] = {
    "exit": extract["exit"],
    "dim": extract_read["nim"]["dim"],
    "ndim": extract_read["nim"]["ndim"],
    "nvox": extract_read["nim"]["nvox"],
    "data_hex_after_load": extract_read["nim"]["data_hex_after_load"],
}

# dim[0] out of range, and a zero in the middle of the dim array.
dim_edge = {}
base_dim = fx("dim_rank3_2x3x2.nii")
for label, patches in (
        ("dim0_zero", [(40, struct.pack("<h", 0))]),
        ("dim0_eight", [(40, struct.pack("<h", 8))]),
        ("dim0_negative", [(40, struct.pack("<h", -1))]),
        ("dim1_zero", [(42, struct.pack("<h", 0))]),
        ("dim1_negative", [(42, struct.pack("<h", -3))]),
        ("dim2_zero_mid_array", [(44, struct.pack("<h", 0))]),
        ("dim_all_32767", [(42, struct.pack("<hhh", 32767, 32767, 32767))])):
    dst = fx(f"dimedge_{label}.nii")
    edit(base_dim, dst, patches)
    dim_edge[label] = {
        "fixture": f"fixtures/dimedge_{label}.nii",
        "readhdr_check_on": probe_full("readhdr", dst, 1),
        "readhdr_check_off": probe_full("readhdr", dst, 0),
        "read": probe_full("read", dst),
        "nifti_tool_disp_hdr": ntool("-disp_hdr", "-infiles", dst),
    }

# pixdim gets silently repaired too, and a port that carries the file's value
# through will disagree with the reference on the voxel spacing.
pixdim = {}
# qform_code sits at offset 252 and gates the whole quaternion branch: with
# it at 0 the loader never touches nim->qfac, so the qfac cases below set it
# to 1 (NIFTI_XFORM_SCANNER_ANAT) to reach the code that reads pixdim[0].
QFORM = (252, struct.pack("<h", 1))
for label, patches in (
        ("pixdim1_zero", [(80, struct.pack("<f", 0.0))]),
        ("pixdim2_inf", [(84, struct.pack("<f", float("inf")))]),
        ("pixdim3_nan", [(88, b"\x00\x00\xc0\x7f")]),
        ("pixdim1_negative", [(80, struct.pack("<f", -2.0))]),
        ("qform_code_zero", []),
        ("qfac_minus_one", [QFORM, (76, struct.pack("<f", -1.0))]),
        ("qfac_zero", [QFORM, (76, struct.pack("<f", 0.0))]),
        ("qfac_plus_one", [QFORM, (76, struct.pack("<f", 1.0))])):
    dst = fx(f"pixdim_{label}.nii")
    edit(fx("dim_rank3_2x3x2.nii"), dst, patches)
    read = probe("read", dst)
    # readhdr does not report pixdim, so the eight floats are read straight
    # off the fixture to sit next to what the loader made of them. JSON has
    # no infinity or NaN, so those become strings.
    raw = open(dst, "rb").read()[76:108]
    on_disk = [struct.unpack("<f", raw[i * 4:i * 4 + 4])[0] for i in range(8)]
    pixdim[label] = {
        "fixture": f"fixtures/pixdim_{label}.nii",
        "on_disk_pixdim": [v if v == v and abs(v) < 1e308 else str(v)
                           for v in on_disk],
        "nim_pixdim": read["nim"]["pixdim"],
        "nim_qfac": read["nim"]["qfac"],
        "qform_code": read["nim"]["qform_code"],
    }

# ======================================================================
# 10. Refusals, weighted the same as the successes.
# ======================================================================
good = fx("ver_n1_single.nii")
refusals = {}


def refuse(label, dst, patches=None, truncate=None, src=None, raw=None):
    path = fx(dst)
    if raw is not None:
        with open(path, "wb") as f:
            f.write(raw)
    else:
        edit(src or good, path, patches or [], truncate=truncate)
    refusals[label] = {
        "fixture": f"fixtures/{dst}",
        "file_bytes": os.path.getsize(path),
        "hdrver": probe("hdrver", path),
        "readhdr_check_on": probe_full("readhdr", path, 1),
        "read": probe_full("read", path),
        "nifti_tool_disp_hdr": ntool("-disp_hdr", "-infiles", path),
    }


refuse("empty_file", "bad_empty.nii", raw=b"")
refuse("one_byte", "bad_onebyte.nii", raw=b"\x5c")
refuse("truncated_to_100_bytes", "bad_trunc100.nii", truncate=100)
refuse("truncated_to_347_bytes", "bad_trunc347.nii", truncate=347)
refuse("truncated_to_348_bytes_no_extender", "bad_trunc348.nii", truncate=348)
refuse("header_complete_but_no_pixels", "bad_nodata.nii", truncate=352)
refuse("half_the_pixels", "bad_halfdata.nii", truncate=352 + 6)
refuse("sizeof_hdr_zero", "bad_sizeof0.nii", [(0, struct.pack("<i", 0))])
refuse("sizeof_hdr_349", "bad_sizeof349.nii", [(0, struct.pack("<i", 349))])
refuse("sizeof_hdr_540_with_nifti1_magic", "bad_sizeof540.nii",
       [(0, struct.pack("<i", 540))])
refuse("sizeof_hdr_byteswapped_348", "bad_sizeof_swapped.nii",
       [(0, struct.pack(">i", 348))])
refuse("magic_garbage", "bad_magic.nii", [(344, b"XY\x00\x00")])
refuse("magic_np1_without_nul", "bad_magic_nonul.nii", [(344, b"n+1x")])
refuse("magic_lowercase_n_plus_2", "bad_magic_n2_in_n1.nii", [(344, b"n+2\x00")])
refuse("datatype_unknown_code_3", "bad_dt3.nii", [(70, struct.pack("<h", 3))])
refuse("datatype_code_9999", "bad_dt9999.nii", [(70, struct.pack("<h", 9999))])
refuse("datatype_binary_code_1", "bad_dt1.nii", [(70, struct.pack("<h", 1))])
refuse("bitpix_disagrees_with_datatype", "bad_bitpix.nii",
       [(72, struct.pack("<h", 64))])
refuse("vox_offset_negative", "bad_voxoff_neg.nii",
       [(108, struct.pack("<f", -8.0))])
refuse("vox_offset_below_348", "bad_voxoff_small.nii",
       [(108, struct.pack("<f", 100.0))])
refuse("vox_offset_past_eof", "bad_voxoff_eof.nii",
       [(108, struct.pack("<f", 1.0e6))])
refuse("vox_offset_not_integral", "bad_voxoff_frac.nii",
       [(108, struct.pack("<f", 352.5))])


# ======================================================================
# 11b. Float and complex values with magnitudes the printer can show.
# ======================================================================
# The byte ramp is the right fixture for proving byte order and width, but
# 0x80 0x81 0x82 0x83 read as a float32 is a denormal, and nifti_tool prints
# floats with plain "%f" then trims trailing zeros (clear_float_zeros), so it
# comes out as "0.0" and tells a port nothing. These fixtures carry
# hand-packed IEEE-754 patterns instead: the bytes are an INPUT, and what the
# oracle makes of them is the answer.
FLOAT_VALUES = [0.0, 1.0, -1.0, 0.5, 1.0e-40, float("inf"), float("-inf"),
                float("nan")]
FLOAT_VALUES_LABELS = ["0.0", "1.0", "-1.0", "0.5", "1e-40", "inf", "-inf",
                       "nan"]
FLOAT_DIMS = [3, 4, 2, 1]          # 4x2x1, eight voxels
float_samples = {}
for label, code, fmt in (("float32", 16, "<f"), ("float64", 64, "<d")):
    base = fx("float_" + label)
    make(base, 1, 1, code, FLOAT_DIMS)
    path = base + ".nii"
    payload = b"".join(struct.pack(fmt, v) for v in FLOAT_VALUES)
    edit(path, path, [(352, payload)])
    read = probe("read", path)
    float_samples[label] = {
        "fixture": f"fixtures/float_{label}.nii",
        "values_packed": ["0.0", "1.0", "-1.0", "0.5", "1e-40",
                          "inf", "-inf", "nan"],
        "payload_hex": payload.hex(),
        "data_hex_after_load": read["nim"]["data_hex_after_load"],
        "nim_datatype": read["nim"]["datatype"],
        "disp_ci": ntool("-disp_ci", -1, -1, -1, -1, -1, -1, -1,
                         "-infiles", path),
        "per_voxel": {f"{x},{y},0": ntool("-disp_ci", x, y, 0, -1, -1, -1, -1,
                                          "-infiles", path)["stdout"]
                      for x in range(4) for y in range(2)},
    }
    # And the same payload big-endian, swapped by the library, to prove the
    # float path swaps at the right width.
    be = fx(f"float_{label}_be.nii")
    probe("swapfile", path, be, 1, 352, 4 if code == 16 else 8)
    be_read = probe("read", be)
    float_samples[label]["big_endian"] = {
        "fixture": f"fixtures/float_{label}_be.nii",
        "data_hex_after_load": be_read["nim"]["data_hex_after_load"],
        "matches_little_endian":
            be_read["nim"]["data_hex_after_load"] ==
            read["nim"]["data_hex_after_load"],
        "disp_ci": ntool("-disp_ci", -1, -1, -1, -1, -1, -1, -1,
                         "-infiles", be)["stdout"],
    }

# The library counts the values it rewrote, but only says so at debug > 1.
float_fix_debug = clean(run(
    [PROBE, "read", fx("float_float32.nii"), "0", "--debug"]).stderr)
float_fix_debug = "\n".join(
    ln for ln in float_fix_debug.splitlines() if "bad floats" in ln)

# COMPLEX64 is two float32s per voxel. nifti_tool will not print it, so the
# answer is the loaded bytes; four voxels, each (real, imag).
COMPLEX_VALUES = [(1.0, -1.0), (0.0, 0.5), (float("inf"), 0.0),
                  (2.5, float("nan"))]
cbase = fx("complex64")
make(cbase, 1, 1, 32, [3, 4, 1, 1])
cpath = cbase + ".nii"
cpayload = b"".join(struct.pack("<ff", re, im) for re, im in COMPLEX_VALUES)
edit(cpath, cpath, [(352, cpayload)])
cread = probe("read", cpath)
float_samples["complex64"] = {
    "fixture": "fixtures/complex64.nii",
    "values_packed": [["1.0", "-1.0"], ["0.0", "0.5"], ["inf", "0.0"],
                      ["2.5", "nan"]],
    "payload_hex": cpayload.hex(),
    "data_hex_after_load": cread["nim"]["data_hex_after_load"],
    "nbyper": cread["nim"]["nbyper"],
    "nvox": cread["nim"]["nvox"],
    "disp_ci": ntool("-disp_ci", -1, -1, -1, -1, -1, -1, -1, "-infiles", cpath),
    "note": "nbyper is 8 and nvox is 4, so nvox counts VOXELS not components. "
            "The swapsize for COMPLEX64 is 4, not 8: the two float32 halves "
            "are swapped separately.",
}

# ======================================================================
# 11. Sample values, pinned per datatype at named coordinates.
# ======================================================================
samples = {}
for code in (2, 4, 8, 16, 64, 256, 512, 768, 1024, 1280, 128, 2304, 32):
    key = str(code)
    if key not in dt_roundtrip or "fixture" not in dt_roundtrip[key]:
        continue
    path = os.path.join(ROOT, dt_roundtrip[key]["fixture"])
    per_voxel = {}
    for x in range(2):
        for y in range(3):
            per_voxel[f"{x},{y},0"] = ntool(
                "-disp_ci", x, y, 0, -1, -1, -1, -1, "-infiles", path)["stdout"]
    samples[key] = {
        "name": dt_roundtrip[key]["name"],
        "fixture": dt_roundtrip[key]["fixture"],
        "payload_bytes": fixture_bytes(os.path.basename(path))[352:],
        "voxels": per_voxel,
    }

# ======================================================================
# Assemble.
# ======================================================================
records["libvips_is_not_the_oracle"] = {
    "what": "This is the ONE area in #510 with no libvips oracle, and the "
            "check is re-run on every capture so a future build that gains "
            "libnifti cannot slip past unnoticed. Runbook §20's three-step "
            "rule is followed: the config line, the operator listing, and "
            "then actually running the thing on a real file.",
    "step_1_vips_config": {
        "vips_version": vips_version,
        "nifti_line": nifti_config_line,
        "full_config": config_lines,
    },
    "step_2_operator_listing": {
        "vips_l_lines_mentioning_nifti": nifti_ops,
        "niftiload": {"exit": niftiload_probe.returncode,
                      "stdout": clean(niftiload_probe.stdout),
                      "stderr": clean(niftiload_probe.stderr)},
        "niftisave": {"exit": niftisave_probe.returncode,
                      "stdout": clean(niftisave_probe.stdout),
                      "stderr": clean(niftisave_probe.stderr)},
        "note": "`vips -l | grep -i nifti` comes back completely empty, "
                "which is the answer runbook §8 says to trust; the "
                "`niftiload` / `niftisave` probes below are the same answer "
                "from the other direction. Their exit status and message are "
                "recorded as measured rather than predicted.",
    },
    "step_3_run_it_on_a_real_file": {
        "file": "fixtures/ver_n1_single.nii",
        "exit": vips_on_nii.returncode,
        "stderr": clean(vips_on_nii.stderr),
        "what_happened": "no loader claims it, so the sniffing chain falls "
                         "through to magickload, which guesses TGA and fails",
    },
    "conclusion": "Do not 'fix' this area to use vips. There is nothing "
                  "behind that door.",
}

records["oracle_build"] = {
    "what": "How the oracle was obtained and what version it is. There is no "
            "`vips --vips-config` equivalent to lean on here, and runbook §20 "
            "records that the config line is a build-time claim that names no "
            "library versions anyway, so everything below was measured.",
    "no_homebrew_formula": {
        "checked": "brew search nifti / brew info nifticlib / brew info "
                   "nifti_clib",
        "result": "no formula exists in homebrew-core, so it was built from "
                  "source",
    },
    "source": {
        "repo": git_remote,
        "commit": git_head,
        "git_describe": git_describe,
        "commit_date": git_date,
        "how_obtained": "git clone --depth 200 "
                        "https://github.com/NIFTI-Imaging/nifti_clib.git",
        "why_master_and_not_the_v3.0.1_tag": "v3.0.1 is from 2020-08-07 and "
                                             "master carries four more years "
                                             "of fixes to nifti2_io.c and "
                                             "nifti_tool.c; the commit SHA "
                                             "above pins it exactly either way",
    },
    "versions_reported_by_the_binary": {
        "nifti_tool_ver": tool_ver,
        "nifti_lib_ver": lib_ver,
        "with_zlib": with_zlib,
        "how": "`nifti_tool -ver`, `-nifti_ver` and `-with_zlib`, run at "
               "capture time; not transcribed from a header",
    },
    "build": {
        "generator": "cmake",
        "cmake_version": run(["cmake", "--version"]).stdout.splitlines()[0],
        "configure": "cmake -S nifti_clib -B build -DCMAKE_BUILD_TYPE=Release "
                     "-DCMAKE_INSTALL_PREFIX=$NIFTI_PREFIX "
                     "-DNIFTI_BUILD_APPLICATIONS=ON -DUSE_NIFTI2_CODE=ON",
        "build_cmd": "cmake --build build -j8 && cmake --install build",
        "compiler": cc_version,
        "probe_cflags": " ".join(CFLAGS).replace(NIFTI_PREFIX, "$NIFTI_PREFIX"),
        "probe_libs": " ".join(a.replace(NIFTI_PREFIX, "$NIFTI_PREFIX")
                               for a in LIBS),
        "probe_compile_exit": compile_proc.returncode,
    },
    "platform": {
        "uname": run(["uname", "-srm"]).stdout.strip(),
        "sw_vers": run(["sw_vers", "-productVersion"]).stdout.strip(),
        "how": "`uname -srm` and `sw_vers -productVersion` at capture time",
    },
    "library_facts_from_probe_env": env,
}

records["datatype_codes"] = {
    "what": "Every datatype code the library will answer for, swept through "
            "nifti_datatype_sizes, nifti_datatype_string, "
            "nifti_datatype_to_string, nifti_datatype_from_string, "
            "nifti_is_inttype, nifti_is_valid_datatype and "
            "nifti_datatype_is_valid for both the Analyze and the NIfTI "
            "dialect. Codes that are NOT valid are included on purpose, "
            "because the refusal boundary is the part a port gets wrong.",
    "table": dt_table,
    "nifti_test_datatype_sizes": {
        "result": env["nifti_test_datatype_sizes"],
        "what_it_checks": "nifti_datatype_sizes() against the nifti_type_list "
                          "table, i.e. the library against itself. It never "
                          "compares either to sizeof, so a platform where "
                          "long double is not 16 bytes still reports 0 errors.",
        "host_widths": {"float": env["sizeof_float"],
                        "double": env["sizeof_double"],
                        "long double": env["sizeof_long_double"]},
    },
    "codes_valid_for_nifti_with_a_width": [r["code"] for r in writable],
    "codes_valid_for_analyze_but_not_nifti":
        [r["code"] for r in dt_table
         if r["valid_for_analyze"] and not r["valid_for_nifti"]],
    "zero_width_but_still_valid":
        [r["code"] for r in dt_table
         if r["valid_for_nifti"] and r["nbyper"] == 0],
}

records["datatype_roundtrip"] = {
    "what": "One file per writable code, all six voxels filled with the same "
            "ramp of consecutive bytes 80 81 82 ... so the ONLY thing that "
            "varies between fixtures is how the code reinterprets those "
            "bytes. `disp_ci_values` is nifti_tool printing them back through "
            "the library's own type dispatch.",
    "disp_ci_only_handles_eight_types": "nifti_tool's act_disp_ci switches on "
        "nim->datatype and accepts only INT8 INT16 INT32 INT64 UINT8 UINT16 "
        "UINT32 FLOAT32 FLOAT64; everything else (the complex types, RGB24, "
        "RGBA32, UINT64, FLOAT128) prints `** dataset ... has unknown type` "
        "on stderr and no values at all. The header still loads and the "
        "pixels still come back through nifti_image_read, so this is a limit "
        "of the TOOL, not of the format or the library. The "
        "data_hex_after_load field is the oracle's answer for those types.",
    "dims": DIMS,
    "fill": "byte i of the payload is (0x80 + i) & 0xff, so the most "
            "significant byte of every value carries a set sign bit and a "
            "signed/unsigned mix-up cannot hide behind small positives",
    "by_code": dt_roundtrip,
}

records["nifti1_vs_nifti2"] = {
    "what": "What separates the two versions on disk, and how the loader "
            "tells them apart.",
    "sizeof_hdr": {
        "nifti_1_header": env["sizeof_nifti_1_header"],
        "nifti_2_header": env["sizeof_nifti_2_header"],
        "nifti_analyze75": env["sizeof_nifti_analyze75"],
    },
    "magic": {
        "nifti1_single_file": fixture_bytes("ver_n1_single.nii")[344:348],
        "nifti1_pair": fixture_bytes("pair_n1.hdr")[344:348],
        "nifti2_single_file": fixture_bytes("ver_n2_single.nii")[4:12],
        "nifti2_pair": fixture_bytes("pair_n2.hdr")[4:12],
        "note": "NIfTI-1 puts a 4-byte magic at offset 344, the LAST field of "
                "the 348-byte header. NIfTI-2 moves it to offset 4, the "
                "SECOND field, and makes it 8 bytes: the name, a NUL, then "
                "\\r \\n \\032 \\n. The spec puts that tail there to catch a "
                "file mangled by a text-mode transfer. See "
                "magic_tail_is_never_checked below for whether the reference "
                "implementation actually looks at it.",
    },
    "field_offsets": offsets,
    "header_first_60_bytes": {
        "nifti1": list(n1_head),
        "nifti2": list(n2_head),
    },
    "nifti_header_version": {
        "nifti1_file": n1_hdrver,
        "nifti2_file": n2_hdrver,
        "note": "A NIfTI-2 file is identified from only 348 bytes, because "
                "sizeof_hdr and the magic both live in the first 12. The "
                "function refuses anything shorter than 348 outright, "
                "including a valid NIfTI-2 header read 347 bytes at a time.",
    },
    "magic_tail_is_never_checked": {
        "what": "MEASURED, and it is the opposite of what the spec's "
                "rationale implies. nifti_clib decides the version with the "
                "NIFTI_VERSION macro (nifti1.h:1495), which reads magic[0], "
                "magic[1], magic[2] and magic[3] and nothing else: `n`, then "
                "`i` or `+`, then a digit 1..9, then a NUL. Bytes 4..7 of the "
                "NIfTI-2 magic, the \\r \\n \\032 \\n that exists to detect a "
                "text-mode transfer, are never examined. Both fixtures below "
                "have that tail destroyed and both load normally. A port that "
                "validates the tail will reject files the reference accepts.",
        "one_file_vs_pair_is_magic_1": "NIFTI_ONEFILE (nifti1.h:1506) is "
                "`magic[1] == '+'` alone, so `n+1`/`n+2` mean one file and "
                "`ni1`/`ni2` mean a pair, and nothing else in the header "
                "says which.",
        "mangled_crlf_tail": {
            "fixture": "fixtures/n2_magic_tail_mangled.nii",
            "hdrver": probe("hdrver", fx("n2_magic_tail_mangled.nii")),
            "read": probe_full("read", fx("n2_magic_tail_mangled.nii")),
        },
        "partial_tail": {
            "fixture": "fixtures/n2_magic_tail_partial.nii",
            "hdrver": probe("hdrver", fx("n2_magic_tail_partial.nii")),
            "read": probe_full("read", fx("n2_magic_tail_partial.nii")),
        },
    },
    "vox_offset_and_dim_widths": {
        "nifti1_dim_element": "int16, so no axis can exceed 32767",
        "nifti2_dim_element": "int64",
        "nifti1_vox_offset": "float32",
        "nifti2_vox_offset": "int64",
        "measured_from": "the field_offsets tables above, which report sizeof "
                         "for every field",
        "default_iname_offset_nifti1": pair_records["nifti1_single"]["iname_offset"],
        "default_iname_offset_nifti2": pair_records["nifti2_single"]["iname_offset"],
    },
}

records["analyze75_overlay"] = {
    "what": "NIfTI-1 deliberately reuses the Analyze 7.5 348-byte header, "
            "renaming the fields it needs and leaving the rest alone. This is "
            "the field-by-field overlay, computed by joining the two offsetof "
            "tables the oracle produced, so a port can see exactly which "
            "Analyze field each NIfTI-1 field displaced.",
    "same_size": env["sizeof_nifti_1_header"] == env["sizeof_nifti_analyze75"],
    "overlay": overlay,
    "renamed_fields": [o["nifti1_field"] for o in overlay if not o["same_name"]],
    "magic_all_zero_is_analyze": {
        "fixture": "fixtures/magic_zero_analyze.nii",
        "hdrver": probe("hdrver", fx("magic_zero_analyze.nii")),
        "read": probe_full("read", fx("magic_zero_analyze.nii")),
        "note": "nifti_header_version returns 0 for a 348-byte header whose "
                "magic is not `n+1`, and the loader does not refuse: the "
                "pixels still come back. What it calls the file afterwards "
                "depends on the FILENAME, not the bytes.",
    },
    "the_extension_decides_when_the_magic_does_not": {
        "what": "With ni_ver 0 the loader skips the `nim->nifti_type = "
                "onefile ? 1 : 2` assignment entirely, and "
                "nifti_set_type_from_names then fills it in from the "
                "extension. So the SAME 348 bytes with a zeroed magic become "
                "nifti_type 1 with iname_offset 352 when the file is called "
                ".nii, and nifti_type 0 (ANALYZE) with iname_offset 0 when it "
                "is a .hdr next to a .img. A port that decides the container "
                "from the magic alone will read the pixels from the wrong "
                "offset for one of the two.",
        "as_a_nii": {
            "fixture": "fixtures/magic_zero_analyze.nii",
            "read": probe_full("read", fx("magic_zero_analyze.nii")),
        },
        "as_a_hdr_img_pair": {
            "fixture": "fixtures/magic_zero_pair.hdr",
            "read": probe_full("read", fx("magic_zero_pair.hdr")),
        },
        "note": "scl_slope also differs: for ni_ver 0 the NIfTI-only fields "
                "are never read off the header, so scl_slope comes back 0 "
                "even though the source file had 1.",
    },
    "the_libvips_boundary": {
        "what": "The sibling foreign-analyze capture (PR #645) records that "
                "libvips reads the 348-byte Mayo header BIG-endian only. A "
                "big-endian NIfTI-1 pair is therefore byte-compatible with "
                "what analyzeload wants, and the ONLY thing keeping them "
                "apart is the four-byte extender NIfTI appends to the .hdr, "
                "which makes it 352 bytes where analyze2vips demands exactly "
                "348. Strip those four bytes and libvips loads a NIfTI file "
                "as Analyze, silently, ignoring scl_slope and every other "
                "NIfTI-only field.",
        "big_endian_pair_352_byte_hdr": {
            "fixture": "fixtures/pair_be.hdr",
            "vipsheader_exit": vips_on_be_pair_352.returncode,
            "vipsheader_stderr": clean(vips_on_be_pair_352.stderr),
        },
        "big_endian_pair_348_byte_hdr": {
            "fixture": "fixtures/pair_be348.hdr",
            "vipsheader_exit": vips_on_be_pair_348.returncode,
            "vipsheader_stdout": clean(vips_on_be_pair_348.stdout),
        },
        "little_endian_pair": {
            "fixture": "fixtures/pair_be_src.hdr",
            "vipsheader_exit": vips_on_le_pair.returncode,
            "vipsheader_stderr": clean(vips_on_le_pair.stderr),
            "why": "sizeof_hdr 348 read big-endian is 1543503872, so "
                   "vips__isanalyze rejects it and the chain falls through "
                   "to magickload",
        },
    },
}

records["single_file_and_pair"] = {
    "what": "The two containers, what distinguishes them, and what a missing "
            "half does.",
    "forms": pair_records,
    "rules_measured": {
        "single_file_magic_nifti1": "n+1\\0",
        "paired_magic_nifti1": "ni1\\0",
        "single_file_magic_nifti2": "n+2\\0\\r\\n\\032\\n",
        "paired_magic_nifti2": "ni2\\0\\r\\n\\032\\n",
        "single_file_data_offset_nifti1":
            pair_records["nifti1_single"]["iname_offset"],
        "paired_data_offset_nifti1":
            pair_records["nifti1_pair"]["iname_offset"],
        "single_file_data_offset_nifti2":
            pair_records["nifti2_single"]["iname_offset"],
        "paired_data_offset_nifti2":
            pair_records["nifti2_pair"]["iname_offset"],
        "hdr_size_when_paired":
            pair_records["nifti1_pair"]["header_bytes"],
        "note": "The paired .hdr is still 352 bytes, not 348: the four-byte "
                "extender is written either way. The pixels start at byte 0 "
                "of the .img, so vox_offset is 0 for a pair and 352 (or 544) "
                "for a single file.",
    },
    "nifti_type_codes": {
        "0": "ANALYZE",
        "1": "NIFTI-1, one file",
        "2": "NIFTI-1, two files",
        "3": "NIFTI-ASCII, one file",
        "4": "NIFTI-2, one file",
        "5": "NIFTI-2, two files",
        "source": "nifti2_io.h, and every value above was read back off a "
                  "written file rather than assumed",
    },
    "missing_half": missing,
    "crossed_magic": crossed,
}

records["scl_slope_and_inter"] = {
    "what": "The scaling pair, and the special case that looks like a bug. "
            "nifti1.h says `if the scl_slope field is nonzero, then each "
            "voxel value should be scaled as y = scl_slope * x + scl_inter`, "
            "so a slope of ZERO means no scaling at all, NOT a multiply by "
            "zero. Every consumer has to special-case it.",
    "where_the_rule_lives": {
        "core_library": "nifti2_io.c does NOT apply scaling. nifti_image_load "
                        "hands back raw voxels and leaves y = slope*x + inter "
                        "to the caller, so nim->scl_slope and nim->scl_inter "
                        "are carried, not consumed. `disp_ci_values` below is "
                        "therefore UNSCALED in every row.",
        "the_rule_implemented": "fsliolib/fslio.c, FslGetVolumeAsScaledDouble "
                                "and FslGetBufferAsScaledDouble: "
                                "`if (scl_slope == 0) { slope = 1.0; inter = "
                                "0.0; }` else use both. That is the reference "
                                "spelling of the zero rule.",
        "spec_carve_outs": "nifti1.h also says the scaling is to be IGNORED "
                           "for DT_RGB24, and applied to both parts of a "
                           "complex type.",
    },
    "cases": scaling,
    "the_write_side_asymmetry": {
        "what": "nifti_convert_nim2n1hdr writes the pair only when the slope "
                "is non-zero (`if (nim->scl_slope != 0.0) { nhdr.scl_slope = "
                "...; nhdr.scl_inter = ...; }`), so a NIfTI-1 file written "
                "with slope 0 and inter 7 comes back with inter 0 as well. "
                "nifti_convert_nim2n2hdr assigns both unconditionally, so the "
                "SAME nifti_image written as NIfTI-2 keeps the 7. Compare "
                "slope_0_inter_7_nifti1 against slope_0_inter_7_nifti2.",
        "nifti1_kept_inter": scaling["slope_0_inter_7_nifti1"]["header_scl_inter"],
        "nifti2_kept_inter": scaling["slope_0_inter_7_nifti2"]["header_scl_inter"],
    },
    "non_finite_slope_becomes_zero": {
        "what": "nifti2_io.h wraps every float coming off the header in "
                "FIXED_FLOAT, defined as `isfinite(x) ? (x) : 0`. So an "
                "infinity or a NaN in scl_slope arrives at the caller as a "
                "plain 0.0, which is exactly the value that means 'do not "
                "scale'. A port that carries the NaN through will produce NaN "
                "voxels where the reference produces the raw ones.",
        "inf": scaling["slope_inf"],
        "nan": scaling["slope_nan"],
    },
}

records["byte_order"] = {
    "what": "The format is endianness-sensitive and there is no flag for it. "
            "Detection is entirely by sentinel: read sizeof_hdr as a native "
            "int32, and if it is neither 348 nor 540, byte-swap it and try "
            "again. Whichever way it matched decides both the version and the "
            "byte order, and the loader then swaps the header AND the pixel "
            "data in place. Both directions are captured; the big-endian "
            "fixtures were produced by the library's own nifti_swap_as_nifti1 "
            "/ nifti_swap_as_nifti2 and nifti_swap_Nbytes, not by hand.",
    "sentinel": {
        "nifti1": "sizeof_hdr == 348",
        "nifti2": "sizeof_hdr == 540",
        "why_it_is_unambiguous": "348 byte-swapped is 1543503872 and 540 "
                                 "byte-swapped is 469827584; neither is 348 "
                                 "or 540, so at most one of the four "
                                 "interpretations can match",
        "then": "the magic decides ANALYZE (0) vs NIFTI-1 (1) vs NIFTI-2 (2); "
                "a 348-byte header with a magic that is neither empty nor "
                "`n+1`/`ni1` is refused outright",
    },
    "cases": byte_order,
}

records["dimensionality"] = {
    "what": "dim[0] is the RANK, 1..7, and dim[1..dim[0]] are the extents. "
            "nvox is their product. Every axis above the rank is ignored, "
            "and nothing is folded into anything: unlike libvips's "
            "analyzeload, which flattens every axis above 2 into the height, "
            "nifti_clib keeps all seven and leaves the reshaping to the "
            "caller.",
    "ranks": dims,
    "collapsing_to_2d": {
        "what": "nifti_read_collapsed_image is how a consumer that wants one "
                "2D plane out of a 3D or 4D volume gets it: pass -1 for the "
                "axes to keep whole and an index for the axes to fix. "
                "nifti_tool exposes it as -disp_ci (print) and -cci (write a "
                "new dataset). The extracted plane's own dim[] is what a 2D "
                "consumer ends up holding.",
        "source_volume": "fixtures/dim_rank4_2x3x2x2.nii, uint8, values 1..24",
        "cases": collapse,
    },
    "edge_cases": dim_edge,
    "what_the_edge_cases_show": {
        "dim0_zero_is_accepted": "dim[0] == 0 passes nifti_hdr1_looks_good "
            "AND loads, coming back as dim [0,1,1,1,0,0,0,0] with nvox 1. "
            "Every extent is discarded. A port that treats rank 0 as an "
            "error will diverge from the reference, which treats it as a "
            "one-voxel image.",
        "dim0_out_of_range_is_refused": "dim[0] of 8 or -1 fails in "
            "nifti_convert_n1hdr2nim with `bad dim[0]`, before any pixel is "
            "touched.",
        "a_zero_extent_inside_the_rank_is_silently_clamped": "dim[2] == 0 "
            "with dim[0] == 3 fails nifti_hdr1_looks_good (returns 0) and "
            "STILL LOADS, with dim[2] rewritten to 1 and nvox dropping from "
            "12 to 4. nifti_image_read does not consult looks_good, so the "
            "check and the load disagree. dim[1] == 0 is refused, though, so "
            "the clamp is not uniform across axes.",
        "dims_beyond_the_rank_are_forced_to_1": "nifti_convert_n1hdr2nim "
            "rewrites every dim[ii] for ii > dim[0] that is neither 0 nor 1 "
            "to 1, so garbage above the rank cannot propagate into nvox.",
        "impossible_product_is_caught_at_the_pixels": "32767^3 passes the "
            "header check and fails in nifti_read_buffer with a byte-count "
            "warning, not in the header.",
    },
    "pixdim_and_qfac": {
        "what": "pixdim is repaired on the way in, not carried. "
                "nifti_convert_n1hdr2nim runs `if (pixdim[ii] == 0.0 || "
                "!IS_GOOD_FLOAT(pixdim[ii])) pixdim[ii] = 1.0` for every ii "
                "in 1..dim[0], so a zero, an infinity or a NaN spacing all "
                "become 1.0. A NEGATIVE spacing is NOT repaired, though: "
                "-2.0 comes through as -2.0, so the repair is not a "
                "plausibility check, it is a divide-by-zero guard. pixdim[0] "
                "is qfac rather than a spacing: it lands in nim->qfac and "
                "nim->pixdim[0] is left at 0. And qfac is only computed at "
                "all when qform_code is non-zero, because the whole "
                "quaternion branch is gated on it; with qform_code 0, "
                "nim->qfac stays whatever nifti_make_new_nim left, which is "
                "0 and is not a valid handedness. Otherwise it is "
                "`pixdim[0] < 0 ? -1 : 1`, so a 0 on disk reads as +1.",
        "cases": pixdim,
    },
}

records["read_header_returns_a_header_even_when_the_check_fails"] = {
    "what": "nifti_read_header(hname, &nver, check=1) logs `nifti_1_header "
            "looks bad for file ...` and then RETURNS the header anyway "
            "(nifti2_io.c: `if (check && !nifti_hdr1_looks_good(hresult)) { "
            "LNI_FERR(...); return hresult; }`). A caller that only tests the "
            "pointer for NULL will sail straight past a header the library "
            "just declared bad. It also does NOT byte-swap: the buffer comes "
            "back exactly as it sits on disk, and the swapping happens later "
            "in nifti_convert_n1hdr2nim. Both are visible in the refusals "
            "record below, where readhdr_check_on reports got_header true "
            "with an error on stderr.",
    "example": "refusals.datatype_unknown_code_3.readhdr_check_on",
}

records["refusals"] = {
    "what": "Where the library gives up, and what it says. Weighted the same "
            "as the successes because a port that accepts a broken file is "
            "worse than one that rejects a good one. `readhdr_check_on` is "
            "nifti_read_header with check=1, which runs "
            "nifti_hdr1_looks_good; `read` is the full nifti_image_read, "
            "which also has to find and size the pixels.",
    "cases": refusals,
}

records["sample_values"] = {
    "what": "Actual voxel values at named coordinates, not just header "
            "fields. Same six-voxel ramp in every datatype, read back through "
            "nifti_tool's -disp_ci so the numbers come out of the library's "
            "own type dispatch. `payload_bytes` is what is on disk after byte "
            "352, so the mapping from bytes to values is fully pinned.",
    "layout": "column-major: -disp_ci I J K takes I along dim[1] and J along "
              "dim[2], and the payload runs I fastest",
    "by_code": samples,
    "float_and_complex": {
        "what": "The byte ramp reads as a denormal in float32 and nifti_tool "
                "prints floats with plain %f, so the ramp fixtures show 0.0 "
                "for every float voxel and prove nothing. These carry "
                "hand-packed IEEE-754 patterns instead, including both "
                "infinities and a NaN, so the printer's formatting and the "
                "loader's swap width are both pinned.",
        "printer": "nifti_tool disp_raw_data uses snprintf(\"%f\") for "
                   "FLOAT32 and snprintf(\"%lf\") for FLOAT64, then "
                   "clear_float_zeros trims trailing zeros but never the "
                   "digit immediately after the point. That is why 1 prints "
                   "as 1.0 and 1e-40 prints as 0.0.",
        "values": FLOAT_VALUES_LABELS,
        "non_finite_voxels_are_rewritten_to_zero": {
            "what": "nifti_read_buffer walks the loaded buffer for FLOAT32, "
                    "COMPLEX64, FLOAT64 and COMPLEX128 and sets every value "
                    "that fails IS_GOOD_FLOAT (isfinite) to 0, counting them. "
                    "So an infinity or a NaN stored in the file NEVER reaches "
                    "the caller: it arrives as a plain zero. Compare "
                    "payload_hex against data_hex_after_load below, where the "
                    "last three float32 words go from 0000807f 000080ff "
                    "0000c07f to 00000000 00000000 00000000.",
            "not_applied_to": "FLOAT128 and COMPLEX256 are not in that "
                              "switch, so a non-finite long double survives",
            "debug_message": float_fix_debug,
        },
        "cases": float_samples,
    },
}

fixture_files = sorted(os.listdir(FIX))
fixture_total = sum(os.path.getsize(fx(n)) for n in fixture_files)

oracle = {
    "meta": {
        "area": "foreign-nifti",
        "issue": 641,
        "parent_issue": 510,
        "oracle": "nifti_clib (NIFTI-Imaging/nifti_clib), the NIH reference "
                  "implementation of the NIfTI format",
        "oracle_is_not_libvips": True,
        "why_not_libvips": "this libvips build reports `NIfTI load/save with "
                           "libnifti: false` and registers neither niftiload "
                           "nor niftisave; see the "
                           "libvips_is_not_the_oracle record for the full "
                           "three-step evidence, re-measured on every run",
        "vips_version_at_capture": vips_version,
        "vips_nifti_config_line": nifti_config_line,
        "nifti_clib_commit": git_head,
        "nifti_clib_describe": git_describe,
        "nifti_clib_commit_date": git_date,
        "nifti_tool_version": tool_ver,
        "nifti_lib_version": lib_ver,
        "compiler": cc_version,
        "platform": run(["uname", "-srm"]).stdout.strip(),
        "captured_by": "oracle-captures/foreign-nifti/capture.py",
        "harness": "oracle-captures/foreign-nifti/probe.c, compiled against "
                   "libnifti2.a at capture time",
        "how_every_version_was_obtained": "measured at capture time: the "
                                          "nifti_clib identity from `git "
                                          "rev-parse` / `git describe` in the "
                                          "checkout, the library and tool "
                                          "versions from `nifti_tool -ver` "
                                          "and `-nifti_ver`, the compiler "
                                          "from `cc --version`, the platform "
                                          "from `uname -srm`, and the vips "
                                          "line from `vips --vips-config`. "
                                          "Nothing here is transcribed from a "
                                          "brief or a header file.",
        "fixture_count": len(fixture_files),
        "fixture_bytes": fixture_total,
    },
    "notes": [
        "libvips is NOT the oracle for this area and cannot be. If someone "
        "later 'fixes' capture.py to shell out to vips they will get nothing: "
        "the operators do not exist, and a real .nii falls through the "
        "sniffing chain to magickload, which guesses TGA. The "
        "libvips_is_not_the_oracle record re-measures all three steps every "
        "run so a build that gains libnifti announces itself.",
        "The toolchain on this machine moved mid-session (#650): vips went "
        "8.18.4 -> 8.18.6 and several codec libraries shifted under running "
        "lanes. Every version in meta was measured at capture time rather "
        "than assumed, and the vips version is recorded even though vips is "
        "not the oracle, because the ONE thing vips is used for here is "
        "proving it cannot be.",
        "nifti_clib does not apply scl_slope / scl_inter. nifti_image_load "
        "returns raw voxels; the y = slope*x + inter rule, and the 'slope 0 "
        "means no scaling' special case, live in fsliolib/fslio.c. So every "
        "disp_ci value in this file is UNSCALED, and a port that scales in "
        "its loader is doing something the reference does not.",
        "The ramp fill is byte i -> (0x80 + i) & 0xff, identical across every "
        "datatype fixture. That is deliberate: the only variable between "
        "those files is the reinterpretation, so a port with a sign, width or "
        "byte-order error cannot accidentally match.",
        "DT_FLOAT128 (1536) and DT_COMPLEX256 (2048) get nbyper 16 and 32 "
        "out of the datatype table, which is right only where `long double` "
        "is 16 bytes. oracle_build.library_facts_from_probe_env records what "
        "it actually is on this host, and nifti_test_datatype_sizes cannot "
        "catch a mismatch because it checks the table against itself rather "
        "than against sizeof. The files this capture writes for those codes "
        "are still 16 and 32 bytes per voxel, so a port matching the TABLE "
        "matches the files.",
        "The loader REWRITES pixel data. nifti_read_buffer sets every "
        "non-finite FLOAT32, COMPLEX64, FLOAT64 and COMPLEX128 voxel to 0 "
        "before the caller sees it, so an infinity or a NaN stored in a file "
        "never comes back. It repairs pixdim the same way, turning a 0, an "
        "infinity or a NaN spacing into 1.0, though it leaves a NEGATIVE "
        "spacing alone. Neither is announced unless the debug level is "
        "raised. A port that faithfully passes those values through will "
        "disagree with the reference on real files.",
        "When the magic does not say which container it is, the FILENAME "
        "decides. The same 348 bytes with a zeroed magic load as nifti_type "
        "1 with the pixels at byte 352 if the file is called .nii, and as "
        "nifti_type 0 with the pixels at byte 0 of the .img if it is a .hdr. "
        "A .nii carrying the paired `ni1` magic still loads as a single file "
        "for the same reason.",
        "Malformed fixtures were hand-built by poking bytes into a good file, "
        "because the reference implementation will not write a broken one. "
        "Every ANSWER about them still comes from the oracle.",
    ],
    "records": records,
}

with open(os.path.join(ROOT, "oracle.json"), "w") as f:
    json.dump(oracle, f, indent=2, sort_keys=False)
    f.write("\n")

with open(os.path.join(ROOT, "commands.sh"), "w") as f:
    f.write("#!/bin/sh\n")
    f.write("# Every command capture.py ran, in order. Regenerate with\n")
    f.write("# `python3 capture.py` from this directory.\n")
    f.write("#\n")
    f.write("# The oracle is nifti_clib, NOT libvips: this vips build reports\n")
    f.write("# `NIfTI load/save with libnifti: false` and has no niftiload.\n")
    f.write("# nifti_clib was obtained and built like this, once, before any\n")
    f.write("# of the commands below ran:\n")
    f.write("#\n")
    f.write("#   git clone --depth 200 \\\n")
    f.write("#       https://github.com/NIFTI-Imaging/nifti_clib.git\n")
    f.write(f"#   # HEAD {git_head}\n")
    f.write(f"#   # {git_describe}, {git_date}\n")
    f.write("#   cmake -S nifti_clib -B build -DCMAKE_BUILD_TYPE=Release \\\n")
    f.write("#         -DCMAKE_INSTALL_PREFIX=$NIFTI_PREFIX \\\n")
    f.write("#         -DNIFTI_BUILD_APPLICATIONS=ON -DUSE_NIFTI2_CODE=ON\n")
    f.write("#   cmake --build build -j8\n")
    f.write("#   cmake --install build\n")
    f.write("#\n")
    f.write("# NIFTI_PREFIX is where that install landed.\n")
    f.write("set -e\n\n")
    f.write(f"NIFTI_PREFIX={NIFTI_PREFIX}\n\n")
    for c in COMMANDS:
        f.write(c + "\n")

print(f"wrote oracle.json ({len(records)} records), commands.sh "
      f"({len(COMMANDS)} commands), {len(fixture_files)} fixtures "
      f"totalling {fixture_total} bytes")
