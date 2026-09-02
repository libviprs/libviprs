#!/usr/bin/env python3
"""
Oracle capture for the Matlab area (issue #640, sub-issue of #510).

Runs the real installed vips CLI over hand-built MAT-file variables and records
exactly what `matload` does with them. Everything a port needs is here,
because none of it can be derived from the MAT-file spec:

  * the sniff in the SHIPPED BINARY is not the one in the C source: it
    reads 128 bytes and validates the version word and the endian indicator
    as well as the `MATLAB 5.0` prefix, where matlab.c reads ten bytes and
    checks the prefix alone. MAT-4 and MAT-7.3 are never routed to matload
    even though the matio underneath reads MAT-4 perfectly well
  * Matlab is column-major and vips TRANSPOSES on load, so dims[0] is the
    height and dims[1] is the width
  * rank 3 becomes bands, and the third dimension is PLANE-SEPARATE in the
    file and interleaved on the way out
  * the first variable with rank 1..=3 wins, and the search filters on RANK
    ONLY: an unsupported CLASS at that variable is fatal rather than skipped
  * the class -> band format table has eight entries and everything else,
    including char, sparse, int64 and uint64, is refused; the LOGICAL flag
    is not a class and is ignored, so a logical array loads as its storage
    type
  * complex arrays are NOT refused, they read back non-deterministic garbage
  * byte order is declared in the file and matio handles both, so unlike
    Analyze there is nothing here for a port to get backwards
  * a one-band uchar array is tagged MULTIBAND, not b-w, which is the
    opposite of what analyzeload does with the same shape

The oracle binary moved under this capture: Homebrew replaced 8.18.4 with
8.18.6 part-way through, and at least the sniff predicate differs between
them. `meta.vips_version` names what actually produced these numbers.

Writes:
  commands.sh  - every vips CLI command actually executed, in order
  oracle.json  - structured records
  fixtures/    - every `.mat` the records refer to

Re-running needs only the vips binary at VIPS; every input is generated from
scratch by the MAT-5 writer below, deterministically. Nothing outside this
script's own directory is written.
"""
import hashlib
import json
import os
import re
import struct
import subprocess
import sys
import zlib

ROOT = os.path.dirname(os.path.abspath(__file__))
FIX = os.path.join(ROOT, "fixtures")
OUT = os.path.join(ROOT, "outputs")

VIPS = "/opt/homebrew/bin/vips"
VIPSHEADER = "/opt/homebrew/bin/vipsheader"

AREA = "foreign-mat"

# The oracle is pinned: oracle-captures/ORACLE_PIN.json names the libvips
# build this area is measured against, and check() exits before anything is
# written when the binary on the machine disagrees, so a wrong-oracle run
# leaves no half-updated capture behind. #650 is what happened without it,
# #796 is why every area carries it now, and tests/oracle_capture_pins.rs is
# the half of the guard that runs in CI.
sys.path.insert(0, os.path.abspath(os.path.join(ROOT, os.pardir)))
import oracle_pin  # noqa: E402  (needs the path above)

VIPS_VERSION, ORACLE_PIN = oracle_pin.check(AREA, VIPS)

os.makedirs(FIX, exist_ok=True)
os.makedirs(OUT, exist_ok=True)

COMMANDS = []

# This machine's vips prints a heif module-load failure on stderr for every
# single invocation (libheif wants an x265 dylib that is not installed), and
# matio drags in libhdf5, which prints a nine-line diagnostic block when it
# is handed something that is not HDF5. Neither has anything to do with the
# MAT loader, so both are stripped rather than pinned as if they were part
# of the answer. The HDF5 case is recorded on its own in `magic_and_dispatch`.
NOISE = ("VIPS-WARNING", "Referenced from:", "Reason: tried:",
         "libheif", "x265", "unable to load \"/opt/homebrew",
         "HDF5-DIAG:", "major:", "minor:", "#0")


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
        if ln.strip() == "matload: load error":
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
    """`vipsheader` on a path: the SNIFFED path, where vips__mat_ismat has to
    accept the file before matload is ever reached."""
    args = [VIPSHEADER] + (["-a"] if all_fields else []) + [path]
    proc = run(args)
    if proc.returncode != 0:
        return {"exit": proc.returncode, "stderr": clean(proc.stderr)}
    out = {"exit": 0,
           "summary": proc.stdout.splitlines()[0].replace(ROOT + "/", "")}
    if proc.stderr.strip():
        out["stderr"] = clean(proc.stderr)
    if all_fields:
        for line in proc.stdout.splitlines()[1:]:
            if ": " in line:
                name, value = line.split(": ", 1)
                out[name.strip()] = value.strip().replace(ROOT + "/", "")
    return out


def direct(path):
    """`vips matload` called by name, which SKIPS is_a entirely. The
    difference is load-bearing here: matio reads MAT-4 perfectly well, but
    vips__mat_ismat will never route a MAT-4 file to it."""
    dest = os.path.join(OUT, "direct.v")
    proc = vips("matload", path, dest)
    r = {"exit": proc.returncode, "stderr": clean(proc.stderr)}
    if proc.returncode == 0:
        r["header"] = header(dest)["summary"]
        r["pixels_available_at"] = "outputs/direct.v"
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


def stat(path):
    """`vips avg`, which forces the whole variable through the pipeline.
    This is the call that fails when the header was fine and the data is
    not: mat2vips_get_header never touches the data element."""
    proc = vips("avg", path)
    if proc.returncode != 0:
        return {"exit": proc.returncode, "stderr": clean(proc.stderr)}
    return {"exit": 0, "avg": float(proc.stdout)}


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


# ---------------------------------------------------------------------------
# A minimal MAT-5 writer. Level 5 is the only level vips__mat_ismat will
# accept, and writing it by hand is what lets this capture control the byte
# order and build the malformed files the refusal records need. Every field
# is spelled out rather than taken from scipy, so re-running needs nothing
# but the standard library.
# ---------------------------------------------------------------------------
miINT8, miUINT8, miINT16, miUINT16 = 1, 2, 3, 4
miINT32, miUINT32, miSINGLE, miDOUBLE = 5, 6, 7, 9
miINT64, miUINT64, miMATRIX, miCOMPRESSED = 12, 13, 14, 15

mxCELL, mxSTRUCT, mxOBJECT, mxCHAR, mxSPARSE = 1, 2, 3, 4, 5
mxDOUBLE, mxSINGLE, mxINT8, mxUINT8 = 6, 7, 8, 9
mxINT16, mxUINT16, mxINT32, mxUINT32 = 10, 11, 12, 13
mxINT64, mxUINT64 = 14, 15

# matio's MAT_C_* enum has the same numbering as the mx classes above, which
# is why the "unsupported class type N" messages quote these values back.
CLASS_NAME = {
    1: "MAT_C_CELL", 2: "MAT_C_STRUCT", 3: "MAT_C_OBJECT", 4: "MAT_C_CHAR",
    5: "MAT_C_SPARSE", 6: "MAT_C_DOUBLE", 7: "MAT_C_SINGLE",
    8: "MAT_C_INT8", 9: "MAT_C_UINT8", 10: "MAT_C_INT16",
    11: "MAT_C_UINT16", 12: "MAT_C_INT32", 13: "MAT_C_UINT32",
    14: "MAT_C_INT64", 15: "MAT_C_UINT64",
}


def elem(e, mtype, payload):
    """One MAT-5 data element: an 8-byte tag then the payload, padded out to
    the next 8-byte boundary."""
    out = struct.pack(e + "II", mtype, len(payload)) + payload
    return out + b"\x00" * ((-len(payload)) % 8)


def matrix(e, name, cls, mtype, dims, data, imag=None, logical=False):
    """One miMATRIX element: array flags, dimensions, name, real part, and
    optionally an imaginary part."""
    flags = (0x08 if imag is not None else 0) | (0x02 if logical else 0)
    body = elem(e, miUINT32, struct.pack(e + "II", (flags << 8) | cls, 0))
    body += elem(e, miINT32, b"".join(struct.pack(e + "i", d) for d in dims))
    body += elem(e, miINT8, name.encode())
    body += elem(e, mtype, data)
    if imag is not None:
        body += elem(e, mtype, imag)
    return elem(e, miMATRIX, body)


def matfile(name, elems, little=True,
            text=b"MATLAB 5.0 MAT-file, written by libviprs oracle capture"):
    """The 128-byte MAT-5 header then the elements. The endian indicator is
    the last two bytes: `IM` little, `MI` big."""
    e = "<" if little else ">"
    hdr = text.ljust(116, b" ")
    hdr += struct.pack(e + "Q", 0)            # subsystem data offset
    hdr += struct.pack(e + "H", 0x0100)       # version
    hdr += b"IM" if little else b"MI"
    path = os.path.join(FIX, name)
    with open(path, "wb") as f:
        f.write(hdr)
        for x in elems:
            f.write(x)
    return path


records = {}
notes = []

# ---------------------------------------------------------------------------
# 1. The base case: a 2x3 uint8 written column-major, read back transposed.
# ---------------------------------------------------------------------------
# Matlab column-major with dims (rows=2, cols=3): the file order is
# col0=(10,40) col1=(20,50) col2=(30,60), i.e. 10 40 20 50 30 60.
BASE = bytes([10, 40, 20, 50, 30, 60])
base = matfile("base_2x3_uint8.mat",
               [matrix("<", "a", mxUINT8, miUINT8, (2, 3), BASE)])
records["column_major_is_transposed"] = {
    "what": "mat2vips_get_header (matlab.c:190-205) reads height from "
            "dims[0] and width from dims[1], and mat2vips_get_data "
            "(matlab.c:276-300) walks the column-major buffer with a stride "
            "of es * Ysize to build each scanline. So a Matlab 2x3 (2 rows, "
            "3 columns) becomes a vips 3x2 image, and element (r, c) of the "
            "array is pixel (c, r) of the image. That transpose is the "
            "single easiest thing to get backwards in a port, which is why "
            "every value below is asymmetric.",
    "fixture": "fixtures/base_2x3_uint8.mat",
    "sha256": sha256(base),
    "bytes": os.path.getsize(base),
    "matlab_dims_rows_cols": [2, 3],
    "file_order_column_major": list(BASE),
    "header": header(base, all_fields=True),
    "pixels": pixels(base, 3, 2),
}
# The transpose rule, applied in Python to the input and CHECKED against the
# binary, rather than a raster order typed into the record by hand.
_rows, _cols = 2, 3
_measured = [int(px[0]) for px in records["column_major_is_transposed"]["pixels"]]
records["column_major_is_transposed"]["matches_transpose_of_input"] = (
    _measured == [BASE[c * _rows + r] for r in range(_rows) for c in range(_cols)])

# ---------------------------------------------------------------------------
# 2. Magic, and the gap between is_a and the loader.
# ---------------------------------------------------------------------------
magic = {}

# 2a. A genuine MAT-4 file: five int32s then the name then the data. matio
# reads this happily; vips__mat_ismat will never let it near matload.
mat4_body = (struct.pack("<5i", 0, 2, 3, 0, 2)   # type, mrows, ncols, imagf, namelen
             + b"a\x00"
             + struct.pack("<6d", 10.0, 40.0, 20.0, 50.0, 30.0, 60.0))
mat4 = os.path.join(FIX, "level4.mat")
with open(mat4, "wb") as f:
    f.write(mat4_body)
magic["mat_level_4"] = {
    "first_10_bytes": list(mat4_body[:10]),
    "sniffed": header(mat4),
    "direct": direct(mat4),
    "direct_pixels": pixels(os.path.join(OUT, "direct.v"), 3, 2),
    "note": "matio reads MAT-4 and gives the SAME 3x2 image, as double "
            "rather than uchar because MAT-4 stores everything as double. "
            "vips will never reach it through new_from_file: the sniff "
            "rejects the file and the chain falls through to magickload, "
            "which reports a TGA error.",
}

# 2b. A MAT-7.3 file, whose text header says 7.3 and whose body is HDF5.
mat73_body = (b"MATLAB 7.3 MAT-file, Platform: ORACLE, Created on: never"
              .ljust(116, b" ")
              + b"\x00" * 8 + struct.pack("<H", 0x0200) + b"IM"
              + b"\x89HDF\r\n\x1a\n" + b"\x00" * 64)
mat73 = os.path.join(FIX, "level73_hdf5.mat")
with open(mat73, "wb") as f:
    f.write(mat73_body)
magic["mat_level_7_3"] = {
    "first_10_bytes": mat73_body[:10].decode(),
    "sniffed": header(mat73),
    "direct": direct(mat73),
    "note": "the sniff says no twice over: the text header starts "
            "`MATLAB 7.3` rather than the literal `MATLAB 5.0`, and the "
            "version word is 0x0200 rather than 0x0100. A DIRECT matload "
            "gets as far as Mat_Open, which detects the HDF5 signature and "
            "hands the file to libhdf5; libhdf5 then prints a nine-line "
            "HDF5-DIAG block to stderr (stripped from the message below) "
            "before matio gives up. So matload's real-world failure on a "
            "modern `.mat` is noisy and comes from a library vips does not "
            "control. Since MATLAB has written 7.3 by default for large "
            "arrays since R2006b, this is the failure most users will "
            "actually hit.",
}

# 2c. The prefix matches and nothing else does.
magic_only = os.path.join(FIX, "magic_only.mat")
with open(magic_only, "wb") as f:
    f.write(b"MATLAB 5.0" + b"\xff" * 118)
magic["prefix_matches_rest_is_garbage"] = {
    "sniffed": header(magic_only),
    "direct": direct(magic_only),
    "note": "128 bytes and the right ten-byte prefix, and the sniff STILL "
            "refuses, because bytes 124..127 are not a valid "
            "version/endian-indicator pair. Compare "
            "`header_only_no_variables` below, which differs only in those "
            "four bytes and is accepted.",
}

# 2d. A 128-byte header and nothing after it: accepted by the sniff, then
# refused by the loader. This is the shape every untrusted loader has, a
# cheap sniff that says yes and a real parse that says no.
hdr_only = matfile("header_only.mat", [])
magic["header_only_no_variables"] = {
    "sniffed": header(hdr_only),
    "direct": direct(hdr_only),
    "note": "the sniff passes on the 128-byte header alone and the failure "
            "comes out of read_new's search loop. A port has to reproduce "
            "BOTH sides or it will disagree with vips about which loader "
            "owns a file.",
}

# 2d. Nine bytes: too short for the sniff to read.
short9 = os.path.join(FIX, "nine_bytes.mat")
with open(short9, "wb") as f:
    f.write(b"MATLAB 5.")
magic["nine_bytes"] = {
    "sniffed": header(short9),
    "direct": direct(short9),
    "note": "the shipped sniff asks for 128 bytes and requires all 128, so "
            "any file below that length is rejected before the prefix "
            "comparison runs. See sniff_predicate.length_floor.",
}

# 2e. The prefix is a PREFIX, not the whole line.
prefixed = matfile("prefix_only.mat",
                   [matrix("<", "a", mxUINT8, miUINT8, (2, 3), BASE)],
                   text=b"MATLAB 5.0 anything at all can follow here")
magic["prefix_then_arbitrary_text"] = {
    "text": "MATLAB 5.0 anything at all can follow here",
    "sniffed": header(prefixed),
    "note": "bytes 10..123 are free text as far as the sniff is concerned, "
            "so anything may follow the ten-byte prefix as long as the "
            "version and endian indicator at 124..127 are right.",
}

# 2f. One byte off.
#
# The filename is written out per case rather than slugged from the label,
# and that is not a style preference. `label.replace(' ', '_').replace('.', '')`
# maps "matlab 5.0" to `magic_matlab_50.mat` and "MATLAB_5.0" to
# `magic_MATLAB_50.mat`, which are the same path on the case-insensitive
# filesystem this capture is taken on. The third iteration reopened the second
# one's file and truncated it, so the tree ended up with two fixtures where
# this loop writes three, the survivor carried the underscore bytes under the
# lowercase name, and `magic_MATLAB_50.mat` was never committed at all. The
# records here were fine, because each `header(p)` runs before the next
# iteration overwrites anything, so the only casualty was what got persisted.
# Nothing noticed for two months: `include_bytes!` resolved the missing name
# through the same case-insensitive lookup, so the crate built on the capture
# host and failed to compile on every Linux CI runner (issue #977).
for label, name, text in (("MATLAB 5.1", "magic_MATLAB_51.mat", b"MATLAB 5.1 MAT-file"),
                          ("matlab 5.0", "magic_lowercase_50.mat", b"matlab 5.0 MAT-file"),
                          ("MATLAB_5.0", "magic_underscore_50.mat", b"MATLAB_5.0 MAT-file")):
    p = matfile(name, [matrix("<", "a", mxUINT8, miUINT8, (2, 3), BASE)], text=text)
    magic[label] = {"sniffed": header(p), "direct_exit": direct(p)["exit"]}

records["magic_and_dispatch"] = {
    "what": "The sniff is what decides whether a file reaches matload at "
            "all, and THE SHIPPED BINARY DOES NOT DO WHAT THE C SOURCE "
            "SAYS. Worse, it changed under this capture: 8.18.4 agreed with "
            "the source and the 8.18.6 that replaced it does not. See the "
            "sniff_predicate record below for the "
            "measured rule and the disassembly it came from. The loader "
            "underneath is matio's Mat_Open, which is more permissive than "
            "the sniff in one direction (it reads MAT-4) and less in "
            "another (it hands MAT-7.3 to libhdf5 and fails noisily), and "
            "the cases where the two disagree are the ones a port has to "
            "decide about deliberately.",
    "registered_suffixes": [".mat"],
    "priority": 0,
    "cases": magic,
}
notes.append(
    "matload is priority 0 and untrusted (matload.c:120-125). Unlike "
    "analyzeload it does not lower its priority, because the sniff is a "
    "fixed 128-byte read off the front of the file and costs nothing."
)

# ---------------------------------------------------------------------------
# 2z. The sniff predicate, measured rather than transcribed, because the
#     source in the reference checkout does not describe the binary.
# ---------------------------------------------------------------------------
def sniff(name, body):
    """Does this byte string reach matload through vips_foreign_find_load?
    `mat2vips:` in the error means yes, matload was selected and then failed
    on its own terms; anything else means the sniff refused."""
    p = os.path.join(FIX, name)
    with open(p, "wb") as f:
        f.write(body)
    proc = run([VIPSHEADER, p])
    return {"routed_to_matload": "mat2vips" in proc.stderr,
            "exit": proc.returncode,
            "message": clean(proc.stderr).splitlines()[0] if proc.stderr.strip()
            else proc.stdout.strip().replace(ROOT + "/", "")}


def hdr128(text=b"MATLAB 5.0 MAT-file", ver=b"\x00\x01", ind=b"IM"):
    return text.ljust(116, b" ") + b"\x00" * 8 + ver + ind


# Which byte positions matter: corrupt exactly one byte of a good header at
# a time and see whether the file still reaches matload.
sensitive = []
for i in range(128):
    b = bytearray(hdr128())
    b[i] = 0xFF
    if not sniff("probe_byte.mat", bytes(b))["routed_to_matload"]:
        sensitive.append(i)
os.remove(os.path.join(FIX, "probe_byte.mat"))

# The version / endian-indicator pairs.
pairs = {}
for ver, ind in ((b"\x00\x01", b"IM"), (b"\x01\x00", b"IM"),
                 (b"\x01\x00", b"MI"), (b"\x00\x01", b"MI"),
                 (b"\x00\x02", b"IM"), (b"\x02\x00", b"MI"),
                 (b"\x00\x01", b"im"), (b"\x00\x01", b"MM"),
                 (b"\x00\x01", b"\x00\x00")):
    label = f"{ver.hex()}/{ind.decode('latin1')!r}"
    pairs[label] = sniff("probe_pair.mat",
                         hdr128(ver=ver, ind=ind))["routed_to_matload"]
os.remove(os.path.join(FIX, "probe_pair.mat"))

# The length floor.
lengths = {}
for n in (0, 9, 10, 64, 127, 128, 129, 200):
    g = hdr128()
    lengths[str(n)] = sniff("probe_len.mat",
                            g[:n] if n <= 128
                            else g + b"\x00" * (n - 128))["routed_to_matload"]
os.remove(os.path.join(FIX, "probe_len.mat"))

records["sniff_predicate"] = {
    "what": "THE C SOURCE IN THE REFERENCE CHECKOUT IS NOT WHAT RUNS, AND "
            "THIS PARTICULAR RULE CHANGED UNDER US MID-CAPTURE. "
            "vips__mat_ismat at matlab.c:326-336 of "
            "v8.18.0-95-gfe420cf3a reads TEN bytes and compares them to "
            "`MATLAB 5.0`, and nothing else. The installed 8.18.6 dylib "
            "reads ONE HUNDRED AND TWENTY-EIGHT and also validates the "
            "version word and the endian indicator, and 8.18.4, which was "
            "the installed build when this capture started, did NOT. See "
            "measured_change_between_8_18_4_and_8_18_6 below. A port "
            "written from the checkout source would accept a large class "
            "of files 8.18.6 refuses, and would route them to itself "
            "instead of leaving them to the next loader. Everything here "
            "was measured from the binary and then confirmed against its "
            "disassembly.",
    "measured_rule": [
        "vips__get_bytes(filename, buf, 128) must return exactly 128, so a "
        "file shorter than 128 bytes is refused outright",
        "vips_isprefix(\"MATLAB 5.0\", buf), i.e. bytes 0..9 are that exact "
        "case-sensitive string",
        "bytes 126..127 are the endian indicator and must be `IM` or `MI`; "
        "anything else is refused",
        "bytes 124..125 are the version, read in the order the indicator "
        "declares, and must equal 0x0100; 0x0200 (which is what a MAT-7.3 "
        "file carries) is refused",
        "bytes 10..123, the free-text description and the subsystem data "
        "offset, are not looked at at all",
    ],
    "measured_change_between_8_18_4_and_8_18_6": {
        "what": "Homebrew replaced libvips 8.18.4 with 8.18.6 on this "
                "machine part-way through this capture. The same 210-byte "
                "file, `MATLAB 5.0` followed by 200 bytes of 0xff, was "
                "measured under both.",
        "under_8_18_4": "vipsheader printed "
                        "`mat2vips: unable to open \"magic_only.mat\"`, "
                        "i.e. the sniff ACCEPTED it, matload was selected, "
                        "and Mat_Open then refused it",
        "under_8_18_6": "vipsheader prints "
                        "`VipsForeignLoad: \"...\" is not a known file "
                        "format`, i.e. the sniff REFUSED it and no loader "
                        "was selected at all",
        "caveat": "8.18.4 is no longer installed (the Cellar holds only "
                  "8.18.6) so capture.py cannot re-derive the first half. "
                  "It is recorded here as an observation, not as a "
                  "reproducible measurement, and the 8.18.6 half is what "
                  "the rest of this record pins.",
        "why_it_matters": "the runbook pins this epic's oracle at 8.18.4. "
                          "Any capture taken before the upgrade and any "
                          "capture taken after it came from different "
                          "binaries, and at least this one predicate "
                          "differs between them.",
    },
    "disassembly": {
        "command": "otool -tvV -p _vips__mat_ismat "
                   "/opt/homebrew/lib/libvips.42.dylib   # 8.18.6",
        "shape": [
            "bl _vips__get_bytes with w2 = #0x80",
            "cmp x0, #0x80 / b.lt <fail>",
            "bl _vips_isprefix on \"MATLAB 5.0\" / cbz <fail>",
            "ldrb w11, [buf+126] ; ldrb w10, [buf+127]",
            "cmp w11, #0x49 ('I') && cmp w10, #0x4d ('M') -> lo=buf[124], "
            "hi=buf[125]",
            "else cmp w11, #0x4d ('M') && cmp w10, #0x49 ('I') -> lo=buf[125], "
            "hi=buf[124]",
            "else return 0",
            "orr w8, lo, hi, lsl #8 ; cmp w8, #0x100 ; b.ne <fail>",
            "return 1",
        ],
    },
    "byte_positions_that_change_the_answer": sensitive,
    "version_and_indicator_pairs": pairs,
    "length_floor": lengths,
}
notes.append(
    "ORACLE BINARY MOVED. Homebrew replaced libvips 8.18.4 with 8.18.6 on "
    "this machine while this capture was being taken, so every number in "
    "this file came from 8.18.6 even though the epic runbook pins the "
    "oracle at 8.18.4. Only 8.18.6 is in the Cellar now, so 8.18.4 cannot "
    "be re-measured here."
)
notes.append(
    "The MAT sniff in 8.18.6 is STRICTER than both 8.18.4 and "
    "vips__mat_ismat in the v8.18.0-95-gfe420cf3a source tree: 128 bytes "
    "read rather than 10, plus a version and endian-indicator check. "
    "Follow the binary. The sniff_predicate record has the measured rule, "
    "the before/after and the disassembly. vips__isanalyze, disassembled "
    "the same way, DOES match its source."
)

# ---------------------------------------------------------------------------
# 3. Rank: 1, 2 and 3 are loadable, everything else is skipped.
# ---------------------------------------------------------------------------
rank = {}

r1 = matfile("rank1.mat",
             [matrix("<", "a", mxUINT8, miUINT8, (4,),
                     bytes([11, 22, 33, 44]))])
rank["1"] = {"dims": [4], "header": header(r1), "pixels": pixels(r1, 1, 4),
             "note": "width stays at the 1 mat2vips_get_header initialised "
                     "it to (matlab.c:188), height is dims[0]. A rank-1 "
                     "variable is a 1-pixel-wide column. MATLAB itself "
                     "never writes rank 1 (it pads everything to at least "
                     "2 dimensions), so this branch is reachable only from "
                     "a hand-built or non-MATLAB writer, which is exactly "
                     "the case an untrusted loader has to survive."}

rank["2"] = {"dims": [2, 3], "header": header(base),
             "pixels": records["column_major_is_transposed"]["pixels"]}

# Plane-separate in the file: all of band 0, then all of band 1, then band 2.
R3 = bytes([1, 2, 3, 4, 5, 6,
            11, 12, 13, 14, 15, 16,
            21, 22, 23, 24, 25, 26])
r3 = matfile("rank3_2x3x3.mat",
             [matrix("<", "a", mxUINT8, miUINT8, (2, 3, 3), R3)])
rank["3"] = {
    "dims": [2, 3, 3],
    "file_order": list(R3),
    "header": header(r3, all_fields=True),
    "pixels": pixels(r3, 3, 2),
    "note": "dims[2] becomes the BAND count and the file holds the planes "
            "one after another, each one a full column-major 2x3. "
            "mat2vips_get_data reassembles them interleaved with an offset "
            "of b * es * N_PELS (matlab.c:286). The comment at "
            "matlab.c:258-259 calls this out: Matlab images are "
            "plane-separate and vips images are interleaved.",
}

r4 = matfile("rank4_only.mat",
             [matrix("<", "a", mxUINT8, miUINT8, (2, 2, 2, 2), bytes(range(16)))])
rank["4_alone"] = {
    "dims": [2, 2, 2, 2],
    "sniffed": header(r4),
    "direct": direct(r4),
    "note": "read_new (matlab.c:118-140) loops over Mat_VarReadNextInfo and "
            "FREES any variable outside rank 1..=3, then keeps looking. "
            "With nothing else in the file it runs off the end and reports "
            "`no matrix variables`, which is a slightly misleading message "
            "for a file that does contain a matrix.",
}

rank_then = matfile("rank4_then_rank2.mat",
                    [matrix("<", "big", mxUINT8, miUINT8, (2, 2, 2, 2),
                            bytes(range(16))),
                     matrix("<", "small", mxUINT8, miUINT8, (2, 2),
                            bytes([7, 8, 9, 10]))])
rank["4_then_2"] = {
    "header": header(rank_then),
    "pixels": pixels(rank_then, 2, 2),
    "note": "the rank-4 variable is skipped and the rank-2 one loads. Note "
            "that the SECOND variable's name is what ends up in the image "
            "and there is no way to ask for a particular variable: matload "
            "has exactly one argument, the filename.",
}

records["rank_and_variable_selection"] = {
    "what": "matload loads ONE variable and there is no option to choose "
            "which. read_new takes the first with rank in 1..=3 and every "
            "other variable in the file is invisible. The gtkdoc at "
            "matload.c:157-159 says `the first array variable with between "
            "1 and 3 dimensions`, and matlab.c:49-50 has an open question "
            "in the source asking whether that is sensible behaviour. It is "
            "the behaviour, so a port has to match it.",
    "cases": rank,
}

# ---------------------------------------------------------------------------
# 4. The class -> carrier table, and how the search interacts with it.
# ---------------------------------------------------------------------------
SUPPORTED = [
    (mxUINT8, miUINT8, "B", [1, 2, 3, 4]),
    (mxINT8, miINT8, "b", [-1, 2, -3, 4]),
    (mxUINT16, miUINT16, "H", [1, 500, 3, 65535]),
    (mxINT16, miINT16, "h", [-1, 500, -3, 32767]),
    (mxUINT32, miUINT32, "I", [1, 70000, 3, 4294967295]),
    (mxINT32, miINT32, "i", [-1, 70000, -3, 2147483647]),
    (mxSINGLE, miSINGLE, "f", [1.5, -2.25, 3.125, 4.0]),
    (mxDOUBLE, miDOUBLE, "d", [1.5, -2.25, 3.125, 4.0]),
]
carriers = {}
for cls, mtype, fmt, vals in SUPPORTED:
    data = b"".join(struct.pack("<" + fmt, v) for v in vals)
    p = matfile(f"class_{CLASS_NAME[cls].lower()}.mat",
                [matrix("<", "a", cls, mtype, (2, 2), data)])
    carriers[CLASS_NAME[cls]] = {
        "class_type": cls,
        "values_column_major": vals,
        "header": header(p),
        "pixels": pixels(p, 2, 2),
    }

# The three-band spellings, which are where the interpretation tag moves.
for cls, mtype, fmt, one in ((mxUINT8, miUINT8, "B", 1),
                             (mxUINT16, miUINT16, "H", 1000),
                             (mxINT16, miINT16, "h", -1000),
                             (mxSINGLE, miSINGLE, "f", 1.0),
                             (mxDOUBLE, miDOUBLE, "d", 1.0),
                             (mxINT8, miINT8, "b", 1),
                             (mxUINT32, miUINT32, "I", 1),
                             (mxINT32, miINT32, "i", 1)):
    data = b"".join(struct.pack("<" + fmt, one) for _ in range(4 * 3))
    p = matfile(f"bands3_{CLASS_NAME[cls].lower()}.mat",
                [matrix("<", "a", cls, mtype, (2, 2, 3), data)])
    carriers[CLASS_NAME[cls]]["three_band_header"] = header(p)["summary"]

unsupported = {}
for cls, mtype, fmt, n in ((mxINT64, miINT64, "q", 4),
                           (mxUINT64, miUINT64, "Q", 4),
                           (mxCHAR, miUINT16, "H", 4)):
    data = b"".join(struct.pack("<" + fmt, i + 1) for i in range(n))
    p = matfile(f"class_{CLASS_NAME[cls].lower()}.mat",
                [matrix("<", "a", cls, mtype, (2, 2), data)])
    unsupported[CLASS_NAME[cls]] = {"class_type": cls,
                                    "sniffed": header(p),
                                    "direct": direct(p)}

# A sparse array, which needs ir/jc elements rather than a plain data element.
e = "<"
sp_body = (elem(e, miUINT32, struct.pack(e + "II", mxSPARSE, 4))
           + elem(e, miINT32, struct.pack(e + "ii", 2, 2))
           + elem(e, miINT8, b"a")
           + elem(e, miINT32, struct.pack(e + "ii", 0, 1))      # ir
           + elem(e, miINT32, struct.pack(e + "iii", 0, 1, 2))  # jc
           + elem(e, miDOUBLE, struct.pack(e + "dd", 1.0, 2.0)))
sp = matfile("class_mat_c_sparse.mat", [elem(e, miMATRIX, sp_body)])
unsupported["MAT_C_SPARSE"] = {"class_type": mxSPARSE,
                               "sniffed": header(sp), "direct": direct(sp)}

# The logical flag on a uint8 array, which vips does not look at.
logical = matfile("logical_uint8.mat",
                  [matrix("<", "a", mxUINT8, miUINT8, (2, 2),
                          bytes([0, 1, 1, 0]), logical=True)])

# An unsupported class at the FIRST rank-2 variable, with a good one behind
# it. This is the record that proves the search filters on rank only.
i64_first = matfile("int64_then_uint8.mat",
                    [matrix("<", "bad", mxINT64, miINT64, (2, 2),
                            struct.pack("<4q", 1, 2, 3, 4)),
                     matrix("<", "good", mxUINT8, miUINT8, (2, 2),
                            bytes([1, 2, 3, 4]))])

records["class_to_carrier"] = {
    "what": "mat2vips_formats (matlab.c:147-156) is an eight-entry table "
            "and mat2vips_get_header refuses anything not in it with "
            "`unsupported class type %d`. The band format follows the "
            "Matlab class alone; the ARRAY FLAGS word is read only for the "
            "class byte, so the logical bit is ignored and a logical array "
            "loads as whatever its storage class is.",
    "supported": carriers,
    "unsupported": unsupported,
    "logical_flag_is_ignored": {
        "fixture": "fixtures/logical_uint8.mat",
        "flags_word_bit": "0x02 (logical), set",
        "header": header(logical),
        "pixels": pixels(logical, 2, 2),
    },
    "rank_filter_runs_before_the_class_check": {
        "what": "read_new's loop breaks out on the first variable with rank "
                "1..=3 and the class check does not happen until "
                "mat2vips_get_header, by which point the loop is over. So a "
                "file whose first rank-2 variable is int64 FAILS OUTRIGHT "
                "even though a perfectly loadable uint8 variable follows "
                "it. Contrast the rank_4_then_2 case, where the skip does "
                "work. A port that filters on class inside the search loop "
                "would load this file and disagree with vips.",
        "fixture": "fixtures/int64_then_uint8.mat",
        "variables": ["bad: 2x2 int64", "good: 2x2 uint8"],
        "sniffed": header(i64_first),
        "direct": direct(i64_first),
    },
}

# ---------------------------------------------------------------------------
# 5. The interpretation table, stated as measured.
# ---------------------------------------------------------------------------
records["interpretation"] = {
    "what": "mat2vips_pick_interpretation (matlab.c:160-178) is four ifs and "
            "a default. Read them off the headers in class_to_carrier "
            "rather than from the source, because the last two branches are "
            "unreachable: `if (bands > 1) return MULTIBAND` and the final "
            "`return MULTIBAND` do the same thing. The consequence worth "
            "pinning is the ONE-BAND UCHAR case: it gets MULTIBAND, not "
            "b-w, so a greyscale uint8 array loads untagged. Analyze, the "
            "other loader in this pair, tags the same shape b-w.",
    "rules": {
        "bands == 3 and 8-bit (uchar or char)": "sRGB",
        "bands == 3 and ushort or short": "RGB16",
        "bands == 1 and ushort or short": "GREY16",
        "everything else, including 1-band uchar": "MULTIBAND",
    },
    "note": "vips_band_format_is8bit is true for both UCHAR and CHAR, so a "
            "3-band int8 array is tagged sRGB even though half its range is "
            "negative.",
}

# ---------------------------------------------------------------------------
# 6. Byte order, which the file declares and matio honours.
# ---------------------------------------------------------------------------
SHORTS = (1, 256, -1, 4660)
le = matfile("endian_little.mat",
             [matrix("<", "a", mxINT16, miINT16, (2, 2),
                     struct.pack("<4h", *SHORTS))])
be = matfile("endian_big.mat",
             [matrix(">", "a", mxINT16, miINT16, (2, 2),
                     struct.pack(">4h", *SHORTS))],
             little=False)
bogus = matfile("endian_bogus.mat",
                [matrix("<", "a", mxINT16, miINT16, (2, 2),
                        struct.pack("<4h", *SHORTS))])
with open(bogus, "r+b") as f:
    f.seek(126)
    f.write(b"XY")
records["byte_order"] = {
    "what": "Unlike Analyze, MAT-5 declares its own byte order: the last "
            "two bytes of the 128-byte header are `IM` for little-endian "
            "and `MI` for big-endian, and every multi-byte field in the "
            "file follows. vips does not touch any of this; matio swaps "
            "where it needs to and hands vips native-order data. So there "
            "is no byteswap in matlab.c at all, and a port must delegate "
            "byte order to its MAT reader rather than applying one itself. "
            "The two files below hold the same four shorts in opposite "
            "orders and produce identical pixels.",
    "shorts": list(SHORTS),
    "little_endian_IM": {"fixture": "fixtures/endian_little.mat",
                         "sha256": sha256(le),
                         "header": header(le), "pixels": pixels(le, 2, 2)},
    "big_endian_MI": {"fixture": "fixtures/endian_big.mat",
                      "sha256": sha256(be),
                      "header": header(be), "pixels": pixels(be, 2, 2)},
    "bogus_indicator_XY": {"fixture": "fixtures/endian_bogus.mat",
                           "sniffed": header(bogus),
                           "direct": direct(bogus),
                           "note": "an otherwise-valid little-endian file "
                                   "with the two indicator bytes replaced. "
                                   "The sniff REFUSES it, and the chain "
                                   "falls through to magickload, which has "
                                   "a MAT reader of its own and reports "
                                   "from coders/mat.c rather than from "
                                   "vips. A direct matload gets to Mat_Open "
                                   "and fails there."},
}

# ---------------------------------------------------------------------------
# 7. Compression, which MATLAB turns on by default.
# ---------------------------------------------------------------------------
inner = matrix("<", "a", mxUINT8, miUINT8, (2, 3), BASE)
comp = matfile("compressed.mat", [elem("<", miCOMPRESSED, zlib.compress(inner))])
bad_comp = matfile("compressed_corrupt.mat",
                   [elem("<", miCOMPRESSED,
                         zlib.compress(inner)[:8] + b"\x00" * 16)])
records["compressed_elements"] = {
    "what": "Every MATLAB release since 7 writes each variable inside a "
            "miCOMPRESSED (type 15) zlib stream by default, so a real-world "
            "`.mat` is compressed and a port that only reads bare elements "
            "will fail on almost every file it meets. matio inflates them "
            "transparently and vips never sees the difference: the two "
            "records here are the same 2x3 array as the base case.",
    "valid": {"fixture": "fixtures/compressed.mat",
              "bytes": os.path.getsize(comp),
              "header": header(comp), "pixels": pixels(comp, 3, 2)},
    "corrupt_stream": {"fixture": "fixtures/compressed_corrupt.mat",
                       "sniffed": header(bad_comp),
                       "direct": direct(bad_comp)},
}

# ---------------------------------------------------------------------------
# 8. Refusals, and exactly where the load gives up.
# ---------------------------------------------------------------------------
refusals = {}

refusals["no_variables_at_all"] = {
    "fixture": "fixtures/header_only.mat",
    "sniffed": header(hdr_only),
    "direct": direct(hdr_only),
    "note": "a bare 128-byte header. read_new runs Mat_VarReadNextInfo "
            "once, gets NULL, and reports `no matrix variables`.",
}

# A tag claiming a megabyte in a 152-byte file.
big_tag = os.path.join(FIX, "tag_overruns_file.mat")
with open(base, "rb") as f:
    header_bytes = f.read(128)
with open(big_tag, "wb") as f:
    f.write(header_bytes + struct.pack("<II", miMATRIX, 1 << 20) + b"\x00" * 16)
refusals["element_length_overruns_the_file"] = {
    "declared_element_bytes": 1 << 20,
    "actual_bytes_after_tag": 16,
    "sniffed": header(big_tag),
    "direct": direct(big_tag),
}

# A file cut off in the middle of its data element.
trunc = os.path.join(FIX, "truncated.mat")
with open(base, "rb") as f:
    whole = f.read()
with open(trunc, "wb") as f:
    f.write(whole[:-8])
refusals["truncated_mid_data_element"] = {
    "bytes_removed": 8,
    "sniffed": header(trunc),
    "avg": stat(trunc),
    "direct": direct(trunc),
    "give_up_point": "the HEADER load succeeds and reports the full 3x2. "
                     "mat2vips_get_header reads only the dims and class out "
                     "of Mat_VarReadNextInfo, which is an info-only read; "
                     "nothing checks that the data element is present until "
                     "Mat_VarReadDataAll runs in mat2vips_get_data.",
}

# Dimensions the file cannot possibly satisfy.
short_data = matfile("dims_100x100_four_bytes.mat",
                     [matrix("<", "a", mxUINT8, miUINT8, (100, 100),
                             b"\x01\x02\x03\x04")])
refusals["dims_100x100_with_4_bytes_of_data"] = {
    "declared_pixels": 10000,
    "data_bytes": 4,
    "sniffed": header(short_data),
    "avg": stat(short_data),
    "direct": direct(short_data),
}

huge = matfile("dims_100000x100000.mat",
               [matrix("<", "a", mxUINT8, miUINT8, (100000, 100000),
                       b"\x00" * 8)])
refusals["dims_100000x100000"] = {
    "declared_pixels": 100000 * 100000,
    "data_bytes": 8,
    "sniffed": header(huge),
    "direct": direct(huge),
    "note": "the header load reports a 10-gigapixel image without "
            "complaint. matio refuses the allocation when the data is "
            "actually asked for, so the process does not try to reserve "
            "10 GB, but nothing in vips imposed that limit.",
}

# Zero and negative dimensions.
zero = matfile("dim_zero.mat",
               [matrix("<", "a", mxUINT8, miUINT8, (0, 3), b"")])
negative = matfile("dim_negative.mat",
                   [matrix("<", "a", mxUINT8, miUINT8, (-2, 3),
                           bytes([1, 2, 3, 4, 5, 6]))])
refusals["zero_and_negative_dimensions_are_clamped_not_refused"] = {
    "what": "Same defect as analyzeload's: a non-positive dimension reaches "
            "vips_image_init_fields, GObject's range check rejects it, "
            "prints a GLib-GObject-CRITICAL, LEAVES THE PROPERTY AT 1, and "
            "the load carries on and exits 0. A 0x3 array therefore loads "
            "as a 3x1 image reading data that is not there. A port should "
            "refuse, and should record that as a deliberate divergence.",
    "dim0_zero": {"fixture": "fixtures/dim_zero.mat",
                  "sniffed": header(zero), "direct": direct(zero)},
    "dim0_minus_2": {"fixture": "fixtures/dim_negative.mat",
                     "sniffed": header(negative), "direct": direct(negative)},
    "pixels_are_not_pinned": "the clamped images read past a zero-length or "
                             "under-length allocation, so their values are "
                             "not stable and are deliberately not recorded "
                             "as expectations.",
}

records["refusals"] = {
    "what": "matload is registered UNTRUSTED (matload.c:120-123) with the "
            "comment `libmatio is fuzzed, but not by us`, so this is the "
            "half of its behaviour that matters most. vips itself has only "
            "three refusals, all in matlab.c: `unable to open` (Mat_Open "
            "failed, line 112), `no matrix variables` (the search ran off "
            "the end, line 120) and `unsupported class type` (line 211). "
            "Everything else comes out of matio, arrives as "
            "`Mat_VarReadDataAll failed` (line 264) with no detail, and "
            "arrives LATE, at first pixel rather than at header time.",
    "cases": refusals,
}

# ---------------------------------------------------------------------------
# 9. Complex arrays are not refused. They should be.
# ---------------------------------------------------------------------------
cplx = matfile("complex_double.mat",
               [matrix("<", "a", mxDOUBLE, miDOUBLE, (2, 2),
                       struct.pack("<4d", 1.0, 2.0, 3.0, 4.0),
                       imag=struct.pack("<4d", 9.0, 8.0, 7.0, 6.0))])
cplx_runs = []
for _ in range(3):
    proc = vips("getpoint", cplx, "0", "0")
    cplx_runs.append({"exit": proc.returncode,
                      "stdout": proc.stdout.strip(),
                      "stderr": clean(proc.stderr)})
records["complex_arrays_are_not_refused"] = {
    "what": "matload.c:158 says `It will not handle complex images` and "
            "matlab.c:45 lists it as a remaining issue, but there is no "
            "check anywhere. The array flags' complex bit is never read: "
            "mat2vips_get_header maps MAT_C_DOUBLE to VIPS_FORMAT_DOUBLE "
            "and mat2vips_get_data memcpys out of var->data, which for a "
            "complex variable is a mat_complex_split_t holding two "
            "POINTERS. So the pixels are the raw bytes of two heap "
            "addresses. The load exits 0 and the values change from run to "
            "run under ASLR, which is why they are recorded below as "
            "evidence rather than pinned as an expectation. A port must "
            "refuse complex arrays outright, and should say so.",
    "fixture": "fixtures/complex_double.mat",
    "real_part": [1.0, 2.0, 3.0, 4.0],
    "imaginary_part": [9.0, 8.0, 7.0, 6.0],
    "header": header(cplx),
    "getpoint_0_0_three_separate_runs": cplx_runs,
    "values_are_stable_across_runs": len({r["stdout"] for r in cplx_runs}) == 1,
    "expected_if_it_worked": "a 2x2 VIPS_FORMAT_DPCOMPLEX image, or a "
                             "refusal. It produces neither.",
}
notes.append(
    "The complex case is the sharpest reason to treat this loader as "
    "untrusted: it reads heap pointers out as pixel data, exits 0, and the "
    "documentation says the case cannot happen."
)
notes.append(
    "There is NO matsave. The format is load-only, so there is no round "
    "trip to pin and no save behaviour to match."
)

# ---------------------------------------------------------------------------
version = run([VIPS, "--version"]).stdout.strip()
listing = run([VIPS, "-l"])
op_line = [ln.strip() for ln in listing.stdout.splitlines() if "matload" in ln]
config = run([VIPS, "--vips-config"])
config_line = [ln.strip() for ln in config.stdout.replace(",", "\n").splitlines()
               if "atlab" in ln or "matio" in ln]

fixture_bytes = sum(os.path.getsize(os.path.join(FIX, f))
                    for f in os.listdir(FIX))
oracle = {
    "meta": {
        "area": "foreign-mat",
        "issue": 640,
        "parent_issue": 510,
        "vips_version": version,
        "vips_binary": VIPS,
        "operation": "matload",
        "operation_listing": op_line[0] if op_line else None,
        "vips_config": config_line,
        "save_operation": None,
        "captured_by": "oracle-captures/foreign-mat/capture.py",
        "reference_c": "libvips v8.18.0-95-gfe420cf3a for the file and line "
                       "numbers quoted here. The binary every measurement "
                       "came out of is the installed release named in "
                       "vips_version above and is a DIFFERENT ARTEFACT "
                       "from that tree, which is not a formality: see the "
                       "oracle_binary_moved note.",
        "underlying_library": "matio, via Mat_Open / Mat_VarReadNextInfo / "
                              "Mat_VarReadDataAll. Every error message that "
                              "is not one of matlab.c's own three comes "
                              "from there.",
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
            "matio": "libmatio 1.5.30_1, from "
                     "`otool -L /opt/homebrew/lib/libvips.42.dylib` "
                     "(libmatio.14.dylib) resolved through "
                     "/opt/homebrew/opt/libmatio. Not named by "
                     "--vips-config, which prints booleans only.",
            "hdf5": "hdf5 2.2.0, which is what prints the HDF5-DIAG block "
                    "when matio is handed a MAT-7.3 file",
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
