#!/usr/bin/env python3
"""
Oracle capture for the AVIF still-image area (issue #638, under #605).

Runs the real vips CLI over small, deterministic rasters and records what
`heifload` hands back for an AVIF still, plus the ISOBMFF structure of the
file it came out of. #605 is a LOAD-ONLY port, so `heifsave` appears here
only as the thing that manufactured the fixtures: every record that matters
is a statement about the decode side.

Scope is AVIF stills. Not `heifload` parity: `heifload` also accepts HEVC,
AVC and JPEG payloads inside an HEIF container, and `heifsave` writes HEVC by
default. That decision is #605's and this capture does not widen it. Where a
`.heic` would behave differently the difference is written down in `notes`
and NOT captured as a target.

What cannot be derived from the AV1 or HEIF specifications, and so is here:

  * the CARRIER libvips picks per depth. 8-bit comes back `uchar`/`srgb`;
    10-bit and 12-bit come back `ushort`/`rgb16` with the sample LEFT
    JUSTIFIED into 16 bits by `<< (16 - bits_per_pixel)`, so a 10-bit file
    tops out at 65472 and its low 6 bits are always zero. Every header check
    passes if you get this wrong and every pixel is wrong.
  * that the interpretation and the band format are decided by
    `bits_per_pixel > 8` alone, nothing else
  * that a greyscale AVIF still comes back as 3 bands, because heifload
    always decodes to RGB
  * that alpha is a SEPARATE, monochrome AV1 item joined to the colour item
    by an `auxl` reference, and that it goes through the same left-justify
  * that `heifload` throws the nclx colour box away and only lifts an ICC
    profile, so a tagged and an untagged file differ in exactly one field
  * which container boxes a port has to walk to get any of that
  * where `is_a` sniffing draws the line, since a file it rejects does not
    fail, it silently goes to magickload instead

The libvips C line numbers quoted below come from the source tree at the
commit in `meta.reference_c`. Every NUMBER here came out of the binary, which
is a release build and a different artefact from that tree.

Writes:
  commands.sh  - every vips CLI command actually executed, in order
  oracle.json  - structured records
  fixtures/    - the .avif files a port would embed verbatim

Re-running needs only the vips binary at VIPS; every input is generated from
scratch, deterministically, including the ICC profile, which is libvips' own
built-in `srgb` rather than a file off this machine. Nothing outside this
script's own directory is written.
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

# Every command goes through `env -u VIPS_NOVECTOR`. Nothing in the heif
# codec has a vector path to disable -- the left-justify loop at
# heifload.c:1000-1016 is plain scalar C -- but an INHERITED, EMPTY
# VIPS_NOVECTOR still counts as set as far as libvips is concerned, so
# unsetting it costs nothing and removes the question.
ENV_PREFIX = ["env", "-u", "VIPS_NOVECTOR"]

# commands.sh abbreviates the two binaries, which are otherwise 40 of the 65
# characters on every one of its lines. It defines both at the top, so it
# stays a runnable transcript.
SHELL_ALIASES = {
    " ".join(ENV_PREFIX + [VIPS]): "$VIPS",
    " ".join(ENV_PREFIX + [VIPSHEADER]): "$VIPSHEADER",
}


# `(vips:8819): VIPS-WARNING **: 08:41:28.888: msg` -> `VIPS-WARNING **: msg`.
# The pid and the wall clock are the only two things in this capture that
# change between runs, so stripping them makes `python3 capture.py` a byte
# for byte no-op on an unchanged machine, which is what lets anyone else
# check the capture rather than take it on trust.
GLIB_PREFIX = re.compile(r"^\([^)]*:\d+\): (\S+ \*\*): "
                         r"\d\d:\d\d:\d\d\.\d+: ", re.M)


def scrub(text):
    """Drop this directory's absolute path, and the pid and timestamp glib
    stamps onto a warning, so a record is portable and reproducible."""
    text = text.replace(ROOT + "/", "").replace(ROOT, "")
    return GLIB_PREFIX.sub(r"\1: ", text)


def run(args, allow_fail=False):
    """Run a command, logging it for commands.sh."""
    full = ENV_PREFIX + args
    line = " ".join(scrub(a) for a in full)
    for long, short in SHELL_ALIASES.items():
        if line.startswith(long + " "):
            line = short + line[len(long):]
            break
    COMMANDS.append(line)
    proc = subprocess.run(full, capture_output=True, text=True)
    if proc.returncode != 0 and not allow_fail:
        raise SystemExit(f"failed: {' '.join(args)}\n{proc.stderr}")
    return proc


def note(text):
    COMMANDS.append("# " + text)


def vips(*args, allow_fail=False):
    return run([VIPS, *args], allow_fail=allow_fail)


def header(path, all_fields=False, allow_fail=False):
    args = [VIPSHEADER] + (["-a"] if all_fields else []) + [path]
    proc = run(args, allow_fail=True)
    if proc.returncode != 0:
        if not allow_fail:
            raise SystemExit(f"vipsheader failed: {path}\n{proc.stderr}")
        return {"error": scrub(proc.stderr.strip())}
    if not all_fields:
        return {"summary": scrub(proc.stdout.strip())}
    out = {}
    for line in proc.stdout.splitlines()[1:]:
        if ": " in line:
            name, value = line.split(": ", 1)
            out[name.strip()] = scrub(value.strip())
    return out


def field(path, name):
    proc = run([VIPSHEADER, "-f", name, path], allow_fail=True)
    if proc.returncode != 0:
        return None
    return proc.stdout.strip()


def getpoint(path, w, h):
    """Every pixel of a small image, in raster order, as ints. AVIF is
    VIPS_CODING_NONE, so getpoint reports exactly the decoded band count and
    the coded-band overrun that bites the Radiance capture cannot happen."""
    out = []
    for y in range(h):
        for x in range(w):
            txt = vips("getpoint", path, str(x), str(y)).stdout.split()
            out.append([int(float(v)) for v in txt])
    return out


def flat(px):
    return [v for pixel in px for v in pixel]


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


# Every path under fixtures/ this run has already written, and what wrote it.
CLAIMED = {}


def fix_path(name, why):
    """Reserve `fixtures/<name>` for one writer, and refuse a second.

    Issue #779 is what happens without this. Two records wrote DIFFERENT
    images to `fixtures/rgb8.avif`: the bit-depth carrier saved the 16-bit
    ramp narrowed to 8 bits, and the lossless-identity record then saved the
    8-bit ramp straight over the top of it. The later write won, so the
    carrier's row kept the sha256 and the byte count of a file that no longer
    existed anywhere, and its `read_back` and `source_16bit` arrays described
    an artefact nobody could open.

    Renaming the one collision fixes today. This is what stops the next one,
    because a capture that silently loses an artefact is worse than one that
    refuses to finish: the file it loses is the evidence.

    Scoped to fixtures/ on purpose. That is the committed set, the set
    oracle.json hashes, and the set `tests/oracle_capture_pins.rs` can check
    from the other side. Nothing under outputs/ is committed or hashed by
    this area.
    """
    path = os.path.join(FIX, name)
    if path in CLAIMED:
        raise SystemExit(
            f"two records write fixtures/{name}: {CLAIMED[path]} and {why}. "
            f"Give one of them its own name. The later write wins, and the "
            f"earlier record keeps the hash of a file that is no longer "
            f"there (#779).")
    CLAIMED[path] = why
    return path


# ---------------------------------------------------------------------------
# Deterministic sources.
# ---------------------------------------------------------------------------
W, H = 4, 3

# 8-bit steps chosen so both ends of the range and both sides of the midpoint
# appear, and so no two pixels in the 4x3 tile repeat a triple.
U8_STEPS = [(61, 13), (97, 151), (29, 211), (85, 40)]

# 16-bit values chosen to straddle every truncation boundary a 10-bit or a
# 12-bit narrow can land on: the low bits that survive, the low bits that do
# not, and full scale.
U16_VALUES = [0, 1, 255, 256, 1023, 1024, 4095, 4096,
              32767, 32768, 65472, 65535]


def ramp8(w, h, bands):
    data = bytearray()
    for y in range(h):
        for x in range(w):
            for b in range(bands):
                dx, dy = U8_STEPS[b]
                data.append((x * dx + y * dy) % 256)
    return bytes(data)


def ramp16(w, h, bands):
    data = bytearray()
    i = 0
    for _ in range(w * h * bands):
        data += struct.pack("<H", U16_VALUES[i % len(U16_VALUES)])
        i += 1
    return bytes(data)


def unpack16(data):
    return [struct.unpack("<H", data[i:i + 2])[0]
            for i in range(0, len(data), 2)]


def rawload(name, data, w, h, bands, fmt="uchar", interp="srgb"):
    """Write a raw buffer and load it into a `.v` with an interpretation."""
    raw = fix_path(f"{name}.raw", f"rawload({name})")
    with open(raw, "wb") as f:
        f.write(data)
    note(f"fixtures/{name}.raw written by this script "
         f"({w}x{h}x{bands} {fmt}, deterministic ramp)")
    v = os.path.join(OUT, f"{name}.v")
    vips("rawload", raw, v, str(w), str(h), str(bands), "--format", fmt)
    tagged = os.path.join(OUT, f"{name}-{interp}.v")
    vips("copy", v, tagged, "--interpretation", interp)
    return tagged


# ---------------------------------------------------------------------------
# The ISOBMFF walk. A port of #605 has to hand-roll exactly this, so the
# structure of every fixture is recorded, not just its pixels.
# ---------------------------------------------------------------------------

# Plain containers: children start right after the box header.
# Plain containers: children start right after the box header.
CONTAINERS = {b"iprp", b"ipco", b"moov", b"trak", b"mdia", b"minf",
              b"stbl", b"dinf"}
# FullBox containers: 4 bytes of version/flags, then children.
FULL_CONTAINERS = {b"meta", b"iref"}
# Item reference boxes, which live inside `iref` and are NOT FullBoxes.
IREF_KINDS = {b"auxl", b"dimg", b"thmb", b"cdsc", b"base"}


def parse_ftyp(b):
    brands = [b[i:i + 4].decode("latin1") for i in range(8, len(b) - 3, 4)]
    return {"major_brand": b[0:4].decode("latin1"),
            "minor_version": struct.unpack(">I", b[4:8])[0],
            "compatible_brands": "|".join(brands)}


def parse_ispe(b):
    return {"width": struct.unpack(">I", b[4:8])[0],
            "height": struct.unpack(">I", b[8:12])[0]}


def parse_pixi(b):
    n = b[4]
    return {"num_channels": n,
            "bits_per_channel": "|".join(str(v) for v in b[5:5 + n])}


def parse_av1c(b):
    """AV1CodecConfigurationBox, AV1-ISOBMFF 2.3.1. The encoded bit depth and
    the chroma subsampling live here, not in pixi and not in the header."""
    b0, b1, b2, b3 = b[0], b[1], b[2], b[3]
    return {"marker": b0 >> 7, "version": b0 & 0x7F,
            "seq_profile": b1 >> 5, "seq_level_idx_0": b1 & 0x1F,
            "seq_tier_0": (b2 >> 7) & 1,
            "high_bitdepth": (b2 >> 6) & 1,
            "twelve_bit": (b2 >> 5) & 1,
            "monochrome": (b2 >> 4) & 1,
            "chroma_subsampling_x": (b2 >> 3) & 1,
            "chroma_subsampling_y": (b2 >> 2) & 1,
            "chroma_sample_position": b2 & 3,
            "initial_presentation_delay_present": (b3 >> 4) & 1,
            "config_obus_bytes": len(b) - 4}


def parse_colr(b):
    kind = b[0:4].decode("latin1")
    out = {"colour_type": kind}
    if kind == "nclx":
        out.update({"colour_primaries": struct.unpack(">H", b[4:6])[0],
                    "transfer_characteristics": struct.unpack(">H", b[6:8])[0],
                    "matrix_coefficients": struct.unpack(">H", b[8:10])[0],
                    "full_range_flag": (b[10] >> 7) & 1})
    else:
        out["profile_bytes"] = len(b) - 4
        out["profile_sha256"] = hashlib.sha256(b[4:]).hexdigest()
    return out


def parse_auxc(b):
    end = b.index(b"\x00", 4)
    return {"aux_type": b[4:end].decode("latin1"),
            "aux_subtype_bytes": len(b) - end - 1}


def parse_pitm(b):
    wide = b[0] != 0
    return {"item_ID": struct.unpack(">I" if wide else ">H",
                                     b[4:8] if wide else b[4:6])[0]}


def parse_hdlr(b):
    return {"handler_type": b[8:12].decode("latin1")}


def parse_infe(b):
    version = b[0]
    if version < 2:
        return {"version": version, "body": b.hex()}
    if version == 2:
        item_id, p = struct.unpack(">H", b[4:6])[0], 6
    else:
        item_id, p = struct.unpack(">I", b[4:8])[0], 8
    name_end = b.index(b"\x00", p + 6)
    return {"version": version, "item_ID": item_id,
            "protection_index": struct.unpack(">H", b[p:p + 2])[0],
            "item_type": b[p + 2:p + 6].decode("latin1"),
            "item_name": b[p + 6:name_end].decode("latin1")}


def parse_iloc(b):
    """Where each item's bytes actually are. A port needs this to find the
    AV1 payload at all, so it is parsed rather than dumped as hex."""
    version = b[0]
    offset_size, length_size = b[4] >> 4, b[4] & 15
    base_size, index_size = b[5] >> 4, b[5] & 15
    p = 6
    if version < 2:
        count = struct.unpack(">H", b[p:p + 2])[0]
        p += 2
    else:
        count = struct.unpack(">I", b[p:p + 4])[0]
        p += 4

    def take(n):
        nonlocal p
        v = int.from_bytes(b[p:p + n], "big") if n else 0
        p += n
        return v

    items = []
    for _ in range(count):
        item_id = take(2 if version < 2 else 4)
        method = (take(2) & 15) if version in (1, 2) else 0
        take(2)  # data_reference_index
        base = take(base_size)
        extents = struct.unpack(">H", b[p:p + 2])[0]
        p += 2
        parts = []
        for _ in range(extents):
            take(index_size if version in (1, 2) and index_size else 0)
            off = take(offset_size)
            length = take(length_size)
            parts.append(f"{base + off}+{length}")
        items.append(f"{item_id}@" + ",".join(parts))
    return {"version": version, "construction_method": method,
            "offset_size": offset_size, "length_size": length_size,
            "items": "|".join(items)}


def parse_ipma(b):
    """Which properties each item claims, in ipco order, 1-based. A `*`
    marks the essential flag."""
    version, flags = b[0], int.from_bytes(b[1:4], "big")
    count = struct.unpack(">I", b[4:8])[0]
    p = 8
    entries = []
    for _ in range(count):
        if version < 1:
            item_id = struct.unpack(">H", b[p:p + 2])[0]
            p += 2
        else:
            item_id = struct.unpack(">I", b[p:p + 4])[0]
            p += 4
        n, p = b[p], p + 1
        props = []
        for _ in range(n):
            if flags & 1:
                v = struct.unpack(">H", b[p:p + 2])[0]
                p += 2
                idx, ess = v & 0x7FFF, v >> 15
            else:
                v, p = b[p], p + 1
                idx, ess = v & 0x7F, v >> 7
            props.append(f"{idx}{'*' if ess else ''}")
        entries.append(f"{item_id}=" + ",".join(props))
    return {"version": version, "flags": flags,
            "entries": "|".join(entries)}


def parse_iref_kind(b):
    n = struct.unpack(">H", b[2:4])[0]
    return {"from_item_ID": struct.unpack(">H", b[0:2])[0],
            "to_item_IDs": "|".join(
                str(struct.unpack(">H", b[4 + 2 * i:6 + 2 * i])[0])
                for i in range(n))}


LEAF_PARSERS = {
    b"ftyp": parse_ftyp, b"ispe": parse_ispe, b"pixi": parse_pixi,
    b"av1C": parse_av1c, b"colr": parse_colr, b"auxC": parse_auxc,
    b"pitm": parse_pitm, b"infe": parse_infe, b"ipma": parse_ipma,
    b"iloc": parse_iloc, b"hdlr": parse_hdlr,
    b"auxl": parse_iref_kind, b"dimg": parse_iref_kind,
    b"thmb": parse_iref_kind, b"cdsc": parse_iref_kind,
}


def _walk(d, start, end, depth, out):
    """One line per box: `<indent><type> <size> k=v k=v ...`. A dict per box
    triples the size of oracle.json for no extra fact."""
    p = start
    while p + 8 <= end:
        size = struct.unpack(">I", d[p:p + 4])[0]
        typ = d[p + 4:p + 8]
        hdr = 8
        if size == 1:
            size = struct.unpack(">Q", d[p + 8:p + 16])[0]
            hdr = 16
        elif size == 0:
            size = end - p
        pad = "  " * depth
        name = typ.decode("latin1")
        if size < hdr or p + size > end:
            out.append(f"{pad}{name} {size} TRUNCATED "
                       f"available={end - p}")
            return
        body = d[p + hdr:p + size]
        fields = {}
        kids = None
        if typ in CONTAINERS:
            kids = p + hdr
        elif typ in FULL_CONTAINERS:
            kids = p + hdr + 4
        elif typ == b"iinf":
            # FullBox, then a 16- or 32-bit entry count, then infe children.
            kids = p + hdr + 4 + (2 if body[0] == 0 else 4)
        elif typ == b"mdat":
            fields = {"payload_sha256": hashlib.sha256(body).hexdigest()}
        elif typ in LEAF_PARSERS:
            fields = LEAF_PARSERS[typ](body)
        else:
            fields = {"body": body.hex()}
        line = f"{pad}{name} {size}"
        for k, v in fields.items():
            line += f" {k}={v}"
        out.append(line)
        if kids is not None:
            _walk(d, kids, p + size, depth + 1, out)
        p += size


def boxes(path):
    with open(path, "rb") as f:
        d = f.read()
    out = []
    _walk(d, 0, len(d), 0, out)
    return out


def box(tree, kind, nth=0):
    """The nth line of a given box type, for records that want one box."""
    hits = [ln for ln in tree if ln.strip().startswith(kind + " ")]
    return hits[nth] if len(hits) > nth else None


def fixture(name, tagged, w, h, *args):
    """Save `tagged` as fixtures/<name>.avif and record everything about
    what comes back out of it."""
    path = fix_path(f"{name}.avif", f"fixture({name})")
    vips("heifsave", tagged, path, *args)
    return path, {
        "fixture": f"fixtures/{name}.avif",
        "sha256": sha256(path),
        "bytes": os.path.getsize(path),
        "saved_with": " ".join(args),
        "header": header(path, all_fields=True),
        "boxes": boxes(path),
        "getpoint": getpoint(path, w, h),
    }


records = {}
notes = []

# ---------------------------------------------------------------------------
# 0. What this build actually is, and what the two operations accept.
#
# `vips heifsave --help` prints the generic driver usage and exits 0, so it
# proves nothing at all about heifsave. The argument surface below is the
# bare invocation, which prints the real one.
# ---------------------------------------------------------------------------
records["build_and_argument_surface"] = {
    "what": "The oracle, checked rather than assumed. `--vips-config` names "
            "the compiled-in support and `vips -l` names the registered "
            "operations and their suffixes; `vips <op> --help` does NOT, it "
            "prints generic driver usage and exits 0 even for an operation "
            "that does not exist. The heif codec is a DYNAMIC MODULE here, "
            "so it can be present in the config line and still fail to "
            "dlopen; the argument dumps below only exist because it loaded.",
    "vips_config_heif": [
        line for line in run([VIPS, "--vips-config"]).stdout.splitlines()
        if "heif" in line.lower()
    ],
    "registered_operations": [
        line.strip()
        for line in run([VIPS, "-l"]).stdout.splitlines()
        if "heif" in line.lower() or "avif" in line.lower()
    ],
    "heifsave_arguments": run([VIPS, "heifsave"], allow_fail=True).stdout,
    "heifload_arguments": run([VIPS, "heifload"], allow_fail=True).stdout,
}

# ---------------------------------------------------------------------------
# 1. THE BIT DEPTH QUESTION.
#
# One 4x3 ushort ramp saved at 8, 10 and 12 bits, and read back. This is the
# record that decides which PixelFormat #605 targets.
# ---------------------------------------------------------------------------
src16 = ramp16(W, H, 3)
src16_in = unpack16(src16)
tagged16 = rawload("rgb16_src", src16, W, H, 3, fmt="ushort", interp="rgb16")

depth = {}
for bits in (8, 10, 12):
    # `rgb8` belongs to record 2, which saves the 8-bit ramp under it. This
    # row saves the 16-bit ramp NARROWED to 8 bits, which is a different
    # image, so it needs a different name. Both used to be called `rgb8` and
    # the second write won (#779). 10 and 12 keep their names: nothing else
    # claims those, and #605 embeds them.
    name = "rgb8_narrowed" if bits == 8 else f"rgb{bits}"
    path, rec = fixture(name, tagged16, W, H, "--bitdepth", str(bits),
                        "--lossless", "--keep", "none")
    out = flat(rec["getpoint"])
    hdr = rec["header"]
    shift = 16 - bits
    rec["source_16bit"] = src16_in
    rec["read_back"] = out
    rec["format"] = hdr.get("format")
    rec["interpretation"] = hdr.get("interpretation")
    rec["bits_per_sample"] = int(hdr["bits-per-sample"])
    rec["heif_bitdepth_deprecated"] = int(field(path, "heif-bitdepth"))
    # Derived from the two MEASURED arrays, not from a table anyone typed.
    rec["is_narrow_then_left_justify"] = \
        out == [(v >> shift) << shift for v in src16_in]
    rec["is_plain_right_shift"] = out == [v >> shift for v in src16_in]
    rec["is_full_scale_rescale"] = \
        out == [round((v >> shift) * 65535 / ((1 << bits) - 1))
                for v in src16_in]
    rec["max_representable"] = max(out)
    rec["low_bits_always_zero"] = shift if all(
        v & ((1 << shift) - 1) == 0 for v in out) else 0
    depth[str(bits)] = rec

records["bit_depth_carrier"] = {
    "what": "THE record. One ushort rgb16 ramp written at 8, 10 and 12 "
            "bits and read straight back, so each row is a full round trip "
            "and the two halves have to be read apart. At 8 bits the carrier "
            "is UCHAR/srgb and the load side is a memcpy, so the row shows "
            "only heifsave's narrow, a plain `>> 8` (heifsave.c:394-411). At "
            "10 and 12 the carrier is USHORT/RGB16 and the load side LEFT "
            "JUSTIFIES: heifload.c:1000-1016 does `v = ((p[0] << 8) | p[1]) "
            "<< (16 - bits_per_pixel)` on the big-endian samples libheif "
            "hands back, which composes with heifsave's `>> (16 - bitdepth)` "
            "into `(v >> s) << s`. That is what the booleans below test, "
            "each computed from the two measured arrays rather than from a "
            "table anyone typed. Load side alone, in the form a port needs "
            "it: a decoder that gives you 0..1023 for 10-bit must return "
            "`sample << 6`, so the low 6 bits are always zero and the "
            "maximum is 65472. For 12-bit it is `sample << 4` and 65520.",
    "carrier_by_bitdepth": {"8": "uchar / srgb", "10": "ushort / rgb16",
                            "12": "ushort / rgb16"},
    "why_the_8_bit_fixture_is_not_called_rgb8": "Because `fixtures/rgb8.avif` "
        "is record 2's file, saved from the 8-BIT ramp, and this row is the "
        "16-bit ramp narrowed to 8. They are different images and they used "
        "to share the name: this row wrote first, record 2 overwrote it, and "
        "this row went on recording the sha256 and the byte count of a file "
        "that was no longer in the tree (#779). fixtures/rgb8_narrowed.avif "
        "is that file, under a name nothing else claims. capture.py now "
        "refuses to write any fixture name twice, so the next collision "
        "stops the capture instead of quietly losing an artefact.",
    "why_it_matters": "A port that decodes 10-bit AVIF into u8, or into u16 "
                      "rescaled to full 0..65535 range, passes every header "
                      "assertion (4x3, 3 bands, ushort, rgb16, "
                      "bits-per-sample 10) and produces wrong pixels "
                      "everywhere. The carrier is u16 and the scale is "
                      "`value << (16 - bits)`.",
    "c_reference": "heifload.c:748-761 picks the format and interpretation "
                   "from `bits_per_pixel > 8` and nothing else; "
                   "heifload.c:1000-1016 does the byte swap and the shift; "
                   "vips__heif_chroma at heifload.c:296-310 is what asks "
                   "libheif for RRGGBB_BE above 8 bits.",
    "by_bitdepth": depth,
}

# ---------------------------------------------------------------------------
# 2. 8-bit is an exact identity, with and without alpha.
# ---------------------------------------------------------------------------
eight = {}
for name, bands in (("rgb8", 3), ("rgba8", 4)):
    src = ramp8(W, H, bands)
    tagged = rawload(f"{name}_src", src, W, H, bands, interp="srgb")
    path, rec = fixture(name, tagged, W, H,
                        "--bitdepth", "8", "--lossless", "--keep", "none")
    rec["source_bytes"] = list(src)
    rec["bands_in"] = bands
    rec["bands_out"] = len(rec["getpoint"][0])
    rec["identity"] = flat(rec["getpoint"]) == list(src)
    eight[name] = rec
    assert rec["identity"], name

records["eight_bit_lossless_identity"] = {
    "what": "`heifsave --lossless --bitdepth 8` to a `.avif` and straight "
            "back is the identity, for 3 bands and for 4. AV1 lossless with "
            "matrix_coefficients 0 (identity/GBR) and 4:4:4 chroma stores "
            "the RGB planes as-is, so there is no colour conversion to "
            "disagree about. A correct decoder reproduces these bytes "
            "exactly rather than approximately.",
    "cases": eight,
}

# ---------------------------------------------------------------------------
# 3. Alpha above 8 bits, and where the alpha plane actually lives.
# ---------------------------------------------------------------------------
src16a = ramp16(W, H, 4)
tagged16a = rawload("rgba16_src", src16a, W, H, 4,
                    fmt="ushort", interp="rgb16")
_, rgba10 = fixture("rgba10", tagged16a, W, H,
                    "--bitdepth", "10", "--lossless", "--keep", "none")
a_in = unpack16(src16a)
a_out = flat(rgba10["getpoint"])
rgba10["source_16bit"] = a_in
rgba10["bands_out"] = len(rgba10["getpoint"][0])
rgba10["alpha_takes_the_same_left_justify"] = \
    a_out == [(v >> 6) << 6 for v in a_in]

records["alpha"] = {
    "what": "Alpha survives at 8 bits exactly (record 2) and at 10 bits "
            "through the same `(v >> 6) << 6`, so the alpha band is not "
            "special-cased on the way back. Structurally it is not part of "
            "the colour image at all: the container carries a SECOND av1C "
            "item with monochrome=1 and its own pixi, tagged by an auxC of "
            "`urn:mpeg:mpegB:cicp:systems:auxiliary:alpha` and joined to the "
            "primary item by an `iref` of type `auxl`. A port walking the "
            "meta box has to follow that reference; there is no fourth plane "
            "in the colour item to find.",
    "c_reference": "heifload.c:585-589 sets bands to 4 from "
                   "heif_image_handle_has_alpha_channel and nothing else; "
                   "heifsave.c:293 sets save_alpha_channel from Bands > 3.",
    "where_to_look": "the `boxes` of eight_bit_lossless_identity.cases."
                     "rgba8 and of alpha.rgba10 below: two av1C, two pixi, "
                     "an auxC, and an iref/auxl from item 2 back to item 1",
    "rgba10": rgba10,
}

# ---------------------------------------------------------------------------
# 4. Greyscale comes back as three bands.
# ---------------------------------------------------------------------------
grey_src = ramp8(W, H, 1)
grey_tagged = rawload("grey8_src", grey_src, W, H, 1, interp="b-w")
_, grey = fixture("grey8", grey_tagged, W, H,
                  "--bitdepth", "8", "--lossless", "--keep", "none")
grey["source_bytes"] = list(grey_src)
grey["bands_in"] = 1
grey["bands_out"] = len(grey["getpoint"][0])
grey["luminance_repeated"] = (
    [p[0] for p in grey["getpoint"]] == list(grey_src)
    and all(p[0] == p[1] == p[2] for p in grey["getpoint"]))

records["greyscale_promotes_to_rgb"] = {
    "what": "A 1-band b-w uchar image saved to `.avif` loads back as 3 bands "
            "srgb with the luminance repeated. Two separate reasons, and "
            "only the second is about loading: `heifsave` is registered "
            "`rgb alpha`, so foreign.c:1479-1481 colourspaces a mono source "
            "up to sRGB before it ever reaches the encoder; and heifload "
            "always decodes to RGB regardless of what it was given "
            "(heifload.c:763-765, `FIXME .. we always decode to RGB in "
            "generate`). So even a genuinely monochrome AVIF -- which this "
            "saver cannot write, but an encoder elsewhere can -- comes back "
            "as 3 bands. A port must not return 1 band for one.",
    "record": grey,
}

# ---------------------------------------------------------------------------
# 5. The default bitdepth follows the INTERPRETATION, not the declared
#    default and not the band format.
# ---------------------------------------------------------------------------
u8_tagged = os.path.join(OUT, "rgb8_src-srgb.v")
tagged16_srgb = rawload("rgb16_srgb_src", src16, W, H, 3,
                        fmt="ushort", interp="srgb")
defaults = {}
for label, tagged in (("uchar srgb", u8_tagged),
                      ("ushort rgb16", tagged16),
                      ("ushort srgb", tagged16_srgb)):
    out = os.path.join(OUT, f"default_{label.replace(' ', '_')}.avif")
    vips("heifsave", tagged, out, "--lossless", "--keep", "none")
    defaults[label] = {
        "summary": header(out)["summary"],
        "bits_per_sample": int(field(out, "bits-per-sample")),
        "read_back": flat(getpoint(out, W, H)),
    }

records["default_bitdepth_follows_interpretation"] = {
    "what": "`heifsave` declares `bitdepth` with a default of 12, and that "
            "default is overwritten in build: heifsave.c:536-541 sets it to "
            "12 only when the READY image is tagged RGB16 or GREY16, and to "
            "8 otherwise. Measured, a uchar srgb source gets 8 and a ushort "
            "source gets 12. The third row is the interesting one: a ushort "
            "image tagged plain `srgb` ALSO gets 12, because foreign.c:1476-"
            "1481 colourspaces any ushort source to RGB16 before heifsave "
            "sees it, keying off the ORIGINAL BandFmt (foreign.c:1403). "
            "That has a consequence for reading heifsave.c: the "
            "`vips_bitdepth = 8` arm at heifsave.c:400-403 and 417-420, and "
            "the negative shift it would compute for bitdepth 10, are "
            "unreachable through the file saver.",
    "cases": defaults,
}

# ---------------------------------------------------------------------------
# 6. An 8-bit source written at 10 or 12 bits does NOT reach full scale.
# ---------------------------------------------------------------------------
# The ramp never reaches 255, and 255 is the value the claim is about, so
# this record gets its own source: the two ends and both sides of the middle.
U8_EXTREMES = [0, 1, 127, 128, 254, 255]
u8_src = bytes(U8_EXTREMES[i % len(U8_EXTREMES)] for i in range(W * H * 3))
u8_ext_tagged = rawload("u8_extremes_src", u8_src, W, H, 3, interp="srgb")
promote = {}
for bits in (10, 12):
    out = os.path.join(OUT, f"u8_at_{bits}.avif")
    vips("heifsave", u8_ext_tagged, out,
         "--bitdepth", str(bits), "--lossless", "--keep", "none")
    got = flat(getpoint(out, W, H))
    promote[str(bits)] = {
        "summary": header(out)["summary"],
        "bits_per_sample": int(field(out, "bits-per-sample")),
        "source_bytes": list(u8_src),
        "read_back": got,
        "is_shift_left_8": got == [v << 8 for v in u8_src],
        "is_replicate_257": got == [v * 257 for v in u8_src],
        "max_read_back": max(got),
    }

records["eight_bit_source_at_deeper_bitdepth"] = {
    "what": "`heifsave --bitdepth 10` (or 12) on a UCHAR source shifts left "
            "by `bitdepth - 8` on the way in (heifsave.c:379-392) and "
            "heifload shifts left by `16 - bitdepth` on the way out, so the "
            "two compose to exactly `value << 8`. Measured, 255 comes back "
            "as 65280 and not 65535: it is a shift, never a `* 257` "
            "replicate, so an 8-bit white does not survive as a 16-bit "
            "white. Recorded "
            "because it is the obvious place to 'fix' a round trip that "
            "looks lossy and it would be wrong to.",
    "cases": promote,
}

# ---------------------------------------------------------------------------
# 7. Colour information: nclx, ICC, and what heifload does with each.
# ---------------------------------------------------------------------------
colour = {}

# Untagged and lossless: heifsave writes an nclx box because lossless needs
# identity matrix coefficients.
colour["lossless_untagged"] = {
    "see": "eight_bit_lossless_identity.cases.rgb8 for the full boxes and "
           "header; the colr line and the absence of an ICC field are the "
           "point here",
    "colr": box(eight["rgb8"]["boxes"], "colr").strip(),
    "has_icc_profile_data": "icc-profile-data" in eight["rgb8"]["header"],
}

# Untagged and lossy: no colour box at all.
lossy = fix_path("rgb8_q50_420.avif", "colour_information.lossy_untagged")
vips("heifsave", u8_tagged, lossy, "--Q", "50", "--keep", "none")
colour["lossy_untagged"] = {
    "fixture": "fixtures/rgb8_q50_420.avif",
    "sha256": sha256(lossy),
    "bytes": os.path.getsize(lossy),
    "boxes": boxes(lossy),
    "header": header(lossy, all_fields=True),
    "getpoint": getpoint(lossy, W, H),
    "has_icc_profile_data": False,
}

# Tagged with libvips' own built-in sRGB profile, so this reproduces without
# any file off this machine.
icc = fix_path("rgb8_icc.avif", "colour_information.icc_tagged")
vips("heifsave", u8_tagged, icc,
     "--lossless", "--profile", "srgb", "--keep", "none")
colour["icc_tagged"] = {
    "fixture": "fixtures/rgb8_icc.avif",
    "sha256": sha256(icc),
    "bytes": os.path.getsize(icc),
    "profile": "libvips built-in `srgb`, via `heifsave --profile srgb`",
    "boxes": boxes(icc),
    "header": header(icc, all_fields=True),
    "getpoint": getpoint(icc, W, H),
}

# Re-saving the tagged file with no --profile takes the embedded one.
icc2 = os.path.join(OUT, "icc_resave.avif")
vips("heifsave", icc, icc2, "--lossless", "--keep", "none")
colour["icc_survives_a_resave"] = {
    "what": "heifsave.c:286-289 falls back to the image's own "
            "VIPS_META_ICC_NAME when no --profile is given",
    "colr": box(boxes(icc2), "colr").strip(),
    "icc_profile_data": header(icc2, all_fields=True).get("icc-profile-data"),
}

records["colour_information"] = {
    "what": "AVIF can carry colour two ways and libvips uses exactly one of "
            "them on the way back. Lossless save writes a `colr` box of type "
            "`nclx` with matrix_coefficients 0, because "
            "heif_matrix_coefficients_RGB_GBR is what makes AV1 lossless "
            "identity (heifsave.c:295-310); a LOSSY save writes no `colr` "
            "box at all, since that branch is guarded on `heif->lossless`. "
            "Either way heifload THROWS THE NCLX AWAY: heifload.c:719-721 "
            "logs `heifload: ignoring nclx profile` and attaches nothing. "
            "Only a `rICC` or `prof` colr becomes `icc-profile-data` "
            "(heifload.c:694-717). So a tagged and an untagged file differ "
            "in exactly one header field and not in the interpretation, "
            "which stays srgb at 8 bits and rgb16 above regardless "
            "(heifload.c:754-761). Note also that supplying an ICC "
            "DISPLACES the nclx box rather than joining it: the icc_tagged "
            "case below is lossless and still has only one colr.",
    "cases": colour,
}

# ---------------------------------------------------------------------------
# 8. Chroma subsampling: the argument exists on this build, so sweep it.
# ---------------------------------------------------------------------------
sub = {}
sweep = [
    ("q50_auto", ["--Q", "50"]),
    ("q89_auto", ["--Q", "89"]),
    ("q90_auto", ["--Q", "90"]),
    ("q50_off", ["--Q", "50", "--subsample-mode", "off"]),
    ("q90_on", ["--Q", "90", "--subsample-mode", "on"]),
    ("lossless_auto", ["--lossless"]),
    ("lossless_on", ["--lossless", "--subsample-mode", "on"]),
]
for label, args in sweep:
    out = os.path.join(OUT, f"sub_{label}.avif")
    vips("heifsave", u8_tagged, out, *args, "--keep", "none")
    sub[label] = {
        "saved_with": " ".join(args),
        "av1C": box(boxes(out), "av1C").strip(),
        "colr": (box(boxes(out), "colr") or "").strip() or None,
        "bytes": os.path.getsize(out),
    }

q90 = fix_path("rgb8_q90_444.avif", "chroma_subsampling.fixture_444")
vips("heifsave", u8_tagged, q90, "--Q", "90", "--keep", "none")
records["chroma_subsampling"] = {
    "what": "`heifsave` on this build does expose `subsample-mode` with "
            "`auto`, `on` and `off`, so this is a real axis rather than an "
            "assumed one. Measured through the av1C box's own subsampling "
            "flags rather than by guessing from file size: `auto` is 4:2:0 "
            "below Q 90 and 4:4:4 at Q 90 and above (heifsave.c:608-612 "
            "picks the encoder's `chroma` string as `Q >= 90 ? 444 : 420`), "
            "`on` forces 4:2:0 at any Q, `off` forces 4:4:4, and `lossless` "
            "overrides the argument to `off` (heifsave.c:530-532) so the "
            "lossless_on row is 4:4:4 despite asking for subsampling. "
            "seq_profile moves with it: 0 (Main) for 4:2:0 8-bit, 1 (High) "
            "for 4:4:4 8-bit, 2 (Professional) for 12-bit. This is the axis "
            "a decoder is most likely to get subtly wrong, because a 4:2:0 "
            "file needs chroma upsampling and a 4:4:4 one does not.",
    "fixture_444": "fixtures/rgb8_q90_444.avif",
    "fixture_444_sha256": sha256(q90),
    "fixture_444_boxes": boxes(q90),
    "fixture_444_getpoint": getpoint(q90, W, H),
    "sweep": sub,
}

# ---------------------------------------------------------------------------
# 9. Odd dimensions with 4:2:0, where the chroma planes do not divide evenly.
# ---------------------------------------------------------------------------
odd_src = ramp8(3, 3, 3)
odd_tagged = rawload("odd3x3_src", odd_src, 3, 3, 3, interp="srgb")
odd = fix_path("odd3x3_q50.avif", "odd_dimensions")
vips("heifsave", odd_tagged, odd, "--Q", "50", "--keep", "none")
records["odd_dimensions"] = {
    "what": "A 3x3 image at the default Q, so 4:2:0 over an odd width and an "
            "odd height. The chroma planes are 2x2 and the reconstruction "
            "has to handle the half-covered right column and bottom row. "
            "Pinned because it is where a hand-rolled upsampler diverges "
            "first and the header gives no hint that it has.",
    "fixture": "fixtures/odd3x3_q50.avif",
    "sha256": sha256(odd),
    "bytes": os.path.getsize(odd),
    "source_bytes": list(odd_src),
    "header": header(odd, all_fields=True),
    "boxes": boxes(odd),
    "getpoint": getpoint(odd, 3, 3),
}

one = os.path.join(OUT, "one1x1.avif")
vips("black", os.path.join(OUT, "one.v"), "1", "1", "--bands", "3")
vips("copy", os.path.join(OUT, "one.v"), os.path.join(OUT, "one-srgb.v"),
     "--interpretation", "srgb")
p1 = vips("heifsave", os.path.join(OUT, "one-srgb.v"), one,
          "--lossless", "--keep", "none", allow_fail=True)
records["odd_dimensions"]["one_by_one"] = {
    "exit": p1.returncode,
    "stderr": scrub(p1.stderr.strip()).splitlines()[:2],
    "header": header(one, all_fields=True, allow_fail=True),
    "boxes": boxes(one) if os.path.exists(one) else None,
}

# ---------------------------------------------------------------------------
# 10. The `.avif` suffix decides the codec, not the --compression argument.
# ---------------------------------------------------------------------------
codec = {}
for label, fname, args in (
    ("avif_default", "codec_default.avif", []),
    ("avif_asking_for_hevc", "codec_hevc.avif", ["--compression", "hevc"]),
    ("avif_asking_for_avc", "codec_avc.avif", ["--compression", "avc"]),
    ("heic_default", "codec_default.heic", []),
):
    out = os.path.join(OUT, fname)
    proc = vips("heifsave", u8_tagged, out, *args,
                "--lossless", "--keep", "none", allow_fail=True)
    codec[label] = {
        "saved_with": " ".join(args),
        "exit": proc.returncode,
        "heif_compression_read_back": field(out, "heif-compression")
        if proc.returncode == 0 else None,
        "ftyp": boxes(out)[0] if proc.returncode == 0 else None,
        "av1C_or_hvcC": next(
            (ln.strip() for ln in boxes(out)
             if ln.strip().startswith(("av1C", "hvcC"))), None)
        if proc.returncode == 0 else None,
        "stderr": scrub(proc.stderr.strip()).splitlines()[:1],
    }

records["filename_decides_the_codec"] = {
    "what": "`heifsave` defaults `compression` to hevc (heifsave.c:856) and "
            "then heifsave.c:889-890 overrides it to AV1 for any filename "
            "ending `.avif`, AFTER the argument has been parsed, so "
            "`--compression hevc` on a `.avif` is silently ignored. Measured "
            "both ways round. The brand follows: `.avif` gets a major brand "
            "of `avif` with `mif1 avif miaf` compatible, `.heic` gets "
            "`heix`. This is why #605 says to capture with an AVIF payload "
            "specifically rather than a generic `.heic`: the same operation "
            "and the same arguments produce a file a pure-Rust AV1 decoder "
            "cannot read, purely from the extension.",
    "heic_note": "The `.heic` row is here as the contrast only. HEIC is NOT "
                 "a target of #605 and nothing else in this capture touches "
                 "it. Same code path, different payload codec, and the "
                 "heifload side is identical apart from `heif-compression`.",
    "cases": codec,
}

# ---------------------------------------------------------------------------
# 11. Which AV1 encoders this libheif actually offers.
# ---------------------------------------------------------------------------
encoders = {}
for enc in ("auto", "aom", "rav1e", "svt", "x265"):
    out = os.path.join(OUT, f"enc_{enc}.avif")
    proc = vips("heifsave", u8_tagged, out, "--encoder", enc,
                "--lossless", "--keep", "none", allow_fail=True)
    encoders[enc] = {
        "exit": proc.returncode,
        "bytes": os.path.getsize(out) if proc.returncode == 0 else None,
        "sha256": sha256(out) if proc.returncode == 0 else None,
        "warned_not_found": "could not find" in proc.stderr,
        "stderr": scrub(proc.stderr.strip()).splitlines()[:1],
    }
for enc, rec in encoders.items():
    rec["same_bytes_as_auto"] = rec["sha256"] == encoders["auto"]["sha256"]

records["encoder_selection"] = {
    "what": "`heifsave --encoder` offers auto/aom/rav1e/svt/x265, but the "
            "enum is the SUPERSET libvips knows about rather than what this "
            "libheif was built with, and asking for a missing one is NOT an "
            "error: heifsave.c:565-567 emits a `g_warning` and "
            "heifsave.c:571-576 then falls back to "
            "heif_context_get_encoder_for_format, so the exit status is 0 "
            "and you silently get the default encoder. Measured, every row "
            "here writes the SAME bytes, so this build has exactly one AV1 "
            "encoder and `--encoder rav1e` is a no-op with a warning. Worth "
            "pinning because a differential that selects an encoder and "
            "trusts the exit code is not testing what it thinks it is. "
            "Every fixture in this directory came out of whichever encoder "
            "these rows agree on. Only the bytes depend on it, not the "
            "decoded pixels.",
    "cases": encoders,
}

# ---------------------------------------------------------------------------
# 12. The bitdepth argument domain.
# ---------------------------------------------------------------------------
domain = {}
for bits in (7, 8, 9, 10, 11, 12, 13, 16):
    out = os.path.join(OUT, f"bd_{bits}.avif")
    proc = vips("heifsave", tagged16, out, "--bitdepth", str(bits),
                "--lossless", "--keep", "none", allow_fail=True)
    domain[str(bits)] = {
        "exit": proc.returncode,
        "stderr": scrub(proc.stderr.strip()).splitlines()[:2],
        "bits_per_sample_read_back": field(out, "bits-per-sample")
        if proc.returncode == 0 else None,
    }

records["bitdepth_argument_domain"] = {
    "what": "Two gates, and only one of them fails. The GObject property "
            "is declared 8..12 (heifsave.c:789-795), so 7, 13 and 16 are "
            "rejected by the property system with a GLib-GObject-CRITICAL "
            "on stderr and then IGNORED: exit status 0, and the file comes "
            "out at whatever the interpretation-derived default was, 12 "
            "here. 9 and 11 are inside the property range and reach "
            "heifsave.c:545-551, which knows AVIF has only 8, 10 and 12, and "
            "that one is a real error with exit 1. So `--bitdepth 16` "
            "quietly writes a 12-bit file. A port validating a bitdepth it "
            "read out of a file should accept exactly {8, 10, 12}.",
    "cases": domain,
}

# ---------------------------------------------------------------------------
# 13. Metadata: what `--keep all` synthesises out of nothing.
# ---------------------------------------------------------------------------
keepall = os.path.join(OUT, "keep_all.avif")
vips("heifsave", u8_tagged, keepall, "--lossless")
records["metadata_default_keep"] = {
    "what": "Every fixture here is saved `--keep none`. With the DEFAULT "
            "keep, heifsave writes an EXIF block the source raster never "
            "carried, synthesised from the resolution fields: "
            "XResolution/YResolution/ResolutionUnit plus an Exif 2.1 version "
            "and the pixel dimensions. heifload lifts it back as `exif-data` "
            "and libexif expands it into `exif-ifd*` fields. Worth pinning "
            "because a differential that saves with defaults and compares "
            "byte counts will see this and blame the codec.",
    "header": header(keepall, all_fields=True),
    "extra_items": [ln for ln in boxes(keepall)
                    if "Exif" in ln or "infe" in ln or ln.startswith("mdat")],
}

# ---------------------------------------------------------------------------
# 14. Malformed and truncated input. A rejected file does not fail loudly.
# ---------------------------------------------------------------------------
good = os.path.join(FIX, "rgb8.avif")
with open(good, "rb") as f:
    good_bytes = f.read()

bad = {}


def probe_bad(name, data, keep_as_fixture=False):
    path = (fix_path(name, f"probe_bad({name})") if keep_as_fixture
            else os.path.join(OUT, name))
    with open(path, "wb") as f:
        f.write(data)
    note(f"{'fixtures' if keep_as_fixture else 'outputs'}/{name} written by "
         f"this script by damaging fixtures/rgb8.avif")
    proc = run([VIPSHEADER, "-a", path], allow_fail=True)
    err = scrub(proc.stderr.strip()).splitlines()
    px = vips("getpoint", path, "1", "0", allow_fail=True)
    entry = {
        "bytes": len(data),
        "exit": proc.returncode,
        "vips_loader": field(path, "vips-loader"),
        "stderr_first": err[:1],
        "stderr_last": err[-1:],
        "distinct_stderr_lines": len(set(err)),
        # 61 97 29 is what the undamaged fixtures/rgb8.avif reads here, so
        # this says whether the damage reached the pixels or only the header.
        "getpoint_1_0": [int(float(v)) for v in px.stdout.split()],
        "pixel_exit": px.returncode,
        "pixel_error": (scrub(px.stderr.strip()).splitlines()[-1:]
                        if px.returncode != 0 else []),
    }
    if keep_as_fixture:
        entry["fixture"] = f"fixtures/{name}"
        entry["sha256"] = sha256(path)
    bad[name] = entry
    return entry


# Cut off inside the meta box.
probe_bad("truncated.avif", good_bytes[:int(len(good_bytes) * 0.6)],
          keep_as_fixture=True)
# Cut off inside the ftyp box.
probe_bad("truncated_ftyp.avif", good_bytes[:10])
# Nothing but a valid ftyp.
probe_bad("ftyp_only.avif", good_bytes[:28])
# Empty.
probe_bad("empty.avif", b"")
# The AVIF SEQUENCE brand, which is deliberately absent from heif_magic.
probe_bad("brand_avis.avif", good_bytes[:8] + b"avis" + good_bytes[12:],
          keep_as_fixture=True)
# A brand nothing knows.
probe_bad("brand_zzzz.avif", good_bytes[:8] + b"zzzz" + good_bytes[12:])
# A first box length that is not a multiple of 4, which is_a rejects even
# though the brand is right.
probe_bad("ftyp_len_29.avif", struct.pack(">I", 29) + good_bytes[4:])
# A first box length over 2048, same.
probe_bad("ftyp_len_4096.avif", struct.pack(">I", 4096) + good_bytes[4:])
# Body replaced with zeroes, header intact.
probe_bad("zeroed_mdat.avif",
          good_bytes[:-49] + b"\x00" * 49)

records["malformed_and_truncated"] = {
    "header_is_not_a_decode": "zeroed_mdat.avif is the case to read first. "
                              "Its container is intact, so vipsheader "
                              "succeeds and reports 4x3 uchar srgb heifload "
                              "with every heif-* field present, and the "
                              "decode fails only when pixels are pulled, "
                              "with `Corrupt frame detected` and one `error "
                              "in tile 0 x N` warning per strip. A port must "
                              "not treat a parsed header as a decodable "
                              "image, and must fail at generate rather than "
                              "returning zeros.",
    "what": "The other half of this record is that a file libvips does "
            "not RECOGNISE does not fail: `vips_foreign_load_heif_is_a` "
            "(heifload.c:400-425) requires the first 4 bytes to be a "
            "big-endian box length that is a multiple of 4 and no larger "
            "than 2048, and bytes 4..12 to equal one of ten literal `ftyp` "
            "magics (heifload.c:380-391). Fail any of that and vips falls "
            "through to the NEXT loader, which on this build is magickload, "
            "and the file may well load anyway through ImageMagick's own "
            "HEIF delegate -- with a different loader name in the header and "
            "different behaviour. `ftypavis`, the AVIF SEQUENCE brand, is "
            "the live example: it is missing from the magic list, so an "
            "otherwise valid file goes to magickload. A genuinely truncated "
            "file that DOES pass is_a fails inside libheif instead, with a "
            "`No 'meta' box` error preceded by a run of `bad seek to N` "
            "warnings from the vips source as it hunts for the end. And "
            "when the fallback DOES land, it can be invisible: "
            "brand_avis.avif goes to magickload and returns the same pixels "
            "as the original, so nothing but `vips-loader` and the missing "
            "heif-* fields says anything happened.",
    "fail_on_note": "`heifload`'s `fail-on` argument does not change any of "
                    "this: the truncated file fails while reading the "
                    "header, before fail-on has anything to gate.",
    "reference_pixel_1_0": "the undamaged fixtures/rgb8.avif reads 61 97 29 "
                           "at (1,0); a case below that loads and reports "
                           "something else decoded different pixels out of "
                           "the same payload",
    "cases": bad,
}

# ---------------------------------------------------------------------------
# 15. What the ftyp brand does and does not decide.
# ---------------------------------------------------------------------------
brands = {}
for brand in ("heic", "heix", "hevc", "heim", "heis", "hevm", "hevs",
              "mif1", "msf1", "avif", "avis", "mif2"):
    out = os.path.join(OUT, f"brand_{brand}.avif")
    with open(out, "wb") as f:
        f.write(good_bytes[:8] + brand.encode() + good_bytes[12:])
    note(f"outputs/brand_{brand}.avif written by this script: "
         f"fixtures/rgb8.avif with its ftyp major brand replaced")
    hdr = header(out, all_fields=True, allow_fail=True)
    brands[brand] = {
        "in_heif_magic_list": brand in ("heic", "heix", "hevc", "heim",
                                        "heis", "hevm", "hevs", "mif1",
                                        "msf1", "avif"),
        "vips_loader": hdr.get("vips-loader"),
        "summary": header(out, allow_fail=True).get("summary"),
        "heif_compression": hdr.get("heif-compression"),
        "getpoint_1_0": [int(float(v)) for v in
                         vips("getpoint", out, "1", "0",
                              allow_fail=True).stdout.split()],
    }

records["ftyp_brand_is_a_gate_not_a_codec_selector"] = {
    "what": "The same AV1 payload behind each of the twelve brands below, "
            "and the answer splits in two. DECODING ignores the brand: all "
            "ten brands in heif_magic load through heifload and return pixel "
            "(1,0) identical to the untouched file, because the codec comes "
            "from the item's `infe` type (`av01`) and its av1C property. "
            "LABELLING is the brand and nothing else: heifload.c:733-741 "
            "sniffs 12 bytes, runs heif_main_brand over them, and sets "
            "`heif-compression` to av1 only for `avif` or `avis`, defaulting "
            "to `hevc` otherwise -- so nine of these ten AV1 files are "
            "labelled `hevc` while decoding perfectly. `heif-compression` is "
            "a brand echo, NOT codec detection, and a port must not believe "
            "it. The two brands outside the list fall through to another "
            "loader entirely.",
    "reference_pixel_1_0": "the untouched fixtures/rgb8.avif reads "
                           "61 97 29 at (1,0)",
    "cases": brands,
}

# ---------------------------------------------------------------------------
# 16. Still-image page fields, and what the load arguments do to a still.
# ---------------------------------------------------------------------------
pages = {}
for label, spec in (("default", ""), ("page=0", "[page=0]"),
                    ("n=-1", "[n=-1]"), ("n=2", "[n=2]"),
                    ("page=1", "[page=1]"),
                    ("thumbnail", "[thumbnail=true]")):
    h = header(good + spec, all_fields=True, allow_fail=True)
    pages[label] = {
        "summary": header(good + spec, allow_fail=True).get("summary"),
        "n_pages": h.get("n-pages"),
        "heif_primary": h.get("heif-primary"),
        "bits_per_sample": h.get("bits-per-sample"),
        "error": (h["error"].splitlines()[-1] if "error" in h else None),
    }

records["still_image_page_fields"] = {
    "what": "An AVIF still reports `n-pages: 1` and `heif-primary: 0`, and "
            "`[n=-1]` is the same image rather than a toilet roll, so the "
            "page model #569 covers has nothing to do here. `[page=1]` is "
            "out of range and errors; `[thumbnail=true]` on a file with no "
            "thumbnail item falls back to the primary image rather than "
            "failing. Recorded so a port knows which of these have to be "
            "accepted and which have to be refused.",
    "cases": pages,
}

# ---------------------------------------------------------------------------
# Provenance of the encoded bytes: the vips version alone does not fix them,
# libheif and the AV1 encoder do.
# ---------------------------------------------------------------------------
def keg(name):
    """`vips --vips-config` prints `HEIC/AVIF load/save with libheif: true
    (dynamic module: true)` and no version at all, for libheif or for the AV1
    encoder behind it, so neither can be measured out of vips. These come
    from the Homebrew keg each `opt` symlink resolves to, which is the
    library vips is actually linked against. Recorded because they, not the
    vips version, are what fixes the encoded bytes and the decoder that
    produces the carrier this whole capture is about."""
    p = f"/opt/homebrew/opt/{name}"
    return os.path.basename(os.path.realpath(p)) if os.path.exists(p) else None


version = run([VIPS, "--version"]).stdout.strip()

notes.append(
    "THE VERSION IN meta IS MEASURED, NOT INHERITED. This machine's vips "
    "moved from 8.18.4 to 8.18.6 DURING this capture, under a `brew upgrade` "
    "I did not start, taking libheif from 1.23.1 to 1.23.2 and x265 from 4.2 "
    "to 4.3 with it, and deleting the 8.18.4 keg. Every fixture and every "
    "number in this file was produced by a single clean re-run AFTER that "
    "upgrade, on the version meta records. Earlier exploratory probes ran on "
    "8.18.4 and none of them survive here."
)
notes.append(
    "The upgrade did not move the answer, as far as I could check before the "
    "8.18.4 keg stopped loading (its libultrahdr had gone too). The 10-bit "
    "lossless fixture, the one this capture exists for, came out "
    "BYTE-IDENTICAL on both versions: sha256 "
    "053eadef26480dc8a24af96e654272d197157f5ddec68c490406a2c0392b2001. That "
    "is one datum, not a proof, and it is the only cross-version comparison "
    "I was able to run."
)
notes.append(
    "So this area records 8.18.6 while every pre-existing area in "
    "oracle-captures/ (convolution, foreign-radiance, foreign-webp, "
    "foreign-gif) records 8.18.4, as do the FITS, EXR and JPEG XL captures "
    "in flight alongside this one. Those numbers are still true of when "
    "those areas were taken; they are simply no longer true of what is "
    "installed. Reconciling that is tracked separately and deliberately not "
    "done here: this file states the truth about itself rather than "
    "inheriting a stale claim."
)
notes.append(
    "`vips heifsave --help` and `vips heifload --help` print the generic "
    "driver usage and exit 0. They are not an existence test and they are not "
    "an argument list. The bare `vips heifsave` invocation is, and its "
    "output is in build_and_argument_surface."
)
notes.append(
    "The heif codec is a DYNAMIC MODULE on this build. `--vips-config` says "
    "`true (dynamic module: true)` whether or not the module can actually "
    "dlopen, and when its x265 dependency went missing mid-session every heif "
    "operation vanished while the config line stayed exactly the same. Check "
    "that an operation RUNS, not that the config claims it."
)
notes.append(
    "Everything ran under `env -u VIPS_NOVECTOR`. Nothing in the heif path is "
    "vectorised, so it changes no number here, but an inherited empty "
    "VIPS_NOVECTOR would still have counted as set and it costs nothing to "
    "remove the question."
)
notes.append(
    "HEIC is out of scope for #605 and is not a target here. Where it would "
    "differ: `.heic` keeps the default `hevc` compression instead of being "
    "overridden to AV1 (see filename_decides_the_codec), its major brand is "
    "`heix` rather than "
    "`avif`, and its payload needs an HEVC decoder, which is the whole reason "
    "#498 closed. The heifload side -- carrier, left justify, band count, "
    "nclx handling -- is shared code and behaves the same."
)
notes.append(
    "`heifsave_buffer` and `heifsave_target` register only `.heic` and "
    "`.heif` (see build_and_argument_surface), and the `.avif` compression "
    "override lives in "
    "`vips_foreign_save_heif_file_build`, so it does not apply to them. "
    "Writing AVIF to a target needs an explicit `--compression av1`."
)

oracle = {
    "meta": {
        "area": "foreign-avif",
        "issues": [638, 605],
        "scope": "AVIF still images, load side. Not heifload parity: "
                 "heifload also reads HEVC, AVC and JPEG payloads and "
                 "heifsave writes HEVC by default. heifsave appears here "
                 "only as the thing that made the fixtures.",
        "vips_version": version,
        "vips_binary": VIPS,
        "libheif": keg("libheif"),
        "av1_encoder": keg("aom"),
        "version_provenance": "vips_version is `vips --version` at capture "
                              "time. `vips --vips-config` reports libheif as "
                              "`true (dynamic module: true)` with NO version "
                              "number, so libheif and av1_encoder are the "
                              "Homebrew kegs /opt/homebrew/opt/libheif and "
                              "/opt/homebrew/opt/aom resolve to, which is "
                              "what vips is linked against. See notes: this "
                              "machine's vips moved mid-capture and the "
                              "version here is measured, not inherited from "
                              "any brief.",
        "captured_by": "oracle-captures/foreign-avif/capture.py",
        "reference_c": "libvips v8.18.0-95-gfe420cf3a for the file and line "
                       "numbers quoted above; the binary every number here "
                       "came out of is a later release build and is not the "
                       "same artefact",
    },
    "notes": notes,
    "records": records,
}

INLINE = {}


def inline_scalars(obj):
    """Mark arrays of numbers so they serialise on ONE line. json.dumps with
    indent=2 puts a 36-sample pixel dump on 38 lines, which is three times
    the bytes and much harder to read than the row it represents.

    Every json.dumps below passes allow_nan=False, this one included: the
    inlined leaves are where a float actually lives, so guarding only the
    top-level dump would leave the pixel rows unguarded (#682)."""
    if isinstance(obj, dict):
        return {k: inline_scalars(v) for k, v in obj.items()}
    if isinstance(obj, list) and obj:
        flatish = all(isinstance(x, (int, float, bool)) for x in obj)
        rows = all(isinstance(x, list) and
                   all(isinstance(y, (int, float)) for y in x) for x in obj)
        if flatish or rows:
            key = f"\u0000{len(INLINE)}\u0000"
            INLINE[key] = (
                "[" + ", ".join(json.dumps(x, allow_nan=False)
                                for x in obj) + "]"
                if rows else json.dumps(obj, allow_nan=False))
            return key
        return [inline_scalars(x) for x in obj]
    return obj


text = json.dumps(inline_scalars(oracle), indent=2, allow_nan=False)
for key, value in INLINE.items():
    text = text.replace(json.dumps(key, allow_nan=False), value)
with open(os.path.join(ROOT, "oracle.json"), "w") as f:
    f.write(text + "\n")

with open(os.path.join(ROOT, "commands.sh"), "w") as f:
    f.write("#!/bin/sh\n")
    f.write("# Every command capture.py ran, in order. Regenerate with\n")
    f.write("# `python3 capture.py` from this directory.\n")
    f.write("#\n")
    f.write("# VIPS_NOVECTOR is UNSET rather than blanked: libvips tests\n")
    f.write("# whether the variable exists, so an empty value still counts\n")
    f.write("# as set. Nothing in the heif path is vectorised, but the\n")
    f.write("# habit costs nothing.\n")
    f.write("set -e\n\n")
    f.write(f'VIPS="env -u VIPS_NOVECTOR {VIPS}"\n')
    f.write(f'VIPSHEADER="env -u VIPS_NOVECTOR {VIPSHEADER}"\n\n')
    for c in COMMANDS:
        f.write(c + "\n")

fixtures = sorted(n for n in os.listdir(FIX) if n.endswith(".avif"))
total = sum(os.path.getsize(os.path.join(FIX, n)) for n in fixtures)
print(f"{len(records)} records, {len(COMMANDS)} commands, "
      f"{len(fixtures)} fixtures, {total} fixture bytes")
