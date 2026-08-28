#!/usr/bin/env python3
"""
Oracle capture for the FITS area (issue #505).

Runs the real vips 8.18.4 CLI, built with `cfitsio: true` against cfitsio
4.6.4, over deterministic fixtures and over hand-authored FITS files, and
records exactly what it does. `src/fits.rs` is pinned against these records
rather than against the FITS standard, because two of the three things the
module has to get right are not in the standard at all:

  * the vertical flip vips wraps the codec in (`vips_flip` in
    `foreign/fitsload.c`, and again before the write in
    `foreign/fitssave.c`), so the first row in a file is the bottom row of
    the image;
  * cfitsio's *equivalent* type, which is what decides the carrier a file
    loads to. `fits_get_img_equivtype` (called at `foreign/fits.c:246`)
    reports the type the BSCALE/BZERO-corrected values need, not the type
    the array is stored in, so BITPIX alone does not determine the answer.

The third is the generated header, which is written by cfitsio rather than
by vips, so the only way to know what it looks like is to save a file and
read the bytes back. That is what `header_cards` records.

Two probing notes, both recorded in `notes`:

 1. `vips fitsload --help` prints generic driver usage and exits 0, so it is
    not an existence test. `vips -l | grep -i fits` is, and it lists
    `fitsload`, `fitsload_source` and `fitssave`.
 2. `vips__fits_isfits` (`foreign/fits.c:526-548`) is not a magic-byte
    check: it opens the file with cfitsio and reports whether that worked.
    libviprs cannot reproduce that in a 16-byte sniff, so it matches the
    `SIMPLE  =` prefix the standard requires of the first card instead.
    `first_card_prefix` records what a real saved file opens with.

Writes:
  commands.sh  - every vips CLI command actually executed, in order
  oracle.json  - structured records

Re-running needs the vips binary at VIPS. Every input is generated from
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

BLOCK = 2880
CARD = 80

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


def vipsheader(*args, allow_fail=False):
    return run([VIPSHEADER, *args], allow_fail=allow_fail)


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def cards(path):
    """Every 80-column card in the first header unit, up to and including
    END, with trailing blanks trimmed."""
    with open(path, "rb") as f:
        blob = f.read()
    out = []
    for i in range(0, len(blob), CARD):
        text = blob[i:i + CARD].decode("ascii", "replace").rstrip(" ")
        out.append(text)
        if text == "END":
            break
    return out


def data_bytes(path, header_units=1):
    """The data segment, as hex, for a file whose header occupies
    `header_units` blocks."""
    with open(path, "rb") as f:
        blob = f.read()
    return blob[header_units * BLOCK:].hex()


def getpoint(path, w, h):
    """Every pixel, row-major, as `vips getpoint` reports it."""
    rows = []
    for y in range(h):
        row = []
        for x in range(w):
            proc = vips("getpoint", path, str(x), str(y))
            row.append([float(v) for v in proc.stdout.split()])
        rows.append(row)
    return rows


def header_fields(path):
    proc = vipsheader("-a", path, allow_fail=True)
    if proc.returncode != 0:
        return {"error": proc.stderr.strip()}
    fields = {}
    for line in proc.stdout.splitlines():
        if ": " in line:
            key, value = line.split(": ", 1)
            fields[key] = value
    return fields


# ---------------------------------------------------------------------------
# Hand-authored FITS, so the capture can pose questions vips cannot be asked
# to write: signed integers, 64-bit types, and scalings that move the
# cfitsio equivalent type.
# ---------------------------------------------------------------------------

def card(keyword, value=None, comment=None, raw=None):
    if raw is not None:
        return raw.ljust(CARD)[:CARD]
    text = keyword.ljust(8)
    if value is not None:
        text += "= " + str(value).rjust(20)
        if comment:
            text += " / " + comment
    return text.ljust(CARD)[:CARD]


def unit(card_list, payload):
    header = "".join(card_list) + card("END")
    header += " " * ((BLOCK - len(header) % BLOCK) % BLOCK)
    payload += b"\0" * ((BLOCK - len(payload) % BLOCK) % BLOCK)
    return header.encode("ascii") + payload


def simple(path, bitpix, naxes, payload, extra=()):
    body = [card("SIMPLE", "T"), card("BITPIX", bitpix), card("NAXIS", len(naxes))]
    for i, n in enumerate(naxes):
        body.append(card("NAXIS%d" % (i + 1), n))
    body.extend(extra)
    with open(path, "wb") as f:
        f.write(unit(body, payload))
    return path


W, H = 4, 3
RAMP = [(y * W + x) for y in range(H) for x in range(W)]

records = {}

# ---------------------------------------------------------------------------
# 1. What vips writes, for every carrier libviprs can reach.
# ---------------------------------------------------------------------------

# Deterministic inputs, in formats vips reads without any FITS involvement.
with open(os.path.join(FIX, "mono8.pgm"), "wb") as f:
    f.write(b"P5\n%d %d\n255\n" % (W, H) + bytes((v * 7) % 256 for v in RAMP))
with open(os.path.join(FIX, "rgb8.ppm"), "wb") as f:
    body = b"".join(bytes([(v * 3) % 256, (v * 5 + 1) % 256, (v * 7 + 2) % 256])
                    for v in RAMP)
    f.write(b"P6\n%d %d\n255\n" % (W, H) + body)
with open(os.path.join(FIX, "mono16.pgm"), "wb") as f:
    body = b"".join(struct.pack(">H", (v * 4097) % 65536) for v in RAMP)
    f.write(b"P5\n%d %d\n65535\n" % (W, H) + body)
with open(os.path.join(FIX, "rgba8.raw"), "wb") as f:
    f.write(b"".join(bytes([(v * 3) % 256, (v * 5 + 1) % 256,
                            (v * 7 + 2) % 256, (v * 11 + 3) % 256])
                     for v in RAMP))

saves = {}

for name, build in (
    ("mono_uchar", lambda dst: vips("copy", os.path.join(FIX, "mono8.pgm"), dst)),
    ("rgb_uchar", lambda dst: vips("copy", os.path.join(FIX, "rgb8.ppm"), dst)),
    ("mono_ushort", lambda dst: vips("copy", os.path.join(FIX, "mono16.pgm"), dst)),
):
    dst = os.path.join(OUT, name + ".fits")
    if os.path.exists(dst):
        os.unlink(dst)
    build(dst)
    saves[name] = dst

# 4-band uchar: rawload gives a multiband image with no interpretation guess.
dst = os.path.join(OUT, "rgba_uchar.fits")
if os.path.exists(dst):
    os.unlink(dst)
vips("rawload", os.path.join(FIX, "rgba8.raw"), os.path.join(OUT, "rgba8.v"),
     str(W), str(H), "4")
vips("copy", os.path.join(OUT, "rgba8.v"), dst)
saves["rgba_uchar"] = dst

# Float, one band and three.
for name, src in (("mono_float", "mono8.pgm"), ("rgb_float", "rgb8.ppm")):
    dst = os.path.join(OUT, name + ".fits")
    if os.path.exists(dst):
        os.unlink(dst)
    vips("cast", os.path.join(FIX, src), os.path.join(OUT, name + ".v"), "float")
    vips("copy", os.path.join(OUT, name + ".v"), dst)
    saves[name] = dst

for name, path in saves.items():
    records["save_" + name] = {
        "what": f"`vips fitssave` on a {name.replace('_', ' ')} image. The header "
                "is generated by cfitsio 4.6.4, not by vips, so these cards are "
                "what libviprs has to spell byte for byte.",
        "file": os.path.relpath(path, ROOT),
        "sha256": sha256(path),
        "size": os.path.getsize(path),
        "header_cards": cards(path),
        "data_hex": data_bytes(path)[:2 * 96],
        "loads_back_as": header_fields(path),
        "pixels": getpoint(path, W, H),
    }

records["save_mono_is_naxis_2"] = {
    "what": "A single-band image is written with NAXIS = 2 and no NAXIS3 "
            "(`vips_fits_set_header`, fits.c:716, the 6/1/23 ewelot change). "
            "Everything else gets NAXIS = 3 with the band count in NAXIS3.",
    "mono_naxis": [c for c in cards(saves["mono_uchar"]) if c.startswith("NAXIS")],
    "rgb_naxis": [c for c in cards(saves["rgb_uchar"]) if c.startswith("NAXIS")],
    "rgba_naxis": [c for c in cards(saves["rgba_uchar"]) if c.startswith("NAXIS")],
}

records["save_ushort_declares_bzero"] = {
    "what": "vips promotes short to ushort on save (`bandfmt_fits`, "
            "fitssave.c) and cfitsio writes USHORT_IMG as BITPIX 16 with "
            "BZERO 32768, which is the FITS standard's unsigned-16 "
            "convention. The stored samples are the unsigned values minus "
            "32768, big-endian.",
    "cards": [c for c in cards(saves["mono_ushort"])
              if c.startswith(("BITPIX", "BZERO", "BSCALE"))],
    "unsigned_samples": [(v * 4097) % 65536 for v in RAMP],
    "stored_hex": data_bytes(saves["mono_ushort"])[:2 * 24],
}

records["scan_order_is_bottom_up"] = {
    "what": "The first row in the file is the BOTTOM row of the image: "
            "`vips_foreign_load_fits_load` (fitsload.c) runs vips__fits_read "
            "then vips_flip(VIPS_DIRECTION_VERTICAL), and "
            "vips_foreign_save_fits_build (fitssave.c) flips before writing. "
            "The mono8 fixture's rows are 0,7,14,21 / 28,35,42,49 / "
            "56,63,70,77, and the data segment opens with the last of those.",
    "image_rows": [[(v * 7) % 256 for v in RAMP[r * W:(r + 1) * W]] for r in range(H)],
    "file_order_hex": data_bytes(saves["mono_uchar"])[:2 * 12],
}

records["bands_are_planes"] = {
    "what": "NAXIS3 is the band count and each band is a whole plane, which "
            "is why vips_fits_generate reads one band of one line at a time "
            "and scatters it (fits.c:456-507). The rgb fixture's red plane "
            "comes first, bottom row leading.",
    "red_plane_bottom_row": [(v * 3) % 256 for v in RAMP[2 * W:3 * W]],
    "file_order_hex": data_bytes(saves["rgb_uchar"])[:2 * 12],
}

# ---------------------------------------------------------------------------
# 2. What each BITPIX and scaling pair loads as. This is the carrier table.
# ---------------------------------------------------------------------------

signed = [v - 3 for v in RAMP]
bitpix_cases = {
    "bitpix_8": (8, [W, H], bytes((v + 3) & 0xFF for v in signed), ()),
    "bitpix_16_signed": (16, [W, H],
                         b"".join(struct.pack(">h", v) for v in signed), ()),
    "bitpix_16_unsigned": (16, [W, H],
                           b"".join(struct.pack(">h", v) for v in signed),
                           (card("BZERO", 32768), card("BSCALE", 1))),
    "bitpix_32_signed": (32, [W, H],
                         b"".join(struct.pack(">i", v * 100000) for v in signed), ()),
    "bitpix_32_unsigned": (32, [W, H],
                           b"".join(struct.pack(">i", v) for v in signed),
                           (card("BZERO", 2147483648), card("BSCALE", 1))),
    "bitpix_64": (64, [W, H],
                  b"".join(struct.pack(">q", v) for v in signed), ()),
    "bitpix_minus_32": (-32, [W, H],
                        b"".join(struct.pack(">f", float(v)) for v in signed), ()),
    "bitpix_minus_64": (-64, [W, H],
                        b"".join(struct.pack(">d", v * 1.5) for v in signed), ()),
    "bitpix_8_signed_byte": (8, [W, H], bytes((v + 3) & 0xFF for v in signed),
                             (card("BZERO", -128), card("BSCALE", 1))),
    "bitpix_8_rescaled": (8, [W, H], bytes((v + 3) & 0xFF for v in signed),
                          (card("BSCALE", 2), card("BZERO", 10))),
    "bitpix_minus_32_rescaled": (-32, [W, H],
                                 b"".join(struct.pack(">f", float(v)) for v in signed),
                                 (card("BSCALE", 2), card("BZERO", 10))),
}

carriers = {}
for name, (bitpix, naxes, payload, extra) in bitpix_cases.items():
    path = simple(os.path.join(FIX, name + ".fits"), bitpix, naxes, payload, extra)
    fields = header_fields(path)
    entry = {
        "bitpix": bitpix,
        "extra_cards": [c.rstrip() for c in extra],
        "vips": fields,
    }
    if "error" not in fields:
        entry["pixels"] = getpoint(path, W, H)
    carriers[name] = entry

records["carrier_table"] = {
    "what": "What each BITPIX and BSCALE/BZERO pair loads as. cfitsio's "
            "fits_get_img_equivtype reports the type the SCALED values need, "
            "so BITPIX alone does not decide the carrier: bitpix_8_rescaled "
            "is stored as bytes and loads as `short`, and "
            "bitpix_8_signed_byte resolves to SBYTE_IMG (10), which has no "
            "row in vips's table (fits.c:196-204) and is refused. "
            "bitpix_minus_32_rescaled shows that a float array keeps its "
            "carrier and has the scaling applied instead.",
    "cases": carriers,
}

# ---------------------------------------------------------------------------
# 3. Structural rules: axis counts, empty header units, header length.
# ---------------------------------------------------------------------------

structural = {}

path = simple(os.path.join(FIX, "naxis_1.fits"), 8, [W], bytes(range(W)))
structural["naxis_1"] = header_fields(path)

path = simple(os.path.join(FIX, "naxis_4_empty.fits"), 8, [W, H, 1, 1],
              bytes(range(W * H)))
structural["naxis_4_higher_axes_empty"] = header_fields(path)

path = simple(os.path.join(FIX, "naxis_4_full.fits"), 8, [W, H, 1, 2],
              bytes(range(W * H * 2)))
structural["naxis_4_higher_axis_of_2"] = header_fields(path)

path = simple(os.path.join(FIX, "naxis_11.fits"), 8, [W, H] + [1] * 9,
              bytes(range(W * H)))
structural["naxis_11"] = header_fields(path)

path = simple(os.path.join(FIX, "bands_5.fits"), 8, [W, H, 5],
              bytes(range(W * H * 5)))
structural["bands_5"] = header_fields(path)

records["structural_rules"] = {
    "what": "The axis rules from vips_fits_get_header (fits.c:260-291). "
            "NAXIS 1, 2 and 3 fall through into each other, 4 through 10 are "
            "accepted only when every axis above the third is exactly 1, and "
            "anything higher is `bad number of axis`.",
    "cases": structural,
}

# An empty primary unit followed by an IMAGE extension, which is the layout
# that makes the header walk at fits.c:223-239 load-bearing.
primary = unit([card("SIMPLE", "T"), card("BITPIX", 8), card("NAXIS", 0),
                card("EXTEND", "T")], b"")
extension = unit([card("XTENSION", "'IMAGE   '"), card("BITPIX", 8),
                  card("NAXIS", 2), card("NAXIS1", W), card("NAXIS2", H),
                  card("PCOUNT", 0), card("GCOUNT", 1)],
                 bytes((v + 3) & 0xFF for v in signed))
multi = os.path.join(FIX, "multi_unit.fits")
with open(multi, "wb") as f:
    f.write(primary + extension)

records["empty_primary_unit_is_walked_past"] = {
    "what": "A primary header unit with NAXIS = 0 carries no data, and vips "
            "walks forward to the next unit (fits.c:223-239). The `fits-` "
            "records that come back are the LOADED unit's, not the "
            "primary's, so the first one here is XTENSION rather than SIMPLE.",
    "file": os.path.relpath(multi, ROOT),
    "vips": header_fields(multi),
    "pixels": getpoint(multi, W, H),
}

# A header longer than one block, to pin that every card before END is
# attached and that the unit is a whole number of blocks.
long_cards = [card("SIMPLE", "T"), card("BITPIX", 8), card("NAXIS", 2),
              card("NAXIS1", W), card("NAXIS2", H)]
for i in range(40):
    long_cards.append(card("COMMENT", raw="COMMENT   filler line %d" % i))
long_path = os.path.join(FIX, "long_header.fits")
with open(long_path, "wb") as f:
    f.write(unit(long_cards, bytes((v + 3) & 0xFF for v in signed)))

long_fields = header_fields(long_path)
records["header_spans_whole_blocks"] = {
    "what": "A 45-card header occupies two 2880-byte blocks and every card "
            "before END is attached as fits-0 .. fits-44, in file order, "
            "with trailing blanks trimmed. END itself and the blank fill "
            "after it are not attached.",
    "file": os.path.relpath(long_path, ROOT),
    "size": os.path.getsize(long_path),
    "attached_count": len([k for k in long_fields if k.startswith("fits-")]),
    "first": long_fields.get("fits-0"),
    "last": long_fields.get("fits-44"),
    "past_last": long_fields.get("fits-45"),
}

# ---------------------------------------------------------------------------
# 4. The save-side refusal, and how the operations are actually named.
# ---------------------------------------------------------------------------

vips("copy", os.path.join(FIX, "mono8.pgm"), os.path.join(OUT, "cplx.v"))
proc = vips("cast", os.path.join(OUT, "cplx.v"), os.path.join(OUT, "cplx2.v"),
            "complex")
proc = vips("copy", os.path.join(OUT, "cplx2.v"),
            os.path.join(OUT, "cplx.fits"), allow_fail=True)
records["save_refuses_complex"] = {
    "what": "vips's own save-side refusal, from vips_fits_set_header "
            "(fits.c:724-728). Complex has no BITPIX, and vips does not "
            "promote it to one. libviprs has no complex carrier either, so "
            "this case cannot arise there.",
    "stderr": proc.stderr.strip(),
    "returncode": proc.returncode,
}

listing = subprocess.run([VIPS, "-l"], capture_output=True, text=True).stdout
COMMANDS.append("vips -l | grep -i fits")
records["operation_names"] = {
    "what": "`vips <op> --help` prints generic driver usage and exits 0 for "
            "some operations, so it is not an existence test. This is the "
            "registered-operation listing instead.",
    "lines": [line.strip() for line in listing.splitlines() if "fits" in line.lower()],
}

config = subprocess.run([VIPS, "--vips-config"], capture_output=True, text=True).stdout
COMMANDS.append("vips --vips-config | tr ',' '\\n' | grep -i cfitsio")
records["build_config"] = {
    "what": "This build really does have FITS compiled in, so it can serve "
            "as the oracle (runbook section 8).",
    "line": next((p.strip() for p in config.replace("\n", ",").split(",")
                  if "cfitsio" in p.lower()), None),
}

records["first_card_prefix"] = {
    "what": "What a saved file's first nine bytes are, which is the prefix "
            "crate::source::sniff matches. vips itself does not sniff: "
            "vips__fits_isfits (fits.c:526-548) opens the file with cfitsio "
            "and reports whether that worked.",
    "prefix": cards(saves["mono_uchar"])[0][:9],
}

version = subprocess.run([VIPS, "--version"], capture_output=True, text=True).stdout.strip()
oracle = {
    "meta": {
        "area": "foreign-fits",
        "issue": 505,
        "vips_version": version,
        "vips_binary": VIPS,
        "captured_by": "oracle-captures/foreign-fits/capture.py",
        "reference_c": "libvips v8.18.0-95-gfe420cf3a for line numbers; the "
                       "binary is the 8.18.4 release and is not the same "
                       "artefact, so every number here comes from the binary",
        "cfitsio_version": "4.6.4 (Homebrew), which is what generates every "
                           "header card recorded under save_*",
    },
    "notes": [
        "`vips fitsload --help` prints generic driver usage and exits 0, so "
        "it is not an existence test; operation_names has the `vips -l` "
        "listing instead (runbook section 8).",
        "The header cards under save_* are written by cfitsio, not by vips. "
        "vips filters them back out on the way in (`vips_fits_basic`, "
        "fits.c:596-613) precisely because cfitsio regenerates them, so a "
        "libviprs save has to spell them the same way or a round trip stops "
        "being byte-exact.",
        "cfitsio's fits_get_img_equivtype reports the type the BSCALE/BZERO-"
        "corrected values need rather than the stored type, so the carrier a "
        "file loads to is not a function of its BITPIX alone. carrier_table "
        "is the measured mapping.",
        "getpoint values are printed row-major in IMAGE order, which is the "
        "reverse of the order the rows appear in the file.",
    ],
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
    f.write("set -e\n\n")
    for c in COMMANDS:
        f.write(c + "\n")

print(f"wrote oracle.json ({len(records)} records) and commands.sh "
      f"({len(COMMANDS)} commands)")
