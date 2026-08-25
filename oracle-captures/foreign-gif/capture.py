#!/usr/bin/env python3
"""
Oracle capture for the still-image GIF area (issues #570, #571).

Runs the real vips 8.18.4 CLI over generated sources and over the reference
suite's GIFs, and records exactly what it does. What `src/gif.rs` is pinned
against lives here, because none of it can be derived by reading the GIF89a
spec: the rules belong to libvips' loader and to cgif plus libimagequant on
the save side.

The five things this capture exists to settle:

 1. `gifload` emits THREE bands unless some frame declares a transparent
    index (`nsgifload.c:271`, `:431-432`), and a frame whose pixel data runs
    off the end of the file counts as declaring one, because the rows that
    never arrived stay uncomposited.
 2. The canvas outside frame 0's rectangle is transparent black and NOT the
    background colour, even though the header reports one.
 3. `gifsave` caps the palette at `min(255, 1 << bitdepth)`
    (`cgifsave.c:795-796`), and reserves an index for transparency whenever
    the colours do not fill the cap -- so an opaque source with palette
    headroom reloads as FOUR bands.
 4. Alpha is thresholded at 128 and the pixel below it is zeroed entirely
    (`cgifsave.c:538-548`).
 5. `--interlace` reorders the stored rows into GIF's four passes; the
    order is recovered here by LZW-decoding what vips wrote.

One trap this capture deliberately avoids, recorded in `notes`: the existing
`oracle-captures/foreign/` capture tags `loads["cogs.gif"]` with
`"lossy_decoder": true`. GIF's LZW is exactly lossless and fully
deterministic, and the tag would let a real palette-expansion or disposal bug
pass as decoder drift. Every load record here is compared byte for byte
instead.

Writes:
  commands.sh  - every vips CLI command actually executed, in order
  oracle.json  - structured records

Re-running needs the vips binary at VIPS and the reference GIFs under
REFERENCE; every other input is generated from scratch, deterministically.
Nothing outside this script's own directory is written.
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
REFERENCE = ("/Users/rom/workspace/libviprs/libviprs-tests/tmp/"
             "libvips-reference-tests/test-suite/images")
REFERENCE_GIFS = [
    "cogs", "cramps", "trans-x", "truncated", "garden",
    "dispose-background", "dispose-previous", "invalid_multiframe",
]

os.makedirs(FIX, exist_ok=True)
os.makedirs(OUT, exist_ok=True)

COMMANDS = []


def run(args, allow_fail=False):
    """Run a vips command, logging it for commands.sh."""
    COMMANDS.append(" ".join(a.replace(ROOT + "/", "") for a in args))
    proc = subprocess.run(args, capture_output=True, text=True)
    if proc.returncode != 0 and not allow_fail:
        raise SystemExit(f"failed: {args}\n{proc.stderr}")
    return proc.stdout.strip()


def header(path, field=None):
    args = [VIPSHEADER, "-a", path] if field is None else [VIPSHEADER, "-f", field, path]
    out = run(args, allow_fail=True)
    if field is not None:
        return out
    fields = {}
    for line in out.splitlines()[1:]:
        if ": " in line:
            name, value = line.split(": ", 1)
            fields[name.strip()] = value.strip()
    return fields


def raw(vfile):
    """`vips rawsave` a `.v` and return the pixel bytes."""
    out = vfile.replace(".v", ".raw")
    run([VIPS, "rawsave", vfile, out])
    with open(out, "rb") as fh:
        return fh.read()


def write_raw(name, width, height, bands, pixels):
    """Write a raw plane and turn it into an sRGB `.v` vips can save from."""
    path = os.path.join(FIX, name + ".raw")
    with open(path, "wb") as fh:
        fh.write(bytes(pixels))
    plain = os.path.join(OUT, name + "-plain.v")
    tagged = os.path.join(OUT, name + ".v")
    run([VIPS, "rawload", path, plain, str(width), str(height), str(bands),
         "--format", "uchar"])
    run([VIPS, "copy", plain, tagged, "--interpretation", "srgb"])
    return tagged


# --------------------------------------------------------------------------
# Sources
# --------------------------------------------------------------------------

def cycling(width, height, unique):
    """`width * height` pixels cycling through `unique` distinct colours."""
    out = bytearray()
    for i in range(width * height):
        c = i % unique
        out += bytes([(c * 7) % 256, (c * 13) % 256, (c * 29) % 256])
    return out


def gradient(width, height):
    """Pixel (x, y) is `[x * 5, y * 8, (x + y) * 3]`: every pixel distinct,
    every neighbour close, so the quantiser's error is visible."""
    out = bytearray()
    for y in range(height):
        for x in range(width):
            out += bytes([x * 5, y * 8, (x + y) * 3])
    return out


def alpha_ramp(width, height):
    """One colour under an alpha ramp that crosses the 128 threshold."""
    out = bytearray()
    for _ in range(height):
        for x in range(width):
            out += bytes([200, 100, 50, min(255, x * 8)])
    return out


def rows(height, width):
    """Row `y` is colour index `y`, so a stored row order is readable."""
    out = bytearray()
    for y in range(height):
        for _ in range(width):
            out += bytes([y * 30, 255 - y * 30, (y * 17) % 256])
    return out


# --------------------------------------------------------------------------
# GIF wire parsing, so the records are about bytes and not about vipsheader
# --------------------------------------------------------------------------

def parse(path):
    """The header, global colour table and first frame of a GIF."""
    with open(path, "rb") as fh:
        b = fh.read()
    width, height, packed, background, aspect = struct.unpack_from("<HHBBB", b, 6)
    out = {
        "bytes": len(b),
        "magic": b[:6].decode("ascii"),
        "screen": [width, height],
        "global_colour_table_entries": (2 << (packed & 7)) if packed & 0x80 else 0,
        "background_index": background,
        "aspect_byte": aspect,
    }
    p = 13 + 3 * out["global_colour_table_entries"]
    frames = []
    gce = None
    while p < len(b):
        block = b[p]
        if block == 0x21:
            label = b[p + 1]
            p += 2
            subs = []
            while p < len(b) and b[p] != 0:
                n = b[p]
                subs.append(b[p + 1:p + 1 + n])
                p += 1 + n
            p += 1
            if label == 0xF9 and subs:
                d = subs[0]
                gce = {
                    "disposal": (d[0] >> 2) & 7,
                    "transparent": bool(d[0] & 1),
                    "transparent_index": d[3],
                    "delay_cs": struct.unpack_from("<H", d, 1)[0],
                }
            elif label == 0xFF and subs:
                out["application"] = subs[0][:11].decode("latin1")
                if len(subs) > 1 and len(subs[1]) >= 3:
                    out["netscape_loop_count"] = subs[1][1] | (subs[1][2] << 8)
        elif block == 0x2C:
            if p + 10 > len(b):
                frames.append({"truncated": "image descriptor"})
                break
            left, top, fw, fh, flags = struct.unpack_from("<HHHHB", b, p + 1)
            frame = {
                "rect": [left, top, fw, fh],
                "interlaced": bool(flags & 0x40),
                "local_colour_table_entries": (2 << (flags & 7)) if flags & 0x80 else 0,
                "gce": gce,
            }
            p += 10 + 3 * frame["local_colour_table_entries"]
            if p >= len(b):
                frames.append(dict(frame, truncated="lzw minimum code size"))
                break
            frame["lzw_minimum_code_size"] = b[p]
            p += 1
            data = bytearray()
            while p < len(b) and b[p] != 0:
                n = b[p]
                data += b[p + 1:p + 1 + n]
                p += 1 + n
            if p >= len(b):
                frames.append(dict(frame, truncated="image data"))
                break
            p += 1
            frame["lzw_bytes"] = len(data)
            frame["lzw"] = bytes(data)
            frames.append(frame)
            gce = None
        elif block == 0x3B:
            break
        else:
            out["unexpected_block"] = hex(block)
            break
    out["frames_found"] = len(frames)
    out["first_frame"] = {k: v for k, v in frames[0].items() if k != "lzw"} if frames else None
    out["_frames"] = frames
    return out


def lzw_decode(data, min_code_size):
    """Enough of GIF LZW to read the stored row order back out."""
    clear = 1 << min_code_size
    end = clear + 1
    size = min_code_size + 1
    table = {i: bytes([i]) for i in range(clear)}
    nxt = end + 1
    out = bytearray()
    prev = None
    bit = 0
    while bit + size <= len(data) * 8:
        i = bit // 8
        chunk = int.from_bytes(data[i:i + 3].ljust(3, b"\0"), "little")
        code = (chunk >> (bit % 8)) & ((1 << size) - 1)
        bit += size
        if code == clear:
            table = {i: bytes([i]) for i in range(clear)}
            nxt = end + 1
            size = min_code_size + 1
            prev = None
            continue
        if code == end:
            break
        if code in table:
            entry = table[code]
        elif prev is not None:
            entry = prev + prev[:1]
        else:
            break
        out += entry
        if prev is not None:
            table[nxt] = prev + entry[:1]
            nxt += 1
            if nxt == (1 << size) and size < 12:
                size += 1
        prev = entry
    return bytes(out)


def diff(a, b):
    """Mean and maximum absolute per-byte difference."""
    assert len(a) == len(b), (len(a), len(b))
    worst = max((abs(x - y) for x, y in zip(a, b)), default=0)
    mean = sum(abs(x - y) for x, y in zip(a, b)) / max(1, len(a))
    return {"avg_abs_diff": round(mean, 6), "max_abs_diff": worst}


def rgb_only(pixels, count, bands):
    """Drop the alpha band so a 3-band source and a 4-band reload compare."""
    if bands == 3:
        return pixels
    return b"".join(pixels[i * 4:i * 4 + 3] for i in range(count))


# --------------------------------------------------------------------------
# Records
# --------------------------------------------------------------------------

record = {
    "vips_version": run([VIPS, "--version"]),
    "loads": {},
    "saves": {},
    "notes": [],
}

# --- load: every reference GIF, header and pixels -------------------------
for name in REFERENCE_GIFS:
    src = os.path.join(REFERENCE, name + ".gif")
    if not os.path.exists(src):
        continue
    fields = header(src)
    loaded = os.path.join(OUT, "ref-" + name + ".v")
    run([VIPS, "gifload", src, loaded], allow_fail=True)
    pixels = raw(loaded)
    wire = parse(src)
    record["loads"][name + ".gif"] = {
        "bands": int(fields["bands"]),
        "width": int(fields["width"]),
        "height": int(fields["height"]),
        "interpretation": fields["interpretation"],
        "n-pages": int(fields["n-pages"]),
        "loop": int(fields["loop"]),
        "bits-per-sample": int(fields["bits-per-sample"]),
        "interlaced": fields.get("interlaced"),
        "netscape_loop_count": wire.get("netscape_loop_count"),
        "frames_on_the_wire": wire["frames_found"],
        "first_frame": wire["first_frame"],
        "frame_zero_pixel_sha256": hashlib.sha256(pixels).hexdigest(),
        "frame_zero_bytes": len(pixels),
        # GIF's LZW is exactly lossless: this hash is an equality, not a
        # tolerance band. See notes.
        "lossy_decoder": False,
    }

# --- load: the canvas outside frame zero ----------------------------------
# 8x8 screen, background index 2 (blue), one opaque 4x4 red frame at (2,2).
inset = bytearray(b"GIF89a")
inset += struct.pack("<HHBBB", 8, 8, 0x80 | 0x01, 2, 0)
for colour in [(0, 0, 0), (255, 0, 0), (0, 0, 255), (0, 255, 0)]:
    inset += bytes(colour)
inset += bytes([0x21, 0xF9, 4, 0x00, 0, 0, 0, 0])
inset += bytes([0x2C]) + struct.pack("<HHHHB", 2, 2, 4, 4, 0x00)


def lzw_literal(indices, min_code_size):
    """Literal-only LZW: a clear code, one code per pixel, then EOI."""
    clear = 1 << min_code_size
    size = min_code_size + 1
    nxt = clear + 2
    acc = 0
    bits = 0
    out = bytearray()

    def emit(code):
        nonlocal acc, bits
        acc |= code << bits
        bits += size
        while bits >= 8:
            out.append(acc & 0xFF)
            acc >>= 8
            bits -= 8

    emit(clear)
    for index in indices:
        emit(index)
        nxt += 1
        if nxt > (1 << size) and size < 12:
            size += 1
    emit(clear + 1)
    if bits:
        out.append(acc & 0xFF)
    return bytes(out)


payload = lzw_literal([1] * 16, 2)
inset += bytes([2])
for i in range(0, len(payload), 255):
    chunk = payload[i:i + 255]
    inset += bytes([len(chunk)]) + chunk
inset += bytes([0, 0x3B])
inset_path = os.path.join(FIX, "inset.gif")
with open(inset_path, "wb") as fh:
    fh.write(inset)
fields = header(inset_path)
run([VIPS, "gifload", inset_path, os.path.join(OUT, "inset.v")])
inset_pixels = raw(os.path.join(OUT, "inset.v"))
bands = int(fields["bands"])
record["loads"]["inset.gif"] = {
    "bands": bands,
    "background": fields["background"],
    "gif-palette": fields["gif-palette"],
    "bits-per-sample": int(fields["bits-per-sample"]),
    "corner_pixel": list(inset_pixels[:bands]),
    "inset_pixel": list(inset_pixels[(2 * 8 + 2) * bands:(2 * 8 + 2) * bands + bands]),
    "note": ("the corner is transparent black even though the header reports "
             "a blue background, and the file is 3 bands because no frame "
             "declares transparency"),
}

# --- load: truncation makes a file transparent ----------------------------
whole = rows(8, 8)
whole_v = write_raw("rows8", 8, 8, 3, whole)
whole_gif = os.path.join(OUT, "rows8.gif")
run([VIPS, "gifsave", whole_v, whole_gif, "--bitdepth", "3"])
with open(whole_gif, "rb") as fh:
    whole_bytes = fh.read()
clipped = os.path.join(FIX, "rows8-truncated.gif")
with open(clipped, "wb") as fh:
    fh.write(whole_bytes[:-6])
record["loads"]["truncation_flips_the_band_count"] = {
    "intact_bands": int(header(whole_gif)["bands"]),
    "truncated_bands": int(header(clipped)["bands"]),
    "note": ("libnsgif reports a frame whose data runs out as transparent, "
             "so the rows that never arrived stay uncomposited; vips warns "
             "rather than failing because fail-on defaults to none"),
}

# --- save: bitdepth sizes the colour table --------------------------------
source = write_raw("cycle256", 16, 16, 3, cycling(16, 16, 256))
bitdepths = {}
for bitdepth in range(1, 9):
    path = os.path.join(OUT, f"bitdepth{bitdepth}.gif")
    run([VIPS, "gifsave", source, path, "--bitdepth", str(bitdepth)])
    wire = parse(path)
    bitdepths[bitdepth] = {
        "global_colour_table_entries": wire["global_colour_table_entries"],
        "lzw_minimum_code_size": wire["first_frame"]["lzw_minimum_code_size"],
        "transparent": wire["first_frame"]["gce"]["transparent"],
        "bytes": wire["bytes"],
    }
record["saves"]["bitdepth"] = bitdepths

# --- save: when a transparent index is reserved ---------------------------
reserve = {}
for bitdepth, unique in [(8, 2), (8, 16), (8, 254), (8, 255), (8, 256),
                         (1, 2), (2, 2), (2, 8), (4, 8), (4, 100),
                         (6, 8), (6, 100)]:
    src = write_raw(f"cycle{unique}", 16, 16, 3, cycling(16, 16, unique))
    path = os.path.join(OUT, f"reserve-{bitdepth}-{unique}.gif")
    run([VIPS, "gifsave", src, path, "--bitdepth", str(bitdepth)])
    wire = parse(path)
    reserve[f"bitdepth{bitdepth}_unique{unique}"] = {
        "global_colour_table_entries": wire["global_colour_table_entries"],
        "transparent": wire["first_frame"]["gce"]["transparent"],
        "transparent_index": wire["first_frame"]["gce"]["transparent_index"],
        "reload_bands": int(header(path)["bands"]),
    }
record["saves"]["transparent_index_reservation"] = reserve

# --- save: alpha threshold ------------------------------------------------
alpha_src = write_raw("alpha", 32, 24, 4, alpha_ramp(32, 24))
alpha_gif = os.path.join(OUT, "alpha.gif")
run([VIPS, "gifsave", alpha_src, alpha_gif])
run([VIPS, "gifload", alpha_gif, os.path.join(OUT, "alpha-reload.v")])
alpha_back = raw(os.path.join(OUT, "alpha-reload.v"))
record["saves"]["alpha_threshold"] = {
    "reload_bands": int(header(alpha_gif)["bands"]),
    "row0": [list(alpha_back[x * 4:x * 4 + 4]) for x in range(32)],
    "source_alpha": [min(255, x * 8) for x in range(32)],
    "note": ("cgifsave.c:538-548 promotes alpha >= 128 to 255 and zeroes the "
             "whole pixel below it, colour included"),
}

# --- save: interlace row order --------------------------------------------
rows_v = write_raw("rows8b", 8, 8, 3, rows(8, 8))
progressive = os.path.join(OUT, "rows-progressive.gif")
woven = os.path.join(OUT, "rows-interlaced.gif")
run([VIPS, "gifsave", rows_v, progressive, "--bitdepth", "3"])
run([VIPS, "gifsave", rows_v, woven, "--interlace", "--bitdepth", "3"])


def stored_rows(path):
    wire = parse(path)
    frame = wire["_frames"][0]
    indices = lzw_decode(frame["lzw"], frame["lzw_minimum_code_size"])
    return [indices[r * 8] for r in range(8)]


plain_order = stored_rows(progressive)
woven_order = stored_rows(woven)
# Palette indices are the quantiser's choice, so map them back to source rows
# through the progressive file, where stored row r holds source row r.
source_of = {palette_index: r for r, palette_index in enumerate(plain_order)}
record["saves"]["interlace"] = {
    "progressive_flag": parse(progressive)["first_frame"]["interlaced"],
    "interlaced_flag": parse(woven)["first_frame"]["interlaced"],
    "stored_source_rows": [source_of[i] for i in woven_order],
    "note": "GIF's four passes: rows 0,8,..; 4,12,..; 2,6,..; then every odd row",
}
run([VIPS, "gifload", progressive, os.path.join(OUT, "rows-p.v")])
run([VIPS, "gifload", woven, os.path.join(OUT, "rows-i.v")])
record["saves"]["interlace"]["reload_identical"] = (
    raw(os.path.join(OUT, "rows-p.v")) == raw(os.path.join(OUT, "rows-i.v"))
)

# --- save: dither ---------------------------------------------------------
grad_small = write_raw("greyramp", 32, 24,
                       3, bytes(v for _ in range(24) for x in range(32)
                                for v in [x * 255 // 31] * 3))
dither = {}
undithered = None
for level in ["0", "0.25", "0.5", "0.75", "1"]:
    path = os.path.join(OUT, f"dither{level}.gif")
    run([VIPS, "gifsave", grad_small, path, "--dither", level, "--bitdepth", "2"])
    run([VIPS, "gifload", path, os.path.join(OUT, f"dither{level}.v")])
    pixels = raw(os.path.join(OUT, f"dither{level}.v"))
    if undithered is None:
        undithered = pixels
    changed = sum(1 for i in range(0, len(pixels), 3) if pixels[i] != undithered[i])
    dither[level] = {"pixels_changed_vs_dither_0": changed}
record["saves"]["dither"] = {
    "levels": dither,
    "total_pixels": 32 * 24,
    "note": ("not monotone in pixels changed, so only the dither == 0 "
             "identity is worth pinning as an equality"),
}

# --- save: quantisation error, the one real divergence --------------------
quant = {}
for name, width, height, pixels in [("gradient48x32", 48, 32, gradient(48, 32)),
                                    ("cycle768", 32, 24, cycling(32, 24, 768))]:
    src = write_raw(name, width, height, 3, pixels)
    path = os.path.join(OUT, name + ".gif")
    run([VIPS, "gifsave", src, path])
    run([VIPS, "gifload", path, os.path.join(OUT, name + "-reload.v")])
    back = raw(os.path.join(OUT, name + "-reload.v"))
    bands = int(header(path)["bands"])
    quant[name] = dict(
        diff(bytes(pixels), rgb_only(back, width * height, bands)),
        reload_bands=bands,
        distinct_colours=len({bytes(pixels[i * 3:i * 3 + 3])
                              for i in range(width * height)}),
    )
record["saves"]["quantisation_error"] = {
    "sources": quant,
    "note": ("vips quantises with libimagequant and libviprs with median cut, "
             "so the palettes differ by construction; these are the numbers "
             "src/gif.rs is required to stay in band with"),
}

record["notes"] = [
    "GIF's LZW is exactly lossless and fully deterministic in both "
    "directions. The existing oracle-captures/foreign/ capture tags "
    "loads[\"cogs.gif\"] with \"lossy_decoder\": true, which is wrong and "
    "worse than cosmetic: it would let a real palette-expansion, disposal or "
    "blending bug pass as decoder drift. Every load record here is a "
    "byte-exact pixel hash.",
    "Do not assert on the fourth value vips getpoint prints for a 3-band "
    "image. getpoint.c:105 sizes its output array from the CODED band count "
    "while vips_image_decode has already reduced the buffer, so the extra "
    "value is an out-of-bounds read. No record here uses getpoint.",
    "gifsave never writes a 256-colour palette: cgifsave.c:795-796 caps at "
    "min(255, 1 << bitdepth) to keep one index free for transparency, so a "
    "256-colour source is lossy for vips too.",
    "loop is not the NETSCAPE count. No application extension means loop 1, "
    "a stored count of 0 means loop 0 (forever), a stored count of n means "
    "loop n + 1.",
]

with open(os.path.join(ROOT, "oracle.json"), "w") as fh:
    json.dump(record, fh, indent=2, sort_keys=True)
    fh.write("\n")
with open(os.path.join(ROOT, "commands.sh"), "w") as fh:
    fh.write("#!/bin/sh\n")
    fh.write("# Every vips command capture.py ran, in order. Regenerate with\n")
    fh.write("# `python3 capture.py` from this directory.\nset -e\n\n")
    fh.write("\n".join(COMMANDS))
    fh.write("\n")
print(f"wrote oracle.json ({len(COMMANDS)} vips commands)")
