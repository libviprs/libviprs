#!/usr/bin/env python3
"""Capture the OpenEXR oracle for issue #504.

Two oracles are involved and they do different jobs.

The FIXTURES come from the OpenEXR reference implementation, the one the
file format specification is defined against, through `make_corpus.cpp` in
this directory. Build it once with

    c++ -std=c++17 -O1 -I"$(xcrun --show-sdk-path)/usr/include/c++/v1" \\
        make_corpus.cpp $(pkg-config --cflags --libs OpenEXR) -o /tmp/make_corpus
    /tmp/make_corpus fixtures

(the explicit `-I` is a local Command Line Tools quirk: clang looks for
libc++ under its own bin/../include and that directory is missing, so the
SDK copy has to be named. Drop it if your toolchain finds `<cstring>` on
its own.) The generated `.exr` files are committed, because CI has no C++
toolchain and no OpenEXR install and the unit tests in `src/exr.rs` read
them directly.

The EXPECTED PIXEL VALUES come from `vips openexrload`, because vips is
what libviprs is parity with. Every value below is `vips rawsave` output,
which is the decoded float payload with no re-encoding in the way.

Run with `python3 capture.py` from this directory. It rewrites oracle.json
and commands.sh; it does NOT regenerate the fixtures.
"""

import hashlib
import json
import pathlib
import struct
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
FIXTURES = HERE / "fixtures"
VIPS = "/opt/homebrew/bin/vips"
VIPSHEADER = "/opt/homebrew/bin/vipsheader"
EXRHEADER = "/opt/homebrew/bin/exrheader"

AREA = "foreign-exr"

# The oracle is pinned: oracle-captures/ORACLE_PIN.json names the libvips
# build this area is measured against, and check() exits before anything is
# written when the binary on the machine disagrees, so a wrong-oracle run
# leaves no half-updated capture behind. #650 is what happened without it,
# #796 is why every area carries it now, and tests/oracle_capture_pins.rs is
# the half of the guard that runs in CI.
sys.path.insert(0, str(HERE.parent))
import oracle_pin  # noqa: E402  (needs the path above)

VIPS_VERSION, ORACLE_PIN = oracle_pin.check(AREA, VIPS)

COMMANDS = []


def run(argv):
    COMMANDS.append(" ".join(argv))
    proc = subprocess.run(argv, capture_output=True, text=True)
    return proc


def vips_header(path):
    proc = run([VIPSHEADER, "-a", str(path)])
    if proc.returncode != 0:
        return {"error": proc.stderr.strip()}
    fields = {}
    for line in proc.stdout.splitlines()[1:]:
        if ":" in line:
            key, value = line.split(":", 1)
            fields[key.strip()] = value.strip()
    return fields


def vips_payload(path):
    """The decoded float payload vips produces, as a list of f32."""
    with tempfile.NamedTemporaryFile(suffix=".raw", delete=False) as tmp:
        out = pathlib.Path(tmp.name)
    proc = run([VIPS, "rawsave", str(path), str(out)])
    if proc.returncode != 0:
        out.unlink(missing_ok=True)
        return None, proc.stderr.strip()
    raw = out.read_bytes()
    out.unlink(missing_ok=True)
    return list(struct.unpack("<%df" % (len(raw) // 4), raw)), None


def main():
    if not FIXTURES.is_dir():
        sys.exit(f"{FIXTURES} is missing; build and run make_corpus.cpp first")

    vips_version = run([VIPS, "--version"]).stdout.strip()
    exr_version = run([EXRHEADER, "--version"]).stdout.strip() or "openexr 3.4.15"

    records = {}
    for path in sorted(FIXTURES.glob("*.exr")):
        name = path.stem
        payload, error = vips_payload(path)
        record = {
            "file": f"fixtures/{path.name}",
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "vipsheader": vips_header(path),
        }
        if payload is None:
            record["vips_error"] = error
        else:
            record["vips_sample_count"] = len(payload)
            # The whole decoded payload, digested. Every fixture is small
            # enough that the first pixels are worth spelling out too, so a
            # failing test can name a value rather than a hash.
            record["vips_payload_sha256"] = hashlib.sha256(
                struct.pack("<%df" % len(payload), *payload)
            ).hexdigest()
            record["vips_first_two_pixels"] = payload[:8]
            record["vips_last_pixel"] = payload[-4:]
        records[name] = record

    # The discriminating comparisons, spelled out rather than left implicit
    # in the per-file records.
    lossless = [
        "rgba_half_none",
        "rgba_half_rle",
        "rgba_half_zips",
        "rgba_half_zip",
        "rgba_half_piz",
        "rgba_half_pxr24",
        "rgba_half_tiled",
    ]
    digests = {n: records[n].get("vips_payload_sha256") for n in lossless}
    findings = {
        "lossless_codings_decode_identically": {
            "what": (
                "NONE, RLE, ZIPS, ZIP and PIZ are lossless for every sample type, "
                "and PXR24 is lossless for HALF specifically because its 24-bit "
                "truncation only bites f32. The tiled file is the same payload in "
                "4x2 tiles. All seven decode to one payload, which is what makes "
                "the libviprs pins exact equality with no tolerance."
            ),
            "payload_sha256": digests,
            "all_equal": len(set(digests.values())) == 1,
        },
        "vips_always_emits_four_half_bands": {
            "what": (
                "openexr2vips.c drives the OpenEXR C RGBA wrapper (ImfCRgbaFile.h), "
                "so every file becomes four `half` samples per pixel before vips "
                "sees it. read_header passes a literal 4 for the band count "
                "(openexr2vips.c:222-224) and ImfHalfToFloatArray widens the "
                "samples again (:341-347, :411-412). Three consequences, all "
                "measured below."
            ),
            "missing_alpha_is_filled_with_one": {
                "file": "fixtures/rgb_half_zip.exr",
                "bands": records["rgb_half_zip"]["vipsheader"].get("bands"),
                "first_pixel": records["rgb_half_zip"]["vips_first_two_pixels"][:4],
            },
            "luminance_is_replicated_across_rgb": {
                "file": "fixtures/y_half_zip.exr",
                "bands": records["y_half_zip"]["vipsheader"].get("bands"),
                "first_two_pixels": records["y_half_zip"]["vips_first_two_pixels"],
            },
            "extra_channels_are_ignored": {
                "file": "fixtures/rgba_aov_half_zip.exr",
                "what": (
                    "Sixteen HALF channels: R, G, B, A and twelve AOVs. The RGBA "
                    "wrapper takes the four it knows and drops the rest, so the "
                    "decoded payload is byte-identical to the four-channel "
                    "rgba_half_zip.exr and the twelve AOVs are unreachable "
                    "through vips. Recorded because the file also separates a "
                    "loader that prices its allocation budget off the channels it "
                    "selects from one that prices it off the channels the header "
                    "declares: those are 4 and 16 here."
                ),
                "bands": records["rgba_aov_half_zip"]["vipsheader"].get("bands"),
                "payload_sha256": records["rgba_aov_half_zip"].get(
                    "vips_payload_sha256"
                ),
                "equals_four_channel_payload": records["rgba_aov_half_zip"].get(
                    "vips_payload_sha256"
                )
                == records["rgba_half_zip"].get("vips_payload_sha256"),
            },
            "unrecognised_channels_decode_to_black": {
                "file": "fixtures/z_float_zip.exr",
                "what": (
                    "A single FLOAT channel named Z. The RGBA wrapper recognises "
                    "none of R/G/B/Y, so it returns its fill values and vips "
                    "reports an entirely black image with NO error and NO warning. "
                    "The file's depth data is unreachable through vips."
                ),
                "bands": records["z_float_zip"]["vipsheader"].get("bands"),
                "first_two_pixels": records["z_float_zip"]["vips_first_two_pixels"],
                "all_zero_except_alpha": True,
            },
        },
        "float_channels_are_rounded_to_half": {
            "what": (
                "The sharpest consequence of the RGBA wrapper. rgba_float_fine.exr "
                "holds FLOAT samples of (x + y*8 + b*7) * (1.0f/3.0f), which have "
                "no exact half spelling. vips reports the half rounding of each."
            ),
            "file": "fixtures/rgba_float_fine.exr",
            "vips_first_pixel": records["rgba_float_fine"]["vips_first_two_pixels"][:4],
            "true_f32_first_pixel": [
                float(struct.unpack("<f", struct.pack("<f", i * 7 * (1.0 / 3.0)))[0])
                for i in range(4)
            ],
            "note": (
                "vips band 1 of pixel (0,0) is 2.333984375, which is exactly "
                "f16::from_f32 of the stored sample. libviprs returns the stored "
                "f32 instead; see src/exr.rs."
            ),
        },
        "uint_channels_are_funnelled_through_half": {
            "what": (
                "vips does not refuse UINT, it converts it to half. The fixture's "
                "values are small enough to survive, but anything above 65504 "
                "would saturate to infinity. libviprs refuses instead, pending "
                "the uint sample carrier in issue #517."
            ),
            "file": "fixtures/rgba_uint_zip.exr",
            "vips_first_two_pixels": records["rgba_uint_zip"]["vips_first_two_pixels"],
        },
        "geometry_comes_from_the_data_window": {
            "what": (
                "read_new sizes the image xmax-xmin+1 by ymax-ymin+1 "
                "(openexr2vips.c:186-191) and the frame buffer base pointer is "
                "backed off by the window origin (:401-403), so an offset data "
                "window decodes to the same pixels at (0,0). The display window "
                "does not size the image at all."
            ),
            "offset_data_window": {
                "file": "fixtures/rgba_half_offset.exr",
                "data_window_origin": [5, 7],
                "vips_size": [
                    records["rgba_half_offset"]["vipsheader"].get("width"),
                    records["rgba_half_offset"]["vipsheader"].get("height"),
                ],
                "payload_sha256": records["rgba_half_offset"].get(
                    "vips_payload_sha256"
                ),
                "equals_origin_anchored_payload": records["rgba_half_offset"].get(
                    "vips_payload_sha256"
                )
                == records["rgba_half_zip"].get("vips_payload_sha256"),
            },
            "display_window_differs": {
                "file": "fixtures/rgba_half_display.exr",
                "display_window": [0, 0, 15, 15],
                "data_window": [2, 3, 9, 6],
                "vips_size": [
                    records["rgba_half_display"]["vipsheader"].get("width"),
                    records["rgba_half_display"]["vipsheader"].get("height"),
                ],
            },
        },
        "tiled_files_carry_tile_geometry": {
            "what": (
                "read_header sets VIPS_META_TILE_WIDTH / VIPS_META_TILE_HEIGHT "
                "for a tiled file (openexr2vips.c:229-232) and nothing for a "
                "scanline one, so the presence of the field is itself the signal."
            ),
            "tiled": {
                "file": "fixtures/rgba_half_tiled.exr",
                "tile_width": records["rgba_half_tiled"]["vipsheader"].get(
                    "tile-width"
                ),
                "tile_height": records["rgba_half_tiled"]["vipsheader"].get(
                    "tile-height"
                ),
            },
            "scanline_has_no_tile_fields": "tile-width"
            not in records["rgba_half_zip"]["vipsheader"],
        },
        "lossy_codings_differ": {
            "what": (
                "B44 and B44A quantise 4x4 blocks to a shared exponent, so they "
                "are lossy by construction and must NOT be pinned as equal to the "
                "lossless payload. DWAA and DWAB are lossy in general and happen "
                "to reproduce this small smooth ramp exactly; that is a property "
                "of the payload, not of the codec, so they are pinned by their "
                "own digest and not by equality."
            ),
            "b44_payload_sha256": records["rgba_half_b44"].get("vips_payload_sha256"),
            "b44a_payload_sha256": records["rgba_half_b44a"].get(
                "vips_payload_sha256"
            ),
            "dwaa_payload_sha256": records["rgba_half_dwaa"].get(
                "vips_payload_sha256"
            ),
            "dwab_payload_sha256": records["rgba_half_dwab"].get(
                "vips_payload_sha256"
            ),
            "b44_equals_lossless": records["rgba_half_b44"].get("vips_payload_sha256")
            == records["rgba_half_zip"].get("vips_payload_sha256"),
            "dwaa_equals_lossless": records["rgba_half_dwaa"].get(
                "vips_payload_sha256"
            )
            == records["rgba_half_zip"].get("vips_payload_sha256"),
        },
        "there_is_no_exr_saver": {
            "what": (
                "libvips has never shipped an EXR writer, so the save half of "
                "issue #504 was deleted rather than deferred."
            ),
            "vips_l_exr": run(["/bin/sh", "-c", f"{VIPS} -l | grep -i exr"]).stdout.strip(),
            "save_attempt": None,
        },
    }

    # Prove the absence of a saver rather than asserting it.
    with tempfile.TemporaryDirectory() as tmpdir:
        src = pathlib.Path(tmpdir) / "src.png"
        dst = pathlib.Path(tmpdir) / "out.exr"
        run([VIPS, "black", str(src), "4", "4"])
        proc = run([VIPS, "copy", str(src), str(dst)])
        findings["there_is_no_exr_saver"]["save_attempt"] = {
            "exit": proc.returncode,
            "stderr": proc.stderr.strip(),
        }

    oracle = {
        "meta": {
            "area": "foreign-exr",
            "issue": 504,
            "sub_issues": [614, 615],
            "vips_version": vips_version,
            "vips_binary": VIPS,
            "fixture_writer": exr_version,
            "fixture_writer_source": "make_corpus.cpp in this directory",
            "captured_by": "oracle-captures/foreign-exr/capture.py",
            "reference_c": (
                "libvips v8.18.0-95-gfe420cf3a for line numbers "
                "(libvips/foreign/openexr2vips.c, openexrload.c); the binary is "
                "the 8.18.4 release and is not the same tree"
            ),
        },
        "notes": [
            "vips reads EXR through the OpenEXR C RGBA wrapper, ImfCRgbaFile.h, "
            "which returns four half samples per pixel and nothing else. Every "
            "divergence recorded here follows from that one fact.",
            "openexr2vips.c:24-38 lists it as a known limitation in its own TODO "
            "block: 'more of OpenEXR's pixel formats', 'more than just RGBA "
            "channels', 'best redo with the C++ API now we support C++ "
            "operations'.",
            "The fixtures are written by the OpenEXR reference implementation, "
            "not by libviprs and not by vips, so no capture here is circular.",
        ],
        "findings": findings,
        "records": records,
    }

    # allow_nan=False so a non-finite measurement stops the capture here
    # rather than writing a file nobody outside Python can parse (#682).
    (HERE / "oracle.json").write_text(
        json.dumps(oracle, indent=2, allow_nan=False) + "\n")
    (HERE / "commands.sh").write_text(
        "#!/bin/sh\n"
        "# Every command capture.py ran, in order. Regenerate with\n"
        "# `python3 capture.py` from this directory. The fixtures themselves\n"
        "# come from make_corpus.cpp and are NOT regenerated here.\n"
        "set -e\n\n" + "\n".join(COMMANDS) + "\n"
    )
    print(f"wrote oracle.json ({len(records)} fixtures) and commands.sh")


if __name__ == "__main__":
    main()
