#!/bin/sh
# Every command capture.py ran, in order. Regenerate with
# `python3 capture.py` from this directory.
set -e

/opt/homebrew/bin/vips -l
/opt/homebrew/bin/vips --vips-config
/opt/homebrew/bin/vips rawload fixtures/hdr64.raw outputs/hdr64-raw.v 64 64 3 --format float
/opt/homebrew/bin/vips copy outputs/hdr64-raw.v outputs/hdr64.v --interpretation scrgb
/opt/homebrew/bin/vips uhdrsave outputs/hdr64.v fixtures/uhdr.jpg
/opt/homebrew/bin/vipsheader -a fixtures/uhdr.jpg
/opt/homebrew/bin/vips colourspace outputs/hdr64.v outputs/sdr64.v srgb
/opt/homebrew/bin/vips jpegsave outputs/sdr64.v fixtures/plain.jpg --keep none
# fixtures/no-*.jpg, base-only.jpg, truncated-*.jpg and mpf-graft.jpg are cut from fixtures/uhdr.jpg and fixtures/plain.jpg by this script; vips cannot edit JPEG marker segments
/opt/homebrew/bin/vipsheader fixtures/uhdr.jpg
/opt/homebrew/bin/vips uhdrload fixtures/uhdr.jpg outputs/det-uhdr.jpg.v
/opt/homebrew/bin/vipsheader outputs/det-uhdr.jpg.v
/opt/homebrew/bin/vipsheader -a outputs/det-uhdr.jpg.v
/opt/homebrew/bin/vipsheader fixtures/plain.jpg
/opt/homebrew/bin/vips uhdrload fixtures/plain.jpg outputs/det-plain.jpg.v
/opt/homebrew/bin/vipsheader fixtures/no-mpf.jpg
/opt/homebrew/bin/vips uhdrload fixtures/no-mpf.jpg outputs/det-no-mpf.jpg.v
/opt/homebrew/bin/vipsheader outputs/det-no-mpf.jpg.v
/opt/homebrew/bin/vipsheader -a outputs/det-no-mpf.jpg.v
/opt/homebrew/bin/vipsheader fixtures/no-iso-base.jpg
/opt/homebrew/bin/vips uhdrload fixtures/no-iso-base.jpg outputs/det-no-iso-base.jpg.v
/opt/homebrew/bin/vipsheader outputs/det-no-iso-base.jpg.v
/opt/homebrew/bin/vipsheader -a outputs/det-no-iso-base.jpg.v
/opt/homebrew/bin/vipsheader fixtures/no-iso-gainmap.jpg
/opt/homebrew/bin/vips uhdrload fixtures/no-iso-gainmap.jpg outputs/det-no-iso-gainmap.jpg.v
/opt/homebrew/bin/vipsheader fixtures/base-only.jpg
/opt/homebrew/bin/vips uhdrload fixtures/base-only.jpg outputs/det-base-only.jpg.v
/opt/homebrew/bin/vipsheader fixtures/truncated-gainmap.jpg
/opt/homebrew/bin/vips uhdrload fixtures/truncated-gainmap.jpg outputs/det-truncated-gainmap.jpg.v
/opt/homebrew/bin/vipsheader fixtures/truncated-base.jpg
/opt/homebrew/bin/vips uhdrload fixtures/truncated-base.jpg outputs/det-truncated-base.jpg.v
/opt/homebrew/bin/vipsheader fixtures/mpf-graft.jpg
/opt/homebrew/bin/vips uhdrload fixtures/mpf-graft.jpg outputs/det-mpf-graft.jpg.v
/opt/homebrew/bin/vips jpegload fixtures/uhdr.jpg outputs/via-jpegload.v
cat fixtures/uhdr.jpg | /opt/homebrew/bin/vips copy stdin outputs/via-stdin.v
/opt/homebrew/bin/vipsheader -a outputs/via-jpegload.v
/opt/homebrew/bin/vipsheader outputs/via-stdin.v
/opt/homebrew/bin/vips copy fixtures/uhdr.jpg outputs/uhdr-loaded.v
# outputs/gainmap-data.jpg is the gainmap-data blob lifted out of the .v XML trailer, so its own header can be read
/opt/homebrew/bin/vips jpegload outputs/gainmap-data.jpg outputs/gainmap-data-decoded.v
/opt/homebrew/bin/vips uhdrload fixtures/uhdr.jpg outputs/shrink1.v --shrink 1
/opt/homebrew/bin/vipsheader -a fixtures/uhdr.jpg[shrink=1]
/opt/homebrew/bin/vipsheader -a outputs/shrink1.v
/opt/homebrew/bin/vips uhdrload fixtures/uhdr.jpg outputs/shrink2.v --shrink 2
/opt/homebrew/bin/vipsheader -a fixtures/uhdr.jpg[shrink=2]
/opt/homebrew/bin/vipsheader -a outputs/shrink2.v
/opt/homebrew/bin/vips uhdrload fixtures/uhdr.jpg outputs/shrink3.v --shrink 3
/opt/homebrew/bin/vips uhdrload fixtures/uhdr.jpg outputs/shrink4.v --shrink 4
/opt/homebrew/bin/vipsheader -a fixtures/uhdr.jpg[shrink=4]
/opt/homebrew/bin/vipsheader -a outputs/shrink4.v
/opt/homebrew/bin/vips uhdrload fixtures/uhdr.jpg outputs/shrink8.v --shrink 8
/opt/homebrew/bin/vipsheader -a fixtures/uhdr.jpg[shrink=8]
/opt/homebrew/bin/vipsheader -a outputs/shrink8.v
/opt/homebrew/bin/vips uhdrload fixtures/uhdr.jpg outputs/shrink9.v --shrink 9
/opt/homebrew/bin/vipsheader -a fixtures/uhdr.jpg[shrink=9]
/opt/homebrew/bin/vipsheader -a outputs/shrink9.v
/opt/homebrew/bin/vipsheader outputs/gainmap-data-decoded.v
/opt/homebrew/bin/vips uhdr2scRGB /Users/rom/workspace/libvips/test/test-suite/images/ultra-hdr.jpg outputs/reference-scrgb.v
/opt/homebrew/bin/vips max outputs/reference-scrgb.v
/opt/homebrew/bin/vips min outputs/reference-scrgb.v
/opt/homebrew/bin/vips avg outputs/reference-scrgb.v
/opt/homebrew/bin/vipsheader -a /Users/rom/workspace/libvips/test/test-suite/images/ultra-hdr.jpg
/opt/homebrew/bin/vips rawload fixtures/base16.raw outputs/base16-raw.v 16 1 3 --format uchar
/opt/homebrew/bin/vips copy outputs/base16-raw.v outputs/base16.v --interpretation srgb
/opt/homebrew/bin/vips rawload fixtures/gainmap-mono16-src.raw outputs/gainmap-mono16-src-raw.v 16 1 1 --format uchar
/opt/homebrew/bin/vips copy outputs/gainmap-mono16-src-raw.v outputs/gainmap-mono16-src.v --interpretation b-w
/opt/homebrew/bin/vips jpegsave outputs/gainmap-mono16-src.v fixtures/gainmap-mono16.jpg --Q 100 --subsample-mode off --keep none
/opt/homebrew/bin/vips jpegload fixtures/gainmap-mono16.jpg outputs/gainmap-mono16-decoded.v
for y in $(seq 0 0); do for x in $(seq 0 15); do /opt/homebrew/bin/vips getpoint outputs/gainmap-mono16-decoded.v $x $y; done; done
/opt/homebrew/bin/vips rawload fixtures/reject-mono.raw outputs/reject-mono-raw.v 16 1 1 --format uchar
/opt/homebrew/bin/vips copy outputs/reject-mono-raw.v outputs/reject-mono.v --interpretation b-w
/opt/homebrew/bin/vips rawload fixtures/reject-rgba.raw outputs/reject-rgba-raw.v 16 1 4 --format uchar
/opt/homebrew/bin/vips copy outputs/reject-rgba-raw.v outputs/reject-rgba.v --interpretation srgb
/opt/homebrew/bin/vips rawload fixtures/reject-ushort.raw outputs/reject-ushort-raw.v 16 1 3 --format ushort
/opt/homebrew/bin/vips copy outputs/reject-ushort-raw.v outputs/reject-ushort.v --interpretation rgb16
/opt/homebrew/bin/vipsedit --setext outputs/p-reject-one_band.v   # gainmap-data + gainmap-* arrays, from a heredoc
/opt/homebrew/bin/vips uhdr2scRGB outputs/p-reject-one_band.v outputs/h-reject-one_band.v
/opt/homebrew/bin/vipsedit --setext outputs/p-reject-four_band.v   # gainmap-data + gainmap-* arrays, from a heredoc
/opt/homebrew/bin/vips uhdr2scRGB outputs/p-reject-four_band.v outputs/h-reject-four_band.v
/opt/homebrew/bin/vipsedit --setext outputs/p-reject-ushort.v   # gainmap-data + gainmap-* arrays, from a heredoc
/opt/homebrew/bin/vips uhdr2scRGB outputs/p-reject-ushort.v outputs/h-reject-ushort.v
/opt/homebrew/bin/vipsedit --setext outputs/p-reject-no-gainmap.v   # gainmap-data + gainmap-* arrays, from a heredoc
/opt/homebrew/bin/vips uhdr2scRGB outputs/p-reject-no-gainmap.v outputs/h-reject-no-gainmap.v
/opt/homebrew/bin/vipsedit --setext outputs/p-reject-no-metadata.v   # gainmap-data + gainmap-* arrays, from a heredoc
/opt/homebrew/bin/vips uhdr2scRGB outputs/p-reject-no-metadata.v outputs/h-reject-no-metadata.v
/opt/homebrew/bin/vips rawload fixtures/ramp256.raw outputs/ramp256-raw.v 256 1 3 --format uchar
/opt/homebrew/bin/vips copy outputs/ramp256-raw.v outputs/ramp256.v --interpretation srgb
/opt/homebrew/bin/vips sRGB2scRGB outputs/ramp256.v outputs/ramp256-linear.v
/opt/homebrew/bin/vips rawsave outputs/ramp256-linear.v outputs/ramp256-linear.raw
/opt/homebrew/bin/vips rawload fixtures/gainmap-ramp256-src.raw outputs/gainmap-ramp256-src-raw.v 256 1 1 --format uchar
/opt/homebrew/bin/vips copy outputs/gainmap-ramp256-src-raw.v outputs/gainmap-ramp256-src.v --interpretation b-w
/opt/homebrew/bin/vips jpegsave outputs/gainmap-ramp256-src.v fixtures/gainmap-ramp256.jpg --Q 100 --subsample-mode off --keep none
/opt/homebrew/bin/vips jpegload fixtures/gainmap-ramp256.jpg outputs/gainmap-ramp256-decoded.v
/opt/homebrew/bin/vipsedit --setext outputs/p-identity.v   # gainmap-data + gainmap-* arrays, from a heredoc
/opt/homebrew/bin/vips uhdr2scRGB outputs/p-identity.v outputs/h-identity.v
/opt/homebrew/bin/vips rawsave outputs/h-identity.v outputs/h-identity.raw
/opt/homebrew/bin/vipsedit --setext outputs/p-mono-canonical.v   # gainmap-data + gainmap-* arrays, from a heredoc
/opt/homebrew/bin/vips uhdr2scRGB outputs/p-mono-canonical.v outputs/h-mono-canonical.v
/opt/homebrew/bin/vipsheader outputs/h-mono-canonical.v
for y in $(seq 0 0); do for x in $(seq 0 15); do /opt/homebrew/bin/vips getpoint outputs/h-mono-canonical.v $x $y; done; done
/opt/homebrew/bin/vipsedit --setext outputs/p-mono-gamma.v   # gainmap-data + gainmap-* arrays, from a heredoc
/opt/homebrew/bin/vips uhdr2scRGB outputs/p-mono-gamma.v outputs/h-mono-gamma.v
/opt/homebrew/bin/vipsheader outputs/h-mono-gamma.v
for y in $(seq 0 0); do for x in $(seq 0 15); do /opt/homebrew/bin/vips getpoint outputs/h-mono-gamma.v $x $y; done; done
/opt/homebrew/bin/vipsedit --setext outputs/p-mono-minhalf.v   # gainmap-data + gainmap-* arrays, from a heredoc
/opt/homebrew/bin/vips uhdr2scRGB outputs/p-mono-minhalf.v outputs/h-mono-minhalf.v
/opt/homebrew/bin/vipsheader outputs/h-mono-minhalf.v
for y in $(seq 0 0); do for x in $(seq 0 15); do /opt/homebrew/bin/vips getpoint outputs/h-mono-minhalf.v $x $y; done; done
/opt/homebrew/bin/vipsedit --setext outputs/p-mono-offsets.v   # gainmap-data + gainmap-* arrays, from a heredoc
/opt/homebrew/bin/vips uhdr2scRGB outputs/p-mono-offsets.v outputs/h-mono-offsets.v
/opt/homebrew/bin/vipsheader outputs/h-mono-offsets.v
for y in $(seq 0 0); do for x in $(seq 0 15); do /opt/homebrew/bin/vips getpoint outputs/h-mono-offsets.v $x $y; done; done
/opt/homebrew/bin/vipsedit --setext outputs/p-mono-green-only.v   # gainmap-data + gainmap-* arrays, from a heredoc
/opt/homebrew/bin/vips uhdr2scRGB outputs/p-mono-green-only.v outputs/h-mono-green-only.v
/opt/homebrew/bin/vipsheader outputs/h-mono-green-only.v
for y in $(seq 0 0); do for x in $(seq 0 15); do /opt/homebrew/bin/vips getpoint outputs/h-mono-green-only.v $x $y; done; done
/opt/homebrew/bin/vips rawload fixtures/gainmap-rgb16-src.raw outputs/gainmap-rgb16-src-raw.v 16 1 3 --format uchar
/opt/homebrew/bin/vips copy outputs/gainmap-rgb16-src-raw.v outputs/gainmap-rgb16-src.v --interpretation srgb
/opt/homebrew/bin/vips jpegsave outputs/gainmap-rgb16-src.v fixtures/gainmap-rgb16.jpg --Q 100 --subsample-mode off --keep none
/opt/homebrew/bin/vips jpegload fixtures/gainmap-rgb16.jpg outputs/gainmap-rgb16-decoded.v
for y in $(seq 0 0); do for x in $(seq 0 15); do /opt/homebrew/bin/vips getpoint outputs/gainmap-rgb16-decoded.v $x $y; done; done
/opt/homebrew/bin/vipsedit --setext outputs/p-rgb-canonical.v   # gainmap-data + gainmap-* arrays, from a heredoc
/opt/homebrew/bin/vips uhdr2scRGB outputs/p-rgb-canonical.v outputs/h-rgb-canonical.v
/opt/homebrew/bin/vipsheader outputs/h-rgb-canonical.v
for y in $(seq 0 0); do for x in $(seq 0 15); do /opt/homebrew/bin/vips getpoint outputs/h-rgb-canonical.v $x $y; done; done
/opt/homebrew/bin/vips rawload fixtures/gainmap-mono8-src.raw outputs/gainmap-mono8-src-raw.v 8 1 1 --format uchar
/opt/homebrew/bin/vips copy outputs/gainmap-mono8-src-raw.v outputs/gainmap-mono8-src.v --interpretation b-w
/opt/homebrew/bin/vips jpegsave outputs/gainmap-mono8-src.v fixtures/gainmap-mono8.jpg --Q 100 --subsample-mode off --keep none
/opt/homebrew/bin/vips jpegload fixtures/gainmap-mono8.jpg outputs/gainmap-mono8-decoded.v
for y in $(seq 0 0); do for x in $(seq 0 7); do /opt/homebrew/bin/vips getpoint outputs/gainmap-mono8-decoded.v $x $y; done; done
/opt/homebrew/bin/vips resize outputs/gainmap-mono8-decoded.v outputs/gainmap-mono8-resized.v 2.0 --vscale 1.0 --kernel linear
/opt/homebrew/bin/vipsedit --setext outputs/p-mono-half-size.v   # gainmap-data + gainmap-* arrays, from a heredoc
/opt/homebrew/bin/vips uhdr2scRGB outputs/p-mono-half-size.v outputs/h-mono-half-size.v
/opt/homebrew/bin/vipsheader outputs/h-mono-half-size.v
for y in $(seq 0 0); do for x in $(seq 0 15); do /opt/homebrew/bin/vips getpoint outputs/h-mono-half-size.v $x $y; done; done
for y in $(seq 0 0); do for x in $(seq 0 15); do /opt/homebrew/bin/vips getpoint outputs/gainmap-mono8-resized.v $x $y; done; done
/opt/homebrew/bin/vipsedit --setext outputs/p-deg-minzero.v   # gainmap-data + gainmap-* arrays, from a heredoc
/opt/homebrew/bin/vips uhdr2scRGB outputs/p-deg-minzero.v outputs/h-deg-minzero.v
/opt/homebrew/bin/vipsheader outputs/h-deg-minzero.v
for y in $(seq 0 0); do for x in $(seq 0 15); do /opt/homebrew/bin/vips getpoint outputs/h-deg-minzero.v $x $y; done; done
/opt/homebrew/bin/vipsedit --setext outputs/p-deg-maxzero.v   # gainmap-data + gainmap-* arrays, from a heredoc
/opt/homebrew/bin/vips uhdr2scRGB outputs/p-deg-maxzero.v outputs/h-deg-maxzero.v
/opt/homebrew/bin/vipsheader outputs/h-deg-maxzero.v
for y in $(seq 0 0); do for x in $(seq 0 15); do /opt/homebrew/bin/vips getpoint outputs/h-deg-maxzero.v $x $y; done; done
/opt/homebrew/bin/vipsedit --setext outputs/p-deg-inverted.v   # gainmap-data + gainmap-* arrays, from a heredoc
/opt/homebrew/bin/vips uhdr2scRGB outputs/p-deg-inverted.v outputs/h-deg-inverted.v
/opt/homebrew/bin/vipsheader outputs/h-deg-inverted.v
for y in $(seq 0 0); do for x in $(seq 0 15); do /opt/homebrew/bin/vips getpoint outputs/h-deg-inverted.v $x $y; done; done
/opt/homebrew/bin/vips uhdrsave outputs/sdr64.v outputs/no-metadata.jpg
/opt/homebrew/bin/vips copy fixtures/uhdr.jpg outputs/uhdr-reloaded.v
/opt/homebrew/bin/vips uhdrsave outputs/uhdr-reloaded.v outputs/resaved.jpg
/opt/homebrew/bin/vipsheader -a outputs/resaved.jpg
/opt/homebrew/bin/vips uhdrsave outputs/hdr64.v outputs/q1.jpg --Q 1
/opt/homebrew/bin/vips uhdrsave outputs/hdr64.v outputs/q50.jpg --Q 50
/opt/homebrew/bin/vips uhdrsave outputs/hdr64.v outputs/q75.jpg --Q 75
/opt/homebrew/bin/vips uhdrsave outputs/hdr64.v outputs/q100.jpg --Q 100
/opt/homebrew/bin/vips uhdrsave outputs/hdr64.v outputs/scale1.jpg --gainmap-scale-factor 1
/opt/homebrew/bin/vipsheader -a outputs/scale1.jpg
/opt/homebrew/bin/vipsheader -a outputs/scale1.jpg[shrink=1]
/opt/homebrew/bin/vips uhdrsave outputs/hdr64.v outputs/scale2.jpg --gainmap-scale-factor 2
/opt/homebrew/bin/vipsheader -a outputs/scale2.jpg
/opt/homebrew/bin/vipsheader -a outputs/scale2.jpg[shrink=1]
/opt/homebrew/bin/vips uhdrsave outputs/hdr64.v outputs/scale4.jpg --gainmap-scale-factor 4
/opt/homebrew/bin/vipsheader -a outputs/scale4.jpg
/opt/homebrew/bin/vipsheader -a outputs/scale4.jpg[shrink=1]
/opt/homebrew/bin/vips uhdrsave outputs/hdr64.v outputs/scale8.jpg --gainmap-scale-factor 8
/opt/homebrew/bin/vipsheader -a outputs/scale8.jpg
/opt/homebrew/bin/vipsheader -a outputs/scale8.jpg[shrink=1]
/opt/homebrew/bin/vips uhdrsave outputs/hdr64.v outputs/scale128.jpg --gainmap-scale-factor 128
/opt/homebrew/bin/vipsheader -a outputs/scale128.jpg
/opt/homebrew/bin/vipsheader -a outputs/scale128.jpg[shrink=1]
/opt/homebrew/bin/vips uhdrsave outputs/hdr64.v outputs/scale129.jpg --gainmap-scale-factor 129
/opt/homebrew/bin/vipsheader -a outputs/scale129.jpg
/opt/homebrew/bin/vipsheader -a outputs/scale129.jpg[shrink=1]
/opt/homebrew/bin/vips jpegsave outputs/hdr64.v outputs/routed-by-jpegsave.jpg
/opt/homebrew/bin/vips jpegsave outputs/uhdr-reloaded.v outputs/routed-sdr.jpg
/opt/homebrew/bin/vipsheader outputs/routed-sdr.jpg
/opt/homebrew/bin/vips uhdr2scRGB fixtures/uhdr.jpg outputs/rt-hdr1.v
/opt/homebrew/bin/vips uhdrsave outputs/rt-hdr1.v outputs/rt.jpg
/opt/homebrew/bin/vips uhdr2scRGB outputs/rt.jpg outputs/rt-hdr2.v
/opt/homebrew/bin/vips copy fixtures/uhdr.jpg outputs/rt-sdr1.v
/opt/homebrew/bin/vips copy outputs/resaved.jpg outputs/rt-sdr2.v
/opt/homebrew/bin/vips max outputs/rt-hdr1.v
/opt/homebrew/bin/vips max outputs/rt-hdr2.v
/opt/homebrew/bin/vips subtract outputs/rt-hdr1.v outputs/rt-hdr2.v outputs/hdr-diff.v
/opt/homebrew/bin/vips abs outputs/hdr-diff.v outputs/hdr-absdiff.v
/opt/homebrew/bin/vips avg outputs/hdr-absdiff.v
/opt/homebrew/bin/vips max outputs/hdr-absdiff.v
/opt/homebrew/bin/vips subtract outputs/rt-sdr1.v outputs/rt-sdr2.v outputs/sdr-diff.v
/opt/homebrew/bin/vips abs outputs/sdr-diff.v outputs/sdr-absdiff.v
/opt/homebrew/bin/vips avg outputs/sdr-absdiff.v
/opt/homebrew/bin/vips max outputs/sdr-absdiff.v
/opt/homebrew/bin/vips uhdrload fixtures/truncated-base.jpg outputs/m-truncated-base.jpg.v
/opt/homebrew/bin/vips uhdr2scRGB fixtures/truncated-base.jpg outputs/t-truncated-base.jpg.v
/opt/homebrew/bin/vipsheader fixtures/truncated-base.jpg
/opt/homebrew/bin/vips uhdrload fixtures/truncated-gainmap.jpg outputs/m-truncated-gainmap.jpg.v
/opt/homebrew/bin/vips uhdr2scRGB fixtures/truncated-gainmap.jpg outputs/t-truncated-gainmap.jpg.v
/opt/homebrew/bin/vipsheader fixtures/truncated-gainmap.jpg
/opt/homebrew/bin/vips uhdrload fixtures/base-only.jpg outputs/m-base-only.jpg.v
/opt/homebrew/bin/vips uhdr2scRGB fixtures/base-only.jpg outputs/t-base-only.jpg.v
/opt/homebrew/bin/vipsheader fixtures/base-only.jpg
# fixtures/gainmap-truncated.jpg is the first 120 bytes of fixtures/gainmap-mono16.jpg
/opt/homebrew/bin/vipsedit --setext outputs/p-malformed-gainmap.v   # gainmap-data + gainmap-* arrays, from a heredoc
/opt/homebrew/bin/vips uhdr2scRGB outputs/p-malformed-gainmap.v outputs/h-malformed-gainmap.v
/opt/homebrew/bin/vips --version
/usr/bin/otool -L /opt/homebrew/lib/libvips.42.dylib
