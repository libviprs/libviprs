#!/bin/sh
# Every command capture.py ran, in order. Regenerate with
# `python3 capture.py` from this directory.
set -e

/opt/homebrew/bin/vips rawload fixtures/ll_rgb.raw outputs/ll_rgb.v 4 3 3 --format uchar
/opt/homebrew/bin/vips copy outputs/ll_rgb.v outputs/ll_rgb-srgb.v --interpretation srgb
/opt/homebrew/bin/vips webpsave outputs/ll_rgb-srgb.v fixtures/ll_rgb.webp --lossless --keep none
/opt/homebrew/bin/vips getpoint fixtures/ll_rgb.webp 0 0
/opt/homebrew/bin/vips getpoint fixtures/ll_rgb.webp 1 0
/opt/homebrew/bin/vips getpoint fixtures/ll_rgb.webp 2 0
/opt/homebrew/bin/vips getpoint fixtures/ll_rgb.webp 3 0
/opt/homebrew/bin/vips getpoint fixtures/ll_rgb.webp 0 1
/opt/homebrew/bin/vips getpoint fixtures/ll_rgb.webp 1 1
/opt/homebrew/bin/vips getpoint fixtures/ll_rgb.webp 2 1
/opt/homebrew/bin/vips getpoint fixtures/ll_rgb.webp 3 1
/opt/homebrew/bin/vips getpoint fixtures/ll_rgb.webp 0 2
/opt/homebrew/bin/vips getpoint fixtures/ll_rgb.webp 1 2
/opt/homebrew/bin/vips getpoint fixtures/ll_rgb.webp 2 2
/opt/homebrew/bin/vips getpoint fixtures/ll_rgb.webp 3 2
/opt/homebrew/bin/vipsheader fixtures/ll_rgb.webp
/opt/homebrew/bin/vips rawload fixtures/ll_rgba.raw outputs/ll_rgba.v 4 3 4 --format uchar
/opt/homebrew/bin/vips copy outputs/ll_rgba.v outputs/ll_rgba-srgb.v --interpretation srgb
/opt/homebrew/bin/vips webpsave outputs/ll_rgba-srgb.v fixtures/ll_rgba.webp --lossless --keep none
/opt/homebrew/bin/vips getpoint fixtures/ll_rgba.webp 0 0
/opt/homebrew/bin/vips getpoint fixtures/ll_rgba.webp 1 0
/opt/homebrew/bin/vips getpoint fixtures/ll_rgba.webp 2 0
/opt/homebrew/bin/vips getpoint fixtures/ll_rgba.webp 3 0
/opt/homebrew/bin/vips getpoint fixtures/ll_rgba.webp 0 1
/opt/homebrew/bin/vips getpoint fixtures/ll_rgba.webp 1 1
/opt/homebrew/bin/vips getpoint fixtures/ll_rgba.webp 2 1
/opt/homebrew/bin/vips getpoint fixtures/ll_rgba.webp 3 1
/opt/homebrew/bin/vips getpoint fixtures/ll_rgba.webp 0 2
/opt/homebrew/bin/vips getpoint fixtures/ll_rgba.webp 1 2
/opt/homebrew/bin/vips getpoint fixtures/ll_rgba.webp 2 2
/opt/homebrew/bin/vips getpoint fixtures/ll_rgba.webp 3 2
/opt/homebrew/bin/vipsheader fixtures/ll_rgba.webp
/opt/homebrew/bin/vips rawload fixtures/lossy_rgb.raw outputs/lossy_rgb.v 4 3 3 --format uchar
/opt/homebrew/bin/vips copy outputs/lossy_rgb.v outputs/lossy_rgb-srgb.v --interpretation srgb
/opt/homebrew/bin/vips webpsave outputs/lossy_rgb-srgb.v fixtures/lossy_rgb.webp --keep none
/opt/homebrew/bin/vips getpoint fixtures/lossy_rgb.webp 0 0
/opt/homebrew/bin/vips getpoint fixtures/lossy_rgb.webp 1 0
/opt/homebrew/bin/vips getpoint fixtures/lossy_rgb.webp 2 0
/opt/homebrew/bin/vips getpoint fixtures/lossy_rgb.webp 3 0
/opt/homebrew/bin/vips getpoint fixtures/lossy_rgb.webp 0 1
/opt/homebrew/bin/vips getpoint fixtures/lossy_rgb.webp 1 1
/opt/homebrew/bin/vips getpoint fixtures/lossy_rgb.webp 2 1
/opt/homebrew/bin/vips getpoint fixtures/lossy_rgb.webp 3 1
/opt/homebrew/bin/vips getpoint fixtures/lossy_rgb.webp 0 2
/opt/homebrew/bin/vips getpoint fixtures/lossy_rgb.webp 1 2
/opt/homebrew/bin/vips getpoint fixtures/lossy_rgb.webp 2 2
/opt/homebrew/bin/vips getpoint fixtures/lossy_rgb.webp 3 2
/opt/homebrew/bin/vips rawload fixtures/grey.raw outputs/grey.v 4 3 1 --format uchar
/opt/homebrew/bin/vips copy outputs/grey.v outputs/grey-b-w.v --interpretation b-w
/opt/homebrew/bin/vips webpsave outputs/grey-b-w.v outputs/grey.webp --lossless --keep none
/opt/homebrew/bin/vips getpoint outputs/grey.webp 0 0
/opt/homebrew/bin/vips getpoint outputs/grey.webp 1 0
/opt/homebrew/bin/vips getpoint outputs/grey.webp 2 0
/opt/homebrew/bin/vips getpoint outputs/grey.webp 3 0
/opt/homebrew/bin/vips getpoint outputs/grey.webp 0 1
/opt/homebrew/bin/vips getpoint outputs/grey.webp 1 1
/opt/homebrew/bin/vips getpoint outputs/grey.webp 2 1
/opt/homebrew/bin/vips getpoint outputs/grey.webp 3 1
/opt/homebrew/bin/vips getpoint outputs/grey.webp 0 2
/opt/homebrew/bin/vips getpoint outputs/grey.webp 1 2
/opt/homebrew/bin/vips getpoint outputs/grey.webp 2 2
/opt/homebrew/bin/vips getpoint outputs/grey.webp 3 2
/opt/homebrew/bin/vipsheader outputs/grey.webp
/opt/homebrew/bin/vips rawload fixtures/rgb16.raw outputs/rgb16.v 4 2 3 --format ushort
/opt/homebrew/bin/vips copy outputs/rgb16.v outputs/rgb16-srgb.v --interpretation srgb
/opt/homebrew/bin/vips webpsave outputs/rgb16-srgb.v outputs/rgb16.webp --lossless --keep none
/opt/homebrew/bin/vips getpoint outputs/rgb16.webp 0 0
/opt/homebrew/bin/vips getpoint outputs/rgb16.webp 1 0
/opt/homebrew/bin/vips getpoint outputs/rgb16.webp 2 0
/opt/homebrew/bin/vips getpoint outputs/rgb16.webp 3 0
/opt/homebrew/bin/vips getpoint outputs/rgb16.webp 0 1
/opt/homebrew/bin/vips getpoint outputs/rgb16.webp 1 1
/opt/homebrew/bin/vips getpoint outputs/rgb16.webp 2 1
/opt/homebrew/bin/vips getpoint outputs/rgb16.webp 3 1
/opt/homebrew/bin/vipsheader outputs/rgb16-srgb.v
/opt/homebrew/bin/vipsheader outputs/rgb16.webp
/opt/homebrew/bin/vips rawload fixtures/roll.raw outputs/roll.v 4 9 3 --format uchar
/opt/homebrew/bin/vips copy outputs/roll.v outputs/roll-srgb.v --interpretation srgb
/opt/homebrew/bin/vips webpsave outputs/roll-srgb.v fixtures/anim3.webp --lossless --keep none --page-height 3
/opt/homebrew/bin/vips getpoint fixtures/anim3.webp 0 0
/opt/homebrew/bin/vips getpoint fixtures/anim3.webp 1 0
/opt/homebrew/bin/vips getpoint fixtures/anim3.webp 2 0
/opt/homebrew/bin/vips getpoint fixtures/anim3.webp 3 0
/opt/homebrew/bin/vips getpoint fixtures/anim3.webp 0 1
/opt/homebrew/bin/vips getpoint fixtures/anim3.webp 1 1
/opt/homebrew/bin/vips getpoint fixtures/anim3.webp 2 1
/opt/homebrew/bin/vips getpoint fixtures/anim3.webp 3 1
/opt/homebrew/bin/vips getpoint fixtures/anim3.webp 0 2
/opt/homebrew/bin/vips getpoint fixtures/anim3.webp 1 2
/opt/homebrew/bin/vips getpoint fixtures/anim3.webp 2 2
/opt/homebrew/bin/vips getpoint fixtures/anim3.webp 3 2
/opt/homebrew/bin/vipsheader -a fixtures/anim3.webp
/opt/homebrew/bin/vipsheader fixtures/anim3.webp[n=-1]
# fixtures/meta.webp is built by this script, not by vips: webpsave has no way to attach an arbitrary XMP packet
/opt/homebrew/bin/vipsheader -a fixtures/meta.webp
/opt/homebrew/bin/vips getpoint fixtures/meta.webp 0 0
/opt/homebrew/bin/vips getpoint fixtures/meta.webp 1 0
/opt/homebrew/bin/vips getpoint fixtures/meta.webp 2 0
/opt/homebrew/bin/vips getpoint fixtures/meta.webp 3 0
/opt/homebrew/bin/vips getpoint fixtures/meta.webp 0 1
/opt/homebrew/bin/vips getpoint fixtures/meta.webp 1 1
/opt/homebrew/bin/vips getpoint fixtures/meta.webp 2 1
/opt/homebrew/bin/vips getpoint fixtures/meta.webp 3 1
/opt/homebrew/bin/vips getpoint fixtures/meta.webp 0 2
/opt/homebrew/bin/vips getpoint fixtures/meta.webp 1 2
/opt/homebrew/bin/vips getpoint fixtures/meta.webp 2 2
/opt/homebrew/bin/vips getpoint fixtures/meta.webp 3 2
/opt/homebrew/bin/vips rawload fixtures/icc_src.raw outputs/icc_src.v 4 3 3 --format uchar
/opt/homebrew/bin/vips copy outputs/icc_src.v outputs/icc_src-srgb.v --interpretation srgb
/opt/homebrew/bin/vips webpsave outputs/icc_src-srgb.v outputs/with_icc.webp --lossless --profile /Users/rom/workspace/libvips/test/test-suite/images/sRGB.icm
/opt/homebrew/bin/vipsheader -a outputs/with_icc.webp
/opt/homebrew/bin/vips black outputs/wide16383.v 16383 1 --bands 3
/opt/homebrew/bin/vips copy outputs/wide16383.v outputs/wide16383-srgb.v --interpretation srgb
/opt/homebrew/bin/vips webpsave outputs/wide16383-srgb.v outputs/wide16383.webp --lossless
/opt/homebrew/bin/vips black outputs/wide16384.v 16384 1 --bands 3
/opt/homebrew/bin/vips copy outputs/wide16384.v outputs/wide16384-srgb.v --interpretation srgb
/opt/homebrew/bin/vips webpsave outputs/wide16384-srgb.v outputs/wide16384.webp --lossless
/opt/homebrew/bin/vips black outputs/wide16385.v 16385 1 --bands 3
/opt/homebrew/bin/vips copy outputs/wide16385.v outputs/wide16385-srgb.v --interpretation srgb
/opt/homebrew/bin/vips webpsave outputs/wide16385-srgb.v outputs/wide16385.webp --lossless
/opt/homebrew/bin/vipsheader -a fixtures/viprs_rgb.webp
/opt/homebrew/bin/vips getpoint fixtures/viprs_rgb.webp 0 0
/opt/homebrew/bin/vips getpoint fixtures/viprs_rgb.webp 1 0
/opt/homebrew/bin/vips getpoint fixtures/viprs_rgb.webp 2 0
/opt/homebrew/bin/vips getpoint fixtures/viprs_rgb.webp 3 0
/opt/homebrew/bin/vips getpoint fixtures/viprs_rgb.webp 0 1
/opt/homebrew/bin/vips getpoint fixtures/viprs_rgb.webp 1 1
/opt/homebrew/bin/vips getpoint fixtures/viprs_rgb.webp 2 1
/opt/homebrew/bin/vips getpoint fixtures/viprs_rgb.webp 3 1
/opt/homebrew/bin/vips getpoint fixtures/viprs_rgb.webp 0 2
/opt/homebrew/bin/vips getpoint fixtures/viprs_rgb.webp 1 2
/opt/homebrew/bin/vips getpoint fixtures/viprs_rgb.webp 2 2
/opt/homebrew/bin/vips getpoint fixtures/viprs_rgb.webp 3 2
/opt/homebrew/bin/vipsheader -a fixtures/viprs_rgba.webp
/opt/homebrew/bin/vips getpoint fixtures/viprs_rgba.webp 0 0
/opt/homebrew/bin/vips getpoint fixtures/viprs_rgba.webp 1 0
/opt/homebrew/bin/vips getpoint fixtures/viprs_rgba.webp 2 0
/opt/homebrew/bin/vips getpoint fixtures/viprs_rgba.webp 3 0
/opt/homebrew/bin/vips getpoint fixtures/viprs_rgba.webp 0 1
/opt/homebrew/bin/vips getpoint fixtures/viprs_rgba.webp 1 1
/opt/homebrew/bin/vips getpoint fixtures/viprs_rgba.webp 2 1
/opt/homebrew/bin/vips getpoint fixtures/viprs_rgba.webp 3 1
/opt/homebrew/bin/vips getpoint fixtures/viprs_rgba.webp 0 2
/opt/homebrew/bin/vips getpoint fixtures/viprs_rgba.webp 1 2
/opt/homebrew/bin/vips getpoint fixtures/viprs_rgba.webp 2 2
/opt/homebrew/bin/vips getpoint fixtures/viprs_rgba.webp 3 2
/opt/homebrew/bin/vipsheader -a fixtures/viprs_grey.webp
/opt/homebrew/bin/vips getpoint fixtures/viprs_grey.webp 0 0
/opt/homebrew/bin/vips getpoint fixtures/viprs_grey.webp 1 0
/opt/homebrew/bin/vips getpoint fixtures/viprs_grey.webp 2 0
/opt/homebrew/bin/vips getpoint fixtures/viprs_grey.webp 3 0
/opt/homebrew/bin/vips getpoint fixtures/viprs_grey.webp 0 1
/opt/homebrew/bin/vips getpoint fixtures/viprs_grey.webp 1 1
/opt/homebrew/bin/vips getpoint fixtures/viprs_grey.webp 2 1
/opt/homebrew/bin/vips getpoint fixtures/viprs_grey.webp 3 1
/opt/homebrew/bin/vips getpoint fixtures/viprs_grey.webp 0 2
/opt/homebrew/bin/vips getpoint fixtures/viprs_grey.webp 1 2
/opt/homebrew/bin/vips getpoint fixtures/viprs_grey.webp 2 2
/opt/homebrew/bin/vips getpoint fixtures/viprs_grey.webp 3 2
/opt/homebrew/bin/vipsheader -a fixtures/viprs_meta.webp
/opt/homebrew/bin/vips getpoint fixtures/viprs_meta.webp 0 0
/opt/homebrew/bin/vips getpoint fixtures/viprs_meta.webp 1 0
/opt/homebrew/bin/vips getpoint fixtures/viprs_meta.webp 2 0
/opt/homebrew/bin/vips getpoint fixtures/viprs_meta.webp 3 0
/opt/homebrew/bin/vips getpoint fixtures/viprs_meta.webp 0 1
/opt/homebrew/bin/vips getpoint fixtures/viprs_meta.webp 1 1
/opt/homebrew/bin/vips getpoint fixtures/viprs_meta.webp 2 1
/opt/homebrew/bin/vips getpoint fixtures/viprs_meta.webp 3 1
/opt/homebrew/bin/vips getpoint fixtures/viprs_meta.webp 0 2
/opt/homebrew/bin/vips getpoint fixtures/viprs_meta.webp 1 2
/opt/homebrew/bin/vips getpoint fixtures/viprs_meta.webp 2 2
/opt/homebrew/bin/vips getpoint fixtures/viprs_meta.webp 3 2
/opt/homebrew/bin/vips --version
