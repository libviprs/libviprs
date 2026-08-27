#!/bin/sh
# Every command capture.py ran, in order. Regenerate with
# `python3 capture.py` from this directory.
set -e

/opt/homebrew/bin/vips rawload fixtures/ll_rgb.raw outputs/ll_rgb.v 4 3 3 --format uchar
/opt/homebrew/bin/vips copy outputs/ll_rgb.v outputs/ll_rgb-srgb.v --interpretation srgb
/opt/homebrew/bin/vips jxlsave outputs/ll_rgb-srgb.v fixtures/ll_rgb.jxl --lossless --keep none
/opt/homebrew/bin/vips getpoint fixtures/ll_rgb.jxl 0 0
/opt/homebrew/bin/vips getpoint fixtures/ll_rgb.jxl 1 0
/opt/homebrew/bin/vips getpoint fixtures/ll_rgb.jxl 2 0
/opt/homebrew/bin/vips getpoint fixtures/ll_rgb.jxl 3 0
/opt/homebrew/bin/vips getpoint fixtures/ll_rgb.jxl 0 1
/opt/homebrew/bin/vips getpoint fixtures/ll_rgb.jxl 1 1
/opt/homebrew/bin/vips getpoint fixtures/ll_rgb.jxl 2 1
/opt/homebrew/bin/vips getpoint fixtures/ll_rgb.jxl 3 1
/opt/homebrew/bin/vips getpoint fixtures/ll_rgb.jxl 0 2
/opt/homebrew/bin/vips getpoint fixtures/ll_rgb.jxl 1 2
/opt/homebrew/bin/vips getpoint fixtures/ll_rgb.jxl 2 2
/opt/homebrew/bin/vips getpoint fixtures/ll_rgb.jxl 3 2
/opt/homebrew/bin/vips getpoint outputs/ll_rgb-srgb.v 0 0
/opt/homebrew/bin/vips getpoint outputs/ll_rgb-srgb.v 1 0
/opt/homebrew/bin/vips getpoint outputs/ll_rgb-srgb.v 2 0
/opt/homebrew/bin/vips getpoint outputs/ll_rgb-srgb.v 3 0
/opt/homebrew/bin/vips getpoint outputs/ll_rgb-srgb.v 0 1
/opt/homebrew/bin/vips getpoint outputs/ll_rgb-srgb.v 1 1
/opt/homebrew/bin/vips getpoint outputs/ll_rgb-srgb.v 2 1
/opt/homebrew/bin/vips getpoint outputs/ll_rgb-srgb.v 3 1
/opt/homebrew/bin/vips getpoint outputs/ll_rgb-srgb.v 0 2
/opt/homebrew/bin/vips getpoint outputs/ll_rgb-srgb.v 1 2
/opt/homebrew/bin/vips getpoint outputs/ll_rgb-srgb.v 2 2
/opt/homebrew/bin/vips getpoint outputs/ll_rgb-srgb.v 3 2
/opt/homebrew/bin/vipsheader -a fixtures/ll_rgb.jxl
/opt/homebrew/bin/vips rawload fixtures/ll_rgba.raw outputs/ll_rgba.v 4 3 4 --format uchar
/opt/homebrew/bin/vips copy outputs/ll_rgba.v outputs/ll_rgba-srgb.v --interpretation srgb
/opt/homebrew/bin/vips jxlsave outputs/ll_rgba-srgb.v fixtures/ll_rgba.jxl --lossless --keep none
/opt/homebrew/bin/vips getpoint fixtures/ll_rgba.jxl 0 0
/opt/homebrew/bin/vips getpoint fixtures/ll_rgba.jxl 1 0
/opt/homebrew/bin/vips getpoint fixtures/ll_rgba.jxl 2 0
/opt/homebrew/bin/vips getpoint fixtures/ll_rgba.jxl 3 0
/opt/homebrew/bin/vips getpoint fixtures/ll_rgba.jxl 0 1
/opt/homebrew/bin/vips getpoint fixtures/ll_rgba.jxl 1 1
/opt/homebrew/bin/vips getpoint fixtures/ll_rgba.jxl 2 1
/opt/homebrew/bin/vips getpoint fixtures/ll_rgba.jxl 3 1
/opt/homebrew/bin/vips getpoint fixtures/ll_rgba.jxl 0 2
/opt/homebrew/bin/vips getpoint fixtures/ll_rgba.jxl 1 2
/opt/homebrew/bin/vips getpoint fixtures/ll_rgba.jxl 2 2
/opt/homebrew/bin/vips getpoint fixtures/ll_rgba.jxl 3 2
/opt/homebrew/bin/vips getpoint outputs/ll_rgba-srgb.v 0 0
/opt/homebrew/bin/vips getpoint outputs/ll_rgba-srgb.v 1 0
/opt/homebrew/bin/vips getpoint outputs/ll_rgba-srgb.v 2 0
/opt/homebrew/bin/vips getpoint outputs/ll_rgba-srgb.v 3 0
/opt/homebrew/bin/vips getpoint outputs/ll_rgba-srgb.v 0 1
/opt/homebrew/bin/vips getpoint outputs/ll_rgba-srgb.v 1 1
/opt/homebrew/bin/vips getpoint outputs/ll_rgba-srgb.v 2 1
/opt/homebrew/bin/vips getpoint outputs/ll_rgba-srgb.v 3 1
/opt/homebrew/bin/vips getpoint outputs/ll_rgba-srgb.v 0 2
/opt/homebrew/bin/vips getpoint outputs/ll_rgba-srgb.v 1 2
/opt/homebrew/bin/vips getpoint outputs/ll_rgba-srgb.v 2 2
/opt/homebrew/bin/vips getpoint outputs/ll_rgba-srgb.v 3 2
/opt/homebrew/bin/vipsheader -a fixtures/ll_rgba.jxl
/opt/homebrew/bin/vips rawload fixtures/ll_grey.raw outputs/ll_grey.v 4 3 1 --format uchar
/opt/homebrew/bin/vips copy outputs/ll_grey.v outputs/ll_grey-b-w.v --interpretation b-w
/opt/homebrew/bin/vips jxlsave outputs/ll_grey-b-w.v fixtures/ll_grey.jxl --lossless --keep none
/opt/homebrew/bin/vips getpoint fixtures/ll_grey.jxl 0 0
/opt/homebrew/bin/vips getpoint fixtures/ll_grey.jxl 1 0
/opt/homebrew/bin/vips getpoint fixtures/ll_grey.jxl 2 0
/opt/homebrew/bin/vips getpoint fixtures/ll_grey.jxl 3 0
/opt/homebrew/bin/vips getpoint fixtures/ll_grey.jxl 0 1
/opt/homebrew/bin/vips getpoint fixtures/ll_grey.jxl 1 1
/opt/homebrew/bin/vips getpoint fixtures/ll_grey.jxl 2 1
/opt/homebrew/bin/vips getpoint fixtures/ll_grey.jxl 3 1
/opt/homebrew/bin/vips getpoint fixtures/ll_grey.jxl 0 2
/opt/homebrew/bin/vips getpoint fixtures/ll_grey.jxl 1 2
/opt/homebrew/bin/vips getpoint fixtures/ll_grey.jxl 2 2
/opt/homebrew/bin/vips getpoint fixtures/ll_grey.jxl 3 2
/opt/homebrew/bin/vips getpoint outputs/ll_grey-b-w.v 0 0
/opt/homebrew/bin/vips getpoint outputs/ll_grey-b-w.v 1 0
/opt/homebrew/bin/vips getpoint outputs/ll_grey-b-w.v 2 0
/opt/homebrew/bin/vips getpoint outputs/ll_grey-b-w.v 3 0
/opt/homebrew/bin/vips getpoint outputs/ll_grey-b-w.v 0 1
/opt/homebrew/bin/vips getpoint outputs/ll_grey-b-w.v 1 1
/opt/homebrew/bin/vips getpoint outputs/ll_grey-b-w.v 2 1
/opt/homebrew/bin/vips getpoint outputs/ll_grey-b-w.v 3 1
/opt/homebrew/bin/vips getpoint outputs/ll_grey-b-w.v 0 2
/opt/homebrew/bin/vips getpoint outputs/ll_grey-b-w.v 1 2
/opt/homebrew/bin/vips getpoint outputs/ll_grey-b-w.v 2 2
/opt/homebrew/bin/vips getpoint outputs/ll_grey-b-w.v 3 2
/opt/homebrew/bin/vipsheader -a fixtures/ll_grey.jxl
/opt/homebrew/bin/vips rawload fixtures/ll_rgb16.raw outputs/ll_rgb16.v 4 3 3 --format ushort
/opt/homebrew/bin/vips copy outputs/ll_rgb16.v outputs/ll_rgb16-rgb16.v --interpretation rgb16
/opt/homebrew/bin/vips jxlsave outputs/ll_rgb16-rgb16.v fixtures/ll_rgb16.jxl --lossless --keep none
/opt/homebrew/bin/vips getpoint fixtures/ll_rgb16.jxl 0 0
/opt/homebrew/bin/vips getpoint fixtures/ll_rgb16.jxl 1 0
/opt/homebrew/bin/vips getpoint fixtures/ll_rgb16.jxl 2 0
/opt/homebrew/bin/vips getpoint fixtures/ll_rgb16.jxl 3 0
/opt/homebrew/bin/vips getpoint fixtures/ll_rgb16.jxl 0 1
/opt/homebrew/bin/vips getpoint fixtures/ll_rgb16.jxl 1 1
/opt/homebrew/bin/vips getpoint fixtures/ll_rgb16.jxl 2 1
/opt/homebrew/bin/vips getpoint fixtures/ll_rgb16.jxl 3 1
/opt/homebrew/bin/vips getpoint fixtures/ll_rgb16.jxl 0 2
/opt/homebrew/bin/vips getpoint fixtures/ll_rgb16.jxl 1 2
/opt/homebrew/bin/vips getpoint fixtures/ll_rgb16.jxl 2 2
/opt/homebrew/bin/vips getpoint fixtures/ll_rgb16.jxl 3 2
/opt/homebrew/bin/vips getpoint outputs/ll_rgb16-rgb16.v 0 0
/opt/homebrew/bin/vips getpoint outputs/ll_rgb16-rgb16.v 1 0
/opt/homebrew/bin/vips getpoint outputs/ll_rgb16-rgb16.v 2 0
/opt/homebrew/bin/vips getpoint outputs/ll_rgb16-rgb16.v 3 0
/opt/homebrew/bin/vips getpoint outputs/ll_rgb16-rgb16.v 0 1
/opt/homebrew/bin/vips getpoint outputs/ll_rgb16-rgb16.v 1 1
/opt/homebrew/bin/vips getpoint outputs/ll_rgb16-rgb16.v 2 1
/opt/homebrew/bin/vips getpoint outputs/ll_rgb16-rgb16.v 3 1
/opt/homebrew/bin/vips getpoint outputs/ll_rgb16-rgb16.v 0 2
/opt/homebrew/bin/vips getpoint outputs/ll_rgb16-rgb16.v 1 2
/opt/homebrew/bin/vips getpoint outputs/ll_rgb16-rgb16.v 2 2
/opt/homebrew/bin/vips getpoint outputs/ll_rgb16-rgb16.v 3 2
/opt/homebrew/bin/vipsheader -a fixtures/ll_rgb16.jxl
/opt/homebrew/bin/vips rawload fixtures/ll_f32.raw outputs/ll_f32.v 4 3 3 --format float
/opt/homebrew/bin/vips copy outputs/ll_f32.v outputs/ll_f32-scrgb.v --interpretation scrgb
/opt/homebrew/bin/vips jxlsave outputs/ll_f32-scrgb.v fixtures/ll_f32.jxl --lossless --keep none
/opt/homebrew/bin/vips getpoint fixtures/ll_f32.jxl 0 0
/opt/homebrew/bin/vips getpoint fixtures/ll_f32.jxl 1 0
/opt/homebrew/bin/vips getpoint fixtures/ll_f32.jxl 2 0
/opt/homebrew/bin/vips getpoint fixtures/ll_f32.jxl 3 0
/opt/homebrew/bin/vips getpoint fixtures/ll_f32.jxl 0 1
/opt/homebrew/bin/vips getpoint fixtures/ll_f32.jxl 1 1
/opt/homebrew/bin/vips getpoint fixtures/ll_f32.jxl 2 1
/opt/homebrew/bin/vips getpoint fixtures/ll_f32.jxl 3 1
/opt/homebrew/bin/vips getpoint fixtures/ll_f32.jxl 0 2
/opt/homebrew/bin/vips getpoint fixtures/ll_f32.jxl 1 2
/opt/homebrew/bin/vips getpoint fixtures/ll_f32.jxl 2 2
/opt/homebrew/bin/vips getpoint fixtures/ll_f32.jxl 3 2
/opt/homebrew/bin/vips getpoint outputs/ll_f32-scrgb.v 0 0
/opt/homebrew/bin/vips getpoint outputs/ll_f32-scrgb.v 1 0
/opt/homebrew/bin/vips getpoint outputs/ll_f32-scrgb.v 2 0
/opt/homebrew/bin/vips getpoint outputs/ll_f32-scrgb.v 3 0
/opt/homebrew/bin/vips getpoint outputs/ll_f32-scrgb.v 0 1
/opt/homebrew/bin/vips getpoint outputs/ll_f32-scrgb.v 1 1
/opt/homebrew/bin/vips getpoint outputs/ll_f32-scrgb.v 2 1
/opt/homebrew/bin/vips getpoint outputs/ll_f32-scrgb.v 3 1
/opt/homebrew/bin/vips getpoint outputs/ll_f32-scrgb.v 0 2
/opt/homebrew/bin/vips getpoint outputs/ll_f32-scrgb.v 1 2
/opt/homebrew/bin/vips getpoint outputs/ll_f32-scrgb.v 2 2
/opt/homebrew/bin/vips getpoint outputs/ll_f32-scrgb.v 3 2
/opt/homebrew/bin/vipsheader -a fixtures/ll_f32.jxl
/opt/homebrew/bin/vips rawload fixtures/lossy_rgb.raw outputs/lossy_rgb.v 4 3 3 --format uchar
/opt/homebrew/bin/vips copy outputs/lossy_rgb.v outputs/lossy_rgb-srgb.v --interpretation srgb
/opt/homebrew/bin/vips jxlsave outputs/lossy_rgb-srgb.v fixtures/lossy_rgb.jxl --keep none
/opt/homebrew/bin/vips getpoint fixtures/lossy_rgb.jxl 0 0
/opt/homebrew/bin/vips getpoint fixtures/lossy_rgb.jxl 1 0
/opt/homebrew/bin/vips getpoint fixtures/lossy_rgb.jxl 2 0
/opt/homebrew/bin/vips getpoint fixtures/lossy_rgb.jxl 3 0
/opt/homebrew/bin/vips getpoint fixtures/lossy_rgb.jxl 0 1
/opt/homebrew/bin/vips getpoint fixtures/lossy_rgb.jxl 1 1
/opt/homebrew/bin/vips getpoint fixtures/lossy_rgb.jxl 2 1
/opt/homebrew/bin/vips getpoint fixtures/lossy_rgb.jxl 3 1
/opt/homebrew/bin/vips getpoint fixtures/lossy_rgb.jxl 0 2
/opt/homebrew/bin/vips getpoint fixtures/lossy_rgb.jxl 1 2
/opt/homebrew/bin/vips getpoint fixtures/lossy_rgb.jxl 2 2
/opt/homebrew/bin/vips getpoint fixtures/lossy_rgb.jxl 3 2
/opt/homebrew/bin/vipsheader -a fixtures/lossy_rgb.jxl
/opt/homebrew/bin/vips getpoint outputs/lossy_rgb-srgb.v 0 0
/opt/homebrew/bin/vips getpoint outputs/lossy_rgb-srgb.v 1 0
/opt/homebrew/bin/vips getpoint outputs/lossy_rgb-srgb.v 2 0
/opt/homebrew/bin/vips getpoint outputs/lossy_rgb-srgb.v 3 0
/opt/homebrew/bin/vips getpoint outputs/lossy_rgb-srgb.v 0 1
/opt/homebrew/bin/vips getpoint outputs/lossy_rgb-srgb.v 1 1
/opt/homebrew/bin/vips getpoint outputs/lossy_rgb-srgb.v 2 1
/opt/homebrew/bin/vips getpoint outputs/lossy_rgb-srgb.v 3 1
/opt/homebrew/bin/vips getpoint outputs/lossy_rgb-srgb.v 0 2
/opt/homebrew/bin/vips getpoint outputs/lossy_rgb-srgb.v 1 2
/opt/homebrew/bin/vips getpoint outputs/lossy_rgb-srgb.v 2 2
/opt/homebrew/bin/vips getpoint outputs/lossy_rgb-srgb.v 3 2
/opt/homebrew/bin/vips rawload fixtures/anim3.raw outputs/anim3.v 4 9 3 --format uchar
/opt/homebrew/bin/vips copy outputs/anim3.v outputs/anim3-srgb.v --interpretation srgb
/opt/homebrew/bin/vips jxlsave outputs/anim3-srgb.v fixtures/anim3.jxl --lossless --keep none --page-height 3
/opt/homebrew/bin/vipsheader -a fixtures/anim3.jxl
/opt/homebrew/bin/vipsheader -a fixtures/anim3.jxl[n=-1]
/opt/homebrew/bin/vips getpoint fixtures/anim3.jxl 0 0
/opt/homebrew/bin/vips getpoint fixtures/anim3.jxl 1 0
/opt/homebrew/bin/vips getpoint fixtures/anim3.jxl 2 0
/opt/homebrew/bin/vips getpoint fixtures/anim3.jxl 3 0
/opt/homebrew/bin/vips getpoint fixtures/anim3.jxl 0 1
/opt/homebrew/bin/vips getpoint fixtures/anim3.jxl 1 1
/opt/homebrew/bin/vips getpoint fixtures/anim3.jxl 2 1
/opt/homebrew/bin/vips getpoint fixtures/anim3.jxl 3 1
/opt/homebrew/bin/vips getpoint fixtures/anim3.jxl 0 2
/opt/homebrew/bin/vips getpoint fixtures/anim3.jxl 1 2
/opt/homebrew/bin/vips getpoint fixtures/anim3.jxl 2 2
/opt/homebrew/bin/vips getpoint fixtures/anim3.jxl 3 2
/opt/homebrew/bin/vips getpoint outputs/anim3-srgb.v 0 0
/opt/homebrew/bin/vips getpoint outputs/anim3-srgb.v 1 0
/opt/homebrew/bin/vips getpoint outputs/anim3-srgb.v 2 0
/opt/homebrew/bin/vips getpoint outputs/anim3-srgb.v 3 0
/opt/homebrew/bin/vips getpoint outputs/anim3-srgb.v 0 1
/opt/homebrew/bin/vips getpoint outputs/anim3-srgb.v 1 1
/opt/homebrew/bin/vips getpoint outputs/anim3-srgb.v 2 1
/opt/homebrew/bin/vips getpoint outputs/anim3-srgb.v 3 1
/opt/homebrew/bin/vips getpoint outputs/anim3-srgb.v 0 2
/opt/homebrew/bin/vips getpoint outputs/anim3-srgb.v 1 2
/opt/homebrew/bin/vips getpoint outputs/anim3-srgb.v 2 2
/opt/homebrew/bin/vips getpoint outputs/anim3-srgb.v 3 2
/opt/homebrew/bin/vips getpoint outputs/anim3-srgb.v 0 3
/opt/homebrew/bin/vips getpoint outputs/anim3-srgb.v 1 3
/opt/homebrew/bin/vips getpoint outputs/anim3-srgb.v 2 3
/opt/homebrew/bin/vips getpoint outputs/anim3-srgb.v 3 3
/opt/homebrew/bin/vips getpoint outputs/anim3-srgb.v 0 4
/opt/homebrew/bin/vips getpoint outputs/anim3-srgb.v 1 4
/opt/homebrew/bin/vips getpoint outputs/anim3-srgb.v 2 4
/opt/homebrew/bin/vips getpoint outputs/anim3-srgb.v 3 4
/opt/homebrew/bin/vips getpoint outputs/anim3-srgb.v 0 5
/opt/homebrew/bin/vips getpoint outputs/anim3-srgb.v 1 5
/opt/homebrew/bin/vips getpoint outputs/anim3-srgb.v 2 5
/opt/homebrew/bin/vips getpoint outputs/anim3-srgb.v 3 5
/opt/homebrew/bin/vips getpoint outputs/anim3-srgb.v 0 6
/opt/homebrew/bin/vips getpoint outputs/anim3-srgb.v 1 6
/opt/homebrew/bin/vips getpoint outputs/anim3-srgb.v 2 6
/opt/homebrew/bin/vips getpoint outputs/anim3-srgb.v 3 6
/opt/homebrew/bin/vips getpoint outputs/anim3-srgb.v 0 7
/opt/homebrew/bin/vips getpoint outputs/anim3-srgb.v 1 7
/opt/homebrew/bin/vips getpoint outputs/anim3-srgb.v 2 7
/opt/homebrew/bin/vips getpoint outputs/anim3-srgb.v 3 7
/opt/homebrew/bin/vips getpoint outputs/anim3-srgb.v 0 8
/opt/homebrew/bin/vips getpoint outputs/anim3-srgb.v 1 8
/opt/homebrew/bin/vips getpoint outputs/anim3-srgb.v 2 8
/opt/homebrew/bin/vips getpoint outputs/anim3-srgb.v 3 8
/opt/homebrew/bin/vips jpegsave outputs/ll_rgb-srgb.v outputs/src.jpg -Q 100
/opt/homebrew/bin/vips jxlsave outputs/src.jpg fixtures/meta.jxl --lossless --keep all
/opt/homebrew/bin/vipsheader -a fixtures/meta.jxl
/opt/homebrew/bin/vipsheader -f exif-data fixtures/meta.jxl
/opt/homebrew/bin/vipsheader -f exif-data outputs/src.jpg
/opt/homebrew/bin/vipsheader -f exif-data fixtures/meta.jxl
/opt/homebrew/bin/vipsheader -f exif-data outputs/src.jpg
/opt/homebrew/bin/vips getpoint fixtures/meta.jxl 0 0
/opt/homebrew/bin/vips getpoint fixtures/meta.jxl 1 0
/opt/homebrew/bin/vips getpoint fixtures/meta.jxl 2 0
/opt/homebrew/bin/vips getpoint fixtures/meta.jxl 3 0
/opt/homebrew/bin/vips getpoint fixtures/meta.jxl 0 1
/opt/homebrew/bin/vips getpoint fixtures/meta.jxl 1 1
/opt/homebrew/bin/vips getpoint fixtures/meta.jxl 2 1
/opt/homebrew/bin/vips getpoint fixtures/meta.jxl 3 1
/opt/homebrew/bin/vips getpoint fixtures/meta.jxl 0 2
/opt/homebrew/bin/vips getpoint fixtures/meta.jxl 1 2
/opt/homebrew/bin/vips getpoint fixtures/meta.jxl 2 2
/opt/homebrew/bin/vips getpoint fixtures/meta.jxl 3 2
# hand-built: meta_off0.jxl = JXL container + Exif(offset=0) + xml  + jxlc
/opt/homebrew/bin/vipsheader -a fixtures/meta_off0.jxl
/opt/homebrew/bin/vipsheader -f exif-data fixtures/meta_off0.jxl
/opt/homebrew/bin/vips getpoint fixtures/meta_off0.jxl 0 0
/opt/homebrew/bin/vips getpoint fixtures/meta_off0.jxl 1 0
/opt/homebrew/bin/vips getpoint fixtures/meta_off0.jxl 2 0
/opt/homebrew/bin/vips getpoint fixtures/meta_off0.jxl 3 0
/opt/homebrew/bin/vips getpoint fixtures/meta_off0.jxl 0 1
/opt/homebrew/bin/vips getpoint fixtures/meta_off0.jxl 1 1
/opt/homebrew/bin/vips getpoint fixtures/meta_off0.jxl 2 1
/opt/homebrew/bin/vips getpoint fixtures/meta_off0.jxl 3 1
/opt/homebrew/bin/vips getpoint fixtures/meta_off0.jxl 0 2
/opt/homebrew/bin/vips getpoint fixtures/meta_off0.jxl 1 2
/opt/homebrew/bin/vips getpoint fixtures/meta_off0.jxl 2 2
/opt/homebrew/bin/vips getpoint fixtures/meta_off0.jxl 3 2
/opt/homebrew/bin/vipsheader -f xmp-data fixtures/meta_off0.jxl
# hand-built: meta_off6.jxl = JXL container + Exif(offset=6) + jxlc
/opt/homebrew/bin/vipsheader -a fixtures/meta_off6.jxl
/opt/homebrew/bin/vipsheader -f exif-data fixtures/meta_off6.jxl
/opt/homebrew/bin/vips getpoint fixtures/meta_off6.jxl 0 0
/opt/homebrew/bin/vips getpoint fixtures/meta_off6.jxl 1 0
/opt/homebrew/bin/vips getpoint fixtures/meta_off6.jxl 2 0
/opt/homebrew/bin/vips getpoint fixtures/meta_off6.jxl 3 0
/opt/homebrew/bin/vips getpoint fixtures/meta_off6.jxl 0 1
/opt/homebrew/bin/vips getpoint fixtures/meta_off6.jxl 1 1
/opt/homebrew/bin/vips getpoint fixtures/meta_off6.jxl 2 1
/opt/homebrew/bin/vips getpoint fixtures/meta_off6.jxl 3 1
/opt/homebrew/bin/vips getpoint fixtures/meta_off6.jxl 0 2
/opt/homebrew/bin/vips getpoint fixtures/meta_off6.jxl 1 2
/opt/homebrew/bin/vips getpoint fixtures/meta_off6.jxl 2 2
/opt/homebrew/bin/vips getpoint fixtures/meta_off6.jxl 3 2
# hand-built: meta_badoffset.jxl = JXL container + Exif(offset=999, payload 10 bytes) + jxlc
/opt/homebrew/bin/vipsheader -a fixtures/meta_badoffset.jxl
# hand-built: outputs/meta_prefixed.jxl = JXL container + Exif(payload already carrying the Exif\0\0 prefix) + jxlc
/opt/homebrew/bin/vipsheader -f exif-data outputs/meta_prefixed.jxl
/opt/homebrew/bin/vips rawload fixtures/tiny_1x1.raw outputs/tiny_1x1.v 1 1 3 --format uchar
/opt/homebrew/bin/vips copy outputs/tiny_1x1.v outputs/tiny_1x1-srgb.v --interpretation srgb
/opt/homebrew/bin/vips jxlsave outputs/tiny_1x1-srgb.v outputs/tiny_1x1.jxl --lossless --keep none
/opt/homebrew/bin/vipsheader outputs/tiny_1x1.jxl
/opt/homebrew/bin/vips rawload fixtures/tiny_2x1.raw outputs/tiny_2x1.v 2 1 3 --format uchar
/opt/homebrew/bin/vips copy outputs/tiny_2x1.v outputs/tiny_2x1-srgb.v --interpretation srgb
/opt/homebrew/bin/vips jxlsave outputs/tiny_2x1-srgb.v outputs/tiny_2x1.jxl --lossless --keep none
/opt/homebrew/bin/vipsheader outputs/tiny_2x1.jxl
/opt/homebrew/bin/vips rawload fixtures/tiny_1x2.raw outputs/tiny_1x2.v 1 2 3 --format uchar
/opt/homebrew/bin/vips copy outputs/tiny_1x2.v outputs/tiny_1x2-srgb.v --interpretation srgb
/opt/homebrew/bin/vips jxlsave outputs/tiny_1x2-srgb.v outputs/tiny_1x2.jxl --lossless --keep none
/opt/homebrew/bin/vipsheader outputs/tiny_1x2.jxl
/opt/homebrew/bin/vips rawload fixtures/tiny_2x2.raw outputs/tiny_2x2.v 2 2 3 --format uchar
/opt/homebrew/bin/vips copy outputs/tiny_2x2.v outputs/tiny_2x2-srgb.v --interpretation srgb
/opt/homebrew/bin/vips jxlsave outputs/tiny_2x2-srgb.v outputs/tiny_2x2.jxl --lossless --keep none
/opt/homebrew/bin/vipsheader outputs/tiny_2x2.jxl
/opt/homebrew/bin/vips rawload fixtures/tiny_4x1.raw outputs/tiny_4x1.v 4 1 3 --format uchar
/opt/homebrew/bin/vips copy outputs/tiny_4x1.v outputs/tiny_4x1-srgb.v --interpretation srgb
/opt/homebrew/bin/vips jxlsave outputs/tiny_4x1-srgb.v outputs/tiny_4x1.jxl --lossless --keep none
/opt/homebrew/bin/vipsheader outputs/tiny_4x1.jxl
/opt/homebrew/bin/vipsheader -a outputs/viprs_rgb.jxl
/opt/homebrew/bin/vips getpoint outputs/viprs_rgb.jxl 0 0
/opt/homebrew/bin/vips getpoint outputs/viprs_rgb.jxl 1 0
/opt/homebrew/bin/vips getpoint outputs/viprs_rgb.jxl 2 0
/opt/homebrew/bin/vips getpoint outputs/viprs_rgb.jxl 3 0
/opt/homebrew/bin/vips getpoint outputs/viprs_rgb.jxl 0 1
/opt/homebrew/bin/vips getpoint outputs/viprs_rgb.jxl 1 1
/opt/homebrew/bin/vips getpoint outputs/viprs_rgb.jxl 2 1
/opt/homebrew/bin/vips getpoint outputs/viprs_rgb.jxl 3 1
/opt/homebrew/bin/vips getpoint outputs/viprs_rgb.jxl 0 2
/opt/homebrew/bin/vips getpoint outputs/viprs_rgb.jxl 1 2
/opt/homebrew/bin/vips getpoint outputs/viprs_rgb.jxl 2 2
/opt/homebrew/bin/vips getpoint outputs/viprs_rgb.jxl 3 2
/opt/homebrew/bin/vipsheader -a outputs/viprs_rgba.jxl
/opt/homebrew/bin/vips getpoint outputs/viprs_rgba.jxl 0 0
/opt/homebrew/bin/vips getpoint outputs/viprs_rgba.jxl 1 0
/opt/homebrew/bin/vips getpoint outputs/viprs_rgba.jxl 2 0
/opt/homebrew/bin/vips getpoint outputs/viprs_rgba.jxl 3 0
/opt/homebrew/bin/vips getpoint outputs/viprs_rgba.jxl 0 1
/opt/homebrew/bin/vips getpoint outputs/viprs_rgba.jxl 1 1
/opt/homebrew/bin/vips getpoint outputs/viprs_rgba.jxl 2 1
/opt/homebrew/bin/vips getpoint outputs/viprs_rgba.jxl 3 1
/opt/homebrew/bin/vips getpoint outputs/viprs_rgba.jxl 0 2
/opt/homebrew/bin/vips getpoint outputs/viprs_rgba.jxl 1 2
/opt/homebrew/bin/vips getpoint outputs/viprs_rgba.jxl 2 2
/opt/homebrew/bin/vips getpoint outputs/viprs_rgba.jxl 3 2
/opt/homebrew/bin/vipsheader -a outputs/viprs_grey.jxl
/opt/homebrew/bin/vips getpoint outputs/viprs_grey.jxl 0 0
/opt/homebrew/bin/vips getpoint outputs/viprs_grey.jxl 1 0
/opt/homebrew/bin/vips getpoint outputs/viprs_grey.jxl 2 0
/opt/homebrew/bin/vips getpoint outputs/viprs_grey.jxl 3 0
/opt/homebrew/bin/vips getpoint outputs/viprs_grey.jxl 0 1
/opt/homebrew/bin/vips getpoint outputs/viprs_grey.jxl 1 1
/opt/homebrew/bin/vips getpoint outputs/viprs_grey.jxl 2 1
/opt/homebrew/bin/vips getpoint outputs/viprs_grey.jxl 3 1
/opt/homebrew/bin/vips getpoint outputs/viprs_grey.jxl 0 2
/opt/homebrew/bin/vips getpoint outputs/viprs_grey.jxl 1 2
/opt/homebrew/bin/vips getpoint outputs/viprs_grey.jxl 2 2
/opt/homebrew/bin/vips getpoint outputs/viprs_grey.jxl 3 2
/opt/homebrew/bin/vipsheader -a outputs/viprs_rgb16.jxl
/opt/homebrew/bin/vips getpoint outputs/viprs_rgb16.jxl 0 0
/opt/homebrew/bin/vips getpoint outputs/viprs_rgb16.jxl 1 0
/opt/homebrew/bin/vips getpoint outputs/viprs_rgb16.jxl 2 0
/opt/homebrew/bin/vips getpoint outputs/viprs_rgb16.jxl 3 0
/opt/homebrew/bin/vips getpoint outputs/viprs_rgb16.jxl 0 1
/opt/homebrew/bin/vips getpoint outputs/viprs_rgb16.jxl 1 1
/opt/homebrew/bin/vips getpoint outputs/viprs_rgb16.jxl 2 1
/opt/homebrew/bin/vips getpoint outputs/viprs_rgb16.jxl 3 1
/opt/homebrew/bin/vips getpoint outputs/viprs_rgb16.jxl 0 2
/opt/homebrew/bin/vips getpoint outputs/viprs_rgb16.jxl 1 2
/opt/homebrew/bin/vips getpoint outputs/viprs_rgb16.jxl 2 2
/opt/homebrew/bin/vips getpoint outputs/viprs_rgb16.jxl 3 2
/opt/homebrew/bin/vips --version
