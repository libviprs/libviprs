#!/bin/sh
# Every command capture.py ran, in order. Regenerate with
# `python3 capture.py` from this directory.
set -e

/opt/homebrew/bin/vips rawload outputs/rgb.raw outputs/rgb.v 4 3 3 --format uchar
/opt/homebrew/bin/vips copy outputs/rgb.v outputs/rgb-srgb.v --interpretation srgb
/opt/homebrew/bin/vips jp2ksave outputs/rgb-srgb.v outputs/suffix.jp2 --lossless
/opt/homebrew/bin/vips jp2ksave outputs/rgb-srgb.v outputs/suffix.j2k --lossless
/opt/homebrew/bin/vips jp2ksave outputs/rgb-srgb.v outputs/suffix.jpt --lossless
/opt/homebrew/bin/vips jp2ksave outputs/rgb-srgb.v outputs/suffix.j2c --lossless
/opt/homebrew/bin/vips jp2ksave outputs/rgb-srgb.v outputs/suffix.jpc --lossless
/opt/homebrew/bin/vips rawload outputs/rgb_lossless_src.raw outputs/rgb_lossless_src.v 4 3 3 --format uchar
/opt/homebrew/bin/vips copy outputs/rgb_lossless_src.v outputs/rgb_lossless_src-srgb.v --interpretation srgb
/opt/homebrew/bin/vips jp2ksave outputs/rgb_lossless_src-srgb.v fixtures/rgb_lossless.jp2 --lossless
/opt/homebrew/bin/vipsheader -a fixtures/rgb_lossless.jp2
/opt/homebrew/bin/vips rawsave fixtures/rgb_lossless.jp2 outputs/rgb_lossless.raw
/opt/homebrew/bin/vips getpoint fixtures/rgb_lossless.jp2 0 0
/opt/homebrew/bin/vips getpoint fixtures/rgb_lossless.jp2 1 0
/opt/homebrew/bin/vips getpoint fixtures/rgb_lossless.jp2 2 0
/opt/homebrew/bin/vips getpoint fixtures/rgb_lossless.jp2 3 0
/opt/homebrew/bin/vips getpoint fixtures/rgb_lossless.jp2 0 1
/opt/homebrew/bin/vips getpoint fixtures/rgb_lossless.jp2 1 1
/opt/homebrew/bin/vips getpoint fixtures/rgb_lossless.jp2 2 1
/opt/homebrew/bin/vips getpoint fixtures/rgb_lossless.jp2 3 1
/opt/homebrew/bin/vips getpoint fixtures/rgb_lossless.jp2 0 2
/opt/homebrew/bin/vips getpoint fixtures/rgb_lossless.jp2 1 2
/opt/homebrew/bin/vips getpoint fixtures/rgb_lossless.jp2 2 2
/opt/homebrew/bin/vips getpoint fixtures/rgb_lossless.jp2 3 2
/opt/homebrew/bin/vips rawload outputs/rgba_lossless_src.raw outputs/rgba_lossless_src.v 4 3 4 --format uchar
/opt/homebrew/bin/vips copy outputs/rgba_lossless_src.v outputs/rgba_lossless_src-srgb.v --interpretation srgb
/opt/homebrew/bin/vips jp2ksave outputs/rgba_lossless_src-srgb.v fixtures/rgba_lossless.jp2 --lossless
/opt/homebrew/bin/vipsheader -a fixtures/rgba_lossless.jp2
/opt/homebrew/bin/vips rawsave fixtures/rgba_lossless.jp2 outputs/rgba_lossless.raw
/opt/homebrew/bin/vips getpoint fixtures/rgba_lossless.jp2 0 0
/opt/homebrew/bin/vips getpoint fixtures/rgba_lossless.jp2 1 0
/opt/homebrew/bin/vips getpoint fixtures/rgba_lossless.jp2 2 0
/opt/homebrew/bin/vips getpoint fixtures/rgba_lossless.jp2 3 0
/opt/homebrew/bin/vips getpoint fixtures/rgba_lossless.jp2 0 1
/opt/homebrew/bin/vips getpoint fixtures/rgba_lossless.jp2 1 1
/opt/homebrew/bin/vips getpoint fixtures/rgba_lossless.jp2 2 1
/opt/homebrew/bin/vips getpoint fixtures/rgba_lossless.jp2 3 1
/opt/homebrew/bin/vips getpoint fixtures/rgba_lossless.jp2 0 2
/opt/homebrew/bin/vips getpoint fixtures/rgba_lossless.jp2 1 2
/opt/homebrew/bin/vips getpoint fixtures/rgba_lossless.jp2 2 2
/opt/homebrew/bin/vips getpoint fixtures/rgba_lossless.jp2 3 2
/opt/homebrew/bin/vips rawload outputs/cmyk_lossless_src.raw outputs/cmyk_lossless_src.v 4 3 4 --format uchar
/opt/homebrew/bin/vips copy outputs/cmyk_lossless_src.v outputs/cmyk_lossless_src-cmyk.v --interpretation cmyk
/opt/homebrew/bin/vips jp2ksave outputs/cmyk_lossless_src-cmyk.v fixtures/cmyk_lossless.jp2 --lossless
/opt/homebrew/bin/vipsheader -a fixtures/cmyk_lossless.jp2
/opt/homebrew/bin/vips rawsave fixtures/cmyk_lossless.jp2 outputs/cmyk_lossless.raw
/opt/homebrew/bin/vips getpoint fixtures/cmyk_lossless.jp2 0 0
/opt/homebrew/bin/vips getpoint fixtures/cmyk_lossless.jp2 1 0
/opt/homebrew/bin/vips getpoint fixtures/cmyk_lossless.jp2 2 0
/opt/homebrew/bin/vips getpoint fixtures/cmyk_lossless.jp2 3 0
/opt/homebrew/bin/vips getpoint fixtures/cmyk_lossless.jp2 0 1
/opt/homebrew/bin/vips getpoint fixtures/cmyk_lossless.jp2 1 1
/opt/homebrew/bin/vips getpoint fixtures/cmyk_lossless.jp2 2 1
/opt/homebrew/bin/vips getpoint fixtures/cmyk_lossless.jp2 3 1
/opt/homebrew/bin/vips getpoint fixtures/cmyk_lossless.jp2 0 2
/opt/homebrew/bin/vips getpoint fixtures/cmyk_lossless.jp2 1 2
/opt/homebrew/bin/vips getpoint fixtures/cmyk_lossless.jp2 2 2
/opt/homebrew/bin/vips getpoint fixtures/cmyk_lossless.jp2 3 2
/opt/homebrew/bin/vips rawload outputs/carrier_uchar.raw outputs/carrier_uchar.v 12 1 1 --format uchar
/opt/homebrew/bin/vips copy outputs/carrier_uchar.v outputs/carrier_uchar-b-w.v --interpretation b-w
/opt/homebrew/bin/vips jp2ksave outputs/carrier_uchar-b-w.v outputs/carrier_uchar.jp2 --lossless
/opt/homebrew/bin/vipsheader outputs/carrier_uchar.jp2
/opt/homebrew/bin/vipsheader -a outputs/carrier_uchar.jp2
/opt/homebrew/bin/vips getpoint outputs/carrier_uchar.jp2 0 0
/opt/homebrew/bin/vips getpoint outputs/carrier_uchar.jp2 1 0
/opt/homebrew/bin/vips getpoint outputs/carrier_uchar.jp2 2 0
/opt/homebrew/bin/vips getpoint outputs/carrier_uchar.jp2 3 0
/opt/homebrew/bin/vips getpoint outputs/carrier_uchar.jp2 4 0
/opt/homebrew/bin/vips getpoint outputs/carrier_uchar.jp2 5 0
/opt/homebrew/bin/vips getpoint outputs/carrier_uchar.jp2 6 0
/opt/homebrew/bin/vips getpoint outputs/carrier_uchar.jp2 7 0
/opt/homebrew/bin/vips getpoint outputs/carrier_uchar.jp2 8 0
/opt/homebrew/bin/vips getpoint outputs/carrier_uchar.jp2 9 0
/opt/homebrew/bin/vips getpoint outputs/carrier_uchar.jp2 10 0
/opt/homebrew/bin/vips getpoint outputs/carrier_uchar.jp2 11 0
/opt/homebrew/bin/vips rawload outputs/carrier_char.raw outputs/carrier_char.v 12 1 1 --format char
/opt/homebrew/bin/vips copy outputs/carrier_char.v outputs/carrier_char-b-w.v --interpretation b-w
/opt/homebrew/bin/vips jp2ksave outputs/carrier_char-b-w.v outputs/carrier_char.jp2 --lossless
/opt/homebrew/bin/vipsheader outputs/carrier_char.jp2
/opt/homebrew/bin/vipsheader -a outputs/carrier_char.jp2
/opt/homebrew/bin/vips getpoint outputs/carrier_char.jp2 0 0
/opt/homebrew/bin/vips getpoint outputs/carrier_char.jp2 1 0
/opt/homebrew/bin/vips getpoint outputs/carrier_char.jp2 2 0
/opt/homebrew/bin/vips getpoint outputs/carrier_char.jp2 3 0
/opt/homebrew/bin/vips getpoint outputs/carrier_char.jp2 4 0
/opt/homebrew/bin/vips getpoint outputs/carrier_char.jp2 5 0
/opt/homebrew/bin/vips getpoint outputs/carrier_char.jp2 6 0
/opt/homebrew/bin/vips getpoint outputs/carrier_char.jp2 7 0
/opt/homebrew/bin/vips getpoint outputs/carrier_char.jp2 8 0
/opt/homebrew/bin/vips getpoint outputs/carrier_char.jp2 9 0
/opt/homebrew/bin/vips getpoint outputs/carrier_char.jp2 10 0
/opt/homebrew/bin/vips getpoint outputs/carrier_char.jp2 11 0
/opt/homebrew/bin/vips rawload outputs/carrier_ushort.raw outputs/carrier_ushort.v 12 1 1 --format ushort
/opt/homebrew/bin/vips copy outputs/carrier_ushort.v outputs/carrier_ushort-b-w.v --interpretation b-w
/opt/homebrew/bin/vips jp2ksave outputs/carrier_ushort-b-w.v outputs/carrier_ushort.jp2 --lossless
/opt/homebrew/bin/vipsheader outputs/carrier_ushort.jp2
/opt/homebrew/bin/vipsheader -a outputs/carrier_ushort.jp2
/opt/homebrew/bin/vips getpoint outputs/carrier_ushort.jp2 0 0
/opt/homebrew/bin/vips getpoint outputs/carrier_ushort.jp2 1 0
/opt/homebrew/bin/vips getpoint outputs/carrier_ushort.jp2 2 0
/opt/homebrew/bin/vips getpoint outputs/carrier_ushort.jp2 3 0
/opt/homebrew/bin/vips getpoint outputs/carrier_ushort.jp2 4 0
/opt/homebrew/bin/vips getpoint outputs/carrier_ushort.jp2 5 0
/opt/homebrew/bin/vips getpoint outputs/carrier_ushort.jp2 6 0
/opt/homebrew/bin/vips getpoint outputs/carrier_ushort.jp2 7 0
/opt/homebrew/bin/vips getpoint outputs/carrier_ushort.jp2 8 0
/opt/homebrew/bin/vips getpoint outputs/carrier_ushort.jp2 9 0
/opt/homebrew/bin/vips getpoint outputs/carrier_ushort.jp2 10 0
/opt/homebrew/bin/vips getpoint outputs/carrier_ushort.jp2 11 0
/opt/homebrew/bin/vips rawload outputs/carrier_short.raw outputs/carrier_short.v 12 1 1 --format short
/opt/homebrew/bin/vips copy outputs/carrier_short.v outputs/carrier_short-b-w.v --interpretation b-w
/opt/homebrew/bin/vips jp2ksave outputs/carrier_short-b-w.v outputs/carrier_short.jp2 --lossless
/opt/homebrew/bin/vipsheader outputs/carrier_short.jp2
/opt/homebrew/bin/vipsheader -a outputs/carrier_short.jp2
/opt/homebrew/bin/vips getpoint outputs/carrier_short.jp2 0 0
/opt/homebrew/bin/vips getpoint outputs/carrier_short.jp2 1 0
/opt/homebrew/bin/vips getpoint outputs/carrier_short.jp2 2 0
/opt/homebrew/bin/vips getpoint outputs/carrier_short.jp2 3 0
/opt/homebrew/bin/vips getpoint outputs/carrier_short.jp2 4 0
/opt/homebrew/bin/vips getpoint outputs/carrier_short.jp2 5 0
/opt/homebrew/bin/vips getpoint outputs/carrier_short.jp2 6 0
/opt/homebrew/bin/vips getpoint outputs/carrier_short.jp2 7 0
/opt/homebrew/bin/vips getpoint outputs/carrier_short.jp2 8 0
/opt/homebrew/bin/vips getpoint outputs/carrier_short.jp2 9 0
/opt/homebrew/bin/vips getpoint outputs/carrier_short.jp2 10 0
/opt/homebrew/bin/vips getpoint outputs/carrier_short.jp2 11 0
/opt/homebrew/bin/vips rawload outputs/carrier_uint.raw outputs/carrier_uint.v 12 1 1 --format uint
/opt/homebrew/bin/vips copy outputs/carrier_uint.v outputs/carrier_uint-b-w.v --interpretation b-w
/opt/homebrew/bin/vips jp2ksave outputs/carrier_uint-b-w.v outputs/carrier_uint.jp2 --lossless
/opt/homebrew/bin/vipsheader outputs/carrier_uint.jp2
/opt/homebrew/bin/vipsheader -a outputs/carrier_uint.jp2
/opt/homebrew/bin/vips getpoint outputs/carrier_uint.jp2 0 0
/opt/homebrew/bin/vips getpoint outputs/carrier_uint.jp2 1 0
/opt/homebrew/bin/vips getpoint outputs/carrier_uint.jp2 2 0
/opt/homebrew/bin/vips getpoint outputs/carrier_uint.jp2 3 0
/opt/homebrew/bin/vips getpoint outputs/carrier_uint.jp2 4 0
/opt/homebrew/bin/vips getpoint outputs/carrier_uint.jp2 5 0
/opt/homebrew/bin/vips getpoint outputs/carrier_uint.jp2 6 0
/opt/homebrew/bin/vips getpoint outputs/carrier_uint.jp2 7 0
/opt/homebrew/bin/vips getpoint outputs/carrier_uint.jp2 8 0
/opt/homebrew/bin/vips getpoint outputs/carrier_uint.jp2 9 0
/opt/homebrew/bin/vips getpoint outputs/carrier_uint.jp2 10 0
/opt/homebrew/bin/vips getpoint outputs/carrier_uint.jp2 11 0
/opt/homebrew/bin/vips rawload outputs/carrier_int.raw outputs/carrier_int.v 12 1 1 --format int
/opt/homebrew/bin/vips copy outputs/carrier_int.v outputs/carrier_int-b-w.v --interpretation b-w
/opt/homebrew/bin/vips jp2ksave outputs/carrier_int-b-w.v outputs/carrier_int.jp2 --lossless
/opt/homebrew/bin/vipsheader outputs/carrier_int.jp2
/opt/homebrew/bin/vipsheader -a outputs/carrier_int.jp2
/opt/homebrew/bin/vips getpoint outputs/carrier_int.jp2 0 0
/opt/homebrew/bin/vips getpoint outputs/carrier_int.jp2 1 0
/opt/homebrew/bin/vips getpoint outputs/carrier_int.jp2 2 0
/opt/homebrew/bin/vips getpoint outputs/carrier_int.jp2 3 0
/opt/homebrew/bin/vips getpoint outputs/carrier_int.jp2 4 0
/opt/homebrew/bin/vips getpoint outputs/carrier_int.jp2 5 0
/opt/homebrew/bin/vips getpoint outputs/carrier_int.jp2 6 0
/opt/homebrew/bin/vips getpoint outputs/carrier_int.jp2 7 0
/opt/homebrew/bin/vips getpoint outputs/carrier_int.jp2 8 0
/opt/homebrew/bin/vips getpoint outputs/carrier_int.jp2 9 0
/opt/homebrew/bin/vips getpoint outputs/carrier_int.jp2 10 0
/opt/homebrew/bin/vips getpoint outputs/carrier_int.jp2 11 0
/opt/homebrew/bin/vips rawload outputs/carrier_float.raw outputs/carrier_float.v 12 1 1 --format float
/opt/homebrew/bin/vips copy outputs/carrier_float.v outputs/carrier_float-b-w.v --interpretation b-w
/opt/homebrew/bin/vips jp2ksave outputs/carrier_float-b-w.v outputs/carrier_float.jp2 --lossless
/opt/homebrew/bin/vips rawload outputs/carrier_double.raw outputs/carrier_double.v 12 1 1 --format double
/opt/homebrew/bin/vips copy outputs/carrier_double.v outputs/carrier_double-b-w.v --interpretation b-w
/opt/homebrew/bin/vips jp2ksave outputs/carrier_double-b-w.v outputs/carrier_double.jp2 --lossless
/opt/homebrew/bin/vips rawload outputs/wide_uint.raw outputs/wide_uint.v 5 1 1 --format uint
/opt/homebrew/bin/vips copy outputs/wide_uint.v outputs/wide_uint-grey16.v --interpretation grey16
/opt/homebrew/bin/vips jp2ksave outputs/wide_uint-grey16.v fixtures/uint31.jp2 --lossless
/opt/homebrew/bin/vips getpoint fixtures/uint31.jp2 0 0
/opt/homebrew/bin/vips getpoint fixtures/uint31.jp2 1 0
/opt/homebrew/bin/vips getpoint fixtures/uint31.jp2 2 0
/opt/homebrew/bin/vips getpoint fixtures/uint31.jp2 3 0
/opt/homebrew/bin/vips getpoint fixtures/uint31.jp2 4 0
/opt/homebrew/bin/vipsheader -a fixtures/uint31.jp2
/opt/homebrew/bin/vips rawload outputs/wide_int.raw outputs/wide_int.v 5 1 1 --format int
/opt/homebrew/bin/vips copy outputs/wide_int.v outputs/wide_int-grey16.v --interpretation grey16
/opt/homebrew/bin/vips jp2ksave outputs/wide_int-grey16.v fixtures/int31.jp2 --lossless
/opt/homebrew/bin/vips getpoint fixtures/int31.jp2 0 0
/opt/homebrew/bin/vips getpoint fixtures/int31.jp2 1 0
/opt/homebrew/bin/vips getpoint fixtures/int31.jp2 2 0
/opt/homebrew/bin/vips getpoint fixtures/int31.jp2 3 0
/opt/homebrew/bin/vips getpoint fixtures/int31.jp2 4 0
/opt/homebrew/bin/vipsheader -a fixtures/int31.jp2
/opt/homebrew/bin/vips rawload outputs/onecomp.raw outputs/onecomp.v 4 3 1 --format uchar
/opt/homebrew/bin/vips copy outputs/onecomp.v outputs/onecomp-b-w.v --interpretation b-w
/opt/homebrew/bin/vips jp2ksave outputs/onecomp-b-w.v outputs/onecomp.jp2 --lossless
/opt/homebrew/bin/vips rawload outputs/threecomp16.raw outputs/threecomp16.v 4 3 3 --format ushort
/opt/homebrew/bin/vips copy outputs/threecomp16.v outputs/threecomp16-rgb16.v --interpretation rgb16
/opt/homebrew/bin/vips jp2ksave outputs/threecomp16-rgb16.v outputs/threecomp16.jp2 --lossless
/opt/homebrew/bin/vips getpoint outputs/cs_1_component_8bit_12.jp2 0 0
/opt/homebrew/bin/vipsheader outputs/cs_1_component_8bit_12.jp2
/opt/homebrew/bin/vipsheader -a outputs/cs_1_component_8bit_12.jp2
/opt/homebrew/bin/vips getpoint outputs/cs_1_component_8bit_14.jp2 0 0
/opt/homebrew/bin/vipsheader outputs/cs_1_component_8bit_14.jp2
/opt/homebrew/bin/vipsheader -a outputs/cs_1_component_8bit_14.jp2
/opt/homebrew/bin/vips getpoint outputs/cs_1_component_8bit_16.jp2 0 0
/opt/homebrew/bin/vipsheader outputs/cs_1_component_8bit_16.jp2
/opt/homebrew/bin/vipsheader -a outputs/cs_1_component_8bit_16.jp2
/opt/homebrew/bin/vips getpoint outputs/cs_1_component_8bit_17.jp2 0 0
/opt/homebrew/bin/vipsheader outputs/cs_1_component_8bit_17.jp2
/opt/homebrew/bin/vipsheader -a outputs/cs_1_component_8bit_17.jp2
/opt/homebrew/bin/vips getpoint outputs/cs_1_component_8bit_18.jp2 0 0
/opt/homebrew/bin/vipsheader outputs/cs_1_component_8bit_18.jp2
/opt/homebrew/bin/vipsheader -a outputs/cs_1_component_8bit_18.jp2
/opt/homebrew/bin/vips getpoint outputs/cs_1_component_8bit_24.jp2 0 0
/opt/homebrew/bin/vipsheader outputs/cs_1_component_8bit_24.jp2
/opt/homebrew/bin/vipsheader -a outputs/cs_1_component_8bit_24.jp2
/opt/homebrew/bin/vips getpoint outputs/cs_1_component_8bit_99.jp2 0 0
/opt/homebrew/bin/vipsheader outputs/cs_1_component_8bit_99.jp2
/opt/homebrew/bin/vipsheader -a outputs/cs_1_component_8bit_99.jp2
/opt/homebrew/bin/vips getpoint outputs/cs_3_components_8bit_12.jp2 0 0
/opt/homebrew/bin/vipsheader outputs/cs_3_components_8bit_12.jp2
/opt/homebrew/bin/vipsheader -a outputs/cs_3_components_8bit_12.jp2
/opt/homebrew/bin/vips getpoint outputs/cs_3_components_8bit_14.jp2 0 0
/opt/homebrew/bin/vipsheader outputs/cs_3_components_8bit_14.jp2
/opt/homebrew/bin/vipsheader -a outputs/cs_3_components_8bit_14.jp2
/opt/homebrew/bin/vips getpoint outputs/cs_3_components_8bit_16.jp2 0 0
/opt/homebrew/bin/vipsheader outputs/cs_3_components_8bit_16.jp2
/opt/homebrew/bin/vipsheader -a outputs/cs_3_components_8bit_16.jp2
/opt/homebrew/bin/vips getpoint outputs/cs_3_components_8bit_17.jp2 0 0
/opt/homebrew/bin/vipsheader outputs/cs_3_components_8bit_17.jp2
/opt/homebrew/bin/vipsheader -a outputs/cs_3_components_8bit_17.jp2
/opt/homebrew/bin/vips getpoint outputs/cs_3_components_8bit_18.jp2 0 0
/opt/homebrew/bin/vipsheader outputs/cs_3_components_8bit_18.jp2
/opt/homebrew/bin/vipsheader -a outputs/cs_3_components_8bit_18.jp2
/opt/homebrew/bin/vips getpoint outputs/cs_3_components_8bit_24.jp2 0 0
/opt/homebrew/bin/vipsheader outputs/cs_3_components_8bit_24.jp2
/opt/homebrew/bin/vipsheader -a outputs/cs_3_components_8bit_24.jp2
/opt/homebrew/bin/vips getpoint outputs/cs_3_components_8bit_99.jp2 0 0
/opt/homebrew/bin/vipsheader outputs/cs_3_components_8bit_99.jp2
/opt/homebrew/bin/vipsheader -a outputs/cs_3_components_8bit_99.jp2
/opt/homebrew/bin/vips getpoint outputs/cs_3_components_16bit_12.jp2 0 0
/opt/homebrew/bin/vipsheader outputs/cs_3_components_16bit_12.jp2
/opt/homebrew/bin/vipsheader -a outputs/cs_3_components_16bit_12.jp2
/opt/homebrew/bin/vips getpoint outputs/cs_3_components_16bit_14.jp2 0 0
/opt/homebrew/bin/vipsheader outputs/cs_3_components_16bit_14.jp2
/opt/homebrew/bin/vipsheader -a outputs/cs_3_components_16bit_14.jp2
/opt/homebrew/bin/vips getpoint outputs/cs_3_components_16bit_16.jp2 0 0
/opt/homebrew/bin/vipsheader outputs/cs_3_components_16bit_16.jp2
/opt/homebrew/bin/vipsheader -a outputs/cs_3_components_16bit_16.jp2
/opt/homebrew/bin/vips getpoint outputs/cs_3_components_16bit_17.jp2 0 0
/opt/homebrew/bin/vipsheader outputs/cs_3_components_16bit_17.jp2
/opt/homebrew/bin/vipsheader -a outputs/cs_3_components_16bit_17.jp2
/opt/homebrew/bin/vips getpoint outputs/cs_3_components_16bit_18.jp2 0 0
/opt/homebrew/bin/vipsheader outputs/cs_3_components_16bit_18.jp2
/opt/homebrew/bin/vipsheader -a outputs/cs_3_components_16bit_18.jp2
/opt/homebrew/bin/vips getpoint outputs/cs_3_components_16bit_24.jp2 0 0
/opt/homebrew/bin/vipsheader outputs/cs_3_components_16bit_24.jp2
/opt/homebrew/bin/vipsheader -a outputs/cs_3_components_16bit_24.jp2
/opt/homebrew/bin/vips getpoint outputs/cs_3_components_16bit_99.jp2 0 0
/opt/homebrew/bin/vipsheader outputs/cs_3_components_16bit_99.jp2
/opt/homebrew/bin/vipsheader -a outputs/cs_3_components_16bit_99.jp2
/opt/homebrew/bin/opj_compress -i outputs/depth2u.raw -o fixtures/depth2u.j2k -F 5,1,1,2,u -n 1
/opt/homebrew/bin/vips getpoint fixtures/depth2u.j2k 0 0
/opt/homebrew/bin/vips getpoint fixtures/depth2u.j2k 1 0
/opt/homebrew/bin/vips getpoint fixtures/depth2u.j2k 2 0
/opt/homebrew/bin/vips getpoint fixtures/depth2u.j2k 3 0
/opt/homebrew/bin/vips getpoint fixtures/depth2u.j2k 4 0
/opt/homebrew/bin/vipsheader fixtures/depth2u.j2k
/opt/homebrew/bin/vipsheader -a fixtures/depth2u.j2k
/opt/homebrew/bin/opj_compress -i outputs/depth4u.raw -o fixtures/depth4u.j2k -F 5,1,1,4,u -n 1
/opt/homebrew/bin/vips getpoint fixtures/depth4u.j2k 0 0
/opt/homebrew/bin/vips getpoint fixtures/depth4u.j2k 1 0
/opt/homebrew/bin/vips getpoint fixtures/depth4u.j2k 2 0
/opt/homebrew/bin/vips getpoint fixtures/depth4u.j2k 3 0
/opt/homebrew/bin/vips getpoint fixtures/depth4u.j2k 4 0
/opt/homebrew/bin/vipsheader fixtures/depth4u.j2k
/opt/homebrew/bin/vipsheader -a fixtures/depth4u.j2k
/opt/homebrew/bin/opj_compress -i outputs/depth8u.raw -o fixtures/depth8u.j2k -F 5,1,1,8,u -n 1
/opt/homebrew/bin/vips getpoint fixtures/depth8u.j2k 0 0
/opt/homebrew/bin/vips getpoint fixtures/depth8u.j2k 1 0
/opt/homebrew/bin/vips getpoint fixtures/depth8u.j2k 2 0
/opt/homebrew/bin/vips getpoint fixtures/depth8u.j2k 3 0
/opt/homebrew/bin/vips getpoint fixtures/depth8u.j2k 4 0
/opt/homebrew/bin/vipsheader fixtures/depth8u.j2k
/opt/homebrew/bin/vipsheader -a fixtures/depth8u.j2k
/opt/homebrew/bin/opj_compress -i outputs/depth10u.raw -o fixtures/depth10u.j2k -F 5,1,1,10,u -n 1
/opt/homebrew/bin/vips getpoint fixtures/depth10u.j2k 0 0
/opt/homebrew/bin/vips getpoint fixtures/depth10u.j2k 1 0
/opt/homebrew/bin/vips getpoint fixtures/depth10u.j2k 2 0
/opt/homebrew/bin/vips getpoint fixtures/depth10u.j2k 3 0
/opt/homebrew/bin/vips getpoint fixtures/depth10u.j2k 4 0
/opt/homebrew/bin/vipsheader fixtures/depth10u.j2k
/opt/homebrew/bin/vipsheader -a fixtures/depth10u.j2k
/opt/homebrew/bin/opj_compress -i outputs/depth12u.raw -o fixtures/depth12u.j2k -F 5,1,1,12,u -n 1
/opt/homebrew/bin/vips getpoint fixtures/depth12u.j2k 0 0
/opt/homebrew/bin/vips getpoint fixtures/depth12u.j2k 1 0
/opt/homebrew/bin/vips getpoint fixtures/depth12u.j2k 2 0
/opt/homebrew/bin/vips getpoint fixtures/depth12u.j2k 3 0
/opt/homebrew/bin/vips getpoint fixtures/depth12u.j2k 4 0
/opt/homebrew/bin/vipsheader fixtures/depth12u.j2k
/opt/homebrew/bin/vipsheader -a fixtures/depth12u.j2k
/opt/homebrew/bin/opj_compress -i outputs/depth12s.raw -o fixtures/depth12s.j2k -F 5,1,1,12,s -n 1
/opt/homebrew/bin/vips getpoint fixtures/depth12s.j2k 0 0
/opt/homebrew/bin/vips getpoint fixtures/depth12s.j2k 1 0
/opt/homebrew/bin/vips getpoint fixtures/depth12s.j2k 2 0
/opt/homebrew/bin/vips getpoint fixtures/depth12s.j2k 3 0
/opt/homebrew/bin/vips getpoint fixtures/depth12s.j2k 4 0
/opt/homebrew/bin/vipsheader fixtures/depth12s.j2k
/opt/homebrew/bin/vipsheader -a fixtures/depth12s.j2k
/opt/homebrew/bin/opj_compress -i outputs/depth14u.raw -o fixtures/depth14u.j2k -F 5,1,1,14,u -n 1
/opt/homebrew/bin/vips getpoint fixtures/depth14u.j2k 0 0
/opt/homebrew/bin/vips getpoint fixtures/depth14u.j2k 1 0
/opt/homebrew/bin/vips getpoint fixtures/depth14u.j2k 2 0
/opt/homebrew/bin/vips getpoint fixtures/depth14u.j2k 3 0
/opt/homebrew/bin/vips getpoint fixtures/depth14u.j2k 4 0
/opt/homebrew/bin/vipsheader fixtures/depth14u.j2k
/opt/homebrew/bin/vipsheader -a fixtures/depth14u.j2k
/opt/homebrew/bin/opj_compress -i outputs/depth16u.raw -o fixtures/depth16u.j2k -F 5,1,1,16,u -n 1
/opt/homebrew/bin/vips getpoint fixtures/depth16u.j2k 0 0
/opt/homebrew/bin/vips getpoint fixtures/depth16u.j2k 1 0
/opt/homebrew/bin/vips getpoint fixtures/depth16u.j2k 2 0
/opt/homebrew/bin/vips getpoint fixtures/depth16u.j2k 3 0
/opt/homebrew/bin/vips getpoint fixtures/depth16u.j2k 4 0
/opt/homebrew/bin/vipsheader fixtures/depth16u.j2k
/opt/homebrew/bin/vipsheader -a fixtures/depth16u.j2k
/opt/homebrew/bin/vips rawload outputs/grad16.raw outputs/grad16.v 16 16 3 --format uchar
/opt/homebrew/bin/vips copy outputs/grad16.v outputs/grad16-srgb.v --interpretation srgb
/opt/homebrew/bin/vips jp2ksave outputs/grad16-srgb.v outputs/sub_off_48.jp2 --subsample-mode off --Q 48
/opt/homebrew/bin/vips getpoint outputs/sub_off_48.jp2 0 0
/opt/homebrew/bin/vips getpoint outputs/sub_off_48.jp2 1 0
/opt/homebrew/bin/vips getpoint outputs/sub_off_48.jp2 2 0
/opt/homebrew/bin/vips getpoint outputs/sub_off_48.jp2 3 0
/opt/homebrew/bin/vips getpoint outputs/sub_off_48.jp2 4 0
/opt/homebrew/bin/vips getpoint outputs/sub_off_48.jp2 5 0
/opt/homebrew/bin/vips jp2ksave outputs/grad16-srgb.v outputs/sub_off_90.jp2 --subsample-mode off --Q 90
/opt/homebrew/bin/vips getpoint outputs/sub_off_90.jp2 0 0
/opt/homebrew/bin/vips getpoint outputs/sub_off_90.jp2 1 0
/opt/homebrew/bin/vips getpoint outputs/sub_off_90.jp2 2 0
/opt/homebrew/bin/vips getpoint outputs/sub_off_90.jp2 3 0
/opt/homebrew/bin/vips getpoint outputs/sub_off_90.jp2 4 0
/opt/homebrew/bin/vips getpoint outputs/sub_off_90.jp2 5 0
/opt/homebrew/bin/vips jp2ksave outputs/grad16-srgb.v outputs/sub_on_48.jp2 --subsample-mode on --Q 48
/opt/homebrew/bin/vips getpoint outputs/sub_on_48.jp2 0 0
/opt/homebrew/bin/vips getpoint outputs/sub_on_48.jp2 1 0
/opt/homebrew/bin/vips getpoint outputs/sub_on_48.jp2 2 0
/opt/homebrew/bin/vips getpoint outputs/sub_on_48.jp2 3 0
/opt/homebrew/bin/vips getpoint outputs/sub_on_48.jp2 4 0
/opt/homebrew/bin/vips getpoint outputs/sub_on_48.jp2 5 0
/opt/homebrew/bin/vips jp2ksave outputs/grad16-srgb.v outputs/sub_on_90.jp2 --subsample-mode on --Q 90
/opt/homebrew/bin/vips getpoint outputs/sub_on_90.jp2 0 0
/opt/homebrew/bin/vips getpoint outputs/sub_on_90.jp2 1 0
/opt/homebrew/bin/vips getpoint outputs/sub_on_90.jp2 2 0
/opt/homebrew/bin/vips getpoint outputs/sub_on_90.jp2 3 0
/opt/homebrew/bin/vips getpoint outputs/sub_on_90.jp2 4 0
/opt/homebrew/bin/vips getpoint outputs/sub_on_90.jp2 5 0
/opt/homebrew/bin/vips jp2ksave outputs/grad16-srgb.v outputs/sub_auto_48.jp2 --subsample-mode auto --Q 48
/opt/homebrew/bin/vips getpoint outputs/sub_auto_48.jp2 0 0
/opt/homebrew/bin/vips getpoint outputs/sub_auto_48.jp2 1 0
/opt/homebrew/bin/vips getpoint outputs/sub_auto_48.jp2 2 0
/opt/homebrew/bin/vips getpoint outputs/sub_auto_48.jp2 3 0
/opt/homebrew/bin/vips getpoint outputs/sub_auto_48.jp2 4 0
/opt/homebrew/bin/vips getpoint outputs/sub_auto_48.jp2 5 0
/opt/homebrew/bin/vips jp2ksave outputs/grad16-srgb.v outputs/sub_auto_90.jp2 --subsample-mode auto --Q 90
/opt/homebrew/bin/vips getpoint outputs/sub_auto_90.jp2 0 0
/opt/homebrew/bin/vips getpoint outputs/sub_auto_90.jp2 1 0
/opt/homebrew/bin/vips getpoint outputs/sub_auto_90.jp2 2 0
/opt/homebrew/bin/vips getpoint outputs/sub_auto_90.jp2 3 0
/opt/homebrew/bin/vips getpoint outputs/sub_auto_90.jp2 4 0
/opt/homebrew/bin/vips getpoint outputs/sub_auto_90.jp2 5 0
/opt/homebrew/bin/vips jp2ksave outputs/grad16-srgb.v outputs/sub_lossless_on.jp2 --lossless --subsample-mode on
/opt/homebrew/bin/vips jp2ksave outputs/grad16-srgb.v outputs/sub_lossless_off.jp2 --lossless
/opt/homebrew/bin/vips rawload outputs/auto_1_band_bw.raw outputs/auto_1_band_bw.v 16 16 1 --format uchar
/opt/homebrew/bin/vips copy outputs/auto_1_band_bw.v outputs/auto_1_band_bw-b-w.v --interpretation b-w
/opt/homebrew/bin/vips jp2ksave outputs/auto_1_band_bw-b-w.v outputs/auto_1_band_bw.jp2 --Q 48 --subsample-mode auto
/opt/homebrew/bin/vips rawload outputs/auto_3_band_srgb.raw outputs/auto_3_band_srgb.v 16 16 3 --format uchar
/opt/homebrew/bin/vips copy outputs/auto_3_band_srgb.v outputs/auto_3_band_srgb-srgb.v --interpretation srgb
/opt/homebrew/bin/vips jp2ksave outputs/auto_3_band_srgb-srgb.v outputs/auto_3_band_srgb.jp2 --Q 48 --subsample-mode auto
/opt/homebrew/bin/vips rawload outputs/auto_4_band_srgb.raw outputs/auto_4_band_srgb.v 16 16 4 --format uchar
/opt/homebrew/bin/vips copy outputs/auto_4_band_srgb.v outputs/auto_4_band_srgb-srgb.v --interpretation srgb
/opt/homebrew/bin/vips jp2ksave outputs/auto_4_band_srgb-srgb.v outputs/auto_4_band_srgb.jp2 --Q 48 --subsample-mode auto
/opt/homebrew/bin/vips rawload outputs/auto_3_band_multiband.raw outputs/auto_3_band_multiband.v 16 16 3 --format uchar
/opt/homebrew/bin/vips copy outputs/auto_3_band_multiband.v outputs/auto_3_band_multiband-multiband.v --interpretation multiband
/opt/homebrew/bin/vips jp2ksave outputs/auto_3_band_multiband-multiband.v outputs/auto_3_band_multiband.jp2 --Q 48 --subsample-mode auto
/opt/homebrew/bin/vips rawload outputs/auto_4_band_cmyk.raw outputs/auto_4_band_cmyk.v 16 16 4 --format uchar
/opt/homebrew/bin/vips copy outputs/auto_4_band_cmyk.v outputs/auto_4_band_cmyk-cmyk.v --interpretation cmyk
/opt/homebrew/bin/vips jp2ksave outputs/auto_4_band_cmyk-cmyk.v outputs/auto_4_band_cmyk.jp2 --Q 48 --subsample-mode auto
/opt/homebrew/bin/vips jp2ksave outputs/grad16-srgb.v fixtures/chroma_sub_off.jp2 --Q 90 --subsample-mode off
/opt/homebrew/bin/vipsheader -a fixtures/chroma_sub_off.jp2
/opt/homebrew/bin/vips rawsave fixtures/chroma_sub_off.jp2 outputs/chroma_sub_off.raw
/opt/homebrew/bin/vips getpoint fixtures/chroma_sub_off.jp2 0 0
/opt/homebrew/bin/vips getpoint fixtures/chroma_sub_off.jp2 1 0
/opt/homebrew/bin/vips getpoint fixtures/chroma_sub_off.jp2 2 0
/opt/homebrew/bin/vips getpoint fixtures/chroma_sub_off.jp2 3 0
/opt/homebrew/bin/vips getpoint fixtures/chroma_sub_off.jp2 0 1
/opt/homebrew/bin/vips getpoint fixtures/chroma_sub_off.jp2 15 15
/opt/homebrew/bin/vips jp2ksave outputs/grad16-srgb.v fixtures/chroma_sub_on.jp2 --Q 90 --subsample-mode on
/opt/homebrew/bin/vipsheader -a fixtures/chroma_sub_on.jp2
/opt/homebrew/bin/vips rawsave fixtures/chroma_sub_on.jp2 outputs/chroma_sub_on.raw
/opt/homebrew/bin/vips getpoint fixtures/chroma_sub_on.jp2 0 0
/opt/homebrew/bin/vips getpoint fixtures/chroma_sub_on.jp2 1 0
/opt/homebrew/bin/vips getpoint fixtures/chroma_sub_on.jp2 2 0
/opt/homebrew/bin/vips getpoint fixtures/chroma_sub_on.jp2 3 0
/opt/homebrew/bin/vips getpoint fixtures/chroma_sub_on.jp2 0 1
/opt/homebrew/bin/vips getpoint fixtures/chroma_sub_on.jp2 15 15
/opt/homebrew/bin/vips rawload outputs/tiny.raw outputs/tiny.v 4 2 3 --format uchar
/opt/homebrew/bin/vips copy outputs/tiny.v outputs/tiny-srgb.v --interpretation srgb
/opt/homebrew/bin/vips jp2ksave outputs/tiny-srgb.v fixtures/chroma_tiny_sub_on.jp2 --Q 90 --subsample-mode on
/opt/homebrew/bin/vipsheader -a fixtures/chroma_tiny_sub_on.jp2
/opt/homebrew/bin/vips rawsave fixtures/chroma_tiny_sub_on.jp2 outputs/chroma_tiny.raw
/opt/homebrew/bin/vips getpoint fixtures/chroma_tiny_sub_on.jp2 0 0
/opt/homebrew/bin/vips getpoint fixtures/chroma_tiny_sub_on.jp2 1 0
/opt/homebrew/bin/vips getpoint fixtures/chroma_tiny_sub_on.jp2 2 0
/opt/homebrew/bin/vips getpoint fixtures/chroma_tiny_sub_on.jp2 3 0
/opt/homebrew/bin/vips getpoint fixtures/chroma_tiny_sub_on.jp2 0 1
/opt/homebrew/bin/vips getpoint fixtures/chroma_tiny_sub_on.jp2 1 1
/opt/homebrew/bin/vips getpoint fixtures/chroma_tiny_sub_on.jp2 2 1
/opt/homebrew/bin/vips getpoint fixtures/chroma_tiny_sub_on.jp2 3 1
/opt/homebrew/bin/opj_compress -i outputs/sub420.raw -o fixtures/sub420.j2k -F 8,4,3,8,u@1x1:2x2:2x2 -n 1 -mct 0
/opt/homebrew/bin/vipsheader -a fixtures/sub420.j2k
/opt/homebrew/bin/vips rawsave fixtures/sub420.j2k outputs/sub420.raw
/opt/homebrew/bin/vips getpoint fixtures/sub420.j2k 0 0
/opt/homebrew/bin/vips getpoint fixtures/sub420.j2k 1 0
/opt/homebrew/bin/vips getpoint fixtures/sub420.j2k 2 0
/opt/homebrew/bin/vips getpoint fixtures/sub420.j2k 3 0
/opt/homebrew/bin/vips getpoint fixtures/sub420.j2k 4 0
/opt/homebrew/bin/vips getpoint fixtures/sub420.j2k 5 0
/opt/homebrew/bin/vips getpoint fixtures/sub420.j2k 6 0
/opt/homebrew/bin/vips getpoint fixtures/sub420.j2k 7 0
/opt/homebrew/bin/vips getpoint fixtures/sub420.j2k 0 1
/opt/homebrew/bin/vips getpoint fixtures/sub420.j2k 1 1
/opt/homebrew/bin/vips getpoint fixtures/sub420.j2k 2 1
/opt/homebrew/bin/vips getpoint fixtures/sub420.j2k 3 1
/opt/homebrew/bin/vips getpoint fixtures/sub420.j2k 4 1
/opt/homebrew/bin/vips getpoint fixtures/sub420.j2k 5 1
/opt/homebrew/bin/vips getpoint fixtures/sub420.j2k 6 1
/opt/homebrew/bin/vips getpoint fixtures/sub420.j2k 7 1
/opt/homebrew/bin/vips getpoint fixtures/sub420.j2k 0 2
/opt/homebrew/bin/vips getpoint fixtures/sub420.j2k 1 2
/opt/homebrew/bin/vips getpoint fixtures/sub420.j2k 2 2
/opt/homebrew/bin/vips getpoint fixtures/sub420.j2k 3 2
/opt/homebrew/bin/vips getpoint fixtures/sub420.j2k 4 2
/opt/homebrew/bin/vips getpoint fixtures/sub420.j2k 5 2
/opt/homebrew/bin/vips getpoint fixtures/sub420.j2k 6 2
/opt/homebrew/bin/vips getpoint fixtures/sub420.j2k 7 2
/opt/homebrew/bin/vips getpoint fixtures/sub420.j2k 0 3
/opt/homebrew/bin/vips getpoint fixtures/sub420.j2k 1 3
/opt/homebrew/bin/vips getpoint fixtures/sub420.j2k 2 3
/opt/homebrew/bin/vips getpoint fixtures/sub420.j2k 3 3
/opt/homebrew/bin/vips getpoint fixtures/sub420.j2k 4 3
/opt/homebrew/bin/vips getpoint fixtures/sub420.j2k 5 3
/opt/homebrew/bin/vips getpoint fixtures/sub420.j2k 6 3
/opt/homebrew/bin/vips getpoint fixtures/sub420.j2k 7 3
/opt/homebrew/bin/vipsheader outputs/copy.j2k
/opt/homebrew/bin/vips rawsave outputs/copy.j2k outputs/carrier_copy.j2k.raw
/opt/homebrew/bin/vipsheader outputs/copy.jp2
/opt/homebrew/bin/vips rawsave outputs/copy.jp2 outputs/carrier_copy.jp2.raw
/opt/homebrew/bin/vipsheader outputs/copy.jpt
/opt/homebrew/bin/vips rawsave outputs/copy.jpt outputs/carrier_copy.jpt.raw
/opt/homebrew/bin/vipsheader outputs/copy.png
/opt/homebrew/bin/vips rawsave outputs/copy.png outputs/carrier_copy.png.raw
/opt/homebrew/bin/vipsheader outputs/copy
/opt/homebrew/bin/vips rawsave outputs/copy outputs/carrier_copy.raw
/opt/homebrew/bin/vips rawload outputs/grey37.raw outputs/grey37.v 37 21 1 --format uchar
/opt/homebrew/bin/vips copy outputs/grey37.v outputs/grey37-b-w.v --interpretation b-w
/opt/homebrew/bin/vips jp2ksave outputs/grey37-b-w.v outputs/tile_512x512.jp2 --lossless --tile-width 512 --tile-height 512
/opt/homebrew/bin/vipsheader -a outputs/tile_512x512.jp2
/opt/homebrew/bin/vips rawsave outputs/tile_512x512.jp2 outputs/tile_512x512.raw
/opt/homebrew/bin/vips jp2ksave outputs/grey37-b-w.v outputs/tile_16x16.jp2 --lossless --tile-width 16 --tile-height 16
/opt/homebrew/bin/vipsheader -a outputs/tile_16x16.jp2
/opt/homebrew/bin/vips rawsave outputs/tile_16x16.jp2 outputs/tile_16x16.raw
/opt/homebrew/bin/vips jp2ksave outputs/grey37-b-w.v outputs/tile_8x8.jp2 --lossless --tile-width 8 --tile-height 8
/opt/homebrew/bin/vipsheader -a outputs/tile_8x8.jp2
/opt/homebrew/bin/vips rawsave outputs/tile_8x8.jp2 outputs/tile_8x8.raw
/opt/homebrew/bin/vips jp2ksave outputs/grey37-b-w.v outputs/tile_16x7.jp2 --lossless --tile-width 16 --tile-height 7
/opt/homebrew/bin/vipsheader -a outputs/tile_16x7.jp2
/opt/homebrew/bin/vips rawsave outputs/tile_16x7.jp2 outputs/tile_16x7.raw
/opt/homebrew/bin/vips jp2ksave outputs/grey37-b-w.v fixtures/grey_tile8.jp2 --lossless --tile-width 8 --tile-height 8
/opt/homebrew/bin/vipsheader -a fixtures/grey_tile8.jp2
/opt/homebrew/bin/vips rawsave fixtures/grey_tile8.jp2 outputs/grey_tile8.raw
/opt/homebrew/bin/vips getpoint fixtures/grey_tile8.jp2 7 0
/opt/homebrew/bin/vips getpoint fixtures/grey_tile8.jp2 8 0
/opt/homebrew/bin/vips getpoint fixtures/grey_tile8.jp2 0 7
/opt/homebrew/bin/vips getpoint fixtures/grey_tile8.jp2 0 8
/opt/homebrew/bin/vips getpoint fixtures/grey_tile8.jp2 31 20
/opt/homebrew/bin/vips getpoint fixtures/grey_tile8.jp2 36 20
/opt/homebrew/bin/vips getpoint fixtures/grey_tile8.jp2 36 0
/opt/homebrew/bin/vips jp2ksave outputs/grad16-srgb.v outputs/q1.jp2 --Q 1
/opt/homebrew/bin/vips getpoint outputs/q1.jp2 0 0
/opt/homebrew/bin/vips getpoint outputs/q1.jp2 8 8
/opt/homebrew/bin/vips getpoint outputs/q1.jp2 15 15
/opt/homebrew/bin/vips jp2ksave outputs/grad16-srgb.v outputs/q25.jp2 --Q 25
/opt/homebrew/bin/vips getpoint outputs/q25.jp2 0 0
/opt/homebrew/bin/vips getpoint outputs/q25.jp2 8 8
/opt/homebrew/bin/vips getpoint outputs/q25.jp2 15 15
/opt/homebrew/bin/vips jp2ksave outputs/grad16-srgb.v outputs/q48.jp2 --Q 48
/opt/homebrew/bin/vips getpoint outputs/q48.jp2 0 0
/opt/homebrew/bin/vips getpoint outputs/q48.jp2 8 8
/opt/homebrew/bin/vips getpoint outputs/q48.jp2 15 15
/opt/homebrew/bin/vips jp2ksave outputs/grad16-srgb.v outputs/q75.jp2 --Q 75
/opt/homebrew/bin/vips getpoint outputs/q75.jp2 0 0
/opt/homebrew/bin/vips getpoint outputs/q75.jp2 8 8
/opt/homebrew/bin/vips getpoint outputs/q75.jp2 15 15
/opt/homebrew/bin/vips jp2ksave outputs/grad16-srgb.v outputs/q90.jp2 --Q 90
/opt/homebrew/bin/vips getpoint outputs/q90.jp2 0 0
/opt/homebrew/bin/vips getpoint outputs/q90.jp2 8 8
/opt/homebrew/bin/vips getpoint outputs/q90.jp2 15 15
/opt/homebrew/bin/vips jp2ksave outputs/grad16-srgb.v outputs/q100.jp2 --Q 100
/opt/homebrew/bin/vips getpoint outputs/q100.jp2 0 0
/opt/homebrew/bin/vips getpoint outputs/q100.jp2 8 8
/opt/homebrew/bin/vips getpoint outputs/q100.jp2 15 15
/opt/homebrew/bin/vips jp2ksave outputs/rgb-srgb.v fixtures/rgb_lossy_q48.jp2 --Q 48
/opt/homebrew/bin/vipsheader -a fixtures/rgb_lossy_q48.jp2
/opt/homebrew/bin/vips rawsave fixtures/rgb_lossy_q48.jp2 outputs/rgb_lossy_q48.raw
/opt/homebrew/bin/vips getpoint fixtures/rgb_lossy_q48.jp2 0 0
/opt/homebrew/bin/vips getpoint fixtures/rgb_lossy_q48.jp2 1 0
/opt/homebrew/bin/vips getpoint fixtures/rgb_lossy_q48.jp2 2 0
/opt/homebrew/bin/vips getpoint fixtures/rgb_lossy_q48.jp2 3 0
/opt/homebrew/bin/vips getpoint fixtures/rgb_lossy_q48.jp2 0 1
/opt/homebrew/bin/vips getpoint fixtures/rgb_lossy_q48.jp2 1 1
/opt/homebrew/bin/vips getpoint fixtures/rgb_lossy_q48.jp2 2 1
/opt/homebrew/bin/vips getpoint fixtures/rgb_lossy_q48.jp2 3 1
/opt/homebrew/bin/vips getpoint fixtures/rgb_lossy_q48.jp2 0 2
/opt/homebrew/bin/vips getpoint fixtures/rgb_lossy_q48.jp2 1 2
/opt/homebrew/bin/vips getpoint fixtures/rgb_lossy_q48.jp2 2 2
/opt/homebrew/bin/vips getpoint fixtures/rgb_lossy_q48.jp2 3 2
/opt/homebrew/bin/opj_compress -i outputs/res3.raw -o fixtures/res3.j2k -F 32,24,1,8,u -n 3
/opt/homebrew/bin/vipsheader fixtures/res3.j2k[page=0]
/opt/homebrew/bin/vipsheader -a fixtures/res3.j2k[page=0]
/opt/homebrew/bin/vips rawsave fixtures/res3.j2k[page=0] outputs/res3_page0.raw
/opt/homebrew/bin/vips getpoint fixtures/res3.j2k[page=0] 0 0
/opt/homebrew/bin/vipsheader fixtures/res3.j2k[page=1]
/opt/homebrew/bin/vipsheader -a fixtures/res3.j2k[page=1]
/opt/homebrew/bin/vips rawsave fixtures/res3.j2k[page=1] outputs/res3_page1.raw
/opt/homebrew/bin/vips getpoint fixtures/res3.j2k[page=1] 0 0
/opt/homebrew/bin/vipsheader fixtures/res3.j2k[page=2]
/opt/homebrew/bin/vipsheader -a fixtures/res3.j2k[page=2]
/opt/homebrew/bin/vips rawsave fixtures/res3.j2k[page=2] outputs/res3_page2.raw
/opt/homebrew/bin/vips getpoint fixtures/res3.j2k[page=2] 0 0
/opt/homebrew/bin/vipsheader fixtures/res3.j2k[page=3]
/opt/homebrew/bin/vipsheader -a fixtures/res3.j2k
/opt/homebrew/bin/vips rawsave fixtures/res3.j2k outputs/res3.raw
/opt/homebrew/bin/vips getpoint fixtures/res3.j2k 0 0
/opt/homebrew/bin/vips getpoint fixtures/res3.j2k 31 23
/opt/homebrew/bin/opj_compress -i outputs/origin57.raw -o fixtures/origin57.j2k -F 32,24,1,8,u -n 1 -d 5,7
/opt/homebrew/bin/vipsheader -a fixtures/origin57.j2k
/opt/homebrew/bin/vips rawsave fixtures/origin57.j2k outputs/origin57.raw
/opt/homebrew/bin/vips getpoint fixtures/origin57.j2k 0 0
/opt/homebrew/bin/vips getpoint fixtures/origin57.j2k 26 16
/opt/homebrew/bin/vips jp2ksave outputs/grad16-srgb.v outputs/profile.jp2 --lossless --profile /Users/rom/workspace/libvips/test/test-suite/images/sRGB.icm
/opt/homebrew/bin/vips jp2ksave outputs/grad16-srgb.v outputs/profile_none.jp2 --lossless
/opt/homebrew/bin/vipsheader -a outputs/profile.jp2
/opt/homebrew/bin/vips jp2ksave outputs/grad16-srgb.v outputs/keep_all.jp2 --lossless --keep all
/opt/homebrew/bin/vips jp2ksave outputs/grad16-srgb.v outputs/keep_none.jp2 --lossless --keep none
/opt/homebrew/bin/vips jp2ksave outputs/grad16-srgb.v outputs/keep_icc.jp2 --lossless --keep icc
/opt/homebrew/bin/vips jp2ksave outputs/rgb-srgb.v outputs/iccbase.jp2 --lossless
# fixtures/icc_colr.jp2 is assembled by this script, not by vips: jp2ksave cannot attach a profile at all, so the colr box is rewritten to METH=2 and a uuid XMP box is appended by hand
/opt/homebrew/bin/vipsheader -a fixtures/icc_colr.jp2
/opt/homebrew/bin/vips rawsave fixtures/icc_colr.jp2 outputs/icc_colr.raw
/opt/homebrew/bin/vips getpoint fixtures/icc_colr.jp2 0 0
/opt/homebrew/bin/vips getpoint fixtures/icc_colr.jp2 1 0
/opt/homebrew/bin/vips getpoint fixtures/icc_colr.jp2 2 0
/opt/homebrew/bin/vips getpoint fixtures/icc_colr.jp2 3 0
/opt/homebrew/bin/vips getpoint fixtures/icc_colr.jp2 0 1
/opt/homebrew/bin/vips getpoint fixtures/icc_colr.jp2 1 1
/opt/homebrew/bin/vips getpoint fixtures/icc_colr.jp2 2 1
/opt/homebrew/bin/vips getpoint fixtures/icc_colr.jp2 3 1
/opt/homebrew/bin/vips getpoint fixtures/icc_colr.jp2 0 2
/opt/homebrew/bin/vips getpoint fixtures/icc_colr.jp2 1 2
/opt/homebrew/bin/vips getpoint fixtures/icc_colr.jp2 2 2
/opt/homebrew/bin/vips getpoint fixtures/icc_colr.jp2 3 2
/opt/homebrew/bin/vipsheader fixtures/truncated_at_codestream.jp2
/opt/homebrew/bin/vips avg fixtures/truncated_at_codestream.jp2
/opt/homebrew/bin/vips avg fixtures/truncated_at_codestream.jp2[fail-on=none]
/opt/homebrew/bin/vips avg fixtures/truncated_at_codestream.jp2[fail-on=truncated]
/opt/homebrew/bin/vips avg fixtures/truncated_at_codestream.jp2[fail-on=error]
/opt/homebrew/bin/vips avg fixtures/truncated_at_codestream.jp2[fail-on=warning]
/opt/homebrew/bin/vipsheader fixtures/truncated_in_siz.jp2
/opt/homebrew/bin/vips avg fixtures/truncated_in_siz.jp2
/opt/homebrew/bin/vips avg fixtures/truncated_in_siz.jp2[fail-on=none]
/opt/homebrew/bin/vips avg fixtures/truncated_in_siz.jp2[fail-on=truncated]
/opt/homebrew/bin/vips avg fixtures/truncated_in_siz.jp2[fail-on=error]
/opt/homebrew/bin/vips avg fixtures/truncated_in_siz.jp2[fail-on=warning]
/opt/homebrew/bin/vipsheader fixtures/truncated_in_tile.jp2
/opt/homebrew/bin/vipsheader -a fixtures/truncated_in_tile.jp2
/opt/homebrew/bin/vips avg fixtures/truncated_in_tile.jp2
/opt/homebrew/bin/vips avg fixtures/truncated_in_tile.jp2[fail-on=none]
/opt/homebrew/bin/vips avg fixtures/truncated_in_tile.jp2[fail-on=truncated]
/opt/homebrew/bin/vips avg fixtures/truncated_in_tile.jp2[fail-on=error]
/opt/homebrew/bin/vips avg fixtures/truncated_in_tile.jp2[fail-on=warning]
/opt/homebrew/bin/vipsheader fixtures/truncated_in_boxes.jp2
/opt/homebrew/bin/vips avg fixtures/truncated_in_boxes.jp2
/opt/homebrew/bin/vips avg fixtures/truncated_in_boxes.jp2[fail-on=none]
/opt/homebrew/bin/vips avg fixtures/truncated_in_boxes.jp2[fail-on=truncated]
/opt/homebrew/bin/vips avg fixtures/truncated_in_boxes.jp2[fail-on=error]
/opt/homebrew/bin/vips avg fixtures/truncated_in_boxes.jp2[fail-on=warning]
/opt/homebrew/bin/vipsheader fixtures/zeroed_body.jp2
/opt/homebrew/bin/vips avg fixtures/zeroed_body.jp2
/opt/homebrew/bin/vips avg fixtures/zeroed_body.jp2[fail-on=none]
/opt/homebrew/bin/vips avg fixtures/zeroed_body.jp2[fail-on=truncated]
/opt/homebrew/bin/vips avg fixtures/zeroed_body.jp2[fail-on=error]
/opt/homebrew/bin/vips avg fixtures/zeroed_body.jp2[fail-on=warning]
/opt/homebrew/bin/vipsheader fixtures/not_jp2k.bin
/opt/homebrew/bin/vips avg fixtures/not_jp2k.bin
/opt/homebrew/bin/vips avg fixtures/not_jp2k.bin[fail-on=none]
/opt/homebrew/bin/vips avg fixtures/not_jp2k.bin[fail-on=truncated]
/opt/homebrew/bin/vips avg fixtures/not_jp2k.bin[fail-on=error]
/opt/homebrew/bin/vips avg fixtures/not_jp2k.bin[fail-on=warning]
# the malformed fixtures are cut from fixtures/rgb_lossless.jp2 at box and marker boundaries by this script, so each one reaches a different check
/opt/homebrew/bin/vips --version
/opt/homebrew/bin/vips --vips-config
/opt/homebrew/bin/opj_compress -h
otool -L /opt/homebrew/lib/libvips.42.dylib
