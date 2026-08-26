#!/bin/sh
# Every vips command capture.py ran, in order. Regenerate with
# `python3 capture.py` from this directory.
#
# Many of these are EXPECTED to fail: the refusal records are the point.
set +e

/opt/homebrew/bin/vipsheader fixtures/base_2d_uchar.hdr
/opt/homebrew/bin/vips analyzeload fixtures/base_2d_uchar.hdr outputs/direct.v
/opt/homebrew/bin/vipsheader outputs/direct.v
/opt/homebrew/bin/vipsheader fixtures/base_2d_uchar.img
/opt/homebrew/bin/vips analyzeload fixtures/base_2d_uchar.img outputs/direct.v
/opt/homebrew/bin/vipsheader outputs/direct.v
/opt/homebrew/bin/vipsheader fixtures/base_2d_uchar
/opt/homebrew/bin/vips analyzeload fixtures/base_2d_uchar outputs/direct.v
/opt/homebrew/bin/vipsheader outputs/direct.v
/opt/homebrew/bin/vipsheader fixtures/no_img.hdr
/opt/homebrew/bin/vips avg fixtures/no_img.hdr
/opt/homebrew/bin/vips analyzeload fixtures/no_img.hdr outputs/direct.v
/opt/homebrew/bin/vipsheader fixtures/no_hdr.img
/opt/homebrew/bin/vips analyzeload fixtures/no_hdr.img outputs/direct.v
/opt/homebrew/bin/vipsheader fixtures/dt0.hdr
/opt/homebrew/bin/vips analyzeload fixtures/dt0.hdr outputs/direct.v
/opt/homebrew/bin/vipsheader fixtures/dt1.hdr
/opt/homebrew/bin/vips analyzeload fixtures/dt1.hdr outputs/direct.v
/opt/homebrew/bin/vipsheader fixtures/dt2.hdr
/opt/homebrew/bin/vips analyzeload fixtures/dt2.hdr outputs/direct.v
/opt/homebrew/bin/vipsheader outputs/direct.v
/opt/homebrew/bin/vips getpoint fixtures/dt2.hdr 0 0
/opt/homebrew/bin/vips getpoint fixtures/dt2.hdr 1 0
/opt/homebrew/bin/vips getpoint fixtures/dt2.hdr 0 1
/opt/homebrew/bin/vips getpoint fixtures/dt2.hdr 1 1
/opt/homebrew/bin/vips avg fixtures/dt2.hdr
/opt/homebrew/bin/vipsheader fixtures/dt4.hdr
/opt/homebrew/bin/vips analyzeload fixtures/dt4.hdr outputs/direct.v
/opt/homebrew/bin/vipsheader outputs/direct.v
/opt/homebrew/bin/vips getpoint fixtures/dt4.hdr 0 0
/opt/homebrew/bin/vips getpoint fixtures/dt4.hdr 1 0
/opt/homebrew/bin/vips getpoint fixtures/dt4.hdr 0 1
/opt/homebrew/bin/vips getpoint fixtures/dt4.hdr 1 1
/opt/homebrew/bin/vips avg fixtures/dt4.hdr
/opt/homebrew/bin/vipsheader fixtures/dt8.hdr
/opt/homebrew/bin/vips analyzeload fixtures/dt8.hdr outputs/direct.v
/opt/homebrew/bin/vipsheader outputs/direct.v
/opt/homebrew/bin/vips getpoint fixtures/dt8.hdr 0 0
/opt/homebrew/bin/vips getpoint fixtures/dt8.hdr 1 0
/opt/homebrew/bin/vips getpoint fixtures/dt8.hdr 0 1
/opt/homebrew/bin/vips getpoint fixtures/dt8.hdr 1 1
/opt/homebrew/bin/vips avg fixtures/dt8.hdr
/opt/homebrew/bin/vipsheader fixtures/dt16.hdr
/opt/homebrew/bin/vips analyzeload fixtures/dt16.hdr outputs/direct.v
/opt/homebrew/bin/vipsheader outputs/direct.v
/opt/homebrew/bin/vips getpoint fixtures/dt16.hdr 0 0
/opt/homebrew/bin/vips getpoint fixtures/dt16.hdr 1 0
/opt/homebrew/bin/vips getpoint fixtures/dt16.hdr 0 1
/opt/homebrew/bin/vips getpoint fixtures/dt16.hdr 1 1
/opt/homebrew/bin/vips avg fixtures/dt16.hdr
/opt/homebrew/bin/vipsheader fixtures/dt32.hdr
/opt/homebrew/bin/vips analyzeload fixtures/dt32.hdr outputs/direct.v
/opt/homebrew/bin/vipsheader outputs/direct.v
/opt/homebrew/bin/vips getpoint fixtures/dt32.hdr 0 0
/opt/homebrew/bin/vips getpoint fixtures/dt32.hdr 1 0
/opt/homebrew/bin/vips getpoint fixtures/dt32.hdr 0 1
/opt/homebrew/bin/vips getpoint fixtures/dt32.hdr 1 1
/opt/homebrew/bin/vips avg fixtures/dt32.hdr
/opt/homebrew/bin/vipsheader fixtures/dt64.hdr
/opt/homebrew/bin/vips analyzeload fixtures/dt64.hdr outputs/direct.v
/opt/homebrew/bin/vipsheader outputs/direct.v
/opt/homebrew/bin/vips getpoint fixtures/dt64.hdr 0 0
/opt/homebrew/bin/vips getpoint fixtures/dt64.hdr 1 0
/opt/homebrew/bin/vips getpoint fixtures/dt64.hdr 0 1
/opt/homebrew/bin/vips getpoint fixtures/dt64.hdr 1 1
/opt/homebrew/bin/vips avg fixtures/dt64.hdr
/opt/homebrew/bin/vipsheader fixtures/dt128.hdr
/opt/homebrew/bin/vips analyzeload fixtures/dt128.hdr outputs/direct.v
/opt/homebrew/bin/vipsheader outputs/direct.v
/opt/homebrew/bin/vips getpoint fixtures/dt128.hdr 0 0
/opt/homebrew/bin/vips getpoint fixtures/dt128.hdr 1 0
/opt/homebrew/bin/vips getpoint fixtures/dt128.hdr 0 1
/opt/homebrew/bin/vips getpoint fixtures/dt128.hdr 1 1
/opt/homebrew/bin/vips avg fixtures/dt128.hdr
/opt/homebrew/bin/vipsheader fixtures/dt256.hdr
/opt/homebrew/bin/vips analyzeload fixtures/dt256.hdr outputs/direct.v
/opt/homebrew/bin/vipsheader fixtures/dt511.hdr
/opt/homebrew/bin/vips analyzeload fixtures/dt511.hdr outputs/direct.v
/opt/homebrew/bin/vipsheader fixtures/rank0.hdr
/opt/homebrew/bin/vips analyzeload fixtures/rank0.hdr outputs/direct.v
/opt/homebrew/bin/vipsheader fixtures/rank1.hdr
/opt/homebrew/bin/vips analyzeload fixtures/rank1.hdr outputs/direct.v
/opt/homebrew/bin/vipsheader fixtures/rank2.hdr
/opt/homebrew/bin/vips analyzeload fixtures/rank2.hdr outputs/direct.v
/opt/homebrew/bin/vipsheader outputs/direct.v
/opt/homebrew/bin/vips getpoint fixtures/rank2.hdr 0 0
/opt/homebrew/bin/vips getpoint fixtures/rank2.hdr 1 0
/opt/homebrew/bin/vips getpoint fixtures/rank2.hdr 2 0
/opt/homebrew/bin/vips getpoint fixtures/rank2.hdr 0 1
/opt/homebrew/bin/vips getpoint fixtures/rank2.hdr 1 1
/opt/homebrew/bin/vips getpoint fixtures/rank2.hdr 2 1
/opt/homebrew/bin/vipsheader fixtures/rank3.hdr
/opt/homebrew/bin/vips analyzeload fixtures/rank3.hdr outputs/direct.v
/opt/homebrew/bin/vipsheader outputs/direct.v
/opt/homebrew/bin/vips getpoint fixtures/rank3.hdr 0 0
/opt/homebrew/bin/vips getpoint fixtures/rank3.hdr 1 0
/opt/homebrew/bin/vips getpoint fixtures/rank3.hdr 2 0
/opt/homebrew/bin/vips getpoint fixtures/rank3.hdr 0 1
/opt/homebrew/bin/vips getpoint fixtures/rank3.hdr 1 1
/opt/homebrew/bin/vips getpoint fixtures/rank3.hdr 2 1
/opt/homebrew/bin/vips getpoint fixtures/rank3.hdr 0 2
/opt/homebrew/bin/vips getpoint fixtures/rank3.hdr 1 2
/opt/homebrew/bin/vips getpoint fixtures/rank3.hdr 2 2
/opt/homebrew/bin/vips getpoint fixtures/rank3.hdr 0 3
/opt/homebrew/bin/vips getpoint fixtures/rank3.hdr 1 3
/opt/homebrew/bin/vips getpoint fixtures/rank3.hdr 2 3
/opt/homebrew/bin/vipsheader fixtures/rank4.hdr
/opt/homebrew/bin/vips analyzeload fixtures/rank4.hdr outputs/direct.v
/opt/homebrew/bin/vipsheader outputs/direct.v
/opt/homebrew/bin/vips getpoint fixtures/rank4.hdr 0 0
/opt/homebrew/bin/vips getpoint fixtures/rank4.hdr 1 0
/opt/homebrew/bin/vips getpoint fixtures/rank4.hdr 2 0
/opt/homebrew/bin/vips getpoint fixtures/rank4.hdr 0 1
/opt/homebrew/bin/vips getpoint fixtures/rank4.hdr 1 1
/opt/homebrew/bin/vips getpoint fixtures/rank4.hdr 2 1
/opt/homebrew/bin/vips getpoint fixtures/rank4.hdr 0 2
/opt/homebrew/bin/vips getpoint fixtures/rank4.hdr 1 2
/opt/homebrew/bin/vips getpoint fixtures/rank4.hdr 2 2
/opt/homebrew/bin/vips getpoint fixtures/rank4.hdr 0 3
/opt/homebrew/bin/vips getpoint fixtures/rank4.hdr 1 3
/opt/homebrew/bin/vips getpoint fixtures/rank4.hdr 2 3
/opt/homebrew/bin/vips getpoint fixtures/rank4.hdr 0 4
/opt/homebrew/bin/vips getpoint fixtures/rank4.hdr 1 4
/opt/homebrew/bin/vips getpoint fixtures/rank4.hdr 2 4
/opt/homebrew/bin/vips getpoint fixtures/rank4.hdr 0 5
/opt/homebrew/bin/vips getpoint fixtures/rank4.hdr 1 5
/opt/homebrew/bin/vips getpoint fixtures/rank4.hdr 2 5
/opt/homebrew/bin/vips getpoint fixtures/rank4.hdr 0 6
/opt/homebrew/bin/vips getpoint fixtures/rank4.hdr 1 6
/opt/homebrew/bin/vips getpoint fixtures/rank4.hdr 2 6
/opt/homebrew/bin/vips getpoint fixtures/rank4.hdr 0 7
/opt/homebrew/bin/vips getpoint fixtures/rank4.hdr 1 7
/opt/homebrew/bin/vips getpoint fixtures/rank4.hdr 2 7
/opt/homebrew/bin/vipsheader fixtures/rank7.hdr
/opt/homebrew/bin/vips analyzeload fixtures/rank7.hdr outputs/direct.v
/opt/homebrew/bin/vipsheader outputs/direct.v
/opt/homebrew/bin/vips getpoint fixtures/rank7.hdr 0 0
/opt/homebrew/bin/vips getpoint fixtures/rank7.hdr 1 0
/opt/homebrew/bin/vips getpoint fixtures/rank7.hdr 0 1
/opt/homebrew/bin/vips getpoint fixtures/rank7.hdr 1 1
/opt/homebrew/bin/vips getpoint fixtures/rank7.hdr 0 2
/opt/homebrew/bin/vips getpoint fixtures/rank7.hdr 1 2
/opt/homebrew/bin/vips getpoint fixtures/rank7.hdr 0 3
/opt/homebrew/bin/vips getpoint fixtures/rank7.hdr 1 3
/opt/homebrew/bin/vips getpoint fixtures/rank7.hdr 0 4
/opt/homebrew/bin/vipsheader fixtures/rank8.hdr
/opt/homebrew/bin/vips analyzeload fixtures/rank8.hdr outputs/direct.v
/opt/homebrew/bin/vipsheader fixtures/be_short.hdr
/opt/homebrew/bin/vips getpoint fixtures/be_short.hdr 0 0
/opt/homebrew/bin/vips getpoint fixtures/be_short.hdr 1 0
/opt/homebrew/bin/vips getpoint fixtures/be_short.hdr 0 1
/opt/homebrew/bin/vips getpoint fixtures/be_short.hdr 1 1
/opt/homebrew/bin/vipsheader fixtures/le_short.hdr
/opt/homebrew/bin/vips getpoint fixtures/le_short.hdr 0 0
/opt/homebrew/bin/vips getpoint fixtures/le_short.hdr 1 0
/opt/homebrew/bin/vips getpoint fixtures/le_short.hdr 0 1
/opt/homebrew/bin/vips getpoint fixtures/le_short.hdr 1 1
/opt/homebrew/bin/vipsheader fixtures/le_header.hdr
/opt/homebrew/bin/vips analyzeload fixtures/le_header.hdr outputs/direct.v
/opt/homebrew/bin/vipsheader fixtures/vox_offset_64.hdr
/opt/homebrew/bin/vips getpoint fixtures/vox_offset_64.hdr 0 0
/opt/homebrew/bin/vips getpoint fixtures/vox_offset_64.hdr 1 0
/opt/homebrew/bin/vips getpoint fixtures/vox_offset_64.hdr 2 0
/opt/homebrew/bin/vips getpoint fixtures/vox_offset_64.hdr 0 1
/opt/homebrew/bin/vips getpoint fixtures/vox_offset_64.hdr 1 1
/opt/homebrew/bin/vips getpoint fixtures/vox_offset_64.hdr 2 1
/opt/homebrew/bin/vips avg fixtures/vox_offset_64.hdr
/opt/homebrew/bin/vipsheader fixtures/hdr_349_bytes.hdr
/opt/homebrew/bin/vips analyzeload fixtures/hdr_349_bytes.hdr outputs/direct.v
/opt/homebrew/bin/vipsheader fixtures/hdr_347_bytes.hdr
/opt/homebrew/bin/vips analyzeload fixtures/hdr_347_bytes.hdr outputs/direct.v
/opt/homebrew/bin/vipsheader fixtures/hdr_0_bytes.hdr
/opt/homebrew/bin/vips analyzeload fixtures/hdr_0_bytes.hdr outputs/direct.v
/opt/homebrew/bin/vipsheader fixtures/sizeof_hdr_200.hdr
/opt/homebrew/bin/vips analyzeload fixtures/sizeof_hdr_200.hdr outputs/direct.v
/opt/homebrew/bin/vipsheader fixtures/sizeof_hdr_0.hdr
/opt/homebrew/bin/vips analyzeload fixtures/sizeof_hdr_0.hdr outputs/direct.v
/opt/homebrew/bin/vipsheader fixtures/sizeof_hdr_348.hdr
/opt/homebrew/bin/vips analyzeload fixtures/sizeof_hdr_348.hdr outputs/direct.v
/opt/homebrew/bin/vipsheader outputs/direct.v
/opt/homebrew/bin/vipsheader fixtures/dims_32767.hdr
/opt/homebrew/bin/vips avg fixtures/dims_32767.hdr
/opt/homebrew/bin/vips analyzeload fixtures/dims_32767.hdr outputs/direct.v
/opt/homebrew/bin/vipsheader fixtures/img_truncated.hdr
/opt/homebrew/bin/vips getpoint fixtures/img_truncated.hdr 0 0
/opt/homebrew/bin/vips avg fixtures/img_truncated.hdr
/opt/homebrew/bin/vips analyzeload fixtures/img_truncated.hdr outputs/direct.v
/opt/homebrew/bin/vipsheader fixtures/img_oversize.hdr
/opt/homebrew/bin/vips getpoint fixtures/img_oversize.hdr 0 0
/opt/homebrew/bin/vips getpoint fixtures/img_oversize.hdr 1 0
/opt/homebrew/bin/vips getpoint fixtures/img_oversize.hdr 2 0
/opt/homebrew/bin/vips getpoint fixtures/img_oversize.hdr 0 1
/opt/homebrew/bin/vips getpoint fixtures/img_oversize.hdr 1 1
/opt/homebrew/bin/vips getpoint fixtures/img_oversize.hdr 2 1
/opt/homebrew/bin/vips avg fixtures/img_oversize.hdr
/opt/homebrew/bin/vipsheader fixtures/dim1_negative.hdr
/opt/homebrew/bin/vips avg fixtures/dim1_negative.hdr
/opt/homebrew/bin/vips analyzeload fixtures/dim1_negative.hdr outputs/direct.v
/opt/homebrew/bin/vipsheader outputs/direct.v
/opt/homebrew/bin/vipsheader fixtures/dim2_negative.hdr
/opt/homebrew/bin/vips avg fixtures/dim2_negative.hdr
/opt/homebrew/bin/vips analyzeload fixtures/dim2_negative.hdr outputs/direct.v
/opt/homebrew/bin/vipsheader outputs/direct.v
/opt/homebrew/bin/vipsheader fixtures/dim1_zero.hdr
/opt/homebrew/bin/vips avg fixtures/dim1_zero.hdr
/opt/homebrew/bin/vips analyzeload fixtures/dim1_zero.hdr outputs/direct.v
/opt/homebrew/bin/vipsheader outputs/direct.v
/opt/homebrew/bin/vipsheader fixtures/all_zero_348.hdr
/opt/homebrew/bin/vips analyzeload fixtures/all_zero_348.hdr outputs/direct.v
/opt/homebrew/bin/vipsheader fixtures/all_ff_348.hdr
/opt/homebrew/bin/vips analyzeload fixtures/all_ff_348.hdr outputs/direct.v
/opt/homebrew/bin/vipsheader fixtures/ascii_348.hdr
/opt/homebrew/bin/vips analyzeload fixtures/ascii_348.hdr outputs/direct.v
/opt/homebrew/bin/vipsheader -a fixtures/meta_strings.hdr
/opt/homebrew/bin/vipsheader fixtures/rgb_2d.hdr
/opt/homebrew/bin/vips getpoint fixtures/rgb_2d.hdr 0 0
/opt/homebrew/bin/vips getpoint fixtures/rgb_2d.hdr 1 0
/opt/homebrew/bin/vips getpoint fixtures/rgb_2d.hdr 2 0
/opt/homebrew/bin/vips getpoint fixtures/rgb_2d.hdr 0 1
/opt/homebrew/bin/vips getpoint fixtures/rgb_2d.hdr 1 1
/opt/homebrew/bin/vips getpoint fixtures/rgb_2d.hdr 2 1
/opt/homebrew/bin/vips --vips-config
/opt/homebrew/bin/vips -l
/opt/homebrew/bin/vips --version
