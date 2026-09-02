#!/bin/sh
# Every vips command capture.py ran, in order. Regenerate with
# `python3 capture.py` from this directory.
#
# Many of these are EXPECTED to fail: the refusal records are the point.
set +e

/opt/homebrew/bin/vipsheader -a fixtures/base_2x3_uint8.mat
/opt/homebrew/bin/vips getpoint fixtures/base_2x3_uint8.mat 0 0
/opt/homebrew/bin/vips getpoint fixtures/base_2x3_uint8.mat 1 0
/opt/homebrew/bin/vips getpoint fixtures/base_2x3_uint8.mat 2 0
/opt/homebrew/bin/vips getpoint fixtures/base_2x3_uint8.mat 0 1
/opt/homebrew/bin/vips getpoint fixtures/base_2x3_uint8.mat 1 1
/opt/homebrew/bin/vips getpoint fixtures/base_2x3_uint8.mat 2 1
/opt/homebrew/bin/vipsheader fixtures/level4.mat
/opt/homebrew/bin/vips matload fixtures/level4.mat outputs/direct.v
/opt/homebrew/bin/vipsheader outputs/direct.v
/opt/homebrew/bin/vips getpoint outputs/direct.v 0 0
/opt/homebrew/bin/vips getpoint outputs/direct.v 1 0
/opt/homebrew/bin/vips getpoint outputs/direct.v 2 0
/opt/homebrew/bin/vips getpoint outputs/direct.v 0 1
/opt/homebrew/bin/vips getpoint outputs/direct.v 1 1
/opt/homebrew/bin/vips getpoint outputs/direct.v 2 1
/opt/homebrew/bin/vipsheader fixtures/level73_hdf5.mat
/opt/homebrew/bin/vips matload fixtures/level73_hdf5.mat outputs/direct.v
/opt/homebrew/bin/vipsheader fixtures/magic_only.mat
/opt/homebrew/bin/vips matload fixtures/magic_only.mat outputs/direct.v
/opt/homebrew/bin/vipsheader fixtures/header_only.mat
/opt/homebrew/bin/vips matload fixtures/header_only.mat outputs/direct.v
/opt/homebrew/bin/vipsheader fixtures/nine_bytes.mat
/opt/homebrew/bin/vips matload fixtures/nine_bytes.mat outputs/direct.v
/opt/homebrew/bin/vipsheader fixtures/prefix_only.mat
/opt/homebrew/bin/vipsheader fixtures/magic_MATLAB_51.mat
/opt/homebrew/bin/vips matload fixtures/magic_MATLAB_51.mat outputs/direct.v
/opt/homebrew/bin/vipsheader outputs/direct.v
/opt/homebrew/bin/vipsheader fixtures/magic_lowercase_50.mat
/opt/homebrew/bin/vips matload fixtures/magic_lowercase_50.mat outputs/direct.v
/opt/homebrew/bin/vipsheader outputs/direct.v
/opt/homebrew/bin/vipsheader fixtures/magic_underscore_50.mat
/opt/homebrew/bin/vips matload fixtures/magic_underscore_50.mat outputs/direct.v
/opt/homebrew/bin/vipsheader outputs/direct.v
/opt/homebrew/bin/vipsheader fixtures/probe_byte.mat
/opt/homebrew/bin/vipsheader fixtures/probe_byte.mat
/opt/homebrew/bin/vipsheader fixtures/probe_byte.mat
/opt/homebrew/bin/vipsheader fixtures/probe_byte.mat
/opt/homebrew/bin/vipsheader fixtures/probe_byte.mat
/opt/homebrew/bin/vipsheader fixtures/probe_byte.mat
/opt/homebrew/bin/vipsheader fixtures/probe_byte.mat
/opt/homebrew/bin/vipsheader fixtures/probe_byte.mat
/opt/homebrew/bin/vipsheader fixtures/probe_byte.mat
/opt/homebrew/bin/vipsheader fixtures/probe_byte.mat
/opt/homebrew/bin/vipsheader fixtures/probe_byte.mat
/opt/homebrew/bin/vipsheader fixtures/probe_byte.mat
/opt/homebrew/bin/vipsheader fixtures/probe_byte.mat
/opt/homebrew/bin/vipsheader fixtures/probe_byte.mat
/opt/homebrew/bin/vipsheader fixtures/probe_byte.mat
/opt/homebrew/bin/vipsheader fixtures/probe_byte.mat
/opt/homebrew/bin/vipsheader fixtures/probe_byte.mat
/opt/homebrew/bin/vipsheader fixtures/probe_byte.mat
/opt/homebrew/bin/vipsheader fixtures/probe_byte.mat
/opt/homebrew/bin/vipsheader fixtures/probe_byte.mat
/opt/homebrew/bin/vipsheader fixtures/probe_byte.mat
/opt/homebrew/bin/vipsheader fixtures/probe_byte.mat
/opt/homebrew/bin/vipsheader fixtures/probe_byte.mat
/opt/homebrew/bin/vipsheader fixtures/probe_byte.mat
/opt/homebrew/bin/vipsheader fixtures/probe_byte.mat
/opt/homebrew/bin/vipsheader fixtures/probe_byte.mat
/opt/homebrew/bin/vipsheader fixtures/probe_byte.mat
/opt/homebrew/bin/vipsheader fixtures/probe_byte.mat
/opt/homebrew/bin/vipsheader fixtures/probe_byte.mat
/opt/homebrew/bin/vipsheader fixtures/probe_byte.mat
/opt/homebrew/bin/vipsheader fixtures/probe_byte.mat
/opt/homebrew/bin/vipsheader fixtures/probe_byte.mat
/opt/homebrew/bin/vipsheader fixtures/probe_byte.mat
/opt/homebrew/bin/vipsheader fixtures/probe_byte.mat
/opt/homebrew/bin/vipsheader fixtures/probe_byte.mat
/opt/homebrew/bin/vipsheader fixtures/probe_byte.mat
/opt/homebrew/bin/vipsheader fixtures/probe_byte.mat
/opt/homebrew/bin/vipsheader fixtures/probe_byte.mat
/opt/homebrew/bin/vipsheader fixtures/probe_byte.mat
/opt/homebrew/bin/vipsheader fixtures/probe_byte.mat
/opt/homebrew/bin/vipsheader fixtures/probe_byte.mat
/opt/homebrew/bin/vipsheader fixtures/probe_byte.mat
/opt/homebrew/bin/vipsheader fixtures/probe_byte.mat
/opt/homebrew/bin/vipsheader fixtures/probe_byte.mat
/opt/homebrew/bin/vipsheader fixtures/probe_byte.mat
/opt/homebrew/bin/vipsheader fixtures/probe_byte.mat
/opt/homebrew/bin/vipsheader fixtures/probe_byte.mat
/opt/homebrew/bin/vipsheader fixtures/probe_byte.mat
/opt/homebrew/bin/vipsheader fixtures/probe_byte.mat
/opt/homebrew/bin/vipsheader fixtures/probe_byte.mat
/opt/homebrew/bin/vipsheader fixtures/probe_byte.mat
/opt/homebrew/bin/vipsheader fixtures/probe_byte.mat
/opt/homebrew/bin/vipsheader fixtures/probe_byte.mat
/opt/homebrew/bin/vipsheader fixtures/probe_byte.mat
/opt/homebrew/bin/vipsheader fixtures/probe_byte.mat
/opt/homebrew/bin/vipsheader fixtures/probe_byte.mat
/opt/homebrew/bin/vipsheader fixtures/probe_byte.mat
/opt/homebrew/bin/vipsheader fixtures/probe_byte.mat
/opt/homebrew/bin/vipsheader fixtures/probe_byte.mat
/opt/homebrew/bin/vipsheader fixtures/probe_byte.mat
/opt/homebrew/bin/vipsheader fixtures/probe_byte.mat
/opt/homebrew/bin/vipsheader fixtures/probe_byte.mat
/opt/homebrew/bin/vipsheader fixtures/probe_byte.mat
/opt/homebrew/bin/vipsheader fixtures/probe_byte.mat
/opt/homebrew/bin/vipsheader fixtures/probe_byte.mat
/opt/homebrew/bin/vipsheader fixtures/probe_byte.mat
/opt/homebrew/bin/vipsheader fixtures/probe_byte.mat
/opt/homebrew/bin/vipsheader fixtures/probe_byte.mat
/opt/homebrew/bin/vipsheader fixtures/probe_byte.mat
/opt/homebrew/bin/vipsheader fixtures/probe_byte.mat
/opt/homebrew/bin/vipsheader fixtures/probe_byte.mat
/opt/homebrew/bin/vipsheader fixtures/probe_byte.mat
/opt/homebrew/bin/vipsheader fixtures/probe_byte.mat
/opt/homebrew/bin/vipsheader fixtures/probe_byte.mat
/opt/homebrew/bin/vipsheader fixtures/probe_byte.mat
/opt/homebrew/bin/vipsheader fixtures/probe_byte.mat
/opt/homebrew/bin/vipsheader fixtures/probe_byte.mat
/opt/homebrew/bin/vipsheader fixtures/probe_byte.mat
/opt/homebrew/bin/vipsheader fixtures/probe_byte.mat
/opt/homebrew/bin/vipsheader fixtures/probe_byte.mat
/opt/homebrew/bin/vipsheader fixtures/probe_byte.mat
/opt/homebrew/bin/vipsheader fixtures/probe_byte.mat
/opt/homebrew/bin/vipsheader fixtures/probe_byte.mat
/opt/homebrew/bin/vipsheader fixtures/probe_byte.mat
/opt/homebrew/bin/vipsheader fixtures/probe_byte.mat
/opt/homebrew/bin/vipsheader fixtures/probe_byte.mat
/opt/homebrew/bin/vipsheader fixtures/probe_byte.mat
/opt/homebrew/bin/vipsheader fixtures/probe_byte.mat
/opt/homebrew/bin/vipsheader fixtures/probe_byte.mat
/opt/homebrew/bin/vipsheader fixtures/probe_byte.mat
/opt/homebrew/bin/vipsheader fixtures/probe_byte.mat
/opt/homebrew/bin/vipsheader fixtures/probe_byte.mat
/opt/homebrew/bin/vipsheader fixtures/probe_byte.mat
/opt/homebrew/bin/vipsheader fixtures/probe_byte.mat
/opt/homebrew/bin/vipsheader fixtures/probe_byte.mat
/opt/homebrew/bin/vipsheader fixtures/probe_byte.mat
/opt/homebrew/bin/vipsheader fixtures/probe_byte.mat
/opt/homebrew/bin/vipsheader fixtures/probe_byte.mat
/opt/homebrew/bin/vipsheader fixtures/probe_byte.mat
/opt/homebrew/bin/vipsheader fixtures/probe_byte.mat
/opt/homebrew/bin/vipsheader fixtures/probe_byte.mat
/opt/homebrew/bin/vipsheader fixtures/probe_byte.mat
/opt/homebrew/bin/vipsheader fixtures/probe_byte.mat
/opt/homebrew/bin/vipsheader fixtures/probe_byte.mat
/opt/homebrew/bin/vipsheader fixtures/probe_byte.mat
/opt/homebrew/bin/vipsheader fixtures/probe_byte.mat
/opt/homebrew/bin/vipsheader fixtures/probe_byte.mat
/opt/homebrew/bin/vipsheader fixtures/probe_byte.mat
/opt/homebrew/bin/vipsheader fixtures/probe_byte.mat
/opt/homebrew/bin/vipsheader fixtures/probe_byte.mat
/opt/homebrew/bin/vipsheader fixtures/probe_byte.mat
/opt/homebrew/bin/vipsheader fixtures/probe_byte.mat
/opt/homebrew/bin/vipsheader fixtures/probe_byte.mat
/opt/homebrew/bin/vipsheader fixtures/probe_byte.mat
/opt/homebrew/bin/vipsheader fixtures/probe_byte.mat
/opt/homebrew/bin/vipsheader fixtures/probe_byte.mat
/opt/homebrew/bin/vipsheader fixtures/probe_byte.mat
/opt/homebrew/bin/vipsheader fixtures/probe_byte.mat
/opt/homebrew/bin/vipsheader fixtures/probe_byte.mat
/opt/homebrew/bin/vipsheader fixtures/probe_byte.mat
/opt/homebrew/bin/vipsheader fixtures/probe_byte.mat
/opt/homebrew/bin/vipsheader fixtures/probe_byte.mat
/opt/homebrew/bin/vipsheader fixtures/probe_byte.mat
/opt/homebrew/bin/vipsheader fixtures/probe_byte.mat
/opt/homebrew/bin/vipsheader fixtures/probe_byte.mat
/opt/homebrew/bin/vipsheader fixtures/probe_byte.mat
/opt/homebrew/bin/vipsheader fixtures/probe_byte.mat
/opt/homebrew/bin/vipsheader fixtures/probe_byte.mat
/opt/homebrew/bin/vipsheader fixtures/probe_pair.mat
/opt/homebrew/bin/vipsheader fixtures/probe_pair.mat
/opt/homebrew/bin/vipsheader fixtures/probe_pair.mat
/opt/homebrew/bin/vipsheader fixtures/probe_pair.mat
/opt/homebrew/bin/vipsheader fixtures/probe_pair.mat
/opt/homebrew/bin/vipsheader fixtures/probe_pair.mat
/opt/homebrew/bin/vipsheader fixtures/probe_pair.mat
/opt/homebrew/bin/vipsheader fixtures/probe_pair.mat
/opt/homebrew/bin/vipsheader fixtures/probe_pair.mat
/opt/homebrew/bin/vipsheader fixtures/probe_len.mat
/opt/homebrew/bin/vipsheader fixtures/probe_len.mat
/opt/homebrew/bin/vipsheader fixtures/probe_len.mat
/opt/homebrew/bin/vipsheader fixtures/probe_len.mat
/opt/homebrew/bin/vipsheader fixtures/probe_len.mat
/opt/homebrew/bin/vipsheader fixtures/probe_len.mat
/opt/homebrew/bin/vipsheader fixtures/probe_len.mat
/opt/homebrew/bin/vipsheader fixtures/probe_len.mat
/opt/homebrew/bin/vipsheader fixtures/rank1.mat
/opt/homebrew/bin/vips getpoint fixtures/rank1.mat 0 0
/opt/homebrew/bin/vips getpoint fixtures/rank1.mat 0 1
/opt/homebrew/bin/vips getpoint fixtures/rank1.mat 0 2
/opt/homebrew/bin/vips getpoint fixtures/rank1.mat 0 3
/opt/homebrew/bin/vipsheader fixtures/base_2x3_uint8.mat
/opt/homebrew/bin/vipsheader -a fixtures/rank3_2x3x3.mat
/opt/homebrew/bin/vips getpoint fixtures/rank3_2x3x3.mat 0 0
/opt/homebrew/bin/vips getpoint fixtures/rank3_2x3x3.mat 1 0
/opt/homebrew/bin/vips getpoint fixtures/rank3_2x3x3.mat 2 0
/opt/homebrew/bin/vips getpoint fixtures/rank3_2x3x3.mat 0 1
/opt/homebrew/bin/vips getpoint fixtures/rank3_2x3x3.mat 1 1
/opt/homebrew/bin/vips getpoint fixtures/rank3_2x3x3.mat 2 1
/opt/homebrew/bin/vipsheader fixtures/rank4_only.mat
/opt/homebrew/bin/vips matload fixtures/rank4_only.mat outputs/direct.v
/opt/homebrew/bin/vipsheader fixtures/rank4_then_rank2.mat
/opt/homebrew/bin/vips getpoint fixtures/rank4_then_rank2.mat 0 0
/opt/homebrew/bin/vips getpoint fixtures/rank4_then_rank2.mat 1 0
/opt/homebrew/bin/vips getpoint fixtures/rank4_then_rank2.mat 0 1
/opt/homebrew/bin/vips getpoint fixtures/rank4_then_rank2.mat 1 1
/opt/homebrew/bin/vipsheader fixtures/class_mat_c_uint8.mat
/opt/homebrew/bin/vips getpoint fixtures/class_mat_c_uint8.mat 0 0
/opt/homebrew/bin/vips getpoint fixtures/class_mat_c_uint8.mat 1 0
/opt/homebrew/bin/vips getpoint fixtures/class_mat_c_uint8.mat 0 1
/opt/homebrew/bin/vips getpoint fixtures/class_mat_c_uint8.mat 1 1
/opt/homebrew/bin/vipsheader fixtures/class_mat_c_int8.mat
/opt/homebrew/bin/vips getpoint fixtures/class_mat_c_int8.mat 0 0
/opt/homebrew/bin/vips getpoint fixtures/class_mat_c_int8.mat 1 0
/opt/homebrew/bin/vips getpoint fixtures/class_mat_c_int8.mat 0 1
/opt/homebrew/bin/vips getpoint fixtures/class_mat_c_int8.mat 1 1
/opt/homebrew/bin/vipsheader fixtures/class_mat_c_uint16.mat
/opt/homebrew/bin/vips getpoint fixtures/class_mat_c_uint16.mat 0 0
/opt/homebrew/bin/vips getpoint fixtures/class_mat_c_uint16.mat 1 0
/opt/homebrew/bin/vips getpoint fixtures/class_mat_c_uint16.mat 0 1
/opt/homebrew/bin/vips getpoint fixtures/class_mat_c_uint16.mat 1 1
/opt/homebrew/bin/vipsheader fixtures/class_mat_c_int16.mat
/opt/homebrew/bin/vips getpoint fixtures/class_mat_c_int16.mat 0 0
/opt/homebrew/bin/vips getpoint fixtures/class_mat_c_int16.mat 1 0
/opt/homebrew/bin/vips getpoint fixtures/class_mat_c_int16.mat 0 1
/opt/homebrew/bin/vips getpoint fixtures/class_mat_c_int16.mat 1 1
/opt/homebrew/bin/vipsheader fixtures/class_mat_c_uint32.mat
/opt/homebrew/bin/vips getpoint fixtures/class_mat_c_uint32.mat 0 0
/opt/homebrew/bin/vips getpoint fixtures/class_mat_c_uint32.mat 1 0
/opt/homebrew/bin/vips getpoint fixtures/class_mat_c_uint32.mat 0 1
/opt/homebrew/bin/vips getpoint fixtures/class_mat_c_uint32.mat 1 1
/opt/homebrew/bin/vipsheader fixtures/class_mat_c_int32.mat
/opt/homebrew/bin/vips getpoint fixtures/class_mat_c_int32.mat 0 0
/opt/homebrew/bin/vips getpoint fixtures/class_mat_c_int32.mat 1 0
/opt/homebrew/bin/vips getpoint fixtures/class_mat_c_int32.mat 0 1
/opt/homebrew/bin/vips getpoint fixtures/class_mat_c_int32.mat 1 1
/opt/homebrew/bin/vipsheader fixtures/class_mat_c_single.mat
/opt/homebrew/bin/vips getpoint fixtures/class_mat_c_single.mat 0 0
/opt/homebrew/bin/vips getpoint fixtures/class_mat_c_single.mat 1 0
/opt/homebrew/bin/vips getpoint fixtures/class_mat_c_single.mat 0 1
/opt/homebrew/bin/vips getpoint fixtures/class_mat_c_single.mat 1 1
/opt/homebrew/bin/vipsheader fixtures/class_mat_c_double.mat
/opt/homebrew/bin/vips getpoint fixtures/class_mat_c_double.mat 0 0
/opt/homebrew/bin/vips getpoint fixtures/class_mat_c_double.mat 1 0
/opt/homebrew/bin/vips getpoint fixtures/class_mat_c_double.mat 0 1
/opt/homebrew/bin/vips getpoint fixtures/class_mat_c_double.mat 1 1
/opt/homebrew/bin/vipsheader fixtures/bands3_mat_c_uint8.mat
/opt/homebrew/bin/vipsheader fixtures/bands3_mat_c_uint16.mat
/opt/homebrew/bin/vipsheader fixtures/bands3_mat_c_int16.mat
/opt/homebrew/bin/vipsheader fixtures/bands3_mat_c_single.mat
/opt/homebrew/bin/vipsheader fixtures/bands3_mat_c_double.mat
/opt/homebrew/bin/vipsheader fixtures/bands3_mat_c_int8.mat
/opt/homebrew/bin/vipsheader fixtures/bands3_mat_c_uint32.mat
/opt/homebrew/bin/vipsheader fixtures/bands3_mat_c_int32.mat
/opt/homebrew/bin/vipsheader fixtures/class_mat_c_int64.mat
/opt/homebrew/bin/vips matload fixtures/class_mat_c_int64.mat outputs/direct.v
/opt/homebrew/bin/vipsheader fixtures/class_mat_c_uint64.mat
/opt/homebrew/bin/vips matload fixtures/class_mat_c_uint64.mat outputs/direct.v
/opt/homebrew/bin/vipsheader fixtures/class_mat_c_char.mat
/opt/homebrew/bin/vips matload fixtures/class_mat_c_char.mat outputs/direct.v
/opt/homebrew/bin/vipsheader fixtures/class_mat_c_sparse.mat
/opt/homebrew/bin/vips matload fixtures/class_mat_c_sparse.mat outputs/direct.v
/opt/homebrew/bin/vipsheader fixtures/logical_uint8.mat
/opt/homebrew/bin/vips getpoint fixtures/logical_uint8.mat 0 0
/opt/homebrew/bin/vips getpoint fixtures/logical_uint8.mat 1 0
/opt/homebrew/bin/vips getpoint fixtures/logical_uint8.mat 0 1
/opt/homebrew/bin/vips getpoint fixtures/logical_uint8.mat 1 1
/opt/homebrew/bin/vipsheader fixtures/int64_then_uint8.mat
/opt/homebrew/bin/vips matload fixtures/int64_then_uint8.mat outputs/direct.v
/opt/homebrew/bin/vipsheader fixtures/endian_little.mat
/opt/homebrew/bin/vips getpoint fixtures/endian_little.mat 0 0
/opt/homebrew/bin/vips getpoint fixtures/endian_little.mat 1 0
/opt/homebrew/bin/vips getpoint fixtures/endian_little.mat 0 1
/opt/homebrew/bin/vips getpoint fixtures/endian_little.mat 1 1
/opt/homebrew/bin/vipsheader fixtures/endian_big.mat
/opt/homebrew/bin/vips getpoint fixtures/endian_big.mat 0 0
/opt/homebrew/bin/vips getpoint fixtures/endian_big.mat 1 0
/opt/homebrew/bin/vips getpoint fixtures/endian_big.mat 0 1
/opt/homebrew/bin/vips getpoint fixtures/endian_big.mat 1 1
/opt/homebrew/bin/vipsheader fixtures/endian_bogus.mat
/opt/homebrew/bin/vips matload fixtures/endian_bogus.mat outputs/direct.v
/opt/homebrew/bin/vipsheader fixtures/compressed.mat
/opt/homebrew/bin/vips getpoint fixtures/compressed.mat 0 0
/opt/homebrew/bin/vips getpoint fixtures/compressed.mat 1 0
/opt/homebrew/bin/vips getpoint fixtures/compressed.mat 2 0
/opt/homebrew/bin/vips getpoint fixtures/compressed.mat 0 1
/opt/homebrew/bin/vips getpoint fixtures/compressed.mat 1 1
/opt/homebrew/bin/vips getpoint fixtures/compressed.mat 2 1
/opt/homebrew/bin/vipsheader fixtures/compressed_corrupt.mat
/opt/homebrew/bin/vips matload fixtures/compressed_corrupt.mat outputs/direct.v
/opt/homebrew/bin/vipsheader fixtures/header_only.mat
/opt/homebrew/bin/vips matload fixtures/header_only.mat outputs/direct.v
/opt/homebrew/bin/vipsheader fixtures/tag_overruns_file.mat
/opt/homebrew/bin/vips matload fixtures/tag_overruns_file.mat outputs/direct.v
/opt/homebrew/bin/vipsheader fixtures/truncated.mat
/opt/homebrew/bin/vips avg fixtures/truncated.mat
/opt/homebrew/bin/vips matload fixtures/truncated.mat outputs/direct.v
/opt/homebrew/bin/vipsheader fixtures/dims_100x100_four_bytes.mat
/opt/homebrew/bin/vips avg fixtures/dims_100x100_four_bytes.mat
/opt/homebrew/bin/vips matload fixtures/dims_100x100_four_bytes.mat outputs/direct.v
/opt/homebrew/bin/vipsheader fixtures/dims_100000x100000.mat
/opt/homebrew/bin/vips matload fixtures/dims_100000x100000.mat outputs/direct.v
/opt/homebrew/bin/vipsheader fixtures/dim_zero.mat
/opt/homebrew/bin/vips matload fixtures/dim_zero.mat outputs/direct.v
/opt/homebrew/bin/vipsheader outputs/direct.v
/opt/homebrew/bin/vipsheader fixtures/dim_negative.mat
/opt/homebrew/bin/vips matload fixtures/dim_negative.mat outputs/direct.v
/opt/homebrew/bin/vips getpoint fixtures/complex_double.mat 0 0
/opt/homebrew/bin/vips getpoint fixtures/complex_double.mat 0 0
/opt/homebrew/bin/vips getpoint fixtures/complex_double.mat 0 0
/opt/homebrew/bin/vipsheader fixtures/complex_double.mat
/opt/homebrew/bin/vips --version
/opt/homebrew/bin/vips -l
/opt/homebrew/bin/vips --vips-config
