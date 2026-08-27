#!/bin/sh
# Every command capture.py ran, in order. Regenerate with
# `python3 capture.py` from this directory.
#
# The oracle is nifti_clib, NOT libvips: this vips build reports
# `NIfTI load/save with libnifti: false` and has no niftiload.
# nifti_clib was obtained and built like this, once, before any
# of the commands below ran:
#
#   git clone --depth 200 \
#       https://github.com/NIFTI-Imaging/nifti_clib.git
#   # HEAD 8f72d1165aa62320cc6982d6ddd71a7f6b9924c5
#   # v3.0.1-91-g8f72d11, 2024-12-20 13:35:31 -0600
#   cmake -S nifti_clib -B build -DCMAKE_BUILD_TYPE=Release \
#         -DCMAKE_INSTALL_PREFIX=$NIFTI_PREFIX \
#         -DNIFTI_BUILD_APPLICATIONS=ON -DUSE_NIFTI2_CODE=ON
#   cmake --build build -j8
#   cmake --install build
#
# NIFTI_PREFIX is where that install landed.
set -e

NIFTI_PREFIX=/Users/rom/workspace/nifti-oracle-641/install

/opt/homebrew/bin/vips --version
/opt/homebrew/bin/vips --vips-config
/opt/homebrew/bin/vips -l
/opt/homebrew/bin/vips niftiload --help
/opt/homebrew/bin/vips niftisave --help
git -C /Users/rom/workspace/nifti-oracle-641/nifti_clib rev-parse HEAD
git -C /Users/rom/workspace/nifti-oracle-641/nifti_clib describe --tags
git -C /Users/rom/workspace/nifti-oracle-641/nifti_clib log -1 --format=%ci
git -C /Users/rom/workspace/nifti-oracle-641/nifti_clib config --get remote.origin.url
cc --version
cc -O2 -std=c99 -Wall -Wextra -I $NIFTI_PREFIX/include/nifti -o outputs/probe probe.c $NIFTI_PREFIX/lib/libnifti2.a $NIFTI_PREFIX/lib/libznz.a -lm -lz
$NIFTI_PREFIX/bin/nifti_tool -quiet -ver
$NIFTI_PREFIX/bin/nifti_tool -quiet -nifti_ver
$NIFTI_PREFIX/bin/nifti_tool -quiet -with_zlib
outputs/probe env
outputs/probe datatypes
outputs/probe make fixtures/dt2_uint8 1 1 2 3 2 3 1 0 0 0 0 1.0 0.0
outputs/probe read fixtures/dt2_uint8.nii
$NIFTI_PREFIX/bin/nifti_tool -quiet -disp_ci -1 -1 -1 -1 -1 -1 -1 -infiles fixtures/dt2_uint8.nii
outputs/probe readhdr fixtures/dt2_uint8.nii 1
outputs/probe make fixtures/dt4_int16 1 1 4 3 2 3 1 0 0 0 0 1.0 0.0
outputs/probe read fixtures/dt4_int16.nii
$NIFTI_PREFIX/bin/nifti_tool -quiet -disp_ci -1 -1 -1 -1 -1 -1 -1 -infiles fixtures/dt4_int16.nii
outputs/probe readhdr fixtures/dt4_int16.nii 1
outputs/probe make fixtures/dt8_int32 1 1 8 3 2 3 1 0 0 0 0 1.0 0.0
outputs/probe read fixtures/dt8_int32.nii
$NIFTI_PREFIX/bin/nifti_tool -quiet -disp_ci -1 -1 -1 -1 -1 -1 -1 -infiles fixtures/dt8_int32.nii
outputs/probe readhdr fixtures/dt8_int32.nii 1
outputs/probe make fixtures/dt16_float32 1 1 16 3 2 3 1 0 0 0 0 1.0 0.0
outputs/probe read fixtures/dt16_float32.nii
$NIFTI_PREFIX/bin/nifti_tool -quiet -disp_ci -1 -1 -1 -1 -1 -1 -1 -infiles fixtures/dt16_float32.nii
outputs/probe readhdr fixtures/dt16_float32.nii 1
outputs/probe make fixtures/dt32_complex64 1 1 32 3 2 3 1 0 0 0 0 1.0 0.0
outputs/probe read fixtures/dt32_complex64.nii
$NIFTI_PREFIX/bin/nifti_tool -quiet -disp_ci -1 -1 -1 -1 -1 -1 -1 -infiles fixtures/dt32_complex64.nii
outputs/probe readhdr fixtures/dt32_complex64.nii 1
outputs/probe make fixtures/dt64_float64 1 1 64 3 2 3 1 0 0 0 0 1.0 0.0
outputs/probe read fixtures/dt64_float64.nii
$NIFTI_PREFIX/bin/nifti_tool -quiet -disp_ci -1 -1 -1 -1 -1 -1 -1 -infiles fixtures/dt64_float64.nii
outputs/probe readhdr fixtures/dt64_float64.nii 1
outputs/probe make fixtures/dt128_rgb24 1 1 128 3 2 3 1 0 0 0 0 1.0 0.0
outputs/probe read fixtures/dt128_rgb24.nii
$NIFTI_PREFIX/bin/nifti_tool -quiet -disp_ci -1 -1 -1 -1 -1 -1 -1 -infiles fixtures/dt128_rgb24.nii
outputs/probe readhdr fixtures/dt128_rgb24.nii 1
outputs/probe make fixtures/dt256_int8 1 1 256 3 2 3 1 0 0 0 0 1.0 0.0
outputs/probe read fixtures/dt256_int8.nii
$NIFTI_PREFIX/bin/nifti_tool -quiet -disp_ci -1 -1 -1 -1 -1 -1 -1 -infiles fixtures/dt256_int8.nii
outputs/probe readhdr fixtures/dt256_int8.nii 1
outputs/probe make fixtures/dt512_uint16 1 1 512 3 2 3 1 0 0 0 0 1.0 0.0
outputs/probe read fixtures/dt512_uint16.nii
$NIFTI_PREFIX/bin/nifti_tool -quiet -disp_ci -1 -1 -1 -1 -1 -1 -1 -infiles fixtures/dt512_uint16.nii
outputs/probe readhdr fixtures/dt512_uint16.nii 1
outputs/probe make fixtures/dt768_uint32 1 1 768 3 2 3 1 0 0 0 0 1.0 0.0
outputs/probe read fixtures/dt768_uint32.nii
$NIFTI_PREFIX/bin/nifti_tool -quiet -disp_ci -1 -1 -1 -1 -1 -1 -1 -infiles fixtures/dt768_uint32.nii
outputs/probe readhdr fixtures/dt768_uint32.nii 1
outputs/probe make fixtures/dt1024_int64 1 1 1024 3 2 3 1 0 0 0 0 1.0 0.0
outputs/probe read fixtures/dt1024_int64.nii
$NIFTI_PREFIX/bin/nifti_tool -quiet -disp_ci -1 -1 -1 -1 -1 -1 -1 -infiles fixtures/dt1024_int64.nii
outputs/probe readhdr fixtures/dt1024_int64.nii 1
outputs/probe make fixtures/dt1280_uint64 1 1 1280 3 2 3 1 0 0 0 0 1.0 0.0
outputs/probe read fixtures/dt1280_uint64.nii
$NIFTI_PREFIX/bin/nifti_tool -quiet -disp_ci -1 -1 -1 -1 -1 -1 -1 -infiles fixtures/dt1280_uint64.nii
outputs/probe readhdr fixtures/dt1280_uint64.nii 1
outputs/probe make fixtures/dt1536_float128 1 1 1536 3 2 3 1 0 0 0 0 1.0 0.0
outputs/probe read fixtures/dt1536_float128.nii
$NIFTI_PREFIX/bin/nifti_tool -quiet -disp_ci -1 -1 -1 -1 -1 -1 -1 -infiles fixtures/dt1536_float128.nii
outputs/probe readhdr fixtures/dt1536_float128.nii 1
outputs/probe make fixtures/dt1792_complex128 1 1 1792 3 2 3 1 0 0 0 0 1.0 0.0
outputs/probe read fixtures/dt1792_complex128.nii
$NIFTI_PREFIX/bin/nifti_tool -quiet -disp_ci -1 -1 -1 -1 -1 -1 -1 -infiles fixtures/dt1792_complex128.nii
outputs/probe readhdr fixtures/dt1792_complex128.nii 1
outputs/probe make fixtures/dt2048_complex256 1 1 2048 3 2 3 1 0 0 0 0 1.0 0.0
outputs/probe read fixtures/dt2048_complex256.nii
$NIFTI_PREFIX/bin/nifti_tool -quiet -disp_ci -1 -1 -1 -1 -1 -1 -1 -infiles fixtures/dt2048_complex256.nii
outputs/probe readhdr fixtures/dt2048_complex256.nii 1
outputs/probe make fixtures/dt2304_rgba32 1 1 2304 3 2 3 1 0 0 0 0 1.0 0.0
outputs/probe read fixtures/dt2304_rgba32.nii
$NIFTI_PREFIX/bin/nifti_tool -quiet -disp_ci -1 -1 -1 -1 -1 -1 -1 -infiles fixtures/dt2304_rgba32.nii
outputs/probe readhdr fixtures/dt2304_rgba32.nii 1
outputs/probe offsets
outputs/probe make fixtures/ver_n1_single 1 1 4 3 2 3 1 0 0 0 0 1.0 0.0
outputs/probe make fixtures/ver_n2_single 2 4 4 3 2 3 1 0 0 0 0 1.0 0.0
outputs/probe hdrver fixtures/ver_n1_single.nii
outputs/probe hdrver fixtures/ver_n2_single.nii
outputs/probe make fixtures/magic_zero_pair_src 1 2 4 3 2 3 1 0 0 0 0 1.0 0.0
outputs/probe make fixtures/pair_be_src 1 2 4 3 2 3 1 0 0 0 0 1.0 0.0
outputs/probe swapfile fixtures/pair_be_src.hdr fixtures/pair_be.hdr 1 352 2
outputs/probe swapfile fixtures/pair_be_src.img fixtures/pair_be.img 1 0 2
/opt/homebrew/bin/vipsheader fixtures/pair_be.hdr
/opt/homebrew/bin/vipsheader -a fixtures/pair_be348.hdr
/opt/homebrew/bin/vipsheader fixtures/pair_be_src.hdr
/opt/homebrew/bin/vipsheader fixtures/ver_n1_single.nii
outputs/probe make fixtures/pair_n1 1 2 4 3 2 3 1 0 0 0 0 1.0 0.0
outputs/probe make fixtures/pair_n2 2 5 4 3 2 3 1 0 0 0 0 1.0 0.0
outputs/probe read fixtures/pair_n1.hdr
outputs/probe names fixtures/pair_n1.hdr
outputs/probe read fixtures/pair_n2.hdr
outputs/probe names fixtures/pair_n2.hdr
outputs/probe read fixtures/ver_n1_single.nii
outputs/probe names fixtures/ver_n1_single.nii
outputs/probe read fixtures/ver_n2_single.nii
outputs/probe names fixtures/ver_n2_single.nii
outputs/probe read fixtures/lonely_hdr.hdr
outputs/probe readhdr fixtures/lonely_hdr.hdr 1
outputs/probe names fixtures/lonely_hdr.hdr
$NIFTI_PREFIX/bin/nifti_tool -quiet -disp_hdr -infiles fixtures/lonely_hdr.hdr
outputs/probe read fixtures/lonely_img.img
outputs/probe names fixtures/lonely_img.img
outputs/probe read fixtures/nii_with_ni1_magic.nii
outputs/probe names fixtures/nii_with_ni1_magic.nii
outputs/probe read fixtures/hdr_with_np1_magic.hdr
outputs/probe make fixtures/scl_identity_slope_1_inter_0 1 1 4 3 2 3 1 0 0 0 0 1.0 0.0
outputs/probe read fixtures/scl_identity_slope_1_inter_0.nii
outputs/probe readhdr fixtures/scl_identity_slope_1_inter_0.nii 1
$NIFTI_PREFIX/bin/nifti_tool -quiet -disp_ci -1 -1 -1 -1 -1 -1 -1 -infiles fixtures/scl_identity_slope_1_inter_0.nii
outputs/probe make fixtures/scl_slope_2_inter_minus_3 1 1 4 3 2 3 1 0 0 0 0 2.0 -3.0
outputs/probe read fixtures/scl_slope_2_inter_minus_3.nii
outputs/probe readhdr fixtures/scl_slope_2_inter_minus_3.nii 1
$NIFTI_PREFIX/bin/nifti_tool -quiet -disp_ci -1 -1 -1 -1 -1 -1 -1 -infiles fixtures/scl_slope_2_inter_minus_3.nii
outputs/probe make fixtures/scl_slope_0_inter_7_nifti1 1 1 4 3 2 3 1 0 0 0 0 0.0 7.0
outputs/probe read fixtures/scl_slope_0_inter_7_nifti1.nii
outputs/probe readhdr fixtures/scl_slope_0_inter_7_nifti1.nii 1
$NIFTI_PREFIX/bin/nifti_tool -quiet -disp_ci -1 -1 -1 -1 -1 -1 -1 -infiles fixtures/scl_slope_0_inter_7_nifti1.nii
outputs/probe make fixtures/scl_slope_0_inter_7_nifti2 2 4 4 3 2 3 1 0 0 0 0 0.0 7.0
outputs/probe read fixtures/scl_slope_0_inter_7_nifti2.nii
outputs/probe readhdr fixtures/scl_slope_0_inter_7_nifti2.nii 1
$NIFTI_PREFIX/bin/nifti_tool -quiet -disp_ci -1 -1 -1 -1 -1 -1 -1 -infiles fixtures/scl_slope_0_inter_7_nifti2.nii
outputs/probe make fixtures/scl_negative_slope 1 1 4 3 2 3 1 0 0 0 0 -0.5 100.0
outputs/probe read fixtures/scl_negative_slope.nii
outputs/probe readhdr fixtures/scl_negative_slope.nii 1
$NIFTI_PREFIX/bin/nifti_tool -quiet -disp_ci -1 -1 -1 -1 -1 -1 -1 -infiles fixtures/scl_negative_slope.nii
outputs/probe make fixtures/scl_slope_2_inter_minus_3_nifti2 2 4 4 3 2 3 1 0 0 0 0 2.0 -3.0
outputs/probe read fixtures/scl_slope_2_inter_minus_3_nifti2.nii
outputs/probe readhdr fixtures/scl_slope_2_inter_minus_3_nifti2.nii 1
$NIFTI_PREFIX/bin/nifti_tool -quiet -disp_ci -1 -1 -1 -1 -1 -1 -1 -infiles fixtures/scl_slope_2_inter_minus_3_nifti2.nii
outputs/probe read fixtures/scl_slope_inf.nii
outputs/probe readhdr fixtures/scl_slope_inf.nii 1
outputs/probe read fixtures/scl_slope_nan.nii
outputs/probe readhdr fixtures/scl_slope_nan.nii 1
outputs/probe make fixtures/endian_nifti1_int16_le 1 1 4 3 2 3 1 0 0 0 0 1.0 0.0
outputs/probe swapfile fixtures/endian_nifti1_int16_le.nii fixtures/endian_nifti1_int16_be.nii 1 352 2
outputs/probe read fixtures/endian_nifti1_int16_le.nii
outputs/probe read fixtures/endian_nifti1_int16_be.nii
$NIFTI_PREFIX/bin/nifti_tool -quiet -disp_ci -1 -1 -1 -1 -1 -1 -1 -infiles fixtures/endian_nifti1_int16_le.nii
$NIFTI_PREFIX/bin/nifti_tool -quiet -disp_ci -1 -1 -1 -1 -1 -1 -1 -infiles fixtures/endian_nifti1_int16_be.nii
outputs/probe hdrver fixtures/endian_nifti1_int16_be.nii
outputs/probe make fixtures/endian_nifti1_float32_le 1 1 16 3 2 3 1 0 0 0 0 1.0 0.0
outputs/probe swapfile fixtures/endian_nifti1_float32_le.nii fixtures/endian_nifti1_float32_be.nii 1 352 4
outputs/probe read fixtures/endian_nifti1_float32_le.nii
outputs/probe read fixtures/endian_nifti1_float32_be.nii
$NIFTI_PREFIX/bin/nifti_tool -quiet -disp_ci -1 -1 -1 -1 -1 -1 -1 -infiles fixtures/endian_nifti1_float32_le.nii
$NIFTI_PREFIX/bin/nifti_tool -quiet -disp_ci -1 -1 -1 -1 -1 -1 -1 -infiles fixtures/endian_nifti1_float32_be.nii
outputs/probe hdrver fixtures/endian_nifti1_float32_be.nii
outputs/probe make fixtures/endian_nifti1_float64_le 1 1 64 3 2 3 1 0 0 0 0 1.0 0.0
outputs/probe swapfile fixtures/endian_nifti1_float64_le.nii fixtures/endian_nifti1_float64_be.nii 1 352 8
outputs/probe read fixtures/endian_nifti1_float64_le.nii
outputs/probe read fixtures/endian_nifti1_float64_be.nii
$NIFTI_PREFIX/bin/nifti_tool -quiet -disp_ci -1 -1 -1 -1 -1 -1 -1 -infiles fixtures/endian_nifti1_float64_le.nii
$NIFTI_PREFIX/bin/nifti_tool -quiet -disp_ci -1 -1 -1 -1 -1 -1 -1 -infiles fixtures/endian_nifti1_float64_be.nii
outputs/probe hdrver fixtures/endian_nifti1_float64_be.nii
outputs/probe make fixtures/endian_nifti2_int16_le 2 4 4 3 2 3 1 0 0 0 0 1.0 0.0
outputs/probe swapfile fixtures/endian_nifti2_int16_le.nii fixtures/endian_nifti2_int16_be.nii 2 544 2
outputs/probe read fixtures/endian_nifti2_int16_le.nii
outputs/probe read fixtures/endian_nifti2_int16_be.nii
$NIFTI_PREFIX/bin/nifti_tool -quiet -disp_ci -1 -1 -1 -1 -1 -1 -1 -infiles fixtures/endian_nifti2_int16_le.nii
$NIFTI_PREFIX/bin/nifti_tool -quiet -disp_ci -1 -1 -1 -1 -1 -1 -1 -infiles fixtures/endian_nifti2_int16_be.nii
outputs/probe hdrver fixtures/endian_nifti2_int16_be.nii
outputs/probe swap fixtures/endian_nifti1_int16_le.nii
outputs/probe make fixtures/dim_rank1_6 1 1 2 1 6 0 0 0 0 0 0 1.0 0.0
outputs/probe read fixtures/dim_rank1_6.nii
outputs/probe readhdr fixtures/dim_rank1_6.nii 1
outputs/probe make fixtures/dim_rank2_2x3 1 1 2 2 2 3 0 0 0 0 0 1.0 0.0
outputs/probe read fixtures/dim_rank2_2x3.nii
outputs/probe readhdr fixtures/dim_rank2_2x3.nii 1
outputs/probe make fixtures/dim_rank3_2x3x2 1 1 2 3 2 3 2 0 0 0 0 1.0 0.0
outputs/probe read fixtures/dim_rank3_2x3x2.nii
outputs/probe readhdr fixtures/dim_rank3_2x3x2.nii 1
outputs/probe make fixtures/dim_rank4_2x3x2x2 1 1 2 4 2 3 2 2 0 0 0 1.0 0.0
outputs/probe read fixtures/dim_rank4_2x3x2x2.nii
outputs/probe readhdr fixtures/dim_rank4_2x3x2x2.nii 1
outputs/probe make fixtures/dim_rank5_2x2x2x2x2 1 1 2 5 2 2 2 2 2 0 0 1.0 0.0
outputs/probe read fixtures/dim_rank5_2x2x2x2x2.nii
outputs/probe readhdr fixtures/dim_rank5_2x2x2x2x2.nii 1
outputs/probe make fixtures/dim_rank7_all2 1 1 2 7 2 2 2 2 2 2 2 1.0 0.0
outputs/probe read fixtures/dim_rank7_all2.nii
outputs/probe readhdr fixtures/dim_rank7_all2.nii 1
$NIFTI_PREFIX/bin/nifti_tool -quiet -disp_ci -1 -1 -1 -1 -1 -1 -1 -infiles fixtures/dim_rank4_2x3x2x2.nii
$NIFTI_PREFIX/bin/nifti_tool -quiet -disp_ci -1 -1 -1 0 -1 -1 -1 -infiles fixtures/dim_rank4_2x3x2x2.nii
$NIFTI_PREFIX/bin/nifti_tool -quiet -disp_ci -1 -1 -1 1 -1 -1 -1 -infiles fixtures/dim_rank4_2x3x2x2.nii
$NIFTI_PREFIX/bin/nifti_tool -quiet -disp_ci -1 -1 0 0 -1 -1 -1 -infiles fixtures/dim_rank4_2x3x2x2.nii
$NIFTI_PREFIX/bin/nifti_tool -quiet -disp_ci -1 -1 1 1 -1 -1 -1 -infiles fixtures/dim_rank4_2x3x2x2.nii
$NIFTI_PREFIX/bin/nifti_tool -quiet -disp_ts 1 2 1 -infiles fixtures/dim_rank4_2x3x2x2.nii
$NIFTI_PREFIX/bin/nifti_tool -quiet -disp_ci 1 2 1 1 -1 -1 -1 -infiles fixtures/dim_rank4_2x3x2x2.nii
$NIFTI_PREFIX/bin/nifti_tool -quiet -disp_ci 9 0 0 0 -1 -1 -1 -infiles fixtures/dim_rank4_2x3x2x2.nii
$NIFTI_PREFIX/bin/nifti_tool -quiet -cci -1 -1 0 0 -1 -1 -1 -prefix outputs/plane_z0_t0.nii -infiles fixtures/dim_rank4_2x3x2x2.nii
outputs/probe read outputs/plane_z0_t0.nii
outputs/probe readhdr fixtures/dimedge_dim0_zero.nii 1
outputs/probe readhdr fixtures/dimedge_dim0_zero.nii 0
outputs/probe read fixtures/dimedge_dim0_zero.nii
$NIFTI_PREFIX/bin/nifti_tool -quiet -disp_hdr -infiles fixtures/dimedge_dim0_zero.nii
outputs/probe readhdr fixtures/dimedge_dim0_eight.nii 1
outputs/probe readhdr fixtures/dimedge_dim0_eight.nii 0
outputs/probe read fixtures/dimedge_dim0_eight.nii
$NIFTI_PREFIX/bin/nifti_tool -quiet -disp_hdr -infiles fixtures/dimedge_dim0_eight.nii
outputs/probe readhdr fixtures/dimedge_dim0_negative.nii 1
outputs/probe readhdr fixtures/dimedge_dim0_negative.nii 0
outputs/probe read fixtures/dimedge_dim0_negative.nii
$NIFTI_PREFIX/bin/nifti_tool -quiet -disp_hdr -infiles fixtures/dimedge_dim0_negative.nii
outputs/probe readhdr fixtures/dimedge_dim1_zero.nii 1
outputs/probe readhdr fixtures/dimedge_dim1_zero.nii 0
outputs/probe read fixtures/dimedge_dim1_zero.nii
$NIFTI_PREFIX/bin/nifti_tool -quiet -disp_hdr -infiles fixtures/dimedge_dim1_zero.nii
outputs/probe readhdr fixtures/dimedge_dim1_negative.nii 1
outputs/probe readhdr fixtures/dimedge_dim1_negative.nii 0
outputs/probe read fixtures/dimedge_dim1_negative.nii
$NIFTI_PREFIX/bin/nifti_tool -quiet -disp_hdr -infiles fixtures/dimedge_dim1_negative.nii
outputs/probe readhdr fixtures/dimedge_dim2_zero_mid_array.nii 1
outputs/probe readhdr fixtures/dimedge_dim2_zero_mid_array.nii 0
outputs/probe read fixtures/dimedge_dim2_zero_mid_array.nii
$NIFTI_PREFIX/bin/nifti_tool -quiet -disp_hdr -infiles fixtures/dimedge_dim2_zero_mid_array.nii
outputs/probe readhdr fixtures/dimedge_dim_all_32767.nii 1
outputs/probe readhdr fixtures/dimedge_dim_all_32767.nii 0
outputs/probe read fixtures/dimedge_dim_all_32767.nii
$NIFTI_PREFIX/bin/nifti_tool -quiet -disp_hdr -infiles fixtures/dimedge_dim_all_32767.nii
outputs/probe read fixtures/pixdim_pixdim1_zero.nii
outputs/probe read fixtures/pixdim_pixdim2_inf.nii
outputs/probe read fixtures/pixdim_pixdim3_nan.nii
outputs/probe read fixtures/pixdim_pixdim1_negative.nii
outputs/probe read fixtures/pixdim_qform_code_zero.nii
outputs/probe read fixtures/pixdim_qfac_minus_one.nii
outputs/probe read fixtures/pixdim_qfac_zero.nii
outputs/probe read fixtures/pixdim_qfac_plus_one.nii
outputs/probe hdrver fixtures/bad_empty.nii
outputs/probe readhdr fixtures/bad_empty.nii 1
outputs/probe read fixtures/bad_empty.nii
$NIFTI_PREFIX/bin/nifti_tool -quiet -disp_hdr -infiles fixtures/bad_empty.nii
outputs/probe hdrver fixtures/bad_onebyte.nii
outputs/probe readhdr fixtures/bad_onebyte.nii 1
outputs/probe read fixtures/bad_onebyte.nii
$NIFTI_PREFIX/bin/nifti_tool -quiet -disp_hdr -infiles fixtures/bad_onebyte.nii
outputs/probe hdrver fixtures/bad_trunc100.nii
outputs/probe readhdr fixtures/bad_trunc100.nii 1
outputs/probe read fixtures/bad_trunc100.nii
$NIFTI_PREFIX/bin/nifti_tool -quiet -disp_hdr -infiles fixtures/bad_trunc100.nii
outputs/probe hdrver fixtures/bad_trunc347.nii
outputs/probe readhdr fixtures/bad_trunc347.nii 1
outputs/probe read fixtures/bad_trunc347.nii
$NIFTI_PREFIX/bin/nifti_tool -quiet -disp_hdr -infiles fixtures/bad_trunc347.nii
outputs/probe hdrver fixtures/bad_trunc348.nii
outputs/probe readhdr fixtures/bad_trunc348.nii 1
outputs/probe read fixtures/bad_trunc348.nii
$NIFTI_PREFIX/bin/nifti_tool -quiet -disp_hdr -infiles fixtures/bad_trunc348.nii
outputs/probe hdrver fixtures/bad_nodata.nii
outputs/probe readhdr fixtures/bad_nodata.nii 1
outputs/probe read fixtures/bad_nodata.nii
$NIFTI_PREFIX/bin/nifti_tool -quiet -disp_hdr -infiles fixtures/bad_nodata.nii
outputs/probe hdrver fixtures/bad_halfdata.nii
outputs/probe readhdr fixtures/bad_halfdata.nii 1
outputs/probe read fixtures/bad_halfdata.nii
$NIFTI_PREFIX/bin/nifti_tool -quiet -disp_hdr -infiles fixtures/bad_halfdata.nii
outputs/probe hdrver fixtures/bad_sizeof0.nii
outputs/probe readhdr fixtures/bad_sizeof0.nii 1
outputs/probe read fixtures/bad_sizeof0.nii
$NIFTI_PREFIX/bin/nifti_tool -quiet -disp_hdr -infiles fixtures/bad_sizeof0.nii
outputs/probe hdrver fixtures/bad_sizeof349.nii
outputs/probe readhdr fixtures/bad_sizeof349.nii 1
outputs/probe read fixtures/bad_sizeof349.nii
$NIFTI_PREFIX/bin/nifti_tool -quiet -disp_hdr -infiles fixtures/bad_sizeof349.nii
outputs/probe hdrver fixtures/bad_sizeof540.nii
outputs/probe readhdr fixtures/bad_sizeof540.nii 1
outputs/probe read fixtures/bad_sizeof540.nii
$NIFTI_PREFIX/bin/nifti_tool -quiet -disp_hdr -infiles fixtures/bad_sizeof540.nii
outputs/probe hdrver fixtures/bad_sizeof_swapped.nii
outputs/probe readhdr fixtures/bad_sizeof_swapped.nii 1
outputs/probe read fixtures/bad_sizeof_swapped.nii
$NIFTI_PREFIX/bin/nifti_tool -quiet -disp_hdr -infiles fixtures/bad_sizeof_swapped.nii
outputs/probe hdrver fixtures/bad_magic.nii
outputs/probe readhdr fixtures/bad_magic.nii 1
outputs/probe read fixtures/bad_magic.nii
$NIFTI_PREFIX/bin/nifti_tool -quiet -disp_hdr -infiles fixtures/bad_magic.nii
outputs/probe hdrver fixtures/bad_magic_nonul.nii
outputs/probe readhdr fixtures/bad_magic_nonul.nii 1
outputs/probe read fixtures/bad_magic_nonul.nii
$NIFTI_PREFIX/bin/nifti_tool -quiet -disp_hdr -infiles fixtures/bad_magic_nonul.nii
outputs/probe hdrver fixtures/bad_magic_n2_in_n1.nii
outputs/probe readhdr fixtures/bad_magic_n2_in_n1.nii 1
outputs/probe read fixtures/bad_magic_n2_in_n1.nii
$NIFTI_PREFIX/bin/nifti_tool -quiet -disp_hdr -infiles fixtures/bad_magic_n2_in_n1.nii
outputs/probe hdrver fixtures/bad_dt3.nii
outputs/probe readhdr fixtures/bad_dt3.nii 1
outputs/probe read fixtures/bad_dt3.nii
$NIFTI_PREFIX/bin/nifti_tool -quiet -disp_hdr -infiles fixtures/bad_dt3.nii
outputs/probe hdrver fixtures/bad_dt9999.nii
outputs/probe readhdr fixtures/bad_dt9999.nii 1
outputs/probe read fixtures/bad_dt9999.nii
$NIFTI_PREFIX/bin/nifti_tool -quiet -disp_hdr -infiles fixtures/bad_dt9999.nii
outputs/probe hdrver fixtures/bad_dt1.nii
outputs/probe readhdr fixtures/bad_dt1.nii 1
outputs/probe read fixtures/bad_dt1.nii
$NIFTI_PREFIX/bin/nifti_tool -quiet -disp_hdr -infiles fixtures/bad_dt1.nii
outputs/probe hdrver fixtures/bad_bitpix.nii
outputs/probe readhdr fixtures/bad_bitpix.nii 1
outputs/probe read fixtures/bad_bitpix.nii
$NIFTI_PREFIX/bin/nifti_tool -quiet -disp_hdr -infiles fixtures/bad_bitpix.nii
outputs/probe hdrver fixtures/bad_voxoff_neg.nii
outputs/probe readhdr fixtures/bad_voxoff_neg.nii 1
outputs/probe read fixtures/bad_voxoff_neg.nii
$NIFTI_PREFIX/bin/nifti_tool -quiet -disp_hdr -infiles fixtures/bad_voxoff_neg.nii
outputs/probe hdrver fixtures/bad_voxoff_small.nii
outputs/probe readhdr fixtures/bad_voxoff_small.nii 1
outputs/probe read fixtures/bad_voxoff_small.nii
$NIFTI_PREFIX/bin/nifti_tool -quiet -disp_hdr -infiles fixtures/bad_voxoff_small.nii
outputs/probe hdrver fixtures/bad_voxoff_eof.nii
outputs/probe readhdr fixtures/bad_voxoff_eof.nii 1
outputs/probe read fixtures/bad_voxoff_eof.nii
$NIFTI_PREFIX/bin/nifti_tool -quiet -disp_hdr -infiles fixtures/bad_voxoff_eof.nii
outputs/probe hdrver fixtures/bad_voxoff_frac.nii
outputs/probe readhdr fixtures/bad_voxoff_frac.nii 1
outputs/probe read fixtures/bad_voxoff_frac.nii
$NIFTI_PREFIX/bin/nifti_tool -quiet -disp_hdr -infiles fixtures/bad_voxoff_frac.nii
outputs/probe make fixtures/float_float32 1 1 16 3 4 2 1 0 0 0 0 1.0 0.0
outputs/probe read fixtures/float_float32.nii
$NIFTI_PREFIX/bin/nifti_tool -quiet -disp_ci -1 -1 -1 -1 -1 -1 -1 -infiles fixtures/float_float32.nii
$NIFTI_PREFIX/bin/nifti_tool -quiet -disp_ci 0 0 0 -1 -1 -1 -1 -infiles fixtures/float_float32.nii
$NIFTI_PREFIX/bin/nifti_tool -quiet -disp_ci 0 1 0 -1 -1 -1 -1 -infiles fixtures/float_float32.nii
$NIFTI_PREFIX/bin/nifti_tool -quiet -disp_ci 1 0 0 -1 -1 -1 -1 -infiles fixtures/float_float32.nii
$NIFTI_PREFIX/bin/nifti_tool -quiet -disp_ci 1 1 0 -1 -1 -1 -1 -infiles fixtures/float_float32.nii
$NIFTI_PREFIX/bin/nifti_tool -quiet -disp_ci 2 0 0 -1 -1 -1 -1 -infiles fixtures/float_float32.nii
$NIFTI_PREFIX/bin/nifti_tool -quiet -disp_ci 2 1 0 -1 -1 -1 -1 -infiles fixtures/float_float32.nii
$NIFTI_PREFIX/bin/nifti_tool -quiet -disp_ci 3 0 0 -1 -1 -1 -1 -infiles fixtures/float_float32.nii
$NIFTI_PREFIX/bin/nifti_tool -quiet -disp_ci 3 1 0 -1 -1 -1 -1 -infiles fixtures/float_float32.nii
outputs/probe swapfile fixtures/float_float32.nii fixtures/float_float32_be.nii 1 352 4
outputs/probe read fixtures/float_float32_be.nii
$NIFTI_PREFIX/bin/nifti_tool -quiet -disp_ci -1 -1 -1 -1 -1 -1 -1 -infiles fixtures/float_float32_be.nii
outputs/probe make fixtures/float_float64 1 1 64 3 4 2 1 0 0 0 0 1.0 0.0
outputs/probe read fixtures/float_float64.nii
$NIFTI_PREFIX/bin/nifti_tool -quiet -disp_ci -1 -1 -1 -1 -1 -1 -1 -infiles fixtures/float_float64.nii
$NIFTI_PREFIX/bin/nifti_tool -quiet -disp_ci 0 0 0 -1 -1 -1 -1 -infiles fixtures/float_float64.nii
$NIFTI_PREFIX/bin/nifti_tool -quiet -disp_ci 0 1 0 -1 -1 -1 -1 -infiles fixtures/float_float64.nii
$NIFTI_PREFIX/bin/nifti_tool -quiet -disp_ci 1 0 0 -1 -1 -1 -1 -infiles fixtures/float_float64.nii
$NIFTI_PREFIX/bin/nifti_tool -quiet -disp_ci 1 1 0 -1 -1 -1 -1 -infiles fixtures/float_float64.nii
$NIFTI_PREFIX/bin/nifti_tool -quiet -disp_ci 2 0 0 -1 -1 -1 -1 -infiles fixtures/float_float64.nii
$NIFTI_PREFIX/bin/nifti_tool -quiet -disp_ci 2 1 0 -1 -1 -1 -1 -infiles fixtures/float_float64.nii
$NIFTI_PREFIX/bin/nifti_tool -quiet -disp_ci 3 0 0 -1 -1 -1 -1 -infiles fixtures/float_float64.nii
$NIFTI_PREFIX/bin/nifti_tool -quiet -disp_ci 3 1 0 -1 -1 -1 -1 -infiles fixtures/float_float64.nii
outputs/probe swapfile fixtures/float_float64.nii fixtures/float_float64_be.nii 1 352 8
outputs/probe read fixtures/float_float64_be.nii
$NIFTI_PREFIX/bin/nifti_tool -quiet -disp_ci -1 -1 -1 -1 -1 -1 -1 -infiles fixtures/float_float64_be.nii
outputs/probe read fixtures/float_float32.nii 0 --debug
outputs/probe make fixtures/complex64 1 1 32 3 4 1 1 0 0 0 0 1.0 0.0
outputs/probe read fixtures/complex64.nii
$NIFTI_PREFIX/bin/nifti_tool -quiet -disp_ci -1 -1 -1 -1 -1 -1 -1 -infiles fixtures/complex64.nii
$NIFTI_PREFIX/bin/nifti_tool -quiet -disp_ci 0 0 0 -1 -1 -1 -1 -infiles fixtures/dt2_uint8.nii
$NIFTI_PREFIX/bin/nifti_tool -quiet -disp_ci 0 1 0 -1 -1 -1 -1 -infiles fixtures/dt2_uint8.nii
$NIFTI_PREFIX/bin/nifti_tool -quiet -disp_ci 0 2 0 -1 -1 -1 -1 -infiles fixtures/dt2_uint8.nii
$NIFTI_PREFIX/bin/nifti_tool -quiet -disp_ci 1 0 0 -1 -1 -1 -1 -infiles fixtures/dt2_uint8.nii
$NIFTI_PREFIX/bin/nifti_tool -quiet -disp_ci 1 1 0 -1 -1 -1 -1 -infiles fixtures/dt2_uint8.nii
$NIFTI_PREFIX/bin/nifti_tool -quiet -disp_ci 1 2 0 -1 -1 -1 -1 -infiles fixtures/dt2_uint8.nii
$NIFTI_PREFIX/bin/nifti_tool -quiet -disp_ci 0 0 0 -1 -1 -1 -1 -infiles fixtures/dt4_int16.nii
$NIFTI_PREFIX/bin/nifti_tool -quiet -disp_ci 0 1 0 -1 -1 -1 -1 -infiles fixtures/dt4_int16.nii
$NIFTI_PREFIX/bin/nifti_tool -quiet -disp_ci 0 2 0 -1 -1 -1 -1 -infiles fixtures/dt4_int16.nii
$NIFTI_PREFIX/bin/nifti_tool -quiet -disp_ci 1 0 0 -1 -1 -1 -1 -infiles fixtures/dt4_int16.nii
$NIFTI_PREFIX/bin/nifti_tool -quiet -disp_ci 1 1 0 -1 -1 -1 -1 -infiles fixtures/dt4_int16.nii
$NIFTI_PREFIX/bin/nifti_tool -quiet -disp_ci 1 2 0 -1 -1 -1 -1 -infiles fixtures/dt4_int16.nii
$NIFTI_PREFIX/bin/nifti_tool -quiet -disp_ci 0 0 0 -1 -1 -1 -1 -infiles fixtures/dt8_int32.nii
$NIFTI_PREFIX/bin/nifti_tool -quiet -disp_ci 0 1 0 -1 -1 -1 -1 -infiles fixtures/dt8_int32.nii
$NIFTI_PREFIX/bin/nifti_tool -quiet -disp_ci 0 2 0 -1 -1 -1 -1 -infiles fixtures/dt8_int32.nii
$NIFTI_PREFIX/bin/nifti_tool -quiet -disp_ci 1 0 0 -1 -1 -1 -1 -infiles fixtures/dt8_int32.nii
$NIFTI_PREFIX/bin/nifti_tool -quiet -disp_ci 1 1 0 -1 -1 -1 -1 -infiles fixtures/dt8_int32.nii
$NIFTI_PREFIX/bin/nifti_tool -quiet -disp_ci 1 2 0 -1 -1 -1 -1 -infiles fixtures/dt8_int32.nii
$NIFTI_PREFIX/bin/nifti_tool -quiet -disp_ci 0 0 0 -1 -1 -1 -1 -infiles fixtures/dt16_float32.nii
$NIFTI_PREFIX/bin/nifti_tool -quiet -disp_ci 0 1 0 -1 -1 -1 -1 -infiles fixtures/dt16_float32.nii
$NIFTI_PREFIX/bin/nifti_tool -quiet -disp_ci 0 2 0 -1 -1 -1 -1 -infiles fixtures/dt16_float32.nii
$NIFTI_PREFIX/bin/nifti_tool -quiet -disp_ci 1 0 0 -1 -1 -1 -1 -infiles fixtures/dt16_float32.nii
$NIFTI_PREFIX/bin/nifti_tool -quiet -disp_ci 1 1 0 -1 -1 -1 -1 -infiles fixtures/dt16_float32.nii
$NIFTI_PREFIX/bin/nifti_tool -quiet -disp_ci 1 2 0 -1 -1 -1 -1 -infiles fixtures/dt16_float32.nii
$NIFTI_PREFIX/bin/nifti_tool -quiet -disp_ci 0 0 0 -1 -1 -1 -1 -infiles fixtures/dt64_float64.nii
$NIFTI_PREFIX/bin/nifti_tool -quiet -disp_ci 0 1 0 -1 -1 -1 -1 -infiles fixtures/dt64_float64.nii
$NIFTI_PREFIX/bin/nifti_tool -quiet -disp_ci 0 2 0 -1 -1 -1 -1 -infiles fixtures/dt64_float64.nii
$NIFTI_PREFIX/bin/nifti_tool -quiet -disp_ci 1 0 0 -1 -1 -1 -1 -infiles fixtures/dt64_float64.nii
$NIFTI_PREFIX/bin/nifti_tool -quiet -disp_ci 1 1 0 -1 -1 -1 -1 -infiles fixtures/dt64_float64.nii
$NIFTI_PREFIX/bin/nifti_tool -quiet -disp_ci 1 2 0 -1 -1 -1 -1 -infiles fixtures/dt64_float64.nii
$NIFTI_PREFIX/bin/nifti_tool -quiet -disp_ci 0 0 0 -1 -1 -1 -1 -infiles fixtures/dt256_int8.nii
$NIFTI_PREFIX/bin/nifti_tool -quiet -disp_ci 0 1 0 -1 -1 -1 -1 -infiles fixtures/dt256_int8.nii
$NIFTI_PREFIX/bin/nifti_tool -quiet -disp_ci 0 2 0 -1 -1 -1 -1 -infiles fixtures/dt256_int8.nii
$NIFTI_PREFIX/bin/nifti_tool -quiet -disp_ci 1 0 0 -1 -1 -1 -1 -infiles fixtures/dt256_int8.nii
$NIFTI_PREFIX/bin/nifti_tool -quiet -disp_ci 1 1 0 -1 -1 -1 -1 -infiles fixtures/dt256_int8.nii
$NIFTI_PREFIX/bin/nifti_tool -quiet -disp_ci 1 2 0 -1 -1 -1 -1 -infiles fixtures/dt256_int8.nii
$NIFTI_PREFIX/bin/nifti_tool -quiet -disp_ci 0 0 0 -1 -1 -1 -1 -infiles fixtures/dt512_uint16.nii
$NIFTI_PREFIX/bin/nifti_tool -quiet -disp_ci 0 1 0 -1 -1 -1 -1 -infiles fixtures/dt512_uint16.nii
$NIFTI_PREFIX/bin/nifti_tool -quiet -disp_ci 0 2 0 -1 -1 -1 -1 -infiles fixtures/dt512_uint16.nii
$NIFTI_PREFIX/bin/nifti_tool -quiet -disp_ci 1 0 0 -1 -1 -1 -1 -infiles fixtures/dt512_uint16.nii
$NIFTI_PREFIX/bin/nifti_tool -quiet -disp_ci 1 1 0 -1 -1 -1 -1 -infiles fixtures/dt512_uint16.nii
$NIFTI_PREFIX/bin/nifti_tool -quiet -disp_ci 1 2 0 -1 -1 -1 -1 -infiles fixtures/dt512_uint16.nii
$NIFTI_PREFIX/bin/nifti_tool -quiet -disp_ci 0 0 0 -1 -1 -1 -1 -infiles fixtures/dt768_uint32.nii
$NIFTI_PREFIX/bin/nifti_tool -quiet -disp_ci 0 1 0 -1 -1 -1 -1 -infiles fixtures/dt768_uint32.nii
$NIFTI_PREFIX/bin/nifti_tool -quiet -disp_ci 0 2 0 -1 -1 -1 -1 -infiles fixtures/dt768_uint32.nii
$NIFTI_PREFIX/bin/nifti_tool -quiet -disp_ci 1 0 0 -1 -1 -1 -1 -infiles fixtures/dt768_uint32.nii
$NIFTI_PREFIX/bin/nifti_tool -quiet -disp_ci 1 1 0 -1 -1 -1 -1 -infiles fixtures/dt768_uint32.nii
$NIFTI_PREFIX/bin/nifti_tool -quiet -disp_ci 1 2 0 -1 -1 -1 -1 -infiles fixtures/dt768_uint32.nii
$NIFTI_PREFIX/bin/nifti_tool -quiet -disp_ci 0 0 0 -1 -1 -1 -1 -infiles fixtures/dt1024_int64.nii
$NIFTI_PREFIX/bin/nifti_tool -quiet -disp_ci 0 1 0 -1 -1 -1 -1 -infiles fixtures/dt1024_int64.nii
$NIFTI_PREFIX/bin/nifti_tool -quiet -disp_ci 0 2 0 -1 -1 -1 -1 -infiles fixtures/dt1024_int64.nii
$NIFTI_PREFIX/bin/nifti_tool -quiet -disp_ci 1 0 0 -1 -1 -1 -1 -infiles fixtures/dt1024_int64.nii
$NIFTI_PREFIX/bin/nifti_tool -quiet -disp_ci 1 1 0 -1 -1 -1 -1 -infiles fixtures/dt1024_int64.nii
$NIFTI_PREFIX/bin/nifti_tool -quiet -disp_ci 1 2 0 -1 -1 -1 -1 -infiles fixtures/dt1024_int64.nii
$NIFTI_PREFIX/bin/nifti_tool -quiet -disp_ci 0 0 0 -1 -1 -1 -1 -infiles fixtures/dt1280_uint64.nii
$NIFTI_PREFIX/bin/nifti_tool -quiet -disp_ci 0 1 0 -1 -1 -1 -1 -infiles fixtures/dt1280_uint64.nii
$NIFTI_PREFIX/bin/nifti_tool -quiet -disp_ci 0 2 0 -1 -1 -1 -1 -infiles fixtures/dt1280_uint64.nii
$NIFTI_PREFIX/bin/nifti_tool -quiet -disp_ci 1 0 0 -1 -1 -1 -1 -infiles fixtures/dt1280_uint64.nii
$NIFTI_PREFIX/bin/nifti_tool -quiet -disp_ci 1 1 0 -1 -1 -1 -1 -infiles fixtures/dt1280_uint64.nii
$NIFTI_PREFIX/bin/nifti_tool -quiet -disp_ci 1 2 0 -1 -1 -1 -1 -infiles fixtures/dt1280_uint64.nii
$NIFTI_PREFIX/bin/nifti_tool -quiet -disp_ci 0 0 0 -1 -1 -1 -1 -infiles fixtures/dt128_rgb24.nii
$NIFTI_PREFIX/bin/nifti_tool -quiet -disp_ci 0 1 0 -1 -1 -1 -1 -infiles fixtures/dt128_rgb24.nii
$NIFTI_PREFIX/bin/nifti_tool -quiet -disp_ci 0 2 0 -1 -1 -1 -1 -infiles fixtures/dt128_rgb24.nii
$NIFTI_PREFIX/bin/nifti_tool -quiet -disp_ci 1 0 0 -1 -1 -1 -1 -infiles fixtures/dt128_rgb24.nii
$NIFTI_PREFIX/bin/nifti_tool -quiet -disp_ci 1 1 0 -1 -1 -1 -1 -infiles fixtures/dt128_rgb24.nii
$NIFTI_PREFIX/bin/nifti_tool -quiet -disp_ci 1 2 0 -1 -1 -1 -1 -infiles fixtures/dt128_rgb24.nii
$NIFTI_PREFIX/bin/nifti_tool -quiet -disp_ci 0 0 0 -1 -1 -1 -1 -infiles fixtures/dt2304_rgba32.nii
$NIFTI_PREFIX/bin/nifti_tool -quiet -disp_ci 0 1 0 -1 -1 -1 -1 -infiles fixtures/dt2304_rgba32.nii
$NIFTI_PREFIX/bin/nifti_tool -quiet -disp_ci 0 2 0 -1 -1 -1 -1 -infiles fixtures/dt2304_rgba32.nii
$NIFTI_PREFIX/bin/nifti_tool -quiet -disp_ci 1 0 0 -1 -1 -1 -1 -infiles fixtures/dt2304_rgba32.nii
$NIFTI_PREFIX/bin/nifti_tool -quiet -disp_ci 1 1 0 -1 -1 -1 -1 -infiles fixtures/dt2304_rgba32.nii
$NIFTI_PREFIX/bin/nifti_tool -quiet -disp_ci 1 2 0 -1 -1 -1 -1 -infiles fixtures/dt2304_rgba32.nii
$NIFTI_PREFIX/bin/nifti_tool -quiet -disp_ci 0 0 0 -1 -1 -1 -1 -infiles fixtures/dt32_complex64.nii
$NIFTI_PREFIX/bin/nifti_tool -quiet -disp_ci 0 1 0 -1 -1 -1 -1 -infiles fixtures/dt32_complex64.nii
$NIFTI_PREFIX/bin/nifti_tool -quiet -disp_ci 0 2 0 -1 -1 -1 -1 -infiles fixtures/dt32_complex64.nii
$NIFTI_PREFIX/bin/nifti_tool -quiet -disp_ci 1 0 0 -1 -1 -1 -1 -infiles fixtures/dt32_complex64.nii
$NIFTI_PREFIX/bin/nifti_tool -quiet -disp_ci 1 1 0 -1 -1 -1 -1 -infiles fixtures/dt32_complex64.nii
$NIFTI_PREFIX/bin/nifti_tool -quiet -disp_ci 1 2 0 -1 -1 -1 -1 -infiles fixtures/dt32_complex64.nii
cmake --version
uname -srm
sw_vers -productVersion
outputs/probe hdrver fixtures/n2_magic_tail_mangled.nii
outputs/probe read fixtures/n2_magic_tail_mangled.nii
outputs/probe hdrver fixtures/n2_magic_tail_partial.nii
outputs/probe read fixtures/n2_magic_tail_partial.nii
outputs/probe hdrver fixtures/magic_zero_analyze.nii
outputs/probe read fixtures/magic_zero_analyze.nii
outputs/probe read fixtures/magic_zero_analyze.nii
outputs/probe read fixtures/magic_zero_pair.hdr
uname -srm
