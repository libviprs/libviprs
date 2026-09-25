#!/bin/sh
# Every command capture.py ran, in order. Regenerate with
# `python3 capture.py` from this directory.
#
# VIPS_NOVECTOR is UNSET rather than blanked: libvips tests
# whether the variable exists, so an empty value still counts
# as set. Nothing in the heif path is vectorised, but the
# habit costs nothing.
set -e

VIPS="env -u VIPS_NOVECTOR /opt/homebrew/bin/vips"
VIPSHEADER="env -u VIPS_NOVECTOR /opt/homebrew/bin/vipsheader"

$VIPS --vips-config
$VIPS -l
$VIPS heifsave
$VIPS heifload
# fixtures/rgb16_src.raw written by this script (4x3x3 ushort, deterministic ramp)
$VIPS rawload fixtures/rgb16_src.raw outputs/rgb16_src.v 4 3 3 --format ushort
$VIPS copy outputs/rgb16_src.v outputs/rgb16_src-rgb16.v --interpretation rgb16
$VIPS heifsave outputs/rgb16_src-rgb16.v fixtures/rgb8_narrowed.avif --bitdepth 8 --lossless --keep none
$VIPSHEADER -a fixtures/rgb8_narrowed.avif
$VIPS getpoint fixtures/rgb8_narrowed.avif 0 0
$VIPS getpoint fixtures/rgb8_narrowed.avif 1 0
$VIPS getpoint fixtures/rgb8_narrowed.avif 2 0
$VIPS getpoint fixtures/rgb8_narrowed.avif 3 0
$VIPS getpoint fixtures/rgb8_narrowed.avif 0 1
$VIPS getpoint fixtures/rgb8_narrowed.avif 1 1
$VIPS getpoint fixtures/rgb8_narrowed.avif 2 1
$VIPS getpoint fixtures/rgb8_narrowed.avif 3 1
$VIPS getpoint fixtures/rgb8_narrowed.avif 0 2
$VIPS getpoint fixtures/rgb8_narrowed.avif 1 2
$VIPS getpoint fixtures/rgb8_narrowed.avif 2 2
$VIPS getpoint fixtures/rgb8_narrowed.avif 3 2
$VIPSHEADER -f heif-bitdepth fixtures/rgb8_narrowed.avif
$VIPS heifsave outputs/rgb16_src-rgb16.v fixtures/rgb10.avif --bitdepth 10 --lossless --keep none
$VIPSHEADER -a fixtures/rgb10.avif
$VIPS getpoint fixtures/rgb10.avif 0 0
$VIPS getpoint fixtures/rgb10.avif 1 0
$VIPS getpoint fixtures/rgb10.avif 2 0
$VIPS getpoint fixtures/rgb10.avif 3 0
$VIPS getpoint fixtures/rgb10.avif 0 1
$VIPS getpoint fixtures/rgb10.avif 1 1
$VIPS getpoint fixtures/rgb10.avif 2 1
$VIPS getpoint fixtures/rgb10.avif 3 1
$VIPS getpoint fixtures/rgb10.avif 0 2
$VIPS getpoint fixtures/rgb10.avif 1 2
$VIPS getpoint fixtures/rgb10.avif 2 2
$VIPS getpoint fixtures/rgb10.avif 3 2
$VIPSHEADER -f heif-bitdepth fixtures/rgb10.avif
$VIPS heifsave outputs/rgb16_src-rgb16.v fixtures/rgb12.avif --bitdepth 12 --lossless --keep none
$VIPSHEADER -a fixtures/rgb12.avif
$VIPS getpoint fixtures/rgb12.avif 0 0
$VIPS getpoint fixtures/rgb12.avif 1 0
$VIPS getpoint fixtures/rgb12.avif 2 0
$VIPS getpoint fixtures/rgb12.avif 3 0
$VIPS getpoint fixtures/rgb12.avif 0 1
$VIPS getpoint fixtures/rgb12.avif 1 1
$VIPS getpoint fixtures/rgb12.avif 2 1
$VIPS getpoint fixtures/rgb12.avif 3 1
$VIPS getpoint fixtures/rgb12.avif 0 2
$VIPS getpoint fixtures/rgb12.avif 1 2
$VIPS getpoint fixtures/rgb12.avif 2 2
$VIPS getpoint fixtures/rgb12.avif 3 2
$VIPSHEADER -f heif-bitdepth fixtures/rgb12.avif
# fixtures/rgb8_src.raw written by this script (4x3x3 uchar, deterministic ramp)
$VIPS rawload fixtures/rgb8_src.raw outputs/rgb8_src.v 4 3 3 --format uchar
$VIPS copy outputs/rgb8_src.v outputs/rgb8_src-srgb.v --interpretation srgb
$VIPS heifsave outputs/rgb8_src-srgb.v fixtures/rgb8.avif --bitdepth 8 --lossless --keep none
$VIPSHEADER -a fixtures/rgb8.avif
$VIPS getpoint fixtures/rgb8.avif 0 0
$VIPS getpoint fixtures/rgb8.avif 1 0
$VIPS getpoint fixtures/rgb8.avif 2 0
$VIPS getpoint fixtures/rgb8.avif 3 0
$VIPS getpoint fixtures/rgb8.avif 0 1
$VIPS getpoint fixtures/rgb8.avif 1 1
$VIPS getpoint fixtures/rgb8.avif 2 1
$VIPS getpoint fixtures/rgb8.avif 3 1
$VIPS getpoint fixtures/rgb8.avif 0 2
$VIPS getpoint fixtures/rgb8.avif 1 2
$VIPS getpoint fixtures/rgb8.avif 2 2
$VIPS getpoint fixtures/rgb8.avif 3 2
# fixtures/rgba8_src.raw written by this script (4x3x4 uchar, deterministic ramp)
$VIPS rawload fixtures/rgba8_src.raw outputs/rgba8_src.v 4 3 4 --format uchar
$VIPS copy outputs/rgba8_src.v outputs/rgba8_src-srgb.v --interpretation srgb
$VIPS heifsave outputs/rgba8_src-srgb.v fixtures/rgba8.avif --bitdepth 8 --lossless --keep none
$VIPSHEADER -a fixtures/rgba8.avif
$VIPS getpoint fixtures/rgba8.avif 0 0
$VIPS getpoint fixtures/rgba8.avif 1 0
$VIPS getpoint fixtures/rgba8.avif 2 0
$VIPS getpoint fixtures/rgba8.avif 3 0
$VIPS getpoint fixtures/rgba8.avif 0 1
$VIPS getpoint fixtures/rgba8.avif 1 1
$VIPS getpoint fixtures/rgba8.avif 2 1
$VIPS getpoint fixtures/rgba8.avif 3 1
$VIPS getpoint fixtures/rgba8.avif 0 2
$VIPS getpoint fixtures/rgba8.avif 1 2
$VIPS getpoint fixtures/rgba8.avif 2 2
$VIPS getpoint fixtures/rgba8.avif 3 2
# fixtures/rgba16_src.raw written by this script (4x3x4 ushort, deterministic ramp)
$VIPS rawload fixtures/rgba16_src.raw outputs/rgba16_src.v 4 3 4 --format ushort
$VIPS copy outputs/rgba16_src.v outputs/rgba16_src-rgb16.v --interpretation rgb16
$VIPS heifsave outputs/rgba16_src-rgb16.v fixtures/rgba10.avif --bitdepth 10 --lossless --keep none
$VIPSHEADER -a fixtures/rgba10.avif
$VIPS getpoint fixtures/rgba10.avif 0 0
$VIPS getpoint fixtures/rgba10.avif 1 0
$VIPS getpoint fixtures/rgba10.avif 2 0
$VIPS getpoint fixtures/rgba10.avif 3 0
$VIPS getpoint fixtures/rgba10.avif 0 1
$VIPS getpoint fixtures/rgba10.avif 1 1
$VIPS getpoint fixtures/rgba10.avif 2 1
$VIPS getpoint fixtures/rgba10.avif 3 1
$VIPS getpoint fixtures/rgba10.avif 0 2
$VIPS getpoint fixtures/rgba10.avif 1 2
$VIPS getpoint fixtures/rgba10.avif 2 2
$VIPS getpoint fixtures/rgba10.avif 3 2
# fixtures/grey8_src.raw written by this script (4x3x1 uchar, deterministic ramp)
$VIPS rawload fixtures/grey8_src.raw outputs/grey8_src.v 4 3 1 --format uchar
$VIPS copy outputs/grey8_src.v outputs/grey8_src-b-w.v --interpretation b-w
$VIPS heifsave outputs/grey8_src-b-w.v fixtures/grey8.avif --bitdepth 8 --lossless --keep none
$VIPSHEADER -a fixtures/grey8.avif
$VIPS getpoint fixtures/grey8.avif 0 0
$VIPS getpoint fixtures/grey8.avif 1 0
$VIPS getpoint fixtures/grey8.avif 2 0
$VIPS getpoint fixtures/grey8.avif 3 0
$VIPS getpoint fixtures/grey8.avif 0 1
$VIPS getpoint fixtures/grey8.avif 1 1
$VIPS getpoint fixtures/grey8.avif 2 1
$VIPS getpoint fixtures/grey8.avif 3 1
$VIPS getpoint fixtures/grey8.avif 0 2
$VIPS getpoint fixtures/grey8.avif 1 2
$VIPS getpoint fixtures/grey8.avif 2 2
$VIPS getpoint fixtures/grey8.avif 3 2
# fixtures/rgb16_srgb_src.raw written by this script (4x3x3 ushort, deterministic ramp)
$VIPS rawload fixtures/rgb16_srgb_src.raw outputs/rgb16_srgb_src.v 4 3 3 --format ushort
$VIPS copy outputs/rgb16_srgb_src.v outputs/rgb16_srgb_src-srgb.v --interpretation srgb
$VIPS heifsave outputs/rgb8_src-srgb.v outputs/default_uchar_srgb.avif --lossless --keep none
$VIPSHEADER outputs/default_uchar_srgb.avif
$VIPSHEADER -f bits-per-sample outputs/default_uchar_srgb.avif
$VIPS getpoint outputs/default_uchar_srgb.avif 0 0
$VIPS getpoint outputs/default_uchar_srgb.avif 1 0
$VIPS getpoint outputs/default_uchar_srgb.avif 2 0
$VIPS getpoint outputs/default_uchar_srgb.avif 3 0
$VIPS getpoint outputs/default_uchar_srgb.avif 0 1
$VIPS getpoint outputs/default_uchar_srgb.avif 1 1
$VIPS getpoint outputs/default_uchar_srgb.avif 2 1
$VIPS getpoint outputs/default_uchar_srgb.avif 3 1
$VIPS getpoint outputs/default_uchar_srgb.avif 0 2
$VIPS getpoint outputs/default_uchar_srgb.avif 1 2
$VIPS getpoint outputs/default_uchar_srgb.avif 2 2
$VIPS getpoint outputs/default_uchar_srgb.avif 3 2
$VIPS heifsave outputs/rgb16_src-rgb16.v outputs/default_ushort_rgb16.avif --lossless --keep none
$VIPSHEADER outputs/default_ushort_rgb16.avif
$VIPSHEADER -f bits-per-sample outputs/default_ushort_rgb16.avif
$VIPS getpoint outputs/default_ushort_rgb16.avif 0 0
$VIPS getpoint outputs/default_ushort_rgb16.avif 1 0
$VIPS getpoint outputs/default_ushort_rgb16.avif 2 0
$VIPS getpoint outputs/default_ushort_rgb16.avif 3 0
$VIPS getpoint outputs/default_ushort_rgb16.avif 0 1
$VIPS getpoint outputs/default_ushort_rgb16.avif 1 1
$VIPS getpoint outputs/default_ushort_rgb16.avif 2 1
$VIPS getpoint outputs/default_ushort_rgb16.avif 3 1
$VIPS getpoint outputs/default_ushort_rgb16.avif 0 2
$VIPS getpoint outputs/default_ushort_rgb16.avif 1 2
$VIPS getpoint outputs/default_ushort_rgb16.avif 2 2
$VIPS getpoint outputs/default_ushort_rgb16.avif 3 2
$VIPS heifsave outputs/rgb16_srgb_src-srgb.v outputs/default_ushort_srgb.avif --lossless --keep none
$VIPSHEADER outputs/default_ushort_srgb.avif
$VIPSHEADER -f bits-per-sample outputs/default_ushort_srgb.avif
$VIPS getpoint outputs/default_ushort_srgb.avif 0 0
$VIPS getpoint outputs/default_ushort_srgb.avif 1 0
$VIPS getpoint outputs/default_ushort_srgb.avif 2 0
$VIPS getpoint outputs/default_ushort_srgb.avif 3 0
$VIPS getpoint outputs/default_ushort_srgb.avif 0 1
$VIPS getpoint outputs/default_ushort_srgb.avif 1 1
$VIPS getpoint outputs/default_ushort_srgb.avif 2 1
$VIPS getpoint outputs/default_ushort_srgb.avif 3 1
$VIPS getpoint outputs/default_ushort_srgb.avif 0 2
$VIPS getpoint outputs/default_ushort_srgb.avif 1 2
$VIPS getpoint outputs/default_ushort_srgb.avif 2 2
$VIPS getpoint outputs/default_ushort_srgb.avif 3 2
# fixtures/u8_extremes_src.raw written by this script (4x3x3 uchar, deterministic ramp)
$VIPS rawload fixtures/u8_extremes_src.raw outputs/u8_extremes_src.v 4 3 3 --format uchar
$VIPS copy outputs/u8_extremes_src.v outputs/u8_extremes_src-srgb.v --interpretation srgb
$VIPS heifsave outputs/u8_extremes_src-srgb.v outputs/u8_at_10.avif --bitdepth 10 --lossless --keep none
$VIPS getpoint outputs/u8_at_10.avif 0 0
$VIPS getpoint outputs/u8_at_10.avif 1 0
$VIPS getpoint outputs/u8_at_10.avif 2 0
$VIPS getpoint outputs/u8_at_10.avif 3 0
$VIPS getpoint outputs/u8_at_10.avif 0 1
$VIPS getpoint outputs/u8_at_10.avif 1 1
$VIPS getpoint outputs/u8_at_10.avif 2 1
$VIPS getpoint outputs/u8_at_10.avif 3 1
$VIPS getpoint outputs/u8_at_10.avif 0 2
$VIPS getpoint outputs/u8_at_10.avif 1 2
$VIPS getpoint outputs/u8_at_10.avif 2 2
$VIPS getpoint outputs/u8_at_10.avif 3 2
$VIPSHEADER outputs/u8_at_10.avif
$VIPSHEADER -f bits-per-sample outputs/u8_at_10.avif
$VIPS heifsave outputs/u8_extremes_src-srgb.v outputs/u8_at_12.avif --bitdepth 12 --lossless --keep none
$VIPS getpoint outputs/u8_at_12.avif 0 0
$VIPS getpoint outputs/u8_at_12.avif 1 0
$VIPS getpoint outputs/u8_at_12.avif 2 0
$VIPS getpoint outputs/u8_at_12.avif 3 0
$VIPS getpoint outputs/u8_at_12.avif 0 1
$VIPS getpoint outputs/u8_at_12.avif 1 1
$VIPS getpoint outputs/u8_at_12.avif 2 1
$VIPS getpoint outputs/u8_at_12.avif 3 1
$VIPS getpoint outputs/u8_at_12.avif 0 2
$VIPS getpoint outputs/u8_at_12.avif 1 2
$VIPS getpoint outputs/u8_at_12.avif 2 2
$VIPS getpoint outputs/u8_at_12.avif 3 2
$VIPSHEADER outputs/u8_at_12.avif
$VIPSHEADER -f bits-per-sample outputs/u8_at_12.avif
$VIPS heifsave outputs/rgb8_src-srgb.v fixtures/rgb8_q50_420.avif --Q 50 --keep none
$VIPSHEADER -a fixtures/rgb8_q50_420.avif
$VIPS getpoint fixtures/rgb8_q50_420.avif 0 0
$VIPS getpoint fixtures/rgb8_q50_420.avif 1 0
$VIPS getpoint fixtures/rgb8_q50_420.avif 2 0
$VIPS getpoint fixtures/rgb8_q50_420.avif 3 0
$VIPS getpoint fixtures/rgb8_q50_420.avif 0 1
$VIPS getpoint fixtures/rgb8_q50_420.avif 1 1
$VIPS getpoint fixtures/rgb8_q50_420.avif 2 1
$VIPS getpoint fixtures/rgb8_q50_420.avif 3 1
$VIPS getpoint fixtures/rgb8_q50_420.avif 0 2
$VIPS getpoint fixtures/rgb8_q50_420.avif 1 2
$VIPS getpoint fixtures/rgb8_q50_420.avif 2 2
$VIPS getpoint fixtures/rgb8_q50_420.avif 3 2
$VIPS heifsave outputs/rgb8_src-srgb.v fixtures/rgb8_icc.avif --lossless --profile srgb --keep none
$VIPSHEADER -a fixtures/rgb8_icc.avif
$VIPS getpoint fixtures/rgb8_icc.avif 0 0
$VIPS getpoint fixtures/rgb8_icc.avif 1 0
$VIPS getpoint fixtures/rgb8_icc.avif 2 0
$VIPS getpoint fixtures/rgb8_icc.avif 3 0
$VIPS getpoint fixtures/rgb8_icc.avif 0 1
$VIPS getpoint fixtures/rgb8_icc.avif 1 1
$VIPS getpoint fixtures/rgb8_icc.avif 2 1
$VIPS getpoint fixtures/rgb8_icc.avif 3 1
$VIPS getpoint fixtures/rgb8_icc.avif 0 2
$VIPS getpoint fixtures/rgb8_icc.avif 1 2
$VIPS getpoint fixtures/rgb8_icc.avif 2 2
$VIPS getpoint fixtures/rgb8_icc.avif 3 2
$VIPS heifsave fixtures/rgb8_icc.avif outputs/icc_resave.avif --lossless --keep none
$VIPSHEADER -a outputs/icc_resave.avif
$VIPS heifsave outputs/rgb8_src-srgb.v outputs/sub_q50_auto.avif --Q 50 --keep none
$VIPS heifsave outputs/rgb8_src-srgb.v outputs/sub_q89_auto.avif --Q 89 --keep none
$VIPS heifsave outputs/rgb8_src-srgb.v outputs/sub_q90_auto.avif --Q 90 --keep none
$VIPS heifsave outputs/rgb8_src-srgb.v outputs/sub_q50_off.avif --Q 50 --subsample-mode off --keep none
$VIPS heifsave outputs/rgb8_src-srgb.v outputs/sub_q90_on.avif --Q 90 --subsample-mode on --keep none
$VIPS heifsave outputs/rgb8_src-srgb.v outputs/sub_lossless_auto.avif --lossless --keep none
$VIPS heifsave outputs/rgb8_src-srgb.v outputs/sub_lossless_on.avif --lossless --subsample-mode on --keep none
$VIPS heifsave outputs/rgb8_src-srgb.v fixtures/rgb8_q90_444.avif --Q 90 --keep none
$VIPS getpoint fixtures/rgb8_q90_444.avif 0 0
$VIPS getpoint fixtures/rgb8_q90_444.avif 1 0
$VIPS getpoint fixtures/rgb8_q90_444.avif 2 0
$VIPS getpoint fixtures/rgb8_q90_444.avif 3 0
$VIPS getpoint fixtures/rgb8_q90_444.avif 0 1
$VIPS getpoint fixtures/rgb8_q90_444.avif 1 1
$VIPS getpoint fixtures/rgb8_q90_444.avif 2 1
$VIPS getpoint fixtures/rgb8_q90_444.avif 3 1
$VIPS getpoint fixtures/rgb8_q90_444.avif 0 2
$VIPS getpoint fixtures/rgb8_q90_444.avif 1 2
$VIPS getpoint fixtures/rgb8_q90_444.avif 2 2
$VIPS getpoint fixtures/rgb8_q90_444.avif 3 2
# fixtures/odd3x3_src.raw written by this script (3x3x3 uchar, deterministic ramp)
$VIPS rawload fixtures/odd3x3_src.raw outputs/odd3x3_src.v 3 3 3 --format uchar
$VIPS copy outputs/odd3x3_src.v outputs/odd3x3_src-srgb.v --interpretation srgb
$VIPS heifsave outputs/odd3x3_src-srgb.v fixtures/odd3x3_q50.avif --Q 50 --keep none
$VIPSHEADER -a fixtures/odd3x3_q50.avif
$VIPS getpoint fixtures/odd3x3_q50.avif 0 0
$VIPS getpoint fixtures/odd3x3_q50.avif 1 0
$VIPS getpoint fixtures/odd3x3_q50.avif 2 0
$VIPS getpoint fixtures/odd3x3_q50.avif 0 1
$VIPS getpoint fixtures/odd3x3_q50.avif 1 1
$VIPS getpoint fixtures/odd3x3_q50.avif 2 1
$VIPS getpoint fixtures/odd3x3_q50.avif 0 2
$VIPS getpoint fixtures/odd3x3_q50.avif 1 2
$VIPS getpoint fixtures/odd3x3_q50.avif 2 2
$VIPS black outputs/one.v 1 1 --bands 3
$VIPS copy outputs/one.v outputs/one-srgb.v --interpretation srgb
$VIPS heifsave outputs/one-srgb.v outputs/one1x1.avif --lossless --keep none
$VIPSHEADER -a outputs/one1x1.avif
$VIPS heifsave outputs/rgb8_src-srgb.v outputs/codec_default.avif --lossless --keep none
$VIPSHEADER -f heif-compression outputs/codec_default.avif
$VIPS heifsave outputs/rgb8_src-srgb.v outputs/codec_hevc.avif --compression hevc --lossless --keep none
$VIPSHEADER -f heif-compression outputs/codec_hevc.avif
$VIPS heifsave outputs/rgb8_src-srgb.v outputs/codec_avc.avif --compression avc --lossless --keep none
$VIPSHEADER -f heif-compression outputs/codec_avc.avif
$VIPS heifsave outputs/rgb8_src-srgb.v outputs/codec_default.heic --lossless --keep none
$VIPSHEADER -f heif-compression outputs/codec_default.heic
$VIPS heifsave outputs/rgb8_src-srgb.v outputs/enc_auto.avif --encoder auto --lossless --keep none
$VIPS heifsave outputs/rgb8_src-srgb.v outputs/enc_aom.avif --encoder aom --lossless --keep none
$VIPS heifsave outputs/rgb8_src-srgb.v outputs/enc_rav1e.avif --encoder rav1e --lossless --keep none
$VIPS heifsave outputs/rgb8_src-srgb.v outputs/enc_svt.avif --encoder svt --lossless --keep none
$VIPS heifsave outputs/rgb8_src-srgb.v outputs/enc_x265.avif --encoder x265 --lossless --keep none
$VIPS heifsave outputs/rgb16_src-rgb16.v outputs/bd_7.avif --bitdepth 7 --lossless --keep none
$VIPSHEADER -f bits-per-sample outputs/bd_7.avif
$VIPS heifsave outputs/rgb16_src-rgb16.v outputs/bd_8.avif --bitdepth 8 --lossless --keep none
$VIPSHEADER -f bits-per-sample outputs/bd_8.avif
$VIPS heifsave outputs/rgb16_src-rgb16.v outputs/bd_9.avif --bitdepth 9 --lossless --keep none
$VIPS heifsave outputs/rgb16_src-rgb16.v outputs/bd_10.avif --bitdepth 10 --lossless --keep none
$VIPSHEADER -f bits-per-sample outputs/bd_10.avif
$VIPS heifsave outputs/rgb16_src-rgb16.v outputs/bd_11.avif --bitdepth 11 --lossless --keep none
$VIPS heifsave outputs/rgb16_src-rgb16.v outputs/bd_12.avif --bitdepth 12 --lossless --keep none
$VIPSHEADER -f bits-per-sample outputs/bd_12.avif
$VIPS heifsave outputs/rgb16_src-rgb16.v outputs/bd_13.avif --bitdepth 13 --lossless --keep none
$VIPSHEADER -f bits-per-sample outputs/bd_13.avif
$VIPS heifsave outputs/rgb16_src-rgb16.v outputs/bd_16.avif --bitdepth 16 --lossless --keep none
$VIPSHEADER -f bits-per-sample outputs/bd_16.avif
$VIPS heifsave outputs/rgb8_src-srgb.v outputs/keep_all.avif --lossless
$VIPSHEADER -a outputs/keep_all.avif
# fixtures/truncated.avif written by this script by damaging fixtures/rgb8.avif
$VIPSHEADER -a fixtures/truncated.avif
$VIPS getpoint fixtures/truncated.avif 1 0
$VIPSHEADER -f vips-loader fixtures/truncated.avif
# outputs/truncated_ftyp.avif written by this script by damaging fixtures/rgb8.avif
$VIPSHEADER -a outputs/truncated_ftyp.avif
$VIPS getpoint outputs/truncated_ftyp.avif 1 0
$VIPSHEADER -f vips-loader outputs/truncated_ftyp.avif
# outputs/ftyp_only.avif written by this script by damaging fixtures/rgb8.avif
$VIPSHEADER -a outputs/ftyp_only.avif
$VIPS getpoint outputs/ftyp_only.avif 1 0
$VIPSHEADER -f vips-loader outputs/ftyp_only.avif
# outputs/empty.avif written by this script by damaging fixtures/rgb8.avif
$VIPSHEADER -a outputs/empty.avif
$VIPS getpoint outputs/empty.avif 1 0
$VIPSHEADER -f vips-loader outputs/empty.avif
# fixtures/brand_avis.avif written by this script by damaging fixtures/rgb8.avif
$VIPSHEADER -a fixtures/brand_avis.avif
$VIPS getpoint fixtures/brand_avis.avif 1 0
$VIPSHEADER -f vips-loader fixtures/brand_avis.avif
# outputs/brand_zzzz.avif written by this script by damaging fixtures/rgb8.avif
$VIPSHEADER -a outputs/brand_zzzz.avif
$VIPS getpoint outputs/brand_zzzz.avif 1 0
$VIPSHEADER -f vips-loader outputs/brand_zzzz.avif
# outputs/ftyp_len_29.avif written by this script by damaging fixtures/rgb8.avif
$VIPSHEADER -a outputs/ftyp_len_29.avif
$VIPS getpoint outputs/ftyp_len_29.avif 1 0
$VIPSHEADER -f vips-loader outputs/ftyp_len_29.avif
# outputs/ftyp_len_4096.avif written by this script by damaging fixtures/rgb8.avif
$VIPSHEADER -a outputs/ftyp_len_4096.avif
$VIPS getpoint outputs/ftyp_len_4096.avif 1 0
$VIPSHEADER -f vips-loader outputs/ftyp_len_4096.avif
# outputs/zeroed_mdat.avif written by this script by damaging fixtures/rgb8.avif
$VIPSHEADER -a outputs/zeroed_mdat.avif
$VIPS getpoint outputs/zeroed_mdat.avif 1 0
$VIPSHEADER -f vips-loader outputs/zeroed_mdat.avif
# outputs/brand_heic.avif written by this script: fixtures/rgb8.avif with its ftyp major brand replaced
$VIPSHEADER -a outputs/brand_heic.avif
$VIPSHEADER outputs/brand_heic.avif
$VIPS getpoint outputs/brand_heic.avif 1 0
# outputs/brand_heix.avif written by this script: fixtures/rgb8.avif with its ftyp major brand replaced
$VIPSHEADER -a outputs/brand_heix.avif
$VIPSHEADER outputs/brand_heix.avif
$VIPS getpoint outputs/brand_heix.avif 1 0
# outputs/brand_hevc.avif written by this script: fixtures/rgb8.avif with its ftyp major brand replaced
$VIPSHEADER -a outputs/brand_hevc.avif
$VIPSHEADER outputs/brand_hevc.avif
$VIPS getpoint outputs/brand_hevc.avif 1 0
# outputs/brand_heim.avif written by this script: fixtures/rgb8.avif with its ftyp major brand replaced
$VIPSHEADER -a outputs/brand_heim.avif
$VIPSHEADER outputs/brand_heim.avif
$VIPS getpoint outputs/brand_heim.avif 1 0
# outputs/brand_heis.avif written by this script: fixtures/rgb8.avif with its ftyp major brand replaced
$VIPSHEADER -a outputs/brand_heis.avif
$VIPSHEADER outputs/brand_heis.avif
$VIPS getpoint outputs/brand_heis.avif 1 0
# outputs/brand_hevm.avif written by this script: fixtures/rgb8.avif with its ftyp major brand replaced
$VIPSHEADER -a outputs/brand_hevm.avif
$VIPSHEADER outputs/brand_hevm.avif
$VIPS getpoint outputs/brand_hevm.avif 1 0
# outputs/brand_hevs.avif written by this script: fixtures/rgb8.avif with its ftyp major brand replaced
$VIPSHEADER -a outputs/brand_hevs.avif
$VIPSHEADER outputs/brand_hevs.avif
$VIPS getpoint outputs/brand_hevs.avif 1 0
# outputs/brand_mif1.avif written by this script: fixtures/rgb8.avif with its ftyp major brand replaced
$VIPSHEADER -a outputs/brand_mif1.avif
$VIPSHEADER outputs/brand_mif1.avif
$VIPS getpoint outputs/brand_mif1.avif 1 0
# outputs/brand_msf1.avif written by this script: fixtures/rgb8.avif with its ftyp major brand replaced
$VIPSHEADER -a outputs/brand_msf1.avif
$VIPSHEADER outputs/brand_msf1.avif
$VIPS getpoint outputs/brand_msf1.avif 1 0
# outputs/brand_avif.avif written by this script: fixtures/rgb8.avif with its ftyp major brand replaced
$VIPSHEADER -a outputs/brand_avif.avif
$VIPSHEADER outputs/brand_avif.avif
$VIPS getpoint outputs/brand_avif.avif 1 0
# outputs/brand_avis.avif written by this script: fixtures/rgb8.avif with its ftyp major brand replaced
$VIPSHEADER -a outputs/brand_avis.avif
$VIPSHEADER outputs/brand_avis.avif
$VIPS getpoint outputs/brand_avis.avif 1 0
# outputs/brand_mif2.avif written by this script: fixtures/rgb8.avif with its ftyp major brand replaced
$VIPSHEADER -a outputs/brand_mif2.avif
$VIPSHEADER outputs/brand_mif2.avif
$VIPS getpoint outputs/brand_mif2.avif 1 0
$VIPSHEADER -a fixtures/rgb8.avif
$VIPSHEADER fixtures/rgb8.avif
$VIPSHEADER -a fixtures/rgb8.avif[page=0]
$VIPSHEADER fixtures/rgb8.avif[page=0]
$VIPSHEADER -a fixtures/rgb8.avif[n=-1]
$VIPSHEADER fixtures/rgb8.avif[n=-1]
$VIPSHEADER -a fixtures/rgb8.avif[n=2]
$VIPSHEADER fixtures/rgb8.avif[n=2]
$VIPSHEADER -a fixtures/rgb8.avif[page=1]
$VIPSHEADER fixtures/rgb8.avif[page=1]
$VIPSHEADER -a fixtures/rgb8.avif[thumbnail=true]
$VIPSHEADER fixtures/rgb8.avif[thumbnail=true]
$VIPS --version
