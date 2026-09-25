#!/bin/sh
# Every vips command capture.py ran, in order. Regenerate with
# `python3 capture.py` from this directory.
set -e

/opt/homebrew/bin/vips copy fixtures/mono8.pgm outputs/mono_uchar.fits
/opt/homebrew/bin/vips copy fixtures/rgb8.ppm outputs/rgb_uchar.fits
/opt/homebrew/bin/vips copy fixtures/mono16.pgm outputs/mono_ushort.fits
/opt/homebrew/bin/vips rawload fixtures/rgba8.raw outputs/rgba8.v 4 3 4
/opt/homebrew/bin/vips copy outputs/rgba8.v outputs/rgba_uchar.fits
/opt/homebrew/bin/vips cast fixtures/mono8.pgm outputs/mono_float.v float
/opt/homebrew/bin/vips copy outputs/mono_float.v outputs/mono_float.fits
/opt/homebrew/bin/vips cast fixtures/rgb8.ppm outputs/rgb_float.v float
/opt/homebrew/bin/vips copy outputs/rgb_float.v outputs/rgb_float.fits
/opt/homebrew/bin/vipsheader -a outputs/mono_uchar.fits
/opt/homebrew/bin/vips getpoint outputs/mono_uchar.fits 0 0
/opt/homebrew/bin/vips getpoint outputs/mono_uchar.fits 1 0
/opt/homebrew/bin/vips getpoint outputs/mono_uchar.fits 2 0
/opt/homebrew/bin/vips getpoint outputs/mono_uchar.fits 3 0
/opt/homebrew/bin/vips getpoint outputs/mono_uchar.fits 0 1
/opt/homebrew/bin/vips getpoint outputs/mono_uchar.fits 1 1
/opt/homebrew/bin/vips getpoint outputs/mono_uchar.fits 2 1
/opt/homebrew/bin/vips getpoint outputs/mono_uchar.fits 3 1
/opt/homebrew/bin/vips getpoint outputs/mono_uchar.fits 0 2
/opt/homebrew/bin/vips getpoint outputs/mono_uchar.fits 1 2
/opt/homebrew/bin/vips getpoint outputs/mono_uchar.fits 2 2
/opt/homebrew/bin/vips getpoint outputs/mono_uchar.fits 3 2
/opt/homebrew/bin/vipsheader -a outputs/rgb_uchar.fits
/opt/homebrew/bin/vips getpoint outputs/rgb_uchar.fits 0 0
/opt/homebrew/bin/vips getpoint outputs/rgb_uchar.fits 1 0
/opt/homebrew/bin/vips getpoint outputs/rgb_uchar.fits 2 0
/opt/homebrew/bin/vips getpoint outputs/rgb_uchar.fits 3 0
/opt/homebrew/bin/vips getpoint outputs/rgb_uchar.fits 0 1
/opt/homebrew/bin/vips getpoint outputs/rgb_uchar.fits 1 1
/opt/homebrew/bin/vips getpoint outputs/rgb_uchar.fits 2 1
/opt/homebrew/bin/vips getpoint outputs/rgb_uchar.fits 3 1
/opt/homebrew/bin/vips getpoint outputs/rgb_uchar.fits 0 2
/opt/homebrew/bin/vips getpoint outputs/rgb_uchar.fits 1 2
/opt/homebrew/bin/vips getpoint outputs/rgb_uchar.fits 2 2
/opt/homebrew/bin/vips getpoint outputs/rgb_uchar.fits 3 2
/opt/homebrew/bin/vipsheader -a outputs/mono_ushort.fits
/opt/homebrew/bin/vips getpoint outputs/mono_ushort.fits 0 0
/opt/homebrew/bin/vips getpoint outputs/mono_ushort.fits 1 0
/opt/homebrew/bin/vips getpoint outputs/mono_ushort.fits 2 0
/opt/homebrew/bin/vips getpoint outputs/mono_ushort.fits 3 0
/opt/homebrew/bin/vips getpoint outputs/mono_ushort.fits 0 1
/opt/homebrew/bin/vips getpoint outputs/mono_ushort.fits 1 1
/opt/homebrew/bin/vips getpoint outputs/mono_ushort.fits 2 1
/opt/homebrew/bin/vips getpoint outputs/mono_ushort.fits 3 1
/opt/homebrew/bin/vips getpoint outputs/mono_ushort.fits 0 2
/opt/homebrew/bin/vips getpoint outputs/mono_ushort.fits 1 2
/opt/homebrew/bin/vips getpoint outputs/mono_ushort.fits 2 2
/opt/homebrew/bin/vips getpoint outputs/mono_ushort.fits 3 2
/opt/homebrew/bin/vipsheader -a outputs/rgba_uchar.fits
/opt/homebrew/bin/vips getpoint outputs/rgba_uchar.fits 0 0
/opt/homebrew/bin/vips getpoint outputs/rgba_uchar.fits 1 0
/opt/homebrew/bin/vips getpoint outputs/rgba_uchar.fits 2 0
/opt/homebrew/bin/vips getpoint outputs/rgba_uchar.fits 3 0
/opt/homebrew/bin/vips getpoint outputs/rgba_uchar.fits 0 1
/opt/homebrew/bin/vips getpoint outputs/rgba_uchar.fits 1 1
/opt/homebrew/bin/vips getpoint outputs/rgba_uchar.fits 2 1
/opt/homebrew/bin/vips getpoint outputs/rgba_uchar.fits 3 1
/opt/homebrew/bin/vips getpoint outputs/rgba_uchar.fits 0 2
/opt/homebrew/bin/vips getpoint outputs/rgba_uchar.fits 1 2
/opt/homebrew/bin/vips getpoint outputs/rgba_uchar.fits 2 2
/opt/homebrew/bin/vips getpoint outputs/rgba_uchar.fits 3 2
/opt/homebrew/bin/vipsheader -a outputs/mono_float.fits
/opt/homebrew/bin/vips getpoint outputs/mono_float.fits 0 0
/opt/homebrew/bin/vips getpoint outputs/mono_float.fits 1 0
/opt/homebrew/bin/vips getpoint outputs/mono_float.fits 2 0
/opt/homebrew/bin/vips getpoint outputs/mono_float.fits 3 0
/opt/homebrew/bin/vips getpoint outputs/mono_float.fits 0 1
/opt/homebrew/bin/vips getpoint outputs/mono_float.fits 1 1
/opt/homebrew/bin/vips getpoint outputs/mono_float.fits 2 1
/opt/homebrew/bin/vips getpoint outputs/mono_float.fits 3 1
/opt/homebrew/bin/vips getpoint outputs/mono_float.fits 0 2
/opt/homebrew/bin/vips getpoint outputs/mono_float.fits 1 2
/opt/homebrew/bin/vips getpoint outputs/mono_float.fits 2 2
/opt/homebrew/bin/vips getpoint outputs/mono_float.fits 3 2
/opt/homebrew/bin/vipsheader -a outputs/rgb_float.fits
/opt/homebrew/bin/vips getpoint outputs/rgb_float.fits 0 0
/opt/homebrew/bin/vips getpoint outputs/rgb_float.fits 1 0
/opt/homebrew/bin/vips getpoint outputs/rgb_float.fits 2 0
/opt/homebrew/bin/vips getpoint outputs/rgb_float.fits 3 0
/opt/homebrew/bin/vips getpoint outputs/rgb_float.fits 0 1
/opt/homebrew/bin/vips getpoint outputs/rgb_float.fits 1 1
/opt/homebrew/bin/vips getpoint outputs/rgb_float.fits 2 1
/opt/homebrew/bin/vips getpoint outputs/rgb_float.fits 3 1
/opt/homebrew/bin/vips getpoint outputs/rgb_float.fits 0 2
/opt/homebrew/bin/vips getpoint outputs/rgb_float.fits 1 2
/opt/homebrew/bin/vips getpoint outputs/rgb_float.fits 2 2
/opt/homebrew/bin/vips getpoint outputs/rgb_float.fits 3 2
/opt/homebrew/bin/vipsheader -a fixtures/bitpix_8.fits
/opt/homebrew/bin/vips getpoint fixtures/bitpix_8.fits 0 0
/opt/homebrew/bin/vips getpoint fixtures/bitpix_8.fits 1 0
/opt/homebrew/bin/vips getpoint fixtures/bitpix_8.fits 2 0
/opt/homebrew/bin/vips getpoint fixtures/bitpix_8.fits 3 0
/opt/homebrew/bin/vips getpoint fixtures/bitpix_8.fits 0 1
/opt/homebrew/bin/vips getpoint fixtures/bitpix_8.fits 1 1
/opt/homebrew/bin/vips getpoint fixtures/bitpix_8.fits 2 1
/opt/homebrew/bin/vips getpoint fixtures/bitpix_8.fits 3 1
/opt/homebrew/bin/vips getpoint fixtures/bitpix_8.fits 0 2
/opt/homebrew/bin/vips getpoint fixtures/bitpix_8.fits 1 2
/opt/homebrew/bin/vips getpoint fixtures/bitpix_8.fits 2 2
/opt/homebrew/bin/vips getpoint fixtures/bitpix_8.fits 3 2
/opt/homebrew/bin/vipsheader -a fixtures/bitpix_16_signed.fits
/opt/homebrew/bin/vips getpoint fixtures/bitpix_16_signed.fits 0 0
/opt/homebrew/bin/vips getpoint fixtures/bitpix_16_signed.fits 1 0
/opt/homebrew/bin/vips getpoint fixtures/bitpix_16_signed.fits 2 0
/opt/homebrew/bin/vips getpoint fixtures/bitpix_16_signed.fits 3 0
/opt/homebrew/bin/vips getpoint fixtures/bitpix_16_signed.fits 0 1
/opt/homebrew/bin/vips getpoint fixtures/bitpix_16_signed.fits 1 1
/opt/homebrew/bin/vips getpoint fixtures/bitpix_16_signed.fits 2 1
/opt/homebrew/bin/vips getpoint fixtures/bitpix_16_signed.fits 3 1
/opt/homebrew/bin/vips getpoint fixtures/bitpix_16_signed.fits 0 2
/opt/homebrew/bin/vips getpoint fixtures/bitpix_16_signed.fits 1 2
/opt/homebrew/bin/vips getpoint fixtures/bitpix_16_signed.fits 2 2
/opt/homebrew/bin/vips getpoint fixtures/bitpix_16_signed.fits 3 2
/opt/homebrew/bin/vipsheader -a fixtures/bitpix_16_unsigned.fits
/opt/homebrew/bin/vips getpoint fixtures/bitpix_16_unsigned.fits 0 0
/opt/homebrew/bin/vips getpoint fixtures/bitpix_16_unsigned.fits 1 0
/opt/homebrew/bin/vips getpoint fixtures/bitpix_16_unsigned.fits 2 0
/opt/homebrew/bin/vips getpoint fixtures/bitpix_16_unsigned.fits 3 0
/opt/homebrew/bin/vips getpoint fixtures/bitpix_16_unsigned.fits 0 1
/opt/homebrew/bin/vips getpoint fixtures/bitpix_16_unsigned.fits 1 1
/opt/homebrew/bin/vips getpoint fixtures/bitpix_16_unsigned.fits 2 1
/opt/homebrew/bin/vips getpoint fixtures/bitpix_16_unsigned.fits 3 1
/opt/homebrew/bin/vips getpoint fixtures/bitpix_16_unsigned.fits 0 2
/opt/homebrew/bin/vips getpoint fixtures/bitpix_16_unsigned.fits 1 2
/opt/homebrew/bin/vips getpoint fixtures/bitpix_16_unsigned.fits 2 2
/opt/homebrew/bin/vips getpoint fixtures/bitpix_16_unsigned.fits 3 2
/opt/homebrew/bin/vipsheader -a fixtures/bitpix_32_signed.fits
/opt/homebrew/bin/vips getpoint fixtures/bitpix_32_signed.fits 0 0
/opt/homebrew/bin/vips getpoint fixtures/bitpix_32_signed.fits 1 0
/opt/homebrew/bin/vips getpoint fixtures/bitpix_32_signed.fits 2 0
/opt/homebrew/bin/vips getpoint fixtures/bitpix_32_signed.fits 3 0
/opt/homebrew/bin/vips getpoint fixtures/bitpix_32_signed.fits 0 1
/opt/homebrew/bin/vips getpoint fixtures/bitpix_32_signed.fits 1 1
/opt/homebrew/bin/vips getpoint fixtures/bitpix_32_signed.fits 2 1
/opt/homebrew/bin/vips getpoint fixtures/bitpix_32_signed.fits 3 1
/opt/homebrew/bin/vips getpoint fixtures/bitpix_32_signed.fits 0 2
/opt/homebrew/bin/vips getpoint fixtures/bitpix_32_signed.fits 1 2
/opt/homebrew/bin/vips getpoint fixtures/bitpix_32_signed.fits 2 2
/opt/homebrew/bin/vips getpoint fixtures/bitpix_32_signed.fits 3 2
/opt/homebrew/bin/vipsheader -a fixtures/bitpix_32_unsigned.fits
/opt/homebrew/bin/vips getpoint fixtures/bitpix_32_unsigned.fits 0 0
/opt/homebrew/bin/vips getpoint fixtures/bitpix_32_unsigned.fits 1 0
/opt/homebrew/bin/vips getpoint fixtures/bitpix_32_unsigned.fits 2 0
/opt/homebrew/bin/vips getpoint fixtures/bitpix_32_unsigned.fits 3 0
/opt/homebrew/bin/vips getpoint fixtures/bitpix_32_unsigned.fits 0 1
/opt/homebrew/bin/vips getpoint fixtures/bitpix_32_unsigned.fits 1 1
/opt/homebrew/bin/vips getpoint fixtures/bitpix_32_unsigned.fits 2 1
/opt/homebrew/bin/vips getpoint fixtures/bitpix_32_unsigned.fits 3 1
/opt/homebrew/bin/vips getpoint fixtures/bitpix_32_unsigned.fits 0 2
/opt/homebrew/bin/vips getpoint fixtures/bitpix_32_unsigned.fits 1 2
/opt/homebrew/bin/vips getpoint fixtures/bitpix_32_unsigned.fits 2 2
/opt/homebrew/bin/vips getpoint fixtures/bitpix_32_unsigned.fits 3 2
/opt/homebrew/bin/vipsheader -a fixtures/bitpix_64.fits
/opt/homebrew/bin/vipsheader -a fixtures/bitpix_minus_32.fits
/opt/homebrew/bin/vips getpoint fixtures/bitpix_minus_32.fits 0 0
/opt/homebrew/bin/vips getpoint fixtures/bitpix_minus_32.fits 1 0
/opt/homebrew/bin/vips getpoint fixtures/bitpix_minus_32.fits 2 0
/opt/homebrew/bin/vips getpoint fixtures/bitpix_minus_32.fits 3 0
/opt/homebrew/bin/vips getpoint fixtures/bitpix_minus_32.fits 0 1
/opt/homebrew/bin/vips getpoint fixtures/bitpix_minus_32.fits 1 1
/opt/homebrew/bin/vips getpoint fixtures/bitpix_minus_32.fits 2 1
/opt/homebrew/bin/vips getpoint fixtures/bitpix_minus_32.fits 3 1
/opt/homebrew/bin/vips getpoint fixtures/bitpix_minus_32.fits 0 2
/opt/homebrew/bin/vips getpoint fixtures/bitpix_minus_32.fits 1 2
/opt/homebrew/bin/vips getpoint fixtures/bitpix_minus_32.fits 2 2
/opt/homebrew/bin/vips getpoint fixtures/bitpix_minus_32.fits 3 2
/opt/homebrew/bin/vipsheader -a fixtures/bitpix_minus_64.fits
/opt/homebrew/bin/vips getpoint fixtures/bitpix_minus_64.fits 0 0
/opt/homebrew/bin/vips getpoint fixtures/bitpix_minus_64.fits 1 0
/opt/homebrew/bin/vips getpoint fixtures/bitpix_minus_64.fits 2 0
/opt/homebrew/bin/vips getpoint fixtures/bitpix_minus_64.fits 3 0
/opt/homebrew/bin/vips getpoint fixtures/bitpix_minus_64.fits 0 1
/opt/homebrew/bin/vips getpoint fixtures/bitpix_minus_64.fits 1 1
/opt/homebrew/bin/vips getpoint fixtures/bitpix_minus_64.fits 2 1
/opt/homebrew/bin/vips getpoint fixtures/bitpix_minus_64.fits 3 1
/opt/homebrew/bin/vips getpoint fixtures/bitpix_minus_64.fits 0 2
/opt/homebrew/bin/vips getpoint fixtures/bitpix_minus_64.fits 1 2
/opt/homebrew/bin/vips getpoint fixtures/bitpix_minus_64.fits 2 2
/opt/homebrew/bin/vips getpoint fixtures/bitpix_minus_64.fits 3 2
/opt/homebrew/bin/vipsheader -a fixtures/bitpix_8_signed_byte.fits
/opt/homebrew/bin/vipsheader -a fixtures/bitpix_8_rescaled.fits
/opt/homebrew/bin/vips getpoint fixtures/bitpix_8_rescaled.fits 0 0
/opt/homebrew/bin/vips getpoint fixtures/bitpix_8_rescaled.fits 1 0
/opt/homebrew/bin/vips getpoint fixtures/bitpix_8_rescaled.fits 2 0
/opt/homebrew/bin/vips getpoint fixtures/bitpix_8_rescaled.fits 3 0
/opt/homebrew/bin/vips getpoint fixtures/bitpix_8_rescaled.fits 0 1
/opt/homebrew/bin/vips getpoint fixtures/bitpix_8_rescaled.fits 1 1
/opt/homebrew/bin/vips getpoint fixtures/bitpix_8_rescaled.fits 2 1
/opt/homebrew/bin/vips getpoint fixtures/bitpix_8_rescaled.fits 3 1
/opt/homebrew/bin/vips getpoint fixtures/bitpix_8_rescaled.fits 0 2
/opt/homebrew/bin/vips getpoint fixtures/bitpix_8_rescaled.fits 1 2
/opt/homebrew/bin/vips getpoint fixtures/bitpix_8_rescaled.fits 2 2
/opt/homebrew/bin/vips getpoint fixtures/bitpix_8_rescaled.fits 3 2
/opt/homebrew/bin/vipsheader -a fixtures/bitpix_minus_32_rescaled.fits
/opt/homebrew/bin/vips getpoint fixtures/bitpix_minus_32_rescaled.fits 0 0
/opt/homebrew/bin/vips getpoint fixtures/bitpix_minus_32_rescaled.fits 1 0
/opt/homebrew/bin/vips getpoint fixtures/bitpix_minus_32_rescaled.fits 2 0
/opt/homebrew/bin/vips getpoint fixtures/bitpix_minus_32_rescaled.fits 3 0
/opt/homebrew/bin/vips getpoint fixtures/bitpix_minus_32_rescaled.fits 0 1
/opt/homebrew/bin/vips getpoint fixtures/bitpix_minus_32_rescaled.fits 1 1
/opt/homebrew/bin/vips getpoint fixtures/bitpix_minus_32_rescaled.fits 2 1
/opt/homebrew/bin/vips getpoint fixtures/bitpix_minus_32_rescaled.fits 3 1
/opt/homebrew/bin/vips getpoint fixtures/bitpix_minus_32_rescaled.fits 0 2
/opt/homebrew/bin/vips getpoint fixtures/bitpix_minus_32_rescaled.fits 1 2
/opt/homebrew/bin/vips getpoint fixtures/bitpix_minus_32_rescaled.fits 2 2
/opt/homebrew/bin/vips getpoint fixtures/bitpix_minus_32_rescaled.fits 3 2
/opt/homebrew/bin/vipsheader -a fixtures/naxis_1.fits
/opt/homebrew/bin/vipsheader -a fixtures/naxis_4_empty.fits
/opt/homebrew/bin/vipsheader -a fixtures/naxis_4_full.fits
/opt/homebrew/bin/vipsheader -a fixtures/naxis_11.fits
/opt/homebrew/bin/vipsheader -a fixtures/bands_5.fits
/opt/homebrew/bin/vipsheader -a fixtures/multi_unit.fits
/opt/homebrew/bin/vips getpoint fixtures/multi_unit.fits 0 0
/opt/homebrew/bin/vips getpoint fixtures/multi_unit.fits 1 0
/opt/homebrew/bin/vips getpoint fixtures/multi_unit.fits 2 0
/opt/homebrew/bin/vips getpoint fixtures/multi_unit.fits 3 0
/opt/homebrew/bin/vips getpoint fixtures/multi_unit.fits 0 1
/opt/homebrew/bin/vips getpoint fixtures/multi_unit.fits 1 1
/opt/homebrew/bin/vips getpoint fixtures/multi_unit.fits 2 1
/opt/homebrew/bin/vips getpoint fixtures/multi_unit.fits 3 1
/opt/homebrew/bin/vips getpoint fixtures/multi_unit.fits 0 2
/opt/homebrew/bin/vips getpoint fixtures/multi_unit.fits 1 2
/opt/homebrew/bin/vips getpoint fixtures/multi_unit.fits 2 2
/opt/homebrew/bin/vips getpoint fixtures/multi_unit.fits 3 2
/opt/homebrew/bin/vipsheader -a fixtures/long_header.fits
/opt/homebrew/bin/vips copy fixtures/mono8.pgm outputs/cplx.v
/opt/homebrew/bin/vips cast outputs/cplx.v outputs/cplx2.v complex
/opt/homebrew/bin/vips copy outputs/cplx2.v outputs/cplx.fits
vips -l | grep -i fits
vips --vips-config | tr ',' '\n' | grep -i cfitsio
