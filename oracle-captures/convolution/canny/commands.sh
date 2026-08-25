cat > fixtures/masks/gx_offset128.mat << 'EOF'
2 2 1 128
-1 1
-1 1
EOF
cat > fixtures/masks/gy_offset128.mat << 'EOF'
2 2 1 128
-1 -1
1 1
EOF
cat > fixtures/masks/gx.mat << 'EOF'
2 2 1 0
-1 1
-1 1
EOF
cat > fixtures/masks/gy.mat << 'EOF'
2 2 1 0
-1 -1
1 1
EOF
# fixture fixtures/step9.pgm: 9x9 uchar, columns 0-3 = 0 and 4-8 = 255. A pure Gx edge, the simplest non-trivial case, and the reference for the border behaviour.
# fixture fixtures/square9.pgm: 9x9 uchar, 4x4 white block at the top-left. Its bottom-right corner (4,4) drives both gradient convs into their negative clip, so the uchar polar stage reaches its maximum G of 64 there.
# fixture fixtures/plateau_h.pgm: 9x5 uchar, every row 0 0 0 0 128 255 255 255 255. The half-step at x=4 gives x=4 and x=5 the same G (32) and the same theta (64), so exactly one of the two survives suppression.
# fixture fixtures/plateau_h_rev.pgm: plateau_h mirrored left-right: same plateau, theta 192 instead of 64. The survivor moves to the other side, which is the asymmetry.
# fixture fixtures/plateau_v.pgm: plateau_h transposed: theta 0, plateau on rows 4 and 5.
# fixture fixtures/plateau_v_rev.pgm: plateau_v mirrored top-bottom: theta 128.
# fixture fixtures/disc33.pgm: 33x33 uchar, white disc of radius 12 centred on (16,16). This is the 'white disc on a black background' the source comment describes.
# fixture fixtures/octants26.pgm: 26x26 uchar on a flat 128 background with twenty 2x2 perturbations, each engineered to produce one exact (gx, gy) pair at its bottom-right pixel. Covers all eight octants, the four axis directions, the four diagonals, gx == gy == 0, and sub-LUT-resolution gradients.
# fixture fixtures/noise64.pgm: 64x64 uchar LCG noise. Every one of the 256 atan2 LUT indices is reached at sigma 0.01, and G spans the full uchar range 0..64.
# fixture fixtures/noise16rgb.ppm: 16x16x3 uchar LCG noise. Pins the (w, h, b) round-trip and the per-band independence of the whole operation.
# fixture fixtures/border7.pgm: 7x7 uchar with a white column on the left frame edge and a white row on the bottom frame edge. Both edges sit in the outer ring, where the Extend::Copy embed duplicates neighbours instead of supplying zeros.
#!/bin/sh
# Reproducible vips CLI commands for the canny oracle capture.
# Run from oracle-captures/convolution/canny/ (paths are relative).
set -e

../../../../../../../../../../opt/homebrew/bin/vips copy fixtures/step9.pgm fixtures/step9.v
../../../../../../../../../../opt/homebrew/bin/vips copy fixtures/square9.pgm fixtures/square9.v
../../../../../../../../../../opt/homebrew/bin/vips copy fixtures/plateau_h.pgm fixtures/plateau_h.v
../../../../../../../../../../opt/homebrew/bin/vips copy fixtures/plateau_h_rev.pgm fixtures/plateau_h_rev.v
../../../../../../../../../../opt/homebrew/bin/vips copy fixtures/plateau_v.pgm fixtures/plateau_v.v
../../../../../../../../../../opt/homebrew/bin/vips copy fixtures/plateau_v_rev.pgm fixtures/plateau_v_rev.v
../../../../../../../../../../opt/homebrew/bin/vips copy fixtures/disc33.pgm fixtures/disc33.v
../../../../../../../../../../opt/homebrew/bin/vips copy fixtures/octants26.pgm fixtures/octants26.v
../../../../../../../../../../opt/homebrew/bin/vips copy fixtures/noise64.pgm fixtures/noise64.v
../../../../../../../../../../opt/homebrew/bin/vips copy fixtures/noise16rgb.ppm fixtures/noise16rgb.v
../../../../../../../../../../opt/homebrew/bin/vips copy fixtures/border7.pgm fixtures/border7.v
../../../../../../../../../../opt/homebrew/bin/vips cast fixtures/step9.pgm fixtures/step9_float.v float
../../../../../../../../../../opt/homebrew/bin/vips cast fixtures/step9.pgm fixtures/step9_double.v double
../../../../../../../../../../opt/homebrew/bin/vips cast fixtures/step9.pgm fixtures/step9_ushort.v ushort
../../../../../../../../../../opt/homebrew/bin/vips cast fixtures/octants26.pgm fixtures/octants26_float.v float
../../../../../../../../../../opt/homebrew/bin/vips cast fixtures/square9.pgm fixtures/square9_float.v float
../../../../../../../../../../opt/homebrew/bin/vips cast fixtures/noise16rgb.ppm fixtures/noise16rgb_float.v float
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/step9.v fixtures/outputs/default_step9_float_vector.v --sigma 1.4 --precision float
VIPS_NOVECTOR=1 ../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/step9.v fixtures/outputs/default_step9_float_novector.v --sigma 1.4 --precision float
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/step9.v fixtures/outputs/default_step9_integer_vector.v --sigma 1.4 --precision integer
VIPS_NOVECTOR=1 ../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/step9.v fixtures/outputs/default_step9_integer_novector.v --sigma 1.4 --precision integer
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/step9.v fixtures/outputs/default_step9_approximate_vector.v --sigma 1.4 --precision approximate
VIPS_NOVECTOR=1 ../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/step9.v fixtures/outputs/default_step9_approximate_novector.v --sigma 1.4 --precision approximate
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/square9.v fixtures/outputs/default_square9_float_vector.v --sigma 1.4 --precision float
VIPS_NOVECTOR=1 ../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/square9.v fixtures/outputs/default_square9_float_novector.v --sigma 1.4 --precision float
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/square9.v fixtures/outputs/default_square9_integer_vector.v --sigma 1.4 --precision integer
VIPS_NOVECTOR=1 ../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/square9.v fixtures/outputs/default_square9_integer_novector.v --sigma 1.4 --precision integer
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/disc33.v fixtures/outputs/default_disc33_float_vector.v --sigma 1.4 --precision float
VIPS_NOVECTOR=1 ../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/disc33.v fixtures/outputs/default_disc33_float_novector.v --sigma 1.4 --precision float
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/disc33.v fixtures/outputs/default_disc33_integer_vector.v --sigma 1.4 --precision integer
VIPS_NOVECTOR=1 ../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/disc33.v fixtures/outputs/default_disc33_integer_novector.v --sigma 1.4 --precision integer
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/noise16rgb.v fixtures/outputs/default_noise16rgb_float_vector.v --sigma 1.4 --precision float
VIPS_NOVECTOR=1 ../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/noise16rgb.v fixtures/outputs/default_noise16rgb_float_novector.v --sigma 1.4 --precision float
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/noise16rgb.v fixtures/outputs/default_noise16rgb_integer_vector.v --sigma 1.4 --precision integer
VIPS_NOVECTOR=1 ../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/noise16rgb.v fixtures/outputs/default_noise16rgb_integer_novector.v --sigma 1.4 --precision integer
../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/step9.v fixtures/outputs/fmtblur_uchar_integer_1.4.v 1.4 --precision integer
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/step9.v fixtures/outputs/fmt_uchar_integer_1.4.v --sigma 1.4 --precision integer
../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/step9.v fixtures/outputs/fmtblur_uchar_integer_0.5.v 0.5 --precision integer
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/step9.v fixtures/outputs/fmt_uchar_integer_0.5.v --sigma 0.5 --precision integer
../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/step9.v fixtures/outputs/fmtblur_uchar_integer_0.2.v 0.2 --precision integer
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/step9.v fixtures/outputs/fmt_uchar_integer_0.2.v --sigma 0.2 --precision integer
../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/step9.v fixtures/outputs/fmtblur_uchar_integer_0.19.v 0.19 --precision integer
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/step9.v fixtures/outputs/fmt_uchar_integer_0.19.v --sigma 0.19 --precision integer
../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/step9.v fixtures/outputs/fmtblur_uchar_integer_0.1.v 0.1 --precision integer
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/step9.v fixtures/outputs/fmt_uchar_integer_0.1.v --sigma 0.1 --precision integer
../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/step9.v fixtures/outputs/fmtblur_uchar_float_1.4.v 1.4 --precision float
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/step9.v fixtures/outputs/fmt_uchar_float_1.4.v --sigma 1.4 --precision float
../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/step9.v fixtures/outputs/fmtblur_uchar_float_0.5.v 0.5 --precision float
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/step9.v fixtures/outputs/fmt_uchar_float_0.5.v --sigma 0.5 --precision float
../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/step9.v fixtures/outputs/fmtblur_uchar_float_0.2.v 0.2 --precision float
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/step9.v fixtures/outputs/fmt_uchar_float_0.2.v --sigma 0.2 --precision float
../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/step9.v fixtures/outputs/fmtblur_uchar_float_0.19.v 0.19 --precision float
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/step9.v fixtures/outputs/fmt_uchar_float_0.19.v --sigma 0.19 --precision float
../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/step9.v fixtures/outputs/fmtblur_uchar_float_0.1.v 0.1 --precision float
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/step9.v fixtures/outputs/fmt_uchar_float_0.1.v --sigma 0.1 --precision float
../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/step9.v fixtures/outputs/fmtblur_uchar_approximate_1.4.v 1.4 --precision approximate
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/step9.v fixtures/outputs/fmt_uchar_approximate_1.4.v --sigma 1.4 --precision approximate
../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/step9.v fixtures/outputs/fmtblur_uchar_approximate_0.5.v 0.5 --precision approximate
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/step9.v fixtures/outputs/fmt_uchar_approximate_0.5.v --sigma 0.5 --precision approximate
../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/step9.v fixtures/outputs/fmtblur_uchar_approximate_0.2.v 0.2 --precision approximate
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/step9.v fixtures/outputs/fmt_uchar_approximate_0.2.v --sigma 0.2 --precision approximate
../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/step9.v fixtures/outputs/fmtblur_uchar_approximate_0.19.v 0.19 --precision approximate
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/step9.v fixtures/outputs/fmt_uchar_approximate_0.19.v --sigma 0.19 --precision approximate
../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/step9.v fixtures/outputs/fmtblur_uchar_approximate_0.1.v 0.1 --precision approximate
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/step9.v fixtures/outputs/fmt_uchar_approximate_0.1.v --sigma 0.1 --precision approximate
../../../../../../../../../../opt/homebrew/bin/vips cast fixtures/step9.pgm fixtures/step9_char.v char
../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/step9_char.v fixtures/outputs/fmtblur_char_integer_1.4.v 1.4 --precision integer
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/step9_char.v fixtures/outputs/fmt_char_integer_1.4.v --sigma 1.4 --precision integer
../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/step9_char.v fixtures/outputs/fmtblur_char_integer_0.5.v 0.5 --precision integer
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/step9_char.v fixtures/outputs/fmt_char_integer_0.5.v --sigma 0.5 --precision integer
../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/step9_char.v fixtures/outputs/fmtblur_char_integer_0.2.v 0.2 --precision integer
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/step9_char.v fixtures/outputs/fmt_char_integer_0.2.v --sigma 0.2 --precision integer
../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/step9_char.v fixtures/outputs/fmtblur_char_integer_0.19.v 0.19 --precision integer
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/step9_char.v fixtures/outputs/fmt_char_integer_0.19.v --sigma 0.19 --precision integer
../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/step9_char.v fixtures/outputs/fmtblur_char_integer_0.1.v 0.1 --precision integer
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/step9_char.v fixtures/outputs/fmt_char_integer_0.1.v --sigma 0.1 --precision integer
../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/step9_char.v fixtures/outputs/fmtblur_char_float_1.4.v 1.4 --precision float
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/step9_char.v fixtures/outputs/fmt_char_float_1.4.v --sigma 1.4 --precision float
../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/step9_char.v fixtures/outputs/fmtblur_char_float_0.5.v 0.5 --precision float
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/step9_char.v fixtures/outputs/fmt_char_float_0.5.v --sigma 0.5 --precision float
../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/step9_char.v fixtures/outputs/fmtblur_char_float_0.2.v 0.2 --precision float
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/step9_char.v fixtures/outputs/fmt_char_float_0.2.v --sigma 0.2 --precision float
../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/step9_char.v fixtures/outputs/fmtblur_char_float_0.19.v 0.19 --precision float
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/step9_char.v fixtures/outputs/fmt_char_float_0.19.v --sigma 0.19 --precision float
../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/step9_char.v fixtures/outputs/fmtblur_char_float_0.1.v 0.1 --precision float
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/step9_char.v fixtures/outputs/fmt_char_float_0.1.v --sigma 0.1 --precision float
../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/step9_char.v fixtures/outputs/fmtblur_char_approximate_1.4.v 1.4 --precision approximate
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/step9_char.v fixtures/outputs/fmt_char_approximate_1.4.v --sigma 1.4 --precision approximate
../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/step9_char.v fixtures/outputs/fmtblur_char_approximate_0.5.v 0.5 --precision approximate
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/step9_char.v fixtures/outputs/fmt_char_approximate_0.5.v --sigma 0.5 --precision approximate
../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/step9_char.v fixtures/outputs/fmtblur_char_approximate_0.2.v 0.2 --precision approximate
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/step9_char.v fixtures/outputs/fmt_char_approximate_0.2.v --sigma 0.2 --precision approximate
../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/step9_char.v fixtures/outputs/fmtblur_char_approximate_0.19.v 0.19 --precision approximate
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/step9_char.v fixtures/outputs/fmt_char_approximate_0.19.v --sigma 0.19 --precision approximate
../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/step9_char.v fixtures/outputs/fmtblur_char_approximate_0.1.v 0.1 --precision approximate
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/step9_char.v fixtures/outputs/fmt_char_approximate_0.1.v --sigma 0.1 --precision approximate
../../../../../../../../../../opt/homebrew/bin/vips cast fixtures/step9.pgm fixtures/step9_ushort.v ushort
../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/step9_ushort.v fixtures/outputs/fmtblur_ushort_integer_1.4.v 1.4 --precision integer
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/step9_ushort.v fixtures/outputs/fmt_ushort_integer_1.4.v --sigma 1.4 --precision integer
../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/step9_ushort.v fixtures/outputs/fmtblur_ushort_integer_0.5.v 0.5 --precision integer
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/step9_ushort.v fixtures/outputs/fmt_ushort_integer_0.5.v --sigma 0.5 --precision integer
../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/step9_ushort.v fixtures/outputs/fmtblur_ushort_integer_0.2.v 0.2 --precision integer
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/step9_ushort.v fixtures/outputs/fmt_ushort_integer_0.2.v --sigma 0.2 --precision integer
../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/step9_ushort.v fixtures/outputs/fmtblur_ushort_integer_0.19.v 0.19 --precision integer
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/step9_ushort.v fixtures/outputs/fmt_ushort_integer_0.19.v --sigma 0.19 --precision integer
../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/step9_ushort.v fixtures/outputs/fmtblur_ushort_integer_0.1.v 0.1 --precision integer
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/step9_ushort.v fixtures/outputs/fmt_ushort_integer_0.1.v --sigma 0.1 --precision integer
../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/step9_ushort.v fixtures/outputs/fmtblur_ushort_float_1.4.v 1.4 --precision float
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/step9_ushort.v fixtures/outputs/fmt_ushort_float_1.4.v --sigma 1.4 --precision float
../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/step9_ushort.v fixtures/outputs/fmtblur_ushort_float_0.5.v 0.5 --precision float
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/step9_ushort.v fixtures/outputs/fmt_ushort_float_0.5.v --sigma 0.5 --precision float
../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/step9_ushort.v fixtures/outputs/fmtblur_ushort_float_0.2.v 0.2 --precision float
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/step9_ushort.v fixtures/outputs/fmt_ushort_float_0.2.v --sigma 0.2 --precision float
../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/step9_ushort.v fixtures/outputs/fmtblur_ushort_float_0.19.v 0.19 --precision float
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/step9_ushort.v fixtures/outputs/fmt_ushort_float_0.19.v --sigma 0.19 --precision float
../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/step9_ushort.v fixtures/outputs/fmtblur_ushort_float_0.1.v 0.1 --precision float
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/step9_ushort.v fixtures/outputs/fmt_ushort_float_0.1.v --sigma 0.1 --precision float
../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/step9_ushort.v fixtures/outputs/fmtblur_ushort_approximate_1.4.v 1.4 --precision approximate
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/step9_ushort.v fixtures/outputs/fmt_ushort_approximate_1.4.v --sigma 1.4 --precision approximate
../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/step9_ushort.v fixtures/outputs/fmtblur_ushort_approximate_0.5.v 0.5 --precision approximate
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/step9_ushort.v fixtures/outputs/fmt_ushort_approximate_0.5.v --sigma 0.5 --precision approximate
../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/step9_ushort.v fixtures/outputs/fmtblur_ushort_approximate_0.2.v 0.2 --precision approximate
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/step9_ushort.v fixtures/outputs/fmt_ushort_approximate_0.2.v --sigma 0.2 --precision approximate
../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/step9_ushort.v fixtures/outputs/fmtblur_ushort_approximate_0.19.v 0.19 --precision approximate
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/step9_ushort.v fixtures/outputs/fmt_ushort_approximate_0.19.v --sigma 0.19 --precision approximate
../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/step9_ushort.v fixtures/outputs/fmtblur_ushort_approximate_0.1.v 0.1 --precision approximate
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/step9_ushort.v fixtures/outputs/fmt_ushort_approximate_0.1.v --sigma 0.1 --precision approximate
../../../../../../../../../../opt/homebrew/bin/vips cast fixtures/step9.pgm fixtures/step9_short.v short
../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/step9_short.v fixtures/outputs/fmtblur_short_integer_1.4.v 1.4 --precision integer
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/step9_short.v fixtures/outputs/fmt_short_integer_1.4.v --sigma 1.4 --precision integer
../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/step9_short.v fixtures/outputs/fmtblur_short_integer_0.5.v 0.5 --precision integer
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/step9_short.v fixtures/outputs/fmt_short_integer_0.5.v --sigma 0.5 --precision integer
../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/step9_short.v fixtures/outputs/fmtblur_short_integer_0.2.v 0.2 --precision integer
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/step9_short.v fixtures/outputs/fmt_short_integer_0.2.v --sigma 0.2 --precision integer
../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/step9_short.v fixtures/outputs/fmtblur_short_integer_0.19.v 0.19 --precision integer
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/step9_short.v fixtures/outputs/fmt_short_integer_0.19.v --sigma 0.19 --precision integer
../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/step9_short.v fixtures/outputs/fmtblur_short_integer_0.1.v 0.1 --precision integer
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/step9_short.v fixtures/outputs/fmt_short_integer_0.1.v --sigma 0.1 --precision integer
../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/step9_short.v fixtures/outputs/fmtblur_short_float_1.4.v 1.4 --precision float
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/step9_short.v fixtures/outputs/fmt_short_float_1.4.v --sigma 1.4 --precision float
../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/step9_short.v fixtures/outputs/fmtblur_short_float_0.5.v 0.5 --precision float
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/step9_short.v fixtures/outputs/fmt_short_float_0.5.v --sigma 0.5 --precision float
../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/step9_short.v fixtures/outputs/fmtblur_short_float_0.2.v 0.2 --precision float
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/step9_short.v fixtures/outputs/fmt_short_float_0.2.v --sigma 0.2 --precision float
../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/step9_short.v fixtures/outputs/fmtblur_short_float_0.19.v 0.19 --precision float
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/step9_short.v fixtures/outputs/fmt_short_float_0.19.v --sigma 0.19 --precision float
../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/step9_short.v fixtures/outputs/fmtblur_short_float_0.1.v 0.1 --precision float
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/step9_short.v fixtures/outputs/fmt_short_float_0.1.v --sigma 0.1 --precision float
../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/step9_short.v fixtures/outputs/fmtblur_short_approximate_1.4.v 1.4 --precision approximate
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/step9_short.v fixtures/outputs/fmt_short_approximate_1.4.v --sigma 1.4 --precision approximate
../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/step9_short.v fixtures/outputs/fmtblur_short_approximate_0.5.v 0.5 --precision approximate
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/step9_short.v fixtures/outputs/fmt_short_approximate_0.5.v --sigma 0.5 --precision approximate
../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/step9_short.v fixtures/outputs/fmtblur_short_approximate_0.2.v 0.2 --precision approximate
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/step9_short.v fixtures/outputs/fmt_short_approximate_0.2.v --sigma 0.2 --precision approximate
../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/step9_short.v fixtures/outputs/fmtblur_short_approximate_0.19.v 0.19 --precision approximate
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/step9_short.v fixtures/outputs/fmt_short_approximate_0.19.v --sigma 0.19 --precision approximate
../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/step9_short.v fixtures/outputs/fmtblur_short_approximate_0.1.v 0.1 --precision approximate
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/step9_short.v fixtures/outputs/fmt_short_approximate_0.1.v --sigma 0.1 --precision approximate
../../../../../../../../../../opt/homebrew/bin/vips cast fixtures/step9.pgm fixtures/step9_uint.v uint
../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/step9_uint.v fixtures/outputs/fmtblur_uint_integer_1.4.v 1.4 --precision integer
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/step9_uint.v fixtures/outputs/fmt_uint_integer_1.4.v --sigma 1.4 --precision integer
../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/step9_uint.v fixtures/outputs/fmtblur_uint_integer_0.5.v 0.5 --precision integer
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/step9_uint.v fixtures/outputs/fmt_uint_integer_0.5.v --sigma 0.5 --precision integer
../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/step9_uint.v fixtures/outputs/fmtblur_uint_integer_0.2.v 0.2 --precision integer
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/step9_uint.v fixtures/outputs/fmt_uint_integer_0.2.v --sigma 0.2 --precision integer
../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/step9_uint.v fixtures/outputs/fmtblur_uint_integer_0.19.v 0.19 --precision integer
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/step9_uint.v fixtures/outputs/fmt_uint_integer_0.19.v --sigma 0.19 --precision integer
../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/step9_uint.v fixtures/outputs/fmtblur_uint_integer_0.1.v 0.1 --precision integer
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/step9_uint.v fixtures/outputs/fmt_uint_integer_0.1.v --sigma 0.1 --precision integer
../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/step9_uint.v fixtures/outputs/fmtblur_uint_float_1.4.v 1.4 --precision float
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/step9_uint.v fixtures/outputs/fmt_uint_float_1.4.v --sigma 1.4 --precision float
../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/step9_uint.v fixtures/outputs/fmtblur_uint_float_0.5.v 0.5 --precision float
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/step9_uint.v fixtures/outputs/fmt_uint_float_0.5.v --sigma 0.5 --precision float
../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/step9_uint.v fixtures/outputs/fmtblur_uint_float_0.2.v 0.2 --precision float
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/step9_uint.v fixtures/outputs/fmt_uint_float_0.2.v --sigma 0.2 --precision float
../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/step9_uint.v fixtures/outputs/fmtblur_uint_float_0.19.v 0.19 --precision float
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/step9_uint.v fixtures/outputs/fmt_uint_float_0.19.v --sigma 0.19 --precision float
../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/step9_uint.v fixtures/outputs/fmtblur_uint_float_0.1.v 0.1 --precision float
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/step9_uint.v fixtures/outputs/fmt_uint_float_0.1.v --sigma 0.1 --precision float
../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/step9_uint.v fixtures/outputs/fmtblur_uint_approximate_1.4.v 1.4 --precision approximate
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/step9_uint.v fixtures/outputs/fmt_uint_approximate_1.4.v --sigma 1.4 --precision approximate
../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/step9_uint.v fixtures/outputs/fmtblur_uint_approximate_0.5.v 0.5 --precision approximate
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/step9_uint.v fixtures/outputs/fmt_uint_approximate_0.5.v --sigma 0.5 --precision approximate
../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/step9_uint.v fixtures/outputs/fmtblur_uint_approximate_0.2.v 0.2 --precision approximate
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/step9_uint.v fixtures/outputs/fmt_uint_approximate_0.2.v --sigma 0.2 --precision approximate
../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/step9_uint.v fixtures/outputs/fmtblur_uint_approximate_0.19.v 0.19 --precision approximate
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/step9_uint.v fixtures/outputs/fmt_uint_approximate_0.19.v --sigma 0.19 --precision approximate
../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/step9_uint.v fixtures/outputs/fmtblur_uint_approximate_0.1.v 0.1 --precision approximate
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/step9_uint.v fixtures/outputs/fmt_uint_approximate_0.1.v --sigma 0.1 --precision approximate
../../../../../../../../../../opt/homebrew/bin/vips cast fixtures/step9.pgm fixtures/step9_int.v int
../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/step9_int.v fixtures/outputs/fmtblur_int_integer_1.4.v 1.4 --precision integer
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/step9_int.v fixtures/outputs/fmt_int_integer_1.4.v --sigma 1.4 --precision integer
../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/step9_int.v fixtures/outputs/fmtblur_int_integer_0.5.v 0.5 --precision integer
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/step9_int.v fixtures/outputs/fmt_int_integer_0.5.v --sigma 0.5 --precision integer
../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/step9_int.v fixtures/outputs/fmtblur_int_integer_0.2.v 0.2 --precision integer
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/step9_int.v fixtures/outputs/fmt_int_integer_0.2.v --sigma 0.2 --precision integer
../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/step9_int.v fixtures/outputs/fmtblur_int_integer_0.19.v 0.19 --precision integer
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/step9_int.v fixtures/outputs/fmt_int_integer_0.19.v --sigma 0.19 --precision integer
../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/step9_int.v fixtures/outputs/fmtblur_int_integer_0.1.v 0.1 --precision integer
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/step9_int.v fixtures/outputs/fmt_int_integer_0.1.v --sigma 0.1 --precision integer
../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/step9_int.v fixtures/outputs/fmtblur_int_float_1.4.v 1.4 --precision float
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/step9_int.v fixtures/outputs/fmt_int_float_1.4.v --sigma 1.4 --precision float
../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/step9_int.v fixtures/outputs/fmtblur_int_float_0.5.v 0.5 --precision float
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/step9_int.v fixtures/outputs/fmt_int_float_0.5.v --sigma 0.5 --precision float
../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/step9_int.v fixtures/outputs/fmtblur_int_float_0.2.v 0.2 --precision float
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/step9_int.v fixtures/outputs/fmt_int_float_0.2.v --sigma 0.2 --precision float
../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/step9_int.v fixtures/outputs/fmtblur_int_float_0.19.v 0.19 --precision float
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/step9_int.v fixtures/outputs/fmt_int_float_0.19.v --sigma 0.19 --precision float
../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/step9_int.v fixtures/outputs/fmtblur_int_float_0.1.v 0.1 --precision float
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/step9_int.v fixtures/outputs/fmt_int_float_0.1.v --sigma 0.1 --precision float
../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/step9_int.v fixtures/outputs/fmtblur_int_approximate_1.4.v 1.4 --precision approximate
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/step9_int.v fixtures/outputs/fmt_int_approximate_1.4.v --sigma 1.4 --precision approximate
../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/step9_int.v fixtures/outputs/fmtblur_int_approximate_0.5.v 0.5 --precision approximate
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/step9_int.v fixtures/outputs/fmt_int_approximate_0.5.v --sigma 0.5 --precision approximate
../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/step9_int.v fixtures/outputs/fmtblur_int_approximate_0.2.v 0.2 --precision approximate
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/step9_int.v fixtures/outputs/fmt_int_approximate_0.2.v --sigma 0.2 --precision approximate
../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/step9_int.v fixtures/outputs/fmtblur_int_approximate_0.19.v 0.19 --precision approximate
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/step9_int.v fixtures/outputs/fmt_int_approximate_0.19.v --sigma 0.19 --precision approximate
../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/step9_int.v fixtures/outputs/fmtblur_int_approximate_0.1.v 0.1 --precision approximate
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/step9_int.v fixtures/outputs/fmt_int_approximate_0.1.v --sigma 0.1 --precision approximate
../../../../../../../../../../opt/homebrew/bin/vips cast fixtures/step9.pgm fixtures/step9_float.v float
../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/step9_float.v fixtures/outputs/fmtblur_float_integer_1.4.v 1.4 --precision integer
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/step9_float.v fixtures/outputs/fmt_float_integer_1.4.v --sigma 1.4 --precision integer
../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/step9_float.v fixtures/outputs/fmtblur_float_integer_0.5.v 0.5 --precision integer
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/step9_float.v fixtures/outputs/fmt_float_integer_0.5.v --sigma 0.5 --precision integer
../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/step9_float.v fixtures/outputs/fmtblur_float_integer_0.2.v 0.2 --precision integer
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/step9_float.v fixtures/outputs/fmt_float_integer_0.2.v --sigma 0.2 --precision integer
../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/step9_float.v fixtures/outputs/fmtblur_float_integer_0.19.v 0.19 --precision integer
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/step9_float.v fixtures/outputs/fmt_float_integer_0.19.v --sigma 0.19 --precision integer
../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/step9_float.v fixtures/outputs/fmtblur_float_integer_0.1.v 0.1 --precision integer
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/step9_float.v fixtures/outputs/fmt_float_integer_0.1.v --sigma 0.1 --precision integer
../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/step9_float.v fixtures/outputs/fmtblur_float_float_1.4.v 1.4 --precision float
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/step9_float.v fixtures/outputs/fmt_float_float_1.4.v --sigma 1.4 --precision float
../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/step9_float.v fixtures/outputs/fmtblur_float_float_0.5.v 0.5 --precision float
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/step9_float.v fixtures/outputs/fmt_float_float_0.5.v --sigma 0.5 --precision float
../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/step9_float.v fixtures/outputs/fmtblur_float_float_0.2.v 0.2 --precision float
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/step9_float.v fixtures/outputs/fmt_float_float_0.2.v --sigma 0.2 --precision float
../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/step9_float.v fixtures/outputs/fmtblur_float_float_0.19.v 0.19 --precision float
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/step9_float.v fixtures/outputs/fmt_float_float_0.19.v --sigma 0.19 --precision float
../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/step9_float.v fixtures/outputs/fmtblur_float_float_0.1.v 0.1 --precision float
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/step9_float.v fixtures/outputs/fmt_float_float_0.1.v --sigma 0.1 --precision float
../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/step9_float.v fixtures/outputs/fmtblur_float_approximate_1.4.v 1.4 --precision approximate
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/step9_float.v fixtures/outputs/fmt_float_approximate_1.4.v --sigma 1.4 --precision approximate
../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/step9_float.v fixtures/outputs/fmtblur_float_approximate_0.5.v 0.5 --precision approximate
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/step9_float.v fixtures/outputs/fmt_float_approximate_0.5.v --sigma 0.5 --precision approximate
../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/step9_float.v fixtures/outputs/fmtblur_float_approximate_0.2.v 0.2 --precision approximate
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/step9_float.v fixtures/outputs/fmt_float_approximate_0.2.v --sigma 0.2 --precision approximate
../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/step9_float.v fixtures/outputs/fmtblur_float_approximate_0.19.v 0.19 --precision approximate
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/step9_float.v fixtures/outputs/fmt_float_approximate_0.19.v --sigma 0.19 --precision approximate
../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/step9_float.v fixtures/outputs/fmtblur_float_approximate_0.1.v 0.1 --precision approximate
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/step9_float.v fixtures/outputs/fmt_float_approximate_0.1.v --sigma 0.1 --precision approximate
../../../../../../../../../../opt/homebrew/bin/vips cast fixtures/step9.pgm fixtures/step9_double.v double
../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/step9_double.v fixtures/outputs/fmtblur_double_integer_1.4.v 1.4 --precision integer
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/step9_double.v fixtures/outputs/fmt_double_integer_1.4.v --sigma 1.4 --precision integer
../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/step9_double.v fixtures/outputs/fmtblur_double_integer_0.5.v 0.5 --precision integer
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/step9_double.v fixtures/outputs/fmt_double_integer_0.5.v --sigma 0.5 --precision integer
../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/step9_double.v fixtures/outputs/fmtblur_double_integer_0.2.v 0.2 --precision integer
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/step9_double.v fixtures/outputs/fmt_double_integer_0.2.v --sigma 0.2 --precision integer
../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/step9_double.v fixtures/outputs/fmtblur_double_integer_0.19.v 0.19 --precision integer
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/step9_double.v fixtures/outputs/fmt_double_integer_0.19.v --sigma 0.19 --precision integer
../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/step9_double.v fixtures/outputs/fmtblur_double_integer_0.1.v 0.1 --precision integer
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/step9_double.v fixtures/outputs/fmt_double_integer_0.1.v --sigma 0.1 --precision integer
../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/step9_double.v fixtures/outputs/fmtblur_double_float_1.4.v 1.4 --precision float
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/step9_double.v fixtures/outputs/fmt_double_float_1.4.v --sigma 1.4 --precision float
../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/step9_double.v fixtures/outputs/fmtblur_double_float_0.5.v 0.5 --precision float
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/step9_double.v fixtures/outputs/fmt_double_float_0.5.v --sigma 0.5 --precision float
../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/step9_double.v fixtures/outputs/fmtblur_double_float_0.2.v 0.2 --precision float
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/step9_double.v fixtures/outputs/fmt_double_float_0.2.v --sigma 0.2 --precision float
../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/step9_double.v fixtures/outputs/fmtblur_double_float_0.19.v 0.19 --precision float
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/step9_double.v fixtures/outputs/fmt_double_float_0.19.v --sigma 0.19 --precision float
../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/step9_double.v fixtures/outputs/fmtblur_double_float_0.1.v 0.1 --precision float
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/step9_double.v fixtures/outputs/fmt_double_float_0.1.v --sigma 0.1 --precision float
../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/step9_double.v fixtures/outputs/fmtblur_double_approximate_1.4.v 1.4 --precision approximate
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/step9_double.v fixtures/outputs/fmt_double_approximate_1.4.v --sigma 1.4 --precision approximate
../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/step9_double.v fixtures/outputs/fmtblur_double_approximate_0.5.v 0.5 --precision approximate
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/step9_double.v fixtures/outputs/fmt_double_approximate_0.5.v --sigma 0.5 --precision approximate
../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/step9_double.v fixtures/outputs/fmtblur_double_approximate_0.2.v 0.2 --precision approximate
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/step9_double.v fixtures/outputs/fmt_double_approximate_0.2.v --sigma 0.2 --precision approximate
../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/step9_double.v fixtures/outputs/fmtblur_double_approximate_0.19.v 0.19 --precision approximate
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/step9_double.v fixtures/outputs/fmt_double_approximate_0.19.v --sigma 0.19 --precision approximate
../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/step9_double.v fixtures/outputs/fmtblur_double_approximate_0.1.v 0.1 --precision approximate
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/step9_double.v fixtures/outputs/fmt_double_approximate_0.1.v --sigma 0.1 --precision approximate
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/step9.v fixtures/outputs/override_uchar_s0.1_float_vector.v --sigma 0.1 --precision float
VIPS_NOVECTOR=1 ../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/step9.v fixtures/outputs/override_uchar_s0.1_float_novector.v --sigma 0.1 --precision float
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/step9.v fixtures/outputs/override_uchar_s0.1_integer_vector.v --sigma 0.1 --precision integer
VIPS_NOVECTOR=1 ../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/step9.v fixtures/outputs/override_uchar_s0.1_integer_novector.v --sigma 0.1 --precision integer
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/step9_float.v fixtures/outputs/override_float_s0.1_float_vector.v --sigma 0.1 --precision float
VIPS_NOVECTOR=1 ../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/step9_float.v fixtures/outputs/override_float_s0.1_float_novector.v --sigma 0.1 --precision float
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/step9_float.v fixtures/outputs/override_float_s0.1_integer_vector.v --sigma 0.1 --precision integer
VIPS_NOVECTOR=1 ../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/step9_float.v fixtures/outputs/override_float_s0.1_integer_novector.v --sigma 0.1 --precision integer
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/step9.v fixtures/outputs/override_uchar_s1.4_float_vector.v --sigma 1.4 --precision float
VIPS_NOVECTOR=1 ../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/step9.v fixtures/outputs/override_uchar_s1.4_float_novector.v --sigma 1.4 --precision float
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/step9.v fixtures/outputs/override_uchar_s1.4_integer_vector.v --sigma 1.4 --precision integer
VIPS_NOVECTOR=1 ../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/step9.v fixtures/outputs/override_uchar_s1.4_integer_novector.v --sigma 1.4 --precision integer
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/step9_float.v fixtures/outputs/override_float_s1.4_integer_vector.v --sigma 1.4 --precision integer
VIPS_NOVECTOR=1 ../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/step9_float.v fixtures/outputs/override_float_s1.4_integer_novector.v --sigma 1.4 --precision integer
../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/step9.v fixtures/outputs/override_stages_uchar_s0.1_float_blur_vector.v 0.1 --precision float
../../../../../../../../../../opt/homebrew/bin/vips conv fixtures/outputs/override_stages_uchar_s0.1_float_blur_vector.v fixtures/outputs/override_stages_uchar_s0.1_float_gx_vector.v fixtures/masks/gx_offset128.mat --precision integer
../../../../../../../../../../opt/homebrew/bin/vips conv fixtures/outputs/override_stages_uchar_s0.1_float_blur_vector.v fixtures/outputs/override_stages_uchar_s0.1_float_gy_vector.v fixtures/masks/gy_offset128.mat --precision integer
VIPS_NOVECTOR=1 ../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/step9.v fixtures/outputs/override_stages_uchar_s0.1_float_blur_novector.v 0.1 --precision float
VIPS_NOVECTOR=1 ../../../../../../../../../../opt/homebrew/bin/vips conv fixtures/outputs/override_stages_uchar_s0.1_float_blur_novector.v fixtures/outputs/override_stages_uchar_s0.1_float_gx_novector.v fixtures/masks/gx_offset128.mat --precision integer
VIPS_NOVECTOR=1 ../../../../../../../../../../opt/homebrew/bin/vips conv fixtures/outputs/override_stages_uchar_s0.1_float_blur_novector.v fixtures/outputs/override_stages_uchar_s0.1_float_gy_novector.v fixtures/masks/gy_offset128.mat --precision integer
../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/step9_float.v fixtures/outputs/override_stages_float_s0.1_float_blur_vector.v 0.1 --precision float
../../../../../../../../../../opt/homebrew/bin/vips conv fixtures/outputs/override_stages_float_s0.1_float_blur_vector.v fixtures/outputs/override_stages_float_s0.1_float_gx_vector.v fixtures/masks/gx.mat --precision float
../../../../../../../../../../opt/homebrew/bin/vips conv fixtures/outputs/override_stages_float_s0.1_float_blur_vector.v fixtures/outputs/override_stages_float_s0.1_float_gy_vector.v fixtures/masks/gy.mat --precision float
VIPS_NOVECTOR=1 ../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/step9_float.v fixtures/outputs/override_stages_float_s0.1_float_blur_novector.v 0.1 --precision float
VIPS_NOVECTOR=1 ../../../../../../../../../../opt/homebrew/bin/vips conv fixtures/outputs/override_stages_float_s0.1_float_blur_novector.v fixtures/outputs/override_stages_float_s0.1_float_gx_novector.v fixtures/masks/gx.mat --precision float
VIPS_NOVECTOR=1 ../../../../../../../../../../opt/homebrew/bin/vips conv fixtures/outputs/override_stages_float_s0.1_float_blur_novector.v fixtures/outputs/override_stages_float_s0.1_float_gy_novector.v fixtures/masks/gy.mat --precision float
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/square9.v fixtures/outputs/gmax_square9_uchar_vector.v --sigma 0.01 --precision integer
VIPS_NOVECTOR=1 ../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/square9.v fixtures/outputs/gmax_square9_uchar_novector.v --sigma 0.01 --precision integer
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/square9_float.v fixtures/outputs/gmax_square9_float_vector.v --sigma 0.01 --precision float
VIPS_NOVECTOR=1 ../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/square9_float.v fixtures/outputs/gmax_square9_float_novector.v --sigma 0.01 --precision float
../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/square9.v fixtures/outputs/gmax_stages_square9_uchar_blur_vector.v 0.01 --precision integer
../../../../../../../../../../opt/homebrew/bin/vips conv fixtures/outputs/gmax_stages_square9_uchar_blur_vector.v fixtures/outputs/gmax_stages_square9_uchar_gx_vector.v fixtures/masks/gx_offset128.mat --precision integer
../../../../../../../../../../opt/homebrew/bin/vips conv fixtures/outputs/gmax_stages_square9_uchar_blur_vector.v fixtures/outputs/gmax_stages_square9_uchar_gy_vector.v fixtures/masks/gy_offset128.mat --precision integer
VIPS_NOVECTOR=1 ../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/square9.v fixtures/outputs/gmax_stages_square9_uchar_blur_novector.v 0.01 --precision integer
VIPS_NOVECTOR=1 ../../../../../../../../../../opt/homebrew/bin/vips conv fixtures/outputs/gmax_stages_square9_uchar_blur_novector.v fixtures/outputs/gmax_stages_square9_uchar_gx_novector.v fixtures/masks/gx_offset128.mat --precision integer
VIPS_NOVECTOR=1 ../../../../../../../../../../opt/homebrew/bin/vips conv fixtures/outputs/gmax_stages_square9_uchar_blur_novector.v fixtures/outputs/gmax_stages_square9_uchar_gy_novector.v fixtures/masks/gy_offset128.mat --precision integer
../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/square9_float.v fixtures/outputs/gmax_stages_square9_float_blur_vector.v 0.01 --precision float
../../../../../../../../../../opt/homebrew/bin/vips conv fixtures/outputs/gmax_stages_square9_float_blur_vector.v fixtures/outputs/gmax_stages_square9_float_gx_vector.v fixtures/masks/gx.mat --precision float
../../../../../../../../../../opt/homebrew/bin/vips conv fixtures/outputs/gmax_stages_square9_float_blur_vector.v fixtures/outputs/gmax_stages_square9_float_gy_vector.v fixtures/masks/gy.mat --precision float
VIPS_NOVECTOR=1 ../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/square9_float.v fixtures/outputs/gmax_stages_square9_float_blur_novector.v 0.01 --precision float
VIPS_NOVECTOR=1 ../../../../../../../../../../opt/homebrew/bin/vips conv fixtures/outputs/gmax_stages_square9_float_blur_novector.v fixtures/outputs/gmax_stages_square9_float_gx_novector.v fixtures/masks/gx.mat --precision float
VIPS_NOVECTOR=1 ../../../../../../../../../../opt/homebrew/bin/vips conv fixtures/outputs/gmax_stages_square9_float_blur_novector.v fixtures/outputs/gmax_stages_square9_float_gy_novector.v fixtures/masks/gy.mat --precision float
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/noise64.v fixtures/outputs/gmax_noise64_uchar_vector.v --sigma 0.01 --precision integer
VIPS_NOVECTOR=1 ../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/noise64.v fixtures/outputs/gmax_noise64_uchar_novector.v --sigma 0.01 --precision integer
../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/disc33.v fixtures/outputs/orientation_disc33_uchar_blur_vector.v 1.4 --precision integer
../../../../../../../../../../opt/homebrew/bin/vips conv fixtures/outputs/orientation_disc33_uchar_blur_vector.v fixtures/outputs/orientation_disc33_uchar_gx_vector.v fixtures/masks/gx_offset128.mat --precision integer
../../../../../../../../../../opt/homebrew/bin/vips conv fixtures/outputs/orientation_disc33_uchar_blur_vector.v fixtures/outputs/orientation_disc33_uchar_gy_vector.v fixtures/masks/gy_offset128.mat --precision integer
VIPS_NOVECTOR=1 ../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/disc33.v fixtures/outputs/orientation_disc33_uchar_blur_novector.v 1.4 --precision integer
VIPS_NOVECTOR=1 ../../../../../../../../../../opt/homebrew/bin/vips conv fixtures/outputs/orientation_disc33_uchar_blur_novector.v fixtures/outputs/orientation_disc33_uchar_gx_novector.v fixtures/masks/gx_offset128.mat --precision integer
VIPS_NOVECTOR=1 ../../../../../../../../../../opt/homebrew/bin/vips conv fixtures/outputs/orientation_disc33_uchar_blur_novector.v fixtures/outputs/orientation_disc33_uchar_gy_novector.v fixtures/masks/gy_offset128.mat --precision integer
../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/disc33.v fixtures/outputs/orientation_disc33_float_blur_vector.v 1.4 --precision float
../../../../../../../../../../opt/homebrew/bin/vips conv fixtures/outputs/orientation_disc33_float_blur_vector.v fixtures/outputs/orientation_disc33_float_gx_vector.v fixtures/masks/gx.mat --precision float
../../../../../../../../../../opt/homebrew/bin/vips conv fixtures/outputs/orientation_disc33_float_blur_vector.v fixtures/outputs/orientation_disc33_float_gy_vector.v fixtures/masks/gy.mat --precision float
VIPS_NOVECTOR=1 ../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/disc33.v fixtures/outputs/orientation_disc33_float_blur_novector.v 1.4 --precision float
VIPS_NOVECTOR=1 ../../../../../../../../../../opt/homebrew/bin/vips conv fixtures/outputs/orientation_disc33_float_blur_novector.v fixtures/outputs/orientation_disc33_float_gx_novector.v fixtures/masks/gx.mat --precision float
VIPS_NOVECTOR=1 ../../../../../../../../../../opt/homebrew/bin/vips conv fixtures/outputs/orientation_disc33_float_blur_novector.v fixtures/outputs/orientation_disc33_float_gy_novector.v fixtures/masks/gy.mat --precision float
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/octants26.v fixtures/outputs/octants_uchar_vector.v --sigma 0.01 --precision integer
VIPS_NOVECTOR=1 ../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/octants26.v fixtures/outputs/octants_uchar_novector.v --sigma 0.01 --precision integer
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/octants26_float.v fixtures/outputs/octants_float_vector.v --sigma 0.01 --precision float
VIPS_NOVECTOR=1 ../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/octants26_float.v fixtures/outputs/octants_float_novector.v --sigma 0.01 --precision float
../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/octants26.v fixtures/outputs/octants_stages_uchar_blur_vector.v 0.01 --precision integer
../../../../../../../../../../opt/homebrew/bin/vips conv fixtures/outputs/octants_stages_uchar_blur_vector.v fixtures/outputs/octants_stages_uchar_gx_vector.v fixtures/masks/gx_offset128.mat --precision integer
../../../../../../../../../../opt/homebrew/bin/vips conv fixtures/outputs/octants_stages_uchar_blur_vector.v fixtures/outputs/octants_stages_uchar_gy_vector.v fixtures/masks/gy_offset128.mat --precision integer
VIPS_NOVECTOR=1 ../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/octants26.v fixtures/outputs/octants_stages_uchar_blur_novector.v 0.01 --precision integer
VIPS_NOVECTOR=1 ../../../../../../../../../../opt/homebrew/bin/vips conv fixtures/outputs/octants_stages_uchar_blur_novector.v fixtures/outputs/octants_stages_uchar_gx_novector.v fixtures/masks/gx_offset128.mat --precision integer
VIPS_NOVECTOR=1 ../../../../../../../../../../opt/homebrew/bin/vips conv fixtures/outputs/octants_stages_uchar_blur_novector.v fixtures/outputs/octants_stages_uchar_gy_novector.v fixtures/masks/gy_offset128.mat --precision integer
../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/octants26_float.v fixtures/outputs/octants_stages_float_blur_vector.v 0.01 --precision float
../../../../../../../../../../opt/homebrew/bin/vips conv fixtures/outputs/octants_stages_float_blur_vector.v fixtures/outputs/octants_stages_float_gx_vector.v fixtures/masks/gx.mat --precision float
../../../../../../../../../../opt/homebrew/bin/vips conv fixtures/outputs/octants_stages_float_blur_vector.v fixtures/outputs/octants_stages_float_gy_vector.v fixtures/masks/gy.mat --precision float
VIPS_NOVECTOR=1 ../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/octants26_float.v fixtures/outputs/octants_stages_float_blur_novector.v 0.01 --precision float
VIPS_NOVECTOR=1 ../../../../../../../../../../opt/homebrew/bin/vips conv fixtures/outputs/octants_stages_float_blur_novector.v fixtures/outputs/octants_stages_float_gx_novector.v fixtures/masks/gx.mat --precision float
VIPS_NOVECTOR=1 ../../../../../../../../../../opt/homebrew/bin/vips conv fixtures/outputs/octants_stages_float_blur_novector.v fixtures/outputs/octants_stages_float_gy_novector.v fixtures/masks/gy.mat --precision float
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/plateau_h.v fixtures/outputs/suppress_plateau_h_uchar_vector.v --sigma 0.01 --precision integer
VIPS_NOVECTOR=1 ../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/plateau_h.v fixtures/outputs/suppress_plateau_h_uchar_novector.v --sigma 0.01 --precision integer
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/plateau_h.v fixtures/outputs/suppress_plateau_h_float_vector.v --sigma 0.01 --precision float
VIPS_NOVECTOR=1 ../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/plateau_h.v fixtures/outputs/suppress_plateau_h_float_novector.v --sigma 0.01 --precision float
../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/plateau_h.v fixtures/outputs/suppress_stages_plateau_h_uchar_blur_vector.v 0.01 --precision integer
../../../../../../../../../../opt/homebrew/bin/vips conv fixtures/outputs/suppress_stages_plateau_h_uchar_blur_vector.v fixtures/outputs/suppress_stages_plateau_h_uchar_gx_vector.v fixtures/masks/gx_offset128.mat --precision integer
../../../../../../../../../../opt/homebrew/bin/vips conv fixtures/outputs/suppress_stages_plateau_h_uchar_blur_vector.v fixtures/outputs/suppress_stages_plateau_h_uchar_gy_vector.v fixtures/masks/gy_offset128.mat --precision integer
VIPS_NOVECTOR=1 ../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/plateau_h.v fixtures/outputs/suppress_stages_plateau_h_uchar_blur_novector.v 0.01 --precision integer
VIPS_NOVECTOR=1 ../../../../../../../../../../opt/homebrew/bin/vips conv fixtures/outputs/suppress_stages_plateau_h_uchar_blur_novector.v fixtures/outputs/suppress_stages_plateau_h_uchar_gx_novector.v fixtures/masks/gx_offset128.mat --precision integer
VIPS_NOVECTOR=1 ../../../../../../../../../../opt/homebrew/bin/vips conv fixtures/outputs/suppress_stages_plateau_h_uchar_blur_novector.v fixtures/outputs/suppress_stages_plateau_h_uchar_gy_novector.v fixtures/masks/gy_offset128.mat --precision integer
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/plateau_h_rev.v fixtures/outputs/suppress_plateau_h_rev_uchar_vector.v --sigma 0.01 --precision integer
VIPS_NOVECTOR=1 ../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/plateau_h_rev.v fixtures/outputs/suppress_plateau_h_rev_uchar_novector.v --sigma 0.01 --precision integer
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/plateau_h_rev.v fixtures/outputs/suppress_plateau_h_rev_float_vector.v --sigma 0.01 --precision float
VIPS_NOVECTOR=1 ../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/plateau_h_rev.v fixtures/outputs/suppress_plateau_h_rev_float_novector.v --sigma 0.01 --precision float
../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/plateau_h_rev.v fixtures/outputs/suppress_stages_plateau_h_rev_uchar_blur_vector.v 0.01 --precision integer
../../../../../../../../../../opt/homebrew/bin/vips conv fixtures/outputs/suppress_stages_plateau_h_rev_uchar_blur_vector.v fixtures/outputs/suppress_stages_plateau_h_rev_uchar_gx_vector.v fixtures/masks/gx_offset128.mat --precision integer
../../../../../../../../../../opt/homebrew/bin/vips conv fixtures/outputs/suppress_stages_plateau_h_rev_uchar_blur_vector.v fixtures/outputs/suppress_stages_plateau_h_rev_uchar_gy_vector.v fixtures/masks/gy_offset128.mat --precision integer
VIPS_NOVECTOR=1 ../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/plateau_h_rev.v fixtures/outputs/suppress_stages_plateau_h_rev_uchar_blur_novector.v 0.01 --precision integer
VIPS_NOVECTOR=1 ../../../../../../../../../../opt/homebrew/bin/vips conv fixtures/outputs/suppress_stages_plateau_h_rev_uchar_blur_novector.v fixtures/outputs/suppress_stages_plateau_h_rev_uchar_gx_novector.v fixtures/masks/gx_offset128.mat --precision integer
VIPS_NOVECTOR=1 ../../../../../../../../../../opt/homebrew/bin/vips conv fixtures/outputs/suppress_stages_plateau_h_rev_uchar_blur_novector.v fixtures/outputs/suppress_stages_plateau_h_rev_uchar_gy_novector.v fixtures/masks/gy_offset128.mat --precision integer
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/plateau_v.v fixtures/outputs/suppress_plateau_v_uchar_vector.v --sigma 0.01 --precision integer
VIPS_NOVECTOR=1 ../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/plateau_v.v fixtures/outputs/suppress_plateau_v_uchar_novector.v --sigma 0.01 --precision integer
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/plateau_v.v fixtures/outputs/suppress_plateau_v_float_vector.v --sigma 0.01 --precision float
VIPS_NOVECTOR=1 ../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/plateau_v.v fixtures/outputs/suppress_plateau_v_float_novector.v --sigma 0.01 --precision float
../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/plateau_v.v fixtures/outputs/suppress_stages_plateau_v_uchar_blur_vector.v 0.01 --precision integer
../../../../../../../../../../opt/homebrew/bin/vips conv fixtures/outputs/suppress_stages_plateau_v_uchar_blur_vector.v fixtures/outputs/suppress_stages_plateau_v_uchar_gx_vector.v fixtures/masks/gx_offset128.mat --precision integer
../../../../../../../../../../opt/homebrew/bin/vips conv fixtures/outputs/suppress_stages_plateau_v_uchar_blur_vector.v fixtures/outputs/suppress_stages_plateau_v_uchar_gy_vector.v fixtures/masks/gy_offset128.mat --precision integer
VIPS_NOVECTOR=1 ../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/plateau_v.v fixtures/outputs/suppress_stages_plateau_v_uchar_blur_novector.v 0.01 --precision integer
VIPS_NOVECTOR=1 ../../../../../../../../../../opt/homebrew/bin/vips conv fixtures/outputs/suppress_stages_plateau_v_uchar_blur_novector.v fixtures/outputs/suppress_stages_plateau_v_uchar_gx_novector.v fixtures/masks/gx_offset128.mat --precision integer
VIPS_NOVECTOR=1 ../../../../../../../../../../opt/homebrew/bin/vips conv fixtures/outputs/suppress_stages_plateau_v_uchar_blur_novector.v fixtures/outputs/suppress_stages_plateau_v_uchar_gy_novector.v fixtures/masks/gy_offset128.mat --precision integer
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/plateau_v_rev.v fixtures/outputs/suppress_plateau_v_rev_uchar_vector.v --sigma 0.01 --precision integer
VIPS_NOVECTOR=1 ../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/plateau_v_rev.v fixtures/outputs/suppress_plateau_v_rev_uchar_novector.v --sigma 0.01 --precision integer
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/plateau_v_rev.v fixtures/outputs/suppress_plateau_v_rev_float_vector.v --sigma 0.01 --precision float
VIPS_NOVECTOR=1 ../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/plateau_v_rev.v fixtures/outputs/suppress_plateau_v_rev_float_novector.v --sigma 0.01 --precision float
../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/plateau_v_rev.v fixtures/outputs/suppress_stages_plateau_v_rev_uchar_blur_vector.v 0.01 --precision integer
../../../../../../../../../../opt/homebrew/bin/vips conv fixtures/outputs/suppress_stages_plateau_v_rev_uchar_blur_vector.v fixtures/outputs/suppress_stages_plateau_v_rev_uchar_gx_vector.v fixtures/masks/gx_offset128.mat --precision integer
../../../../../../../../../../opt/homebrew/bin/vips conv fixtures/outputs/suppress_stages_plateau_v_rev_uchar_blur_vector.v fixtures/outputs/suppress_stages_plateau_v_rev_uchar_gy_vector.v fixtures/masks/gy_offset128.mat --precision integer
VIPS_NOVECTOR=1 ../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/plateau_v_rev.v fixtures/outputs/suppress_stages_plateau_v_rev_uchar_blur_novector.v 0.01 --precision integer
VIPS_NOVECTOR=1 ../../../../../../../../../../opt/homebrew/bin/vips conv fixtures/outputs/suppress_stages_plateau_v_rev_uchar_blur_novector.v fixtures/outputs/suppress_stages_plateau_v_rev_uchar_gx_novector.v fixtures/masks/gx_offset128.mat --precision integer
VIPS_NOVECTOR=1 ../../../../../../../../../../opt/homebrew/bin/vips conv fixtures/outputs/suppress_stages_plateau_v_rev_uchar_blur_novector.v fixtures/outputs/suppress_stages_plateau_v_rev_uchar_gy_novector.v fixtures/masks/gy_offset128.mat --precision integer
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/border7.v fixtures/outputs/border7_uchar_vector.v --sigma 0.01 --precision integer
VIPS_NOVECTOR=1 ../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/border7.v fixtures/outputs/border7_uchar_novector.v --sigma 0.01 --precision integer
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/border7.v fixtures/outputs/border7_float_vector.v --sigma 1.4 --precision float
VIPS_NOVECTOR=1 ../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/border7.v fixtures/outputs/border7_float_novector.v --sigma 1.4 --precision float
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/step9.v fixtures/outputs/border_step9_float_vector.v --sigma 1.4 --precision float
VIPS_NOVECTOR=1 ../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/step9.v fixtures/outputs/border_step9_float_novector.v --sigma 1.4 --precision float
../../../../../../../../../../opt/homebrew/bin/vips gaussmat fixtures/outputs/gaussmat_0.01_integer.mat 0.01 0.2 --separable --precision integer
../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/step9.v fixtures/outputs/sigblur_0.01_integer.v 0.01 --precision integer
../../../../../../../../../../opt/homebrew/bin/vips gaussmat fixtures/outputs/gaussmat_0.01_float.mat 0.01 0.2 --separable --precision float
../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/step9.v fixtures/outputs/sigblur_0.01_float.v 0.01 --precision float
../../../../../../../../../../opt/homebrew/bin/vips gaussmat fixtures/outputs/gaussmat_0.1_integer.mat 0.1 0.2 --separable --precision integer
../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/step9.v fixtures/outputs/sigblur_0.1_integer.v 0.1 --precision integer
../../../../../../../../../../opt/homebrew/bin/vips gaussmat fixtures/outputs/gaussmat_0.1_float.mat 0.1 0.2 --separable --precision float
../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/step9.v fixtures/outputs/sigblur_0.1_float.v 0.1 --precision float
../../../../../../../../../../opt/homebrew/bin/vips gaussmat fixtures/outputs/gaussmat_0.19_integer.mat 0.19 0.2 --separable --precision integer
../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/step9.v fixtures/outputs/sigblur_0.19_integer.v 0.19 --precision integer
../../../../../../../../../../opt/homebrew/bin/vips gaussmat fixtures/outputs/gaussmat_0.19_float.mat 0.19 0.2 --separable --precision float
../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/step9.v fixtures/outputs/sigblur_0.19_float.v 0.19 --precision float
../../../../../../../../../../opt/homebrew/bin/vips gaussmat fixtures/outputs/gaussmat_0.2_integer.mat 0.2 0.2 --separable --precision integer
../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/step9.v fixtures/outputs/sigblur_0.2_integer.v 0.2 --precision integer
../../../../../../../../../../opt/homebrew/bin/vips gaussmat fixtures/outputs/gaussmat_0.2_float.mat 0.2 0.2 --separable --precision float
../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/step9.v fixtures/outputs/sigblur_0.2_float.v 0.2 --precision float
../../../../../../../../../../opt/homebrew/bin/vips gaussmat fixtures/outputs/gaussmat_0.5_integer.mat 0.5 0.2 --separable --precision integer
../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/step9.v fixtures/outputs/sigblur_0.5_integer.v 0.5 --precision integer
../../../../../../../../../../opt/homebrew/bin/vips gaussmat fixtures/outputs/gaussmat_0.5_float.mat 0.5 0.2 --separable --precision float
../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/step9.v fixtures/outputs/sigblur_0.5_float.v 0.5 --precision float
../../../../../../../../../../opt/homebrew/bin/vips gaussmat fixtures/outputs/gaussmat_0.8_integer.mat 0.8 0.2 --separable --precision integer
../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/step9.v fixtures/outputs/sigblur_0.8_integer.v 0.8 --precision integer
../../../../../../../../../../opt/homebrew/bin/vips gaussmat fixtures/outputs/gaussmat_0.8_float.mat 0.8 0.2 --separable --precision float
../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/step9.v fixtures/outputs/sigblur_0.8_float.v 0.8 --precision float
../../../../../../../../../../opt/homebrew/bin/vips gaussmat fixtures/outputs/gaussmat_1.0_integer.mat 1.0 0.2 --separable --precision integer
../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/step9.v fixtures/outputs/sigblur_1.0_integer.v 1.0 --precision integer
../../../../../../../../../../opt/homebrew/bin/vips gaussmat fixtures/outputs/gaussmat_1.0_float.mat 1.0 0.2 --separable --precision float
../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/step9.v fixtures/outputs/sigblur_1.0_float.v 1.0 --precision float
../../../../../../../../../../opt/homebrew/bin/vips gaussmat fixtures/outputs/gaussmat_1.2_integer.mat 1.2 0.2 --separable --precision integer
../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/step9.v fixtures/outputs/sigblur_1.2_integer.v 1.2 --precision integer
../../../../../../../../../../opt/homebrew/bin/vips gaussmat fixtures/outputs/gaussmat_1.2_float.mat 1.2 0.2 --separable --precision float
../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/step9.v fixtures/outputs/sigblur_1.2_float.v 1.2 --precision float
../../../../../../../../../../opt/homebrew/bin/vips gaussmat fixtures/outputs/gaussmat_1.4_integer.mat 1.4 0.2 --separable --precision integer
../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/step9.v fixtures/outputs/sigblur_1.4_integer.v 1.4 --precision integer
../../../../../../../../../../opt/homebrew/bin/vips gaussmat fixtures/outputs/gaussmat_1.4_float.mat 1.4 0.2 --separable --precision float
../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/step9.v fixtures/outputs/sigblur_1.4_float.v 1.4 --precision float
../../../../../../../../../../opt/homebrew/bin/vips gaussmat fixtures/outputs/gaussmat_1.6_integer.mat 1.6 0.2 --separable --precision integer
../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/step9.v fixtures/outputs/sigblur_1.6_integer.v 1.6 --precision integer
../../../../../../../../../../opt/homebrew/bin/vips gaussmat fixtures/outputs/gaussmat_1.6_float.mat 1.6 0.2 --separable --precision float
../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/step9.v fixtures/outputs/sigblur_1.6_float.v 1.6 --precision float
../../../../../../../../../../opt/homebrew/bin/vips gaussmat fixtures/outputs/gaussmat_1.8_integer.mat 1.8 0.2 --separable --precision integer
../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/step9.v fixtures/outputs/sigblur_1.8_integer.v 1.8 --precision integer
../../../../../../../../../../opt/homebrew/bin/vips gaussmat fixtures/outputs/gaussmat_1.8_float.mat 1.8 0.2 --separable --precision float
../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/step9.v fixtures/outputs/sigblur_1.8_float.v 1.8 --precision float
../../../../../../../../../../opt/homebrew/bin/vips gaussmat fixtures/outputs/gaussmat_2.0_integer.mat 2.0 0.2 --separable --precision integer
../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/step9.v fixtures/outputs/sigblur_2.0_integer.v 2.0 --precision integer
../../../../../../../../../../opt/homebrew/bin/vips gaussmat fixtures/outputs/gaussmat_2.0_float.mat 2.0 0.2 --separable --precision float
../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/step9.v fixtures/outputs/sigblur_2.0_float.v 2.0 --precision float
../../../../../../../../../../opt/homebrew/bin/vips gaussmat fixtures/outputs/gaussmat_2.5_integer.mat 2.5 0.2 --separable --precision integer
../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/step9.v fixtures/outputs/sigblur_2.5_integer.v 2.5 --precision integer
../../../../../../../../../../opt/homebrew/bin/vips gaussmat fixtures/outputs/gaussmat_2.5_float.mat 2.5 0.2 --separable --precision float
../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/step9.v fixtures/outputs/sigblur_2.5_float.v 2.5 --precision float
../../../../../../../../../../opt/homebrew/bin/vips gaussmat fixtures/outputs/gaussmat_3.0_integer.mat 3.0 0.2 --separable --precision integer
../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/step9.v fixtures/outputs/sigblur_3.0_integer.v 3.0 --precision integer
../../../../../../../../../../opt/homebrew/bin/vips gaussmat fixtures/outputs/gaussmat_3.0_float.mat 3.0 0.2 --separable --precision float
../../../../../../../../../../opt/homebrew/bin/vips gaussblur fixtures/step9.v fixtures/outputs/sigblur_3.0_float.v 3.0 --precision float
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/step9.v fixtures/outputs/sigma_step9_0.01_integer_vector.v --sigma 0.01 --precision integer
VIPS_NOVECTOR=1 ../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/step9.v fixtures/outputs/sigma_step9_0.01_integer_novector.v --sigma 0.01 --precision integer
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/step9.v fixtures/outputs/sigma_step9_0.01_float_vector.v --sigma 0.01 --precision float
VIPS_NOVECTOR=1 ../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/step9.v fixtures/outputs/sigma_step9_0.01_float_novector.v --sigma 0.01 --precision float
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/step9.v fixtures/outputs/sigma_step9_0.1_integer_vector.v --sigma 0.1 --precision integer
VIPS_NOVECTOR=1 ../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/step9.v fixtures/outputs/sigma_step9_0.1_integer_novector.v --sigma 0.1 --precision integer
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/step9.v fixtures/outputs/sigma_step9_0.1_float_vector.v --sigma 0.1 --precision float
VIPS_NOVECTOR=1 ../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/step9.v fixtures/outputs/sigma_step9_0.1_float_novector.v --sigma 0.1 --precision float
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/step9.v fixtures/outputs/sigma_step9_0.19_integer_vector.v --sigma 0.19 --precision integer
VIPS_NOVECTOR=1 ../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/step9.v fixtures/outputs/sigma_step9_0.19_integer_novector.v --sigma 0.19 --precision integer
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/step9.v fixtures/outputs/sigma_step9_0.19_float_vector.v --sigma 0.19 --precision float
VIPS_NOVECTOR=1 ../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/step9.v fixtures/outputs/sigma_step9_0.19_float_novector.v --sigma 0.19 --precision float
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/step9.v fixtures/outputs/sigma_step9_0.2_integer_vector.v --sigma 0.2 --precision integer
VIPS_NOVECTOR=1 ../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/step9.v fixtures/outputs/sigma_step9_0.2_integer_novector.v --sigma 0.2 --precision integer
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/step9.v fixtures/outputs/sigma_step9_0.2_float_vector.v --sigma 0.2 --precision float
VIPS_NOVECTOR=1 ../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/step9.v fixtures/outputs/sigma_step9_0.2_float_novector.v --sigma 0.2 --precision float
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/step9.v fixtures/outputs/sigma_step9_1.4_integer_vector.v --sigma 1.4 --precision integer
VIPS_NOVECTOR=1 ../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/step9.v fixtures/outputs/sigma_step9_1.4_integer_novector.v --sigma 1.4 --precision integer
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/step9.v fixtures/outputs/sigma_step9_1.4_float_vector.v --sigma 1.4 --precision float
VIPS_NOVECTOR=1 ../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/step9.v fixtures/outputs/sigma_step9_1.4_float_novector.v --sigma 1.4 --precision float
vips canny fixtures/step9.v <out> --sigma 0   # exit 0
vips canny fixtures/step9.v <out> --sigma 0.009   # exit 0
vips canny fixtures/step9.v <out> --sigma -1   # exit 0
vips canny fixtures/step9.v <out> --sigma 1000.1   # exit 0
vips canny fixtures/step9.v <out> --sigma 1000   # exit 0
vips canny fixtures/step9.v <out> --sigma 1.4   # exit 0
../../../../../../../../../../opt/homebrew/bin/vips copy fixtures/noise16rgb.v fixtures/noise16rgb_srgb.v --interpretation srgb
../../../../../../../../../../opt/homebrew/bin/vips copy fixtures/noise64.v fixtures/noise64_bw.v --interpretation b-w
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/noise16rgb.v fixtures/outputs/rt_noise16rgb_integer.v --precision integer
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/noise16rgb.v fixtures/outputs/rt_noise16rgb_float.v --precision float
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/noise16rgb_srgb.v fixtures/outputs/rt_noise16rgb_srgb_integer.v --precision integer
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/noise16rgb_srgb.v fixtures/outputs/rt_noise16rgb_srgb_float.v --precision float
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/noise64.v fixtures/outputs/rt_noise64_integer.v --precision integer
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/noise64.v fixtures/outputs/rt_noise64_float.v --precision float
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/noise64_bw.v fixtures/outputs/rt_noise64_bw_integer.v --precision integer
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/noise64_bw.v fixtures/outputs/rt_noise64_bw_float.v --precision float
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/noise16rgb_float.v fixtures/outputs/rt_noise16rgb_float_integer.v --precision integer
../../../../../../../../../../opt/homebrew/bin/vips canny fixtures/noise16rgb_float.v fixtures/outputs/rt_noise16rgb_float_float.v --precision float
# vector/scalar sweep: for every (input, precision, sigma) below, both
#   vips canny <in> <out> --sigma S --precision P
#   VIPS_NOVECTOR=1 vips canny <in> <out> --sigma S --precision P
# see oracle.json -> vector_scalar_sweep for the pairs and the diff counts.
