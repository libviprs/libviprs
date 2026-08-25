#!/bin/sh
# Every vips command capture.py ran, in order. Regenerate with
# `python3 capture.py` from this directory.
set -e

/opt/homebrew/bin/vips --version
/opt/homebrew/bin/vipsheader -a /Users/rom/workspace/libviprs/libviprs-tests/tmp/libvips-reference-tests/test-suite/images/cogs.gif
/opt/homebrew/bin/vips gifload /Users/rom/workspace/libviprs/libviprs-tests/tmp/libvips-reference-tests/test-suite/images/cogs.gif outputs/ref-cogs.v
/opt/homebrew/bin/vips rawsave outputs/ref-cogs.v outputs/ref-cogs.raw
/opt/homebrew/bin/vipsheader -a /Users/rom/workspace/libviprs/libviprs-tests/tmp/libvips-reference-tests/test-suite/images/cramps.gif
/opt/homebrew/bin/vips gifload /Users/rom/workspace/libviprs/libviprs-tests/tmp/libvips-reference-tests/test-suite/images/cramps.gif outputs/ref-cramps.v
/opt/homebrew/bin/vips rawsave outputs/ref-cramps.v outputs/ref-cramps.raw
/opt/homebrew/bin/vipsheader -a /Users/rom/workspace/libviprs/libviprs-tests/tmp/libvips-reference-tests/test-suite/images/trans-x.gif
/opt/homebrew/bin/vips gifload /Users/rom/workspace/libviprs/libviprs-tests/tmp/libvips-reference-tests/test-suite/images/trans-x.gif outputs/ref-trans-x.v
/opt/homebrew/bin/vips rawsave outputs/ref-trans-x.v outputs/ref-trans-x.raw
/opt/homebrew/bin/vipsheader -a /Users/rom/workspace/libviprs/libviprs-tests/tmp/libvips-reference-tests/test-suite/images/truncated.gif
/opt/homebrew/bin/vips gifload /Users/rom/workspace/libviprs/libviprs-tests/tmp/libvips-reference-tests/test-suite/images/truncated.gif outputs/ref-truncated.v
/opt/homebrew/bin/vips rawsave outputs/ref-truncated.v outputs/ref-truncated.raw
/opt/homebrew/bin/vipsheader -a /Users/rom/workspace/libviprs/libviprs-tests/tmp/libvips-reference-tests/test-suite/images/garden.gif
/opt/homebrew/bin/vips gifload /Users/rom/workspace/libviprs/libviprs-tests/tmp/libvips-reference-tests/test-suite/images/garden.gif outputs/ref-garden.v
/opt/homebrew/bin/vips rawsave outputs/ref-garden.v outputs/ref-garden.raw
/opt/homebrew/bin/vipsheader -a /Users/rom/workspace/libviprs/libviprs-tests/tmp/libvips-reference-tests/test-suite/images/dispose-background.gif
/opt/homebrew/bin/vips gifload /Users/rom/workspace/libviprs/libviprs-tests/tmp/libvips-reference-tests/test-suite/images/dispose-background.gif outputs/ref-dispose-background.v
/opt/homebrew/bin/vips rawsave outputs/ref-dispose-background.v outputs/ref-dispose-background.raw
/opt/homebrew/bin/vipsheader -a /Users/rom/workspace/libviprs/libviprs-tests/tmp/libvips-reference-tests/test-suite/images/dispose-previous.gif
/opt/homebrew/bin/vips gifload /Users/rom/workspace/libviprs/libviprs-tests/tmp/libvips-reference-tests/test-suite/images/dispose-previous.gif outputs/ref-dispose-previous.v
/opt/homebrew/bin/vips rawsave outputs/ref-dispose-previous.v outputs/ref-dispose-previous.raw
/opt/homebrew/bin/vipsheader -a /Users/rom/workspace/libviprs/libviprs-tests/tmp/libvips-reference-tests/test-suite/images/invalid_multiframe.gif
/opt/homebrew/bin/vips gifload /Users/rom/workspace/libviprs/libviprs-tests/tmp/libvips-reference-tests/test-suite/images/invalid_multiframe.gif outputs/ref-invalid_multiframe.v
/opt/homebrew/bin/vips rawsave outputs/ref-invalid_multiframe.v outputs/ref-invalid_multiframe.raw
/opt/homebrew/bin/vipsheader -a fixtures/inset.gif
/opt/homebrew/bin/vips gifload fixtures/inset.gif outputs/inset.v
/opt/homebrew/bin/vips rawsave outputs/inset.v outputs/inset.raw
/opt/homebrew/bin/vips rawload fixtures/rows8.raw outputs/rows8-plain.v 8 8 3 --format uchar
/opt/homebrew/bin/vips copy outputs/rows8-plain.v outputs/rows8.v --interpretation srgb
/opt/homebrew/bin/vips gifsave outputs/rows8.v outputs/rows8.gif --bitdepth 3
/opt/homebrew/bin/vipsheader -a outputs/rows8.gif
/opt/homebrew/bin/vipsheader -a fixtures/rows8-truncated.gif
/opt/homebrew/bin/vips rawload fixtures/cycle256.raw outputs/cycle256-plain.v 16 16 3 --format uchar
/opt/homebrew/bin/vips copy outputs/cycle256-plain.v outputs/cycle256.v --interpretation srgb
/opt/homebrew/bin/vips gifsave outputs/cycle256.v outputs/bitdepth1.gif --bitdepth 1
/opt/homebrew/bin/vips gifsave outputs/cycle256.v outputs/bitdepth2.gif --bitdepth 2
/opt/homebrew/bin/vips gifsave outputs/cycle256.v outputs/bitdepth3.gif --bitdepth 3
/opt/homebrew/bin/vips gifsave outputs/cycle256.v outputs/bitdepth4.gif --bitdepth 4
/opt/homebrew/bin/vips gifsave outputs/cycle256.v outputs/bitdepth5.gif --bitdepth 5
/opt/homebrew/bin/vips gifsave outputs/cycle256.v outputs/bitdepth6.gif --bitdepth 6
/opt/homebrew/bin/vips gifsave outputs/cycle256.v outputs/bitdepth7.gif --bitdepth 7
/opt/homebrew/bin/vips gifsave outputs/cycle256.v outputs/bitdepth8.gif --bitdepth 8
/opt/homebrew/bin/vips rawload fixtures/cycle2.raw outputs/cycle2-plain.v 16 16 3 --format uchar
/opt/homebrew/bin/vips copy outputs/cycle2-plain.v outputs/cycle2.v --interpretation srgb
/opt/homebrew/bin/vips gifsave outputs/cycle2.v outputs/reserve-8-2.gif --bitdepth 8
/opt/homebrew/bin/vipsheader -a outputs/reserve-8-2.gif
/opt/homebrew/bin/vips rawload fixtures/cycle16.raw outputs/cycle16-plain.v 16 16 3 --format uchar
/opt/homebrew/bin/vips copy outputs/cycle16-plain.v outputs/cycle16.v --interpretation srgb
/opt/homebrew/bin/vips gifsave outputs/cycle16.v outputs/reserve-8-16.gif --bitdepth 8
/opt/homebrew/bin/vipsheader -a outputs/reserve-8-16.gif
/opt/homebrew/bin/vips rawload fixtures/cycle254.raw outputs/cycle254-plain.v 16 16 3 --format uchar
/opt/homebrew/bin/vips copy outputs/cycle254-plain.v outputs/cycle254.v --interpretation srgb
/opt/homebrew/bin/vips gifsave outputs/cycle254.v outputs/reserve-8-254.gif --bitdepth 8
/opt/homebrew/bin/vipsheader -a outputs/reserve-8-254.gif
/opt/homebrew/bin/vips rawload fixtures/cycle255.raw outputs/cycle255-plain.v 16 16 3 --format uchar
/opt/homebrew/bin/vips copy outputs/cycle255-plain.v outputs/cycle255.v --interpretation srgb
/opt/homebrew/bin/vips gifsave outputs/cycle255.v outputs/reserve-8-255.gif --bitdepth 8
/opt/homebrew/bin/vipsheader -a outputs/reserve-8-255.gif
/opt/homebrew/bin/vips rawload fixtures/cycle256.raw outputs/cycle256-plain.v 16 16 3 --format uchar
/opt/homebrew/bin/vips copy outputs/cycle256-plain.v outputs/cycle256.v --interpretation srgb
/opt/homebrew/bin/vips gifsave outputs/cycle256.v outputs/reserve-8-256.gif --bitdepth 8
/opt/homebrew/bin/vipsheader -a outputs/reserve-8-256.gif
/opt/homebrew/bin/vips rawload fixtures/cycle2.raw outputs/cycle2-plain.v 16 16 3 --format uchar
/opt/homebrew/bin/vips copy outputs/cycle2-plain.v outputs/cycle2.v --interpretation srgb
/opt/homebrew/bin/vips gifsave outputs/cycle2.v outputs/reserve-1-2.gif --bitdepth 1
/opt/homebrew/bin/vipsheader -a outputs/reserve-1-2.gif
/opt/homebrew/bin/vips rawload fixtures/cycle2.raw outputs/cycle2-plain.v 16 16 3 --format uchar
/opt/homebrew/bin/vips copy outputs/cycle2-plain.v outputs/cycle2.v --interpretation srgb
/opt/homebrew/bin/vips gifsave outputs/cycle2.v outputs/reserve-2-2.gif --bitdepth 2
/opt/homebrew/bin/vipsheader -a outputs/reserve-2-2.gif
/opt/homebrew/bin/vips rawload fixtures/cycle8.raw outputs/cycle8-plain.v 16 16 3 --format uchar
/opt/homebrew/bin/vips copy outputs/cycle8-plain.v outputs/cycle8.v --interpretation srgb
/opt/homebrew/bin/vips gifsave outputs/cycle8.v outputs/reserve-2-8.gif --bitdepth 2
/opt/homebrew/bin/vipsheader -a outputs/reserve-2-8.gif
/opt/homebrew/bin/vips rawload fixtures/cycle8.raw outputs/cycle8-plain.v 16 16 3 --format uchar
/opt/homebrew/bin/vips copy outputs/cycle8-plain.v outputs/cycle8.v --interpretation srgb
/opt/homebrew/bin/vips gifsave outputs/cycle8.v outputs/reserve-4-8.gif --bitdepth 4
/opt/homebrew/bin/vipsheader -a outputs/reserve-4-8.gif
/opt/homebrew/bin/vips rawload fixtures/cycle100.raw outputs/cycle100-plain.v 16 16 3 --format uchar
/opt/homebrew/bin/vips copy outputs/cycle100-plain.v outputs/cycle100.v --interpretation srgb
/opt/homebrew/bin/vips gifsave outputs/cycle100.v outputs/reserve-4-100.gif --bitdepth 4
/opt/homebrew/bin/vipsheader -a outputs/reserve-4-100.gif
/opt/homebrew/bin/vips rawload fixtures/cycle8.raw outputs/cycle8-plain.v 16 16 3 --format uchar
/opt/homebrew/bin/vips copy outputs/cycle8-plain.v outputs/cycle8.v --interpretation srgb
/opt/homebrew/bin/vips gifsave outputs/cycle8.v outputs/reserve-6-8.gif --bitdepth 6
/opt/homebrew/bin/vipsheader -a outputs/reserve-6-8.gif
/opt/homebrew/bin/vips rawload fixtures/cycle100.raw outputs/cycle100-plain.v 16 16 3 --format uchar
/opt/homebrew/bin/vips copy outputs/cycle100-plain.v outputs/cycle100.v --interpretation srgb
/opt/homebrew/bin/vips gifsave outputs/cycle100.v outputs/reserve-6-100.gif --bitdepth 6
/opt/homebrew/bin/vipsheader -a outputs/reserve-6-100.gif
/opt/homebrew/bin/vips rawload fixtures/alpha.raw outputs/alpha-plain.v 32 24 4 --format uchar
/opt/homebrew/bin/vips copy outputs/alpha-plain.v outputs/alpha.v --interpretation srgb
/opt/homebrew/bin/vips gifsave outputs/alpha.v outputs/alpha.gif
/opt/homebrew/bin/vips gifload outputs/alpha.gif outputs/alpha-reload.v
/opt/homebrew/bin/vips rawsave outputs/alpha-reload.v outputs/alpha-reload.raw
/opt/homebrew/bin/vipsheader -a outputs/alpha.gif
/opt/homebrew/bin/vips rawload fixtures/rows8b.raw outputs/rows8b-plain.v 8 8 3 --format uchar
/opt/homebrew/bin/vips copy outputs/rows8b-plain.v outputs/rows8b.v --interpretation srgb
/opt/homebrew/bin/vips gifsave outputs/rows8b.v outputs/rows-progressive.gif --bitdepth 3
/opt/homebrew/bin/vips gifsave outputs/rows8b.v outputs/rows-interlaced.gif --interlace --bitdepth 3
/opt/homebrew/bin/vips gifload outputs/rows-progressive.gif outputs/rows-p.v
/opt/homebrew/bin/vips gifload outputs/rows-interlaced.gif outputs/rows-i.v
/opt/homebrew/bin/vips rawsave outputs/rows-p.v outputs/rows-p.raw
/opt/homebrew/bin/vips rawsave outputs/rows-i.v outputs/rows-i.raw
/opt/homebrew/bin/vips rawload fixtures/greyramp.raw outputs/greyramp-plain.v 32 24 3 --format uchar
/opt/homebrew/bin/vips copy outputs/greyramp-plain.v outputs/greyramp.v --interpretation srgb
/opt/homebrew/bin/vips gifsave outputs/greyramp.v outputs/dither0.gif --dither 0 --bitdepth 2
/opt/homebrew/bin/vips gifload outputs/dither0.gif outputs/dither0.v
/opt/homebrew/bin/vips rawsave outputs/dither0.v outputs/dither0.raw
/opt/homebrew/bin/vips gifsave outputs/greyramp.v outputs/dither0.25.gif --dither 0.25 --bitdepth 2
/opt/homebrew/bin/vips gifload outputs/dither0.25.gif outputs/dither0.25.v
/opt/homebrew/bin/vips rawsave outputs/dither0.25.v outputs/dither0.25.raw
/opt/homebrew/bin/vips gifsave outputs/greyramp.v outputs/dither0.5.gif --dither 0.5 --bitdepth 2
/opt/homebrew/bin/vips gifload outputs/dither0.5.gif outputs/dither0.5.v
/opt/homebrew/bin/vips rawsave outputs/dither0.5.v outputs/dither0.5.raw
/opt/homebrew/bin/vips gifsave outputs/greyramp.v outputs/dither0.75.gif --dither 0.75 --bitdepth 2
/opt/homebrew/bin/vips gifload outputs/dither0.75.gif outputs/dither0.75.v
/opt/homebrew/bin/vips rawsave outputs/dither0.75.v outputs/dither0.75.raw
/opt/homebrew/bin/vips gifsave outputs/greyramp.v outputs/dither1.gif --dither 1 --bitdepth 2
/opt/homebrew/bin/vips gifload outputs/dither1.gif outputs/dither1.v
/opt/homebrew/bin/vips rawsave outputs/dither1.v outputs/dither1.raw
/opt/homebrew/bin/vips rawload fixtures/gradient48x32.raw outputs/gradient48x32-plain.v 48 32 3 --format uchar
/opt/homebrew/bin/vips copy outputs/gradient48x32-plain.v outputs/gradient48x32.v --interpretation srgb
/opt/homebrew/bin/vips gifsave outputs/gradient48x32.v outputs/gradient48x32.gif
/opt/homebrew/bin/vips gifload outputs/gradient48x32.gif outputs/gradient48x32-reload.v
/opt/homebrew/bin/vips rawsave outputs/gradient48x32-reload.v outputs/gradient48x32-reload.raw
/opt/homebrew/bin/vipsheader -a outputs/gradient48x32.gif
/opt/homebrew/bin/vips rawload fixtures/cycle768.raw outputs/cycle768-plain.v 32 24 3 --format uchar
/opt/homebrew/bin/vips copy outputs/cycle768-plain.v outputs/cycle768.v --interpretation srgb
/opt/homebrew/bin/vips gifsave outputs/cycle768.v outputs/cycle768.gif
/opt/homebrew/bin/vips gifload outputs/cycle768.gif outputs/cycle768-reload.v
/opt/homebrew/bin/vips rawsave outputs/cycle768-reload.v outputs/cycle768-reload.raw
/opt/homebrew/bin/vipsheader -a outputs/cycle768.gif
