#!/bin/sh
# Every vips command capture.py ran, in order. Regenerate with
# `python3 capture.py` from this directory.
set -e

/opt/homebrew/bin/vipsheader -a fixtures/flat6.hdr
/opt/homebrew/bin/vips rad2float fixtures/flat6.hdr outputs/getpoint.v
/opt/homebrew/bin/vips getpoint outputs/getpoint.v 0 0
/opt/homebrew/bin/vips getpoint outputs/getpoint.v 1 0
/opt/homebrew/bin/vips getpoint outputs/getpoint.v 2 0
/opt/homebrew/bin/vips getpoint outputs/getpoint.v 3 0
/opt/homebrew/bin/vips getpoint outputs/getpoint.v 4 0
/opt/homebrew/bin/vips getpoint outputs/getpoint.v 5 0
/opt/homebrew/bin/vips rawload fixtures/setcolr.raw outputs/setcolr.v 6 2 3 --format float
/opt/homebrew/bin/vips copy outputs/setcolr.v outputs/setcolr-scrgb.v --interpretation scrgb
/opt/homebrew/bin/vips float2rad outputs/setcolr-scrgb.v outputs/setcolr-rad.v
/opt/homebrew/bin/vips rawsave outputs/setcolr-rad.v outputs/setcolr-rad.raw
/opt/homebrew/bin/vips rawload fixtures/rle16.raw outputs/rle16.v 16 1 3 --format float
/opt/homebrew/bin/vips copy outputs/rle16.v outputs/rle16-scrgb.v --interpretation scrgb
/opt/homebrew/bin/vips float2rad outputs/rle16-scrgb.v outputs/rle16-rad.v
/opt/homebrew/bin/vips radsave outputs/rle16-rad.v outputs/rle16.hdr
/opt/homebrew/bin/vips rawload fixtures/size4.raw outputs/size4.v 4 2 3 --format float
/opt/homebrew/bin/vips copy outputs/size4.v outputs/size4-scrgb.v --interpretation scrgb
/opt/homebrew/bin/vips float2rad outputs/size4-scrgb.v outputs/size4-rad.v
/opt/homebrew/bin/vips radsave outputs/size4-rad.v outputs/size4.hdr
/opt/homebrew/bin/vips rawload fixtures/size16.raw outputs/size16.v 16 2 3 --format float
/opt/homebrew/bin/vips copy outputs/size16.v outputs/size16-scrgb.v --interpretation scrgb
/opt/homebrew/bin/vips float2rad outputs/size16-scrgb.v outputs/size16-rad.v
/opt/homebrew/bin/vips radsave outputs/size16-rad.v outputs/size16.hdr
/opt/homebrew/bin/vips rawload fixtures/size40000.raw outputs/size40000.v 40000 2 3 --format float
/opt/homebrew/bin/vips copy outputs/size40000.v outputs/size40000-scrgb.v --interpretation scrgb
/opt/homebrew/bin/vips float2rad outputs/size40000-scrgb.v outputs/size40000-rad.v
/opt/homebrew/bin/vips radsave outputs/size40000-rad.v outputs/size40000.hdr
/opt/homebrew/bin/vipsheader fixtures/orient_-Y_1_+X_6.hdr
/opt/homebrew/bin/vipsheader fixtures/orient_+X_6_+Y_1.hdr
/opt/homebrew/bin/vipsheader fixtures/orient_-Y_1_-X_6.hdr
/opt/homebrew/bin/vips rad2float /Users/rom/workspace/libviprs/libviprs-tests/tmp/libvips-reference-tests/test-suite/images/sample.hdr outputs/inv1.v
/opt/homebrew/bin/vips rawsave outputs/inv1.v outputs/inv1.raw
/opt/homebrew/bin/vips float2rad outputs/inv1.v outputs/invr.v
/opt/homebrew/bin/vips radsave outputs/invr.v outputs/invrt.hdr
/opt/homebrew/bin/vips rad2float outputs/invrt.hdr outputs/inv2.v
/opt/homebrew/bin/vips rawsave outputs/inv2.v outputs/inv2.raw
/opt/homebrew/bin/vipsheader /Users/rom/workspace/libviprs/libviprs-tests/tmp/libvips-reference-tests/test-suite/images/sample.hdr
/opt/homebrew/bin/vips rad2float /Users/rom/workspace/libvips/test/test-suite/images/sample.hdr outputs/inv1.v
/opt/homebrew/bin/vips rawsave outputs/inv1.v outputs/inv1.raw
/opt/homebrew/bin/vips float2rad outputs/inv1.v outputs/invr.v
/opt/homebrew/bin/vips radsave outputs/invr.v outputs/invrt.hdr
/opt/homebrew/bin/vips rad2float outputs/invrt.hdr outputs/inv2.v
/opt/homebrew/bin/vips rawsave outputs/inv2.v outputs/inv2.raw
/opt/homebrew/bin/vipsheader /Users/rom/workspace/libvips/test/test-suite/images/sample.hdr
/opt/homebrew/bin/vips rad2float /Users/rom/workspace/libviprs/libviprs-tests/tmp/libvips-reference-tests/test-suite/images/sample.hdr outputs/trap-f.v
/opt/homebrew/bin/vips radsave outputs/trap-f.v outputs/trap-direct.hdr
/opt/homebrew/bin/vips float2rad outputs/trap-f.v outputs/trap-pair.v
/opt/homebrew/bin/vips radsave outputs/trap-pair.v outputs/trap-pair.hdr
/opt/homebrew/bin/vips max /Users/rom/workspace/libviprs/libviprs-tests/tmp/libvips-reference-tests/test-suite/images/sample.hdr
/opt/homebrew/bin/vips avg /Users/rom/workspace/libviprs/libviprs-tests/tmp/libvips-reference-tests/test-suite/images/sample.hdr
/opt/homebrew/bin/vips max outputs/trap-direct.hdr
/opt/homebrew/bin/vips avg outputs/trap-direct.hdr
/opt/homebrew/bin/vips max outputs/trap-pair.hdr
/opt/homebrew/bin/vips avg outputs/trap-pair.hdr
/opt/homebrew/bin/vips rawload fixtures/mono.raw outputs/mono.v 8 2 1 --format float
/opt/homebrew/bin/vips radsave outputs/mono.v outputs/mono.hdr
/opt/homebrew/bin/vipsheader fixtures/rgbe_magic.hdr
/opt/homebrew/bin/vipsheader -a fixtures/xyze6.hdr
/opt/homebrew/bin/vipsheader -a fixtures/bogusfmt6.hdr
