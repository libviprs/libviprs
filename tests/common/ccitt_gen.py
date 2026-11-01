import subprocess, io, sys, os
from PIL import Image

def ink_small(x, y):  # must match the Rust test
    return ((x + 2 * y) % 11 < 4) or (x >= 30 and y % 2 == 0)

def ink_wide(x, y):
    return (x // 3 + y) % 5 == 0 or x > 1700

def make(w, h, ink, name, comp, extra=()):
    # raw: bit 1 = ink. PIL mode '1' white=1, so give it the ink as 1 and tag WhiteIsZero? keep simple: save, then force photometric via tiffset.
    im = Image.new('1', (w, h), 0)
    px = im.load()
    for y in range(h):
        for x in range(w):
            px[x, y] = 1 if ink(x, y) else 0
    im.save(f'in_{name}.tif', dpi=(200, 200))
    subprocess.check_call(['tiffset', '-s', '262', '0', f'in_{name}.tif'])  # WhiteIsZero: 1 bits are black
    subprocess.check_call(['tiffcp', '-c', comp, '-r', str(h)] + list(extra) + [f'in_{name}.tif', f'out_{name}.tif'])
    # read the strip back
    data = open(f'out_{name}.tif', 'rb').read()
    t = Image.open(f'out_{name}.tif')
    off = t.tag_v2[273][0]
    cnt = t.tag_v2[279][0]
    strip = data[off:off + cnt]
    # check polarity by decoding with libtiff
    subprocess.check_call(['tiffcp', '-c', 'none', f'out_{name}.tif', f'back_{name}.tif'])
    a = Image.open(f'in_{name}.tif').tobytes(); b = Image.open(f'back_{name}.tif').tobytes()
    assert a == b, name
    return strip, subprocess.check_output(['tiffinfo', f'out_{name}.tif']).decode()

cases = [
    ('G4_SMALL', 40, 8, ink_small, 'g4', ()),
    ('G3_1D_SMALL', 40, 8, ink_small, 'g3:1d', ()),
    ('G3_2D_SMALL', 40, 8, ink_small, 'g3:2d', ()),
    ('G3_1D_FILL_SMALL', 40, 8, ink_small, 'g3:1d:fill', ()),
    ('G4_WIDE', 1728, 4, ink_wide, 'g4', ()),
]
out = ['//! CCITT fixtures for tests/pvec_1363_pdf_ccitt.rs. Generated, not hand-edited: see the header of the test.\n#![allow(dead_code)]\n']
for n, w, h, ink, comp, ex in cases:
    s, info = make(w, h, ink, n.lower(), comp, ex)
    print(n, len(s), file=sys.stderr)
    out.append(f'/// `{n}`: {w} by {h}, tiffcp -c {comp}, {len(s)} bytes.')
    out.append(f'pub const {n}: &[u8] = &[')
    for i in range(0, len(s), 16):
        out.append('    ' + ' '.join(f'0x{b:02X},' for b in s[i:i+16]))
    out.append('];\n')
open('ccitt_fixtures.rs', 'w').write('\n'.join(out))
