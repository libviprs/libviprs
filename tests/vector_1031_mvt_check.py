"""Decodes the tiles go-pmtiles returned with the Mapbox decoder and compares
every layer, geometry and property with the model libviprs wrote them from.

Run by vector_1031_interop.sh inside a python container, never on a laptop.
"""
import gzip, json, mapbox_vector_tile as mvt
expected = json.load(open('/out/expected.json'))
checked = 0
for tile in expected:
    name = f"/out/tile_{tile['z']}_{tile['x']}_{tile['y']}.mvt"
    raw = open(name, 'rb').read()
    data = gzip.decompress(raw) if raw[:2] == b'\x1f\x8b' else raw
    got = mvt.decode(data, y_coord_down=True)
    assert sorted(got) == sorted(l['name'] for l in tile['layers']), (name, sorted(got))
    for layer in tile['layers']:
        g = got[layer['name']]
        assert g['extent'] == layer['extent'], (name, layer['name'], g['extent'])
        assert len(g['features']) == len(layer['features'])
        for gf, ef in zip(g['features'], layer['features']):
            assert gf['geometry']['type'] == ef['type'], (name, gf['geometry']['type'], ef['type'])
            assert gf['properties'] == ef['properties'], (name, gf['properties'], ef['properties'])
            c = gf['geometry']['coordinates']
            e = ef['coordinates']
            if ef['type'] == 'Point':
                pts = [c] if not isinstance(c[0], (list, tuple)) else c
                assert [list(map(int, p)) for p in pts] == e, (name, c, e)
            elif ef['type'] == 'LineString':
                got_lines = c if isinstance(c[0][0], (list, tuple)) else [c]
                assert [[list(map(int, p)) for p in l] for l in got_lines] == e, (name, c, e)
            else:
                got_rings = c if isinstance(c[0][0][0], (list, tuple)) else [c]
                for gp, ep in zip(got_rings, e):
                    for gr, er in zip(gp, ep):
                        gr = [list(map(int, p)) for p in gr][:-1]
                        assert sorted(map(tuple, gr)) == sorted(map(tuple, er)), (name, gr, er)
            checked += 1
print(f'python mapbox-vector-tile: {checked} features match expected.json')
