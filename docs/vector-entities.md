# Supported entities

What a DWG entity becomes on the vector path (DWG to MVT to PMTiles), and what is said when it does
not. The rule behind the table is the one in #1028: no silent fidelity loss. An entity that is not
drawn, or drawn less than it is, shows up in the decode report with a code, and a view that ends up
with nothing in it says so (code 108).

Model Space is the production target. Paper space and layouts are refused (code 508, `ViewNotEnabled`)
until the viewport transform and clipping fixtures pass. Supported DWG versions are AC1014 through AC1032
(see `docs/vector-providers.md`); older stamps are refused when the file is opened.

Every row is a committed drawing under `tests/fixtures/cad/g13/` (written by ACadSharp's own writer),
decoded by `tests/vector_1038_entity_matrix.rs` against the native library. The test fails if a row here
no longer matches what the decoder does, or if a fixture has no row. Code 101 (a reader notification) is
on nearly every drawing and is left out of the last column.

| Result | Meaning |
| --- | --- |
| Drawn | Every record of that entity reaches the tiles. |
| Drawn, with a reported gap | Part of it reaches the tiles; the codes say what did not. |
| Not drawn, reported | Nothing of it reaches the tiles; the decode report says so (code 100) and marks the view empty if nothing else was drawn (code 108). |
| Refused | The file is not opened (an unsupported version, or over a limit). |
| Fails, bounded | The decode stops with an error on purpose, and the report says why (code 504). |

Known gaps that stay open: a solid HATCH is drawn as an outline and a patterned one has no pattern
lines (codes 103 and 104), and 2D SOLID is not drawn. Both need a field on the wire record from
libviprs-dep (libviprs-dep#165, #1034). MTEXT is anchored at its bottom-left because the attachment point
is not on the wire either.

| Fixture | Entity | Result | Drawn as | Report codes |
| --- | --- | --- | --- | --- |
| `g11_codepage.dwg` | Text in a non-UTF-8 code page | Drawn | Arc, Circle, Line, Text |  |
| `g11_shapes.dwg` | Mixed shapes | Drawn | Arc, Circle, Line, Text |  |
| `g13_ac1009.dwg` | A pre-R14 drawing (AC1009) | Refused | refused at open |  |
| `g13_arc.dwg` | ARC | Drawn | Arc |  |
| `g13_bad_extents.dwg` | Drawing with wrong stored extents | Drawn | Line |  |
| `g13_circle.dwg` | CIRCLE | Drawn | Circle |  |
| `g13_deep_blocks.dwg` | Nested blocks | Drawn | Line |  |
| `g13_dimension.dwg` | DIMENSION | Drawn, with a reported gap | Line, Text | 100 |
| `g13_dimension_deep.dwg` | DIMENSION (pathological nesting) | Fails, bounded | nothing | 504 |
| `g13_dimension_shallow.dwg` | DIMENSION (shallow nesting) | Drawn, with a reported gap | Line, Text | 100 |
| `g13_ellipse.dwg` | ELLIPSE | Drawn | Ellipse |  |
| `g13_empty_view.dwg` | A view with nothing in it | Not drawn, reported | nothing | 108 |
| `g13_face3d.dwg` | 3DFACE | Not drawn, reported | nothing | 100, 108 |
| `g13_hatch.dwg` | HATCH | Drawn, with a reported gap | Polygon | 103, 104, 502 |
| `g13_insert.dwg` | INSERT (a block reference) | Drawn | Circle, Line |  |
| `g13_leader.dwg` | LEADER | Not drawn, reported | nothing | 100, 108 |
| `g13_line.dwg` | LINE | Drawn | Line |  |
| `g13_long_text.dwg` | Long text | Drawn | Text |  |
| `g13_many_inserts.dwg` | Many repeated INSERTs | Drawn | Arc, Circle, Line |  |
| `g13_mesh.dwg` | MESH | Not drawn, reported | nothing | 100, 108 |
| `g13_mesh_bad_faces.dwg` | MESH with bad faces | Not drawn, reported | nothing | 100, 108 |
| `g13_mirrored_bulge.dwg` | Polyline with a bulge, mirrored | Drawn | Polyline |  |
| `g13_mline.dwg` | MLINE | Not drawn, reported | nothing | 100, 108 |
| `g13_nan_bulge.dwg` | Polyline with a non-finite bulge | Not drawn, reported | nothing | 107, 108 |
| `g13_nonuniform.dwg` | INSERT with a non-uniform scale | Drawn, with a reported gap | Circle | 106 |
| `g13_ocs_mirror.dwg` | Entities in a mirrored OCS | Drawn | Arc, Circle, Ellipse |  |
| `g13_ocs_plane.dwg` | Entities in an OCS plane | Drawn | Arc, Circle, Polyline |  |
| `g13_ocs_rotated.dwg` | Entities in a rotated OCS | Drawn | Arc, Circle, Polyline |  |
| `g13_ocs_skew.dwg` | Entities in a skewed OCS | Drawn, with a reported gap | Circle, Polyline | 106 |
| `g13_point.dwg` | POINT | Not drawn, reported | nothing | 100, 108 |
| `g13_polyface_mesh.dwg` | Polyface mesh | Drawn | Polyline |  |
| `g13_polygon_mesh.dwg` | Polygon mesh | Drawn | Polyline |  |
| `g13_polyline.dwg` | LWPOLYLINE and POLYLINE | Drawn | Polyline |  |
| `g13_ray_xline.dwg` | RAY and XLINE | Drawn, with a reported gap | Line | 100 |
| `g13_scale_16x.dwg` | INSERT scaled 16x | Drawn | Circle, Line |  |
| `g13_scale_1x.dwg` | INSERT scaled 1x | Drawn | Circle, Line |  |
| `g13_scale_4x.dwg` | INSERT scaled 4x | Drawn | Circle, Line |  |
| `g13_slot.dwg` | Slot (rounded polyline) | Drawn | Polyline |  |
| `g13_slot_block.dwg` | Slot inside a block | Drawn | Polyline |  |
| `g13_solid.dwg` | SOLID (2D) | Not drawn, reported | nothing | 100, 108 |
| `g13_spline.dwg` | SPLINE | Drawn | Spline |  |
| `g13_text.dwg` | TEXT and MTEXT | Drawn | Text |  |
| `g13_tolerance.dwg` | TOLERANCE | Not drawn, reported | nothing | 100, 108 |
| `g13_two_entities.dwg` | Two entities | Drawn | Line |  |
| `g13_unsupported.dwg` | An entity kind libviprs has no mapping for | Not drawn, reported | nothing | 100, 108 |
| `g13_wide_polyline.dwg` | Polyline with width | Drawn | Polyline |  |
| `g13_wide_spline.dwg` | Spline with width | Drawn | Spline |  |
| `g13_wipeout.dwg` | WIPEOUT | Not drawn, reported | nothing | 100, 108 |
| `g13_xref.dwg` | XREF block | Not drawn, reported | nothing | 105, 108 |
| `g13_xref_long.dwg` | XREF block (long path) | Not drawn, reported | nothing | 105, 108 |
