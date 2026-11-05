//! Solid fills on the acadrust path (issue #1034, "Solid hatches are supported").
//!
//! The drawings are written in a tempdir-free way, with acadrust's own `DwgWriter`, so no fixture
//! file is committed: a hatch, a 2D SOLID and an MTEXT are built in memory, decoded by
//! `AcadrustDecoder` and tiled. What is checked is the tile, not the primitive, because a fill
//! that reaches the IR and never reaches the `fills` layer is the bug this exists to catch.
#![cfg(feature = "dwg-acadrust")]

use acadrust::entities::{
    AttachmentPoint, BoundaryEdge, BoundaryPath, BoundaryPathFlags, Hatch, HatchPattern,
    HatchPatternLine, HatchStyleType, Line, MText, PolylineEdge, Solid,
};
use acadrust::types::{Vector2, Vector3};
use acadrust::{CadDocument, DwgWriter, EntityType};
use libviprs::cad::acadrust::AcadrustDecoder;
use libviprs::cad::{
    CadDecoder, CadSource, CollectSink, DecodeReport, DiagnosticCode, Primitive, PrimitiveKind,
};
use libviprs::vector::VectorArchive;
use libviprs::vector::mvt::{Geometry, Polygon, Value};
use libviprs::vector::tiler::{CollectTiles, TilerConfig, build_archive, tile_primitives};

const EXTENT: f64 = 4096.0;

fn drawing(entities: Vec<EntityType>) -> Vec<u8> {
    let mut doc = CadDocument::new();
    for e in entities {
        doc.add_entity(e).unwrap();
    }
    DwgWriter::write_to_vec(&doc).unwrap()
}

fn decode(bytes: &[u8]) -> (Vec<Primitive>, DecodeReport) {
    let decoder = AcadrustDecoder::new();
    let drawing = decoder
        .open(CadSource::Bytes(bytes))
        .expect("the drawing opens");
    let (mut sink, mut report) = (CollectSink::new(), DecodeReport::new());
    drawing.decode(0, &mut sink, &mut report).expect("decode");
    (sink.collected().to_vec(), report)
}

fn tiles(prims: &[Primitive], max_zoom: u8) -> CollectTiles {
    let mut out = CollectTiles::default();
    tile_primitives(
        prims,
        &TilerConfig::default().with_max_zoom(max_zoom),
        &mut out,
    )
    .unwrap();
    out
}

/// A frame line that pins the world to 0..1000 on both axes, whatever else is drawn.
fn frame() -> EntityType {
    EntityType::Line(Line::from_coords(0.0, 0.0, 0.0, 1000.0, 1000.0, 0.0))
}

fn ring(points: &[(f64, f64)]) -> BoundaryPath {
    let mut edge = PolylineEdge::new(
        points.iter().map(|&(x, y)| Vector2::new(x, y)).collect(),
        true,
    );
    edge.is_closed = true;
    let mut path = BoundaryPath::with_flags(BoundaryPathFlags::from_bits(2));
    path.add_edge(BoundaryEdge::Polyline(edge));
    path
}

fn square(x0: f64, y0: f64, x1: f64, y1: f64) -> Vec<(f64, f64)> {
    vec![(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
}

fn hatch(solid: bool, rings: &[Vec<(f64, f64)>]) -> EntityType {
    let mut h = if solid {
        Hatch::solid()
    } else {
        let mut p = HatchPattern::new("ANSI31");
        p.add_line(HatchPatternLine {
            angle: std::f64::consts::FRAC_PI_4,
            base_point: Vector2::new(0.0, 0.0),
            offset: Vector2::new(0.0, 3.0),
            dash_lengths: Vec::new(),
        });
        Hatch::with_pattern(p)
    };
    for r in rings {
        h.add_path(ring(r));
    }
    EntityType::Hatch(Box::new(h))
}

fn fills(t: &CollectTiles, z: u8, x: u32, y: u32) -> Vec<Polygon> {
    let Some(tile) = t.tiles.get(&(z, x, y)) else {
        return Vec::new();
    };
    tile.layers
        .iter()
        .filter(|l| l.name == "fills")
        .flat_map(|l| &l.features)
        .flat_map(|f| match &f.geometry {
            Geometry::Polygons(p) => p.clone(),
            other => panic!("the fills layer holds {other:?}"),
        })
        .collect()
}

fn area(ring: &[[i32; 2]]) -> f64 {
    let n = ring.len();
    let twice: f64 = (0..n)
        .map(|i| {
            let (a, b) = (ring[i], ring[(i + 1) % n]);
            f64::from(a[0]) * f64::from(b[1]) - f64::from(b[0]) * f64::from(a[1])
        })
        .sum();
    twice.abs() / 2.0
}

fn filled_area(polys: &[Polygon]) -> f64 {
    polys
        .iter()
        .map(|p| area(&p.exterior) - p.holes.iter().map(|h| area(h)).sum::<f64>())
        .sum()
}

/// Even-odd point test over every ring of every polygon.
fn covers(polys: &[Polygon], p: [f64; 2]) -> bool {
    let mut inside = false;
    for ring in polys
        .iter()
        .flat_map(|g| std::iter::once(&g.exterior).chain(&g.holes))
    {
        let n = ring.len();
        for i in 0..n {
            let (a, b) = (ring[i], ring[(i + 1) % n]);
            let (ay, by) = (f64::from(a[1]), f64::from(b[1]));
            if (ay > p[1]) != (by > p[1]) {
                let x =
                    f64::from(a[0]) + (p[1] - ay) / (by - ay) * (f64::from(b[0]) - f64::from(a[0]));
                if p[0] < x {
                    inside = !inside;
                }
            }
        }
    }
    inside
}

/// Tile units per drawing unit at zoom 0 for a 0..1000 drawing: the world is 1000 plus 2% each side.
fn scale() -> f64 {
    EXTENT / (1000.0 * 1.04)
}

fn text_anchor(t: &CollectTiles) -> Option<String> {
    t.tiles
        .values()
        .flat_map(|tile| &tile.layers)
        .filter(|l| l.name == "text")
        .flat_map(|l| &l.features)
        .find_map(|f| {
            f.properties
                .iter()
                .find_map(|(k, v)| match (k.as_str(), v) {
                    ("anchor", Value::String(s)) => Some(s.clone()),
                    _ => None,
                })
        })
}

#[test]
#[cfg_attr(miri, ignore)] // touches the filesystem, which Miri isolation blocks
fn a_solid_hatch_square_is_a_filled_polygon_of_its_own_area() {
    let (prims, _) = decode(&drawing(vec![
        frame(),
        hatch(true, &[square(100.0, 100.0, 900.0, 900.0)]),
    ]));
    let polys = fills(&tiles(&prims, 0), 0, 0, 0);
    assert_eq!(polys.len(), 1, "one filled polygon");
    let want = (800.0 * scale()).powi(2);
    let got = filled_area(&polys);
    assert!(
        (got - want).abs() / want < 0.01,
        "area {got}, wanted {want}"
    );
}

#[test]
#[cfg_attr(miri, ignore)] // touches the filesystem, which Miri isolation blocks
fn a_patterned_hatch_stays_an_outline_and_says_pattern_only() {
    let (prims, report) = decode(&drawing(vec![
        frame(),
        hatch(false, &[square(100.0, 100.0, 900.0, 900.0)]),
    ]));
    assert!(prims.iter().any(|p| p.kind() == PrimitiveKind::Polygon));
    assert!(
        report
            .diagnostics()
            .iter()
            .any(|d| d.code() == DiagnosticCode::HATCH_PATTERN_ONLY),
        "the missing pattern lines have to be in the report"
    );
    let t = tiles(&prims, 0);
    assert!(fills(&t, 0, 0, 0).is_empty(), "a pattern is not a fill");
    let lines = t.tiles[&(0, 0, 0)].layers.iter().any(|l| l.name == "lines");
    assert!(lines, "the outline is still drawn");
}

#[test]
#[cfg_attr(miri, ignore)] // touches the filesystem, which Miri isolation blocks
fn an_island_is_a_hole_by_even_odd() {
    let (prims, _) = decode(&drawing(vec![
        frame(),
        hatch(
            true,
            &[
                square(100.0, 100.0, 900.0, 900.0),
                square(400.0, 400.0, 600.0, 600.0),
            ],
        ),
    ]));
    let polys = fills(&tiles(&prims, 0), 0, 0, 0);
    assert_eq!(polys.len(), 1, "outer and island are one polygon");
    assert_eq!(polys[0].holes.len(), 1, "with the island as its hole");
    let to_tile = |v: f64| v * scale() + 0.02 / 1.04 * EXTENT;
    // The y axis is flipped on the way into a tile, and the middle of the square is its own mirror.
    let mid = to_tile(500.0);
    assert!(!covers(&polys, [mid, mid]), "the island pixel is empty");
    assert!(
        covers(&polys, [to_tile(200.0), to_tile(200.0)]),
        "the ring around it is filled"
    );
}

#[test]
#[cfg_attr(miri, ignore)] // touches the filesystem, which Miri isolation blocks
fn a_2d_solid_is_filled_in_its_visible_vertex_order() {
    // Stored first, second, third, fourth, drawn first, second, fourth, third: a bow tie if the
    // order is not swapped, and it has no area.
    let solid = Solid::new(
        Vector3::new(100.0, 100.0, 0.0),
        Vector3::new(900.0, 100.0, 0.0),
        Vector3::new(100.0, 900.0, 0.0),
        Vector3::new(900.0, 900.0, 0.0),
    );
    let (prims, _) = decode(&drawing(vec![frame(), EntityType::Solid(solid)]));
    let polys = fills(&tiles(&prims, 0), 0, 0, 0);
    let want = (800.0 * scale()).powi(2);
    let got = filled_area(&polys);
    assert!(
        (got - want).abs() / want < 0.01,
        "area {got}, wanted {want}"
    );
}

#[test]
#[cfg_attr(miri, ignore)] // touches the filesystem, which Miri isolation blocks
fn an_mtext_attachment_point_moves_its_anchor() {
    let anchor = |point| {
        let mut m = MText::with_value("Room 12", Vector3::new(500.0, 500.0, 0.0)).with_height(60.0);
        m.attachment_point = point;
        let (prims, _) = decode(&drawing(vec![frame(), EntityType::MText(Box::new(m))]));
        text_anchor(&tiles(&prims, 0))
    };
    let top_left = anchor(AttachmentPoint::TopLeft);
    let middle = anchor(AttachmentPoint::MiddleCenter);
    assert_eq!(top_left.as_deref(), Some("top-left"));
    assert_eq!(middle.as_deref(), Some("center"));
}

#[test]
#[cfg_attr(miri, ignore)] // touches the filesystem, which Miri isolation blocks
fn a_fill_across_four_tiles_is_closed_and_inside_each_tile() {
    let (prims, _) = decode(&drawing(vec![
        frame(),
        hatch(true, &[square(50.0, 50.0, 950.0, 950.0)]),
    ]));
    let t = tiles(&prims, 1);
    let buffer = f64::from(TilerConfig::default().buffer);
    for (x, y) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
        let polys = fills(&t, 1, x, y);
        assert!(!polys.is_empty(), "tile {x}/{y} has no fill");
        for p in &polys {
            for ring in std::iter::once(&p.exterior).chain(&p.holes) {
                assert!(ring.len() >= 3, "a ring needs three points");
                for q in ring {
                    for c in q {
                        let c = f64::from(*c);
                        assert!(
                            (-buffer..=EXTENT + buffer).contains(&c),
                            "tile {x}/{y} has a point at {c}, outside the extent and its buffer"
                        );
                    }
                }
            }
        }
        assert!(filled_area(&polys) > 0.0, "tile {x}/{y} fills nothing");
    }
}

#[test]
#[cfg_attr(miri, ignore)] // touches the filesystem, which Miri isolation blocks
fn an_archive_declares_the_fills_layer_and_the_anchor_only_when_it_uses_them() {
    let built = |entities: Vec<EntityType>| {
        let (prims, _) = decode(&drawing(entities));
        let dir =
            std::env::temp_dir().join(format!("fill1034-{}-{}", std::process::id(), prims.len()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("a.pmtiles");
        build_archive(&prims, &TilerConfig::default().with_max_zoom(1), &path, "a").unwrap();
        let layers = VectorArchive::try_open(&path).unwrap().layers().to_vec();
        let _ = std::fs::remove_dir_all(&dir);
        layers
    };
    let plain = built(vec![frame()]);
    assert!(plain.iter().all(|l| l.id != "fills"));
    let filled = built(vec![
        frame(),
        hatch(true, &[square(100.0, 100.0, 900.0, 900.0)]),
    ]);
    assert!(filled.iter().any(|l| l.id == "fills"));
    let mut mtext = MText::with_value("Room 12", Vector3::new(500.0, 500.0, 0.0)).with_height(60.0);
    mtext.attachment_point = AttachmentPoint::MiddleCenter;
    let anchored = built(vec![frame(), EntityType::MText(Box::new(mtext))]);
    let text = anchored.iter().find(|l| l.id == "text").unwrap();
    assert!(text.fields.contains_key("anchor"));
}

fn styled(style: HatchStyleType, rings: &[Vec<(f64, f64)>]) -> EntityType {
    let EntityType::Hatch(mut h) = hatch(true, rings) else {
        unreachable!()
    };
    h.style = style;
    EntityType::Hatch(h)
}

#[test]
#[cfg_attr(miri, ignore)] // touches the filesystem, which Miri isolation blocks
fn an_outer_style_hatch_keeps_its_island_as_a_hole() {
    // Outer is AutoCAD's default island detection, and an island in it is not filled.
    let (prims, _) = decode(&drawing(vec![
        frame(),
        styled(
            HatchStyleType::Outer,
            &[
                square(100.0, 100.0, 900.0, 900.0),
                square(400.0, 400.0, 600.0, 600.0),
            ],
        ),
    ]));
    let polys = fills(&tiles(&prims, 0), 0, 0, 0);
    assert_eq!(polys.len(), 1, "outer and island are one polygon");
    assert_eq!(polys[0].holes.len(), 1, "with the island as its hole");
    let mid = 500.0 * scale() + 0.02 / 1.04 * EXTENT;
    assert!(!covers(&polys, [mid, mid]), "the island pixel is empty");
}

#[test]
#[cfg_attr(miri, ignore)] // touches the filesystem, which Miri isolation blocks
fn an_ignore_style_hatch_fills_every_loop_on_its_own() {
    let (prims, _) = decode(&drawing(vec![
        frame(),
        styled(
            HatchStyleType::Ignore,
            &[
                square(100.0, 100.0, 900.0, 900.0),
                square(400.0, 400.0, 600.0, 600.0),
            ],
        ),
    ]));
    let polys = fills(&tiles(&prims, 0), 0, 0, 0);
    assert_eq!(polys.len(), 2, "two loops, two polygons");
    assert!(polys.iter().all(|p| p.holes.is_empty()), "no holes");
    let mid = 500.0 * scale() + 0.02 / 1.04 * EXTENT;
    assert!(covers(&polys[..1], [mid, mid]) || covers(&polys[1..], [mid, mid]));
}

#[test]
#[cfg_attr(miri, ignore)] // touches the filesystem, which Miri isolation blocks
fn a_fill_with_too_many_rings_to_nest_is_filled_ring_by_ring_and_counted() {
    // 2049 small squares in one hatch, one past what a tile nests.
    let rings: Vec<Vec<(f64, f64)>> = (0..2049)
        .map(|i| {
            let (x, y) = (
                f64::from(i % 50) * 19.0 + 20.0,
                f64::from(i / 50) * 19.0 + 20.0,
            );
            square(x, y, x + 10.0, y + 10.0)
        })
        .collect();
    let (prims, _) = decode(&drawing(vec![frame(), hatch(true, &rings)]));
    let mut out = CollectTiles::default();
    let (report, _) =
        tile_primitives(&prims, &TilerConfig::default().with_max_zoom(0), &mut out).unwrap();
    assert_eq!(report.fills_unnested, 1, "the fallback is counted");
    let polys = fills(&out, 0, 0, 0);
    assert_eq!(polys.len(), 2049, "every ring is still drawn");
    assert!(polys.iter().all(|p| p.holes.is_empty()));
}

#[test]
#[cfg_attr(miri, ignore)] // touches the filesystem, which Miri isolation blocks
fn an_island_across_a_tile_edge_is_a_hole_whichever_vertex_comes_first() {
    // The island crosses x = 500, the edge between the two tiles of zoom 1. Starting at (600, 200)
    // puts its first vertex off the left tile, so its clipped ring begins on the tile edge.
    let outer = square(100.0, 100.0, 900.0, 900.0);
    for island in [
        vec![
            (400.0, 200.0),
            (600.0, 200.0),
            (600.0, 300.0),
            (400.0, 300.0),
        ],
        vec![
            (600.0, 200.0),
            (600.0, 300.0),
            (400.0, 300.0),
            (400.0, 200.0),
        ],
        vec![
            (600.0, 300.0),
            (400.0, 300.0),
            (400.0, 200.0),
            (600.0, 200.0),
        ],
    ] {
        let (prims, _) = decode(&drawing(vec![
            frame(),
            hatch(true, &[outer.clone(), island.clone()]),
        ]));
        let t = tiles(&prims, 1);
        for x in [0, 1] {
            let polys = fills(&t, 1, x, 1);
            assert_eq!(
                polys.len(),
                1,
                "tile {x}/1, island starting at {:?}",
                island[0]
            );
            assert_eq!(
                polys[0].holes.len(),
                1,
                "tile {x}/1, island starting at {:?}",
                island[0]
            );
        }
    }
}

#[test]
#[cfg_attr(miri, ignore)] // touches the filesystem, which Miri isolation blocks
fn a_loop_that_cannot_be_built_does_not_turn_the_rest_into_separate_fills() {
    // The first loop has a non-finite vertex, so Polygon::new refuses it. The outer loop and its
    // island that follow are still one fill, and the island is still a hole.
    let broken = vec![(f64::NAN, 0.0), (10.0, 0.0), (10.0, 10.0)];
    let (prims, report) = decode(&drawing(vec![
        frame(),
        hatch(
            true,
            &[
                broken,
                square(100.0, 100.0, 900.0, 900.0),
                square(400.0, 400.0, 600.0, 600.0),
            ],
        ),
    ]));
    assert!(
        report
            .diagnostics()
            .iter()
            .any(|d| d.code() == DiagnosticCode::NON_FINITE_GEOMETRY)
    );
    let polys = fills(&tiles(&prims, 0), 0, 0, 0);
    assert_eq!(polys.len(), 1);
    assert_eq!(polys[0].holes.len(), 1);
}

#[test]
#[cfg_attr(miri, ignore)] // touches the filesystem, which Miri isolation blocks
fn a_fill_too_heavy_to_nest_is_drawn_ring_by_ring_and_counted() {
    // 2000 nested rings of 16 vertices is under the ring cap and over the nesting work budget.
    let ring = |r: f64| -> Vec<(f64, f64)> {
        (0..16)
            .map(|i| {
                let t = f64::from(i) / 16.0 * std::f64::consts::TAU;
                (500.0 + r * t.cos(), 500.0 + r * t.sin())
            })
            .collect()
    };
    let rings: Vec<_> = (0..2000)
        .map(|k| ring(450.0 - f64::from(k) * 0.2))
        .collect();
    let (prims, _) = decode(&drawing(vec![frame(), hatch(true, &rings)]));
    let mut out = CollectTiles::default();
    let (report, _) =
        tile_primitives(&prims, &TilerConfig::default().with_max_zoom(0), &mut out).unwrap();
    assert_eq!(report.fills_unnested, 1, "the budget fallback is counted");
    assert_eq!(
        fills(&out, 0, 0, 0).len(),
        2000,
        "every ring is still drawn"
    );
}

#[test]
#[cfg_attr(miri, ignore)] // touches the filesystem, which Miri isolation blocks
fn an_island_touching_the_outer_loop_is_a_hole_on_every_edge_and_start() {
    let outer = square(100.0, 100.0, 900.0, 900.0);
    for (x0, y0, x1, y1) in [
        (400.0, 100.0, 600.0, 200.0),
        (400.0, 800.0, 600.0, 900.0),
        (100.0, 400.0, 200.0, 600.0),
        (800.0, 400.0, 900.0, 600.0),
    ] {
        let corners = square(x0, y0, x1, y1);
        for start in 0..4 {
            let island: Vec<_> = (0..4).map(|k| corners[(start + k) % 4]).collect();
            let (prims, _) = decode(&drawing(vec![
                frame(),
                hatch(true, &[outer.clone(), island]),
            ]));
            let polys = fills(&tiles(&prims, 0), 0, 0, 0);
            assert_eq!(
                polys.len(),
                1,
                "island {x0},{y0} to {x1},{y1} starting at corner {start}"
            );
            assert_eq!(polys[0].holes.len(), 1, "island {x0},{y0} start {start}");
        }
    }
}
