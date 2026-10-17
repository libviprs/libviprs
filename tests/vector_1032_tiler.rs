//! Geometry tiling, curve tessellation, bounds and the precision budget
//! (issue #1032).
//!
//! The acceptance criteria are numbers, so most tests here measure one and
//! assert it against the bound the module documents. Three of them compare the
//! tiles with something that does not share their code: a pixel render of the
//! source geometry made by a rasteriser written in this file.

use std::collections::BTreeSet;

use libviprs::cad::{Arc, Circle, Ellipse, Line, Polyline, Primitive, Spline, Text};
use libviprs::vector::mvt::{Geometry, Tile};
use libviprs::vector::tessellate;
use libviprs::vector::tiler::{
    CollectTiles, OversizePolicy, TilerConfig, TilerError, WorldTransform, build_archive,
    tile_primitives,
};
use libviprs::vector::{VectorArchive, tiler::TILER_METADATA_KEY};

const Z: [f64; 3] = [0.0, 0.0, 1.0];

fn line(a: [f64; 2], b: [f64; 2]) -> Primitive {
    Line::new([a[0], a[1], 0.0], [b[0], b[1], 0.0])
        .unwrap()
        .into()
}

fn rect(x0: f64, y0: f64, x1: f64, y1: f64) -> Primitive {
    Polyline::new(
        &[[x0, y0, 0.0], [x1, y0, 0.0], [x1, y1, 0.0], [x0, y1, 0.0]],
        true,
        Z,
        &[],
    )
    .unwrap()
    .into()
}

/// A small drawing with every kind of geometry, 1000 x 600.
fn golden() -> Vec<Primitive> {
    let mut p = vec![
        rect(0.0, 0.0, 1000.0, 600.0),
        Circle::new([300.0, 300.0, 0.0], 150.0, Z).unwrap().into(),
        Arc::new([700.0, 300.0, 0.0], 200.0, 0.4, 4.0, Z)
            .unwrap()
            .into(),
        Ellipse::new([500.0, 300.0, 0.0], [120.0, 40.0, 0.0], 0.5, 0.0, 6.0, Z)
            .unwrap()
            .into(),
        Spline::new(
            3,
            0,
            &[0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0],
            &[
                [50.0, 50.0, 0.0],
                [200.0, 500.0, 0.0],
                [400.0, 100.0, 0.0],
                [600.0, 550.0, 0.0],
                [900.0, 80.0, 0.0],
            ],
            &[],
        )
        .unwrap()
        .into(),
        Polyline::new(
            &[
                [100.0, 100.0, 0.0],
                [300.0, 120.0, 0.0],
                [350.0, 250.0, 0.0],
            ],
            false,
            Z,
            &[0.0, 0.7, 0.0],
        )
        .unwrap()
        .into(),
        // Long diagonals cross many tiles at every zoom.
        line([0.0, 0.0], [1000.0, 600.0]),
        line([0.0, 600.0], [1000.0, 0.0]),
        Text::new([100.0, 560.0, 0.0], 30.0, 0.0, "GROUND FLOOR")
            .unwrap()
            .into(),
        Text::new([600.0, 20.0, 0.0], 1.0, std::f64::consts::FRAC_PI_2, "N")
            .unwrap()
            .into(),
    ];
    // A hatch of short parallel lines.
    for i in 0..40 {
        let x = 600.0 + f64::from(i) * 5.0;
        p.push(line([x, 400.0], [x + 30.0, 470.0]));
    }
    p
}

fn small_config(max_zoom: u8) -> TilerConfig {
    TilerConfig::default().with_max_zoom(max_zoom)
}

fn tiles_of(
    prims: &[Primitive],
    config: &TilerConfig,
) -> (
    CollectTiles,
    libviprs::vector::tiler::TilerReport,
    WorldTransform,
) {
    let mut out = CollectTiles::default();
    let (report, world) = tile_primitives(prims, config, &mut out).unwrap();
    (out, report, world)
}

// ---------------------------------------------------------------------------
// Aspect ratio
// ---------------------------------------------------------------------------

#[test]
fn the_transform_keeps_the_aspect_ratio_exactly() {
    for (w, h) in [(1000.0, 100.0), (100.0, 1000.0), (5.0, 5.0), (1.0e7, 3.0e6)] {
        let t =
            WorldTransform::from_bounds([-w / 2.0 + 1.0e6, 0.0, w / 2.0 + 1.0e6, h], 0.02).unwrap();
        let lo = t.unit([-w / 2.0 + 1.0e6, 0.0]);
        let hi = t.unit([w / 2.0 + 1.0e6, h]);
        let (du, dv) = ((hi[0] - lo[0]).abs(), (hi[1] - lo[1]).abs());
        assert!(
            (du / dv - w / h).abs() < 1e-9 * (w / h),
            "{w}x{h}: {du} x {dv}"
        );
        // The drawing sits inside the unit square, centred.
        for c in [lo, hi] {
            assert!((0.0..=1.0).contains(&c[0]) && (0.0..=1.0).contains(&c[1]));
        }
        assert!(((lo[0] + hi[0]) / 2.0 - 0.5).abs() < 1e-12);
        assert!(((lo[1] + hi[1]) / 2.0 - 0.5).abs() < 1e-12);
    }
}

#[test]
fn y_runs_down_in_tiles_and_up_in_the_drawing() {
    let t = WorldTransform::from_bounds([0.0, 0.0, 10.0, 10.0], 0.0).unwrap();
    assert!(t.unit([0.0, 10.0])[1] < t.unit([0.0, 0.0])[1]);
    let p = [3.25, 7.75];
    let back = t.from_unit(t.unit(p));
    assert!((back[0] - p[0]).abs() < 1e-12 && (back[1] - p[1]).abs() < 1e-12);
}

#[test]
fn a_square_in_the_drawing_is_a_square_in_the_tile() {
    let prims = vec![
        rect(0.0, 0.0, 800.0, 200.0),
        rect(100.0, 50.0, 200.0, 150.0),
    ];
    let (tiles, _, _) = tiles_of(&prims, &small_config(0));
    let tile = &tiles.tiles[&(0, 0, 0)];
    let Geometry::Lines(lines) = &tile.layers[0].features[0].geometry else {
        panic!()
    };
    // The inner square: find the run with 5 points whose box is ~100 x 100 units
    // of drawing, i.e. equal in x and y in tile units.
    let boxes: Vec<(i32, i32)> = lines
        .iter()
        .map(|l| {
            let xs: Vec<_> = l.iter().map(|p| p[0]).collect();
            let ys: Vec<_> = l.iter().map(|p| p[1]).collect();
            (
                xs.iter().max().unwrap() - xs.iter().min().unwrap(),
                ys.iter().max().unwrap() - ys.iter().min().unwrap(),
            )
        })
        .collect();
    assert!(
        boxes.iter().any(|(w, h)| (w - h).abs() <= 1 && *w > 100),
        "{boxes:?}"
    );
}

// ---------------------------------------------------------------------------
// No seams or gaps, against an independent rasteriser
// ---------------------------------------------------------------------------

/// Bresenham on integer pixels. Written here so the comparison does not share
/// a line algorithm with the code under test.
fn plot_line(
    img: &mut BTreeSet<(i64, i64)>,
    a: (i64, i64),
    b: (i64, i64),
    keep: &dyn Fn(i64, i64) -> bool,
) {
    let (mut x, mut y) = a;
    let dx = (b.0 - a.0).abs();
    let dy = -(b.1 - a.1).abs();
    let sx = if a.0 < b.0 { 1 } else { -1 };
    let sy = if a.1 < b.1 { 1 } else { -1 };
    let mut err = dx + dy;
    loop {
        if keep(x, y) {
            img.insert((x, y));
        }
        if (x, y) == b {
            break;
        }
        let e2 = 2 * err;
        if e2 >= dy {
            err += dy;
            x += sx;
        }
        if e2 <= dx {
            err += dx;
            y += sy;
        }
    }
}

fn near(set: &BTreeSet<(i64, i64)>, p: (i64, i64), r: i64) -> bool {
    (-r..=r).any(|dx| (-r..=r).any(|dy| set.contains(&(p.0 + dx, p.1 + dy))))
}

/// Renders the source straight to global pixels, with no tiles involved.
fn render_source(
    prims: &[Primitive],
    world: &WorldTransform,
    zoom: u8,
    extent: u32,
    tol_px: f64,
) -> BTreeSet<(i64, i64)> {
    let scale = f64::from(1u32 << zoom) * f64::from(extent);
    let tol = tol_px * world.source_units_per_tile_unit(zoom, extent);
    let mut img = BTreeSet::new();
    let to_px = |p: [f64; 2]| {
        let uv = world.unit(p);
        (
            (uv[0] * scale).round() as i64,
            (uv[1] * scale).round() as i64,
        )
    };
    let keep = |_: i64, _: i64| true;
    for prim in prims {
        let runs: Vec<Vec<[f64; 2]>> = match prim {
            Primitive::Line(l) => vec![vec![
                [l.start().x(), l.start().y()],
                [l.end().x(), l.end().y()],
            ]],
            Primitive::Polyline(p) => {
                let v: Vec<[f64; 3]> = p.vertices().iter().map(|p| p.into_array()).collect();
                vec![
                    tessellate::bulge_run(
                        &v,
                        p.bulges(),
                        p.normal().into_array(),
                        p.is_closed(),
                        tol,
                    )
                    .points,
                ]
            }
            Primitive::Arc(a) => vec![tessellate::arc(a, tol).points],
            Primitive::Circle(c) => vec![tessellate::circle(c, tol).points],
            Primitive::Ellipse(e) => vec![tessellate::ellipse(e, tol).points],
            Primitive::Spline(s) => vec![tessellate::spline(s, tol).unwrap().points],
            _ => continue,
        };
        for run in runs {
            for w in run.windows(2) {
                plot_line(&mut img, to_px(w[0]), to_px(w[1]), &keep);
            }
        }
    }
    img
}

/// Renders the tiles of one zoom to global pixels, each tile keeping only the
/// pixels inside its own extent, as a client compositing tiles would.
fn render_tiles(tiles: &CollectTiles, zoom: u8, extent: u32) -> BTreeSet<(i64, i64)> {
    let e = i64::from(extent);
    let mut img = BTreeSet::new();
    for (&(z, tx, ty), tile) in &tiles.tiles {
        if z != zoom {
            continue;
        }
        let (ox, oy) = (i64::from(tx) * e, i64::from(ty) * e);
        let keep = move |x: i64, y: i64| x >= ox && x < ox + e && y >= oy && y < oy + e;
        for layer in &tile.layers {
            for f in &layer.features {
                if let Geometry::Lines(lines) = &f.geometry {
                    for l in lines {
                        for w in l.windows(2) {
                            plot_line(
                                &mut img,
                                (ox + i64::from(w[0][0]), oy + i64::from(w[0][1])),
                                (ox + i64::from(w[1][0]), oy + i64::from(w[1][1])),
                                &keep,
                            );
                        }
                    }
                }
            }
        }
    }
    img
}

#[test]
fn tiles_composite_to_the_source_with_no_seams_or_gaps() {
    let prims = golden();
    let extent = 256;
    for zoom in [2u8, 3, 4] {
        let cfg = TilerConfig {
            extent,
            buffer: 16,
            ..small_config(zoom)
        };
        let (tiles, _, world) = tiles_of(&prims, &cfg);
        let from_tiles = render_tiles(&tiles, zoom, extent);
        let from_source = render_source(&prims, &world, zoom, extent, 0.25);
        assert!(
            from_tiles.len() > 500,
            "the render is not empty at zoom {zoom}"
        );
        // Every pixel one side draws is within 2 px of one the other draws.
        // Quantisation, simplification and tessellation account for well under
        // that; a missing or misplaced piece of a seam does not.
        let missing = from_source
            .iter()
            .filter(|p| !near(&from_tiles, **p, 2))
            .count();
        let extra = from_tiles
            .iter()
            .filter(|p| !near(&from_source, **p, 2))
            .count();
        assert_eq!(
            missing, 0,
            "zoom {zoom}: {missing} source pixels have no tile pixel near them"
        );
        assert_eq!(
            extra, 0,
            "zoom {zoom}: {extra} tile pixels have no source pixel near them"
        );

        // The seams specifically: every tile edge that source pixels touch has
        // tile pixels there too.
        let e = i64::from(extent);
        let on_edge = |p: &(i64, i64)| {
            (p.0 % e).abs() <= 1
                || (p.0 % e - e).abs() <= 1
                || (p.1 % e).abs() <= 1
                || (p.1 % e - e).abs() <= 1
        };
        let seam_src: Vec<_> = from_source.iter().filter(|p| on_edge(p)).collect();
        assert!(!seam_src.is_empty());
        assert!(
            seam_src.iter().all(|p| near(&from_tiles, **p, 1)),
            "a gap on a tile edge at zoom {zoom}"
        );
    }
}

#[test]
fn a_long_diagonal_is_continuous_across_every_tile_it_crosses() {
    let prims = vec![line([0.0, 0.0], [1000.0, 700.0])];
    let cfg = TilerConfig {
        extent: 256,
        buffer: 8,
        ..small_config(7)
    };
    let (tiles, _, world) = tiles_of(&prims, &cfg);
    // Walk the line in drawing space and check that the tile containing each
    // sample has a point within a unit and a half of it.
    let mut checked = 0;
    for i in 0..=2000 {
        let t = f64::from(i) / 2000.0;
        let p = [t * 1000.0, t * 700.0];
        let uv = world.unit(p);
        let n = f64::from(1u32 << 7);
        let (gx, gy) = (uv[0] * n, uv[1] * n);
        let (tx, ty) = (gx.floor() as u32, gy.floor() as u32);
        let tile = &tiles.tiles[&(7, tx, ty)];
        let Geometry::Lines(lines) = &tile.layers[0].features[0].geometry else {
            panic!()
        };
        let want = [(gx - f64::from(tx)) * 256.0, (gy - f64::from(ty)) * 256.0];
        let ok = lines.iter().any(|l| {
            l.windows(2).any(|w| {
                let (a, b) = (
                    [f64::from(w[0][0]), f64::from(w[0][1])],
                    [f64::from(w[1][0]), f64::from(w[1][1])],
                );
                let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
                let len2 = dx * dx + dy * dy;
                let t = if len2 == 0.0 {
                    0.0
                } else {
                    (((want[0] - a[0]) * dx + (want[1] - a[1]) * dy) / len2).clamp(0.0, 1.0)
                };
                ((want[0] - a[0] - t * dx).powi(2) + (want[1] - a[1] - t * dy).powi(2)).sqrt()
                    <= 1.5
            })
        });
        assert!(ok, "no line near sample {i} in tile 7/{tx}/{ty}");
        checked += 1;
    }
    assert_eq!(checked, 2001);
}

// ---------------------------------------------------------------------------
// Precision, measured
// ---------------------------------------------------------------------------

/// Largest distance from any decoded vertex in a tile to its source vertex,
/// for source vertices that are endpoints of two point runs.
fn vertex_error(
    tiles: &CollectTiles,
    world: &WorldTransform,
    zoom: u8,
    extent: u32,
    sources: &[[f64; 2]],
) -> f64 {
    let mut worst = 0.0f64;
    for src in sources {
        let uv = world.unit(*src);
        let n = f64::from(1u32 << zoom);
        let (tx, ty) = ((uv[0] * n).floor() as u32, (uv[1] * n).floor() as u32);
        let tile = &tiles.tiles[&(zoom, tx, ty)];
        let mut best = f64::INFINITY;
        for layer in &tile.layers {
            for f in &layer.features {
                if let Geometry::Lines(lines) = &f.geometry {
                    for l in lines {
                        for p in l {
                            let back = world.tile_to_source(
                                zoom,
                                tx,
                                ty,
                                [f64::from(p[0]), f64::from(p[1])],
                                extent,
                            );
                            best = best.min(
                                ((back[0] - src[0]).powi(2) + (back[1] - src[1]).powi(2)).sqrt(),
                            );
                        }
                    }
                }
            }
        }
        worst = worst.max(best);
    }
    worst
}

#[test]
fn source_coordinate_error_is_inside_the_budget_at_every_tested_zoom() {
    // A drawing a million units from the origin, with detail 1/1000 of its
    // size: the case a naive f32-style transform loses.
    let off = 1.0e6;
    let verts = [
        [off + 0.0, off + 0.0],
        [off + 1.0, off + 0.0],
        [off + 999.0, off + 599.0],
        [off + 1000.0, off + 600.0],
        [off + 500.25, off + 300.75],
        [off + 500.9, off + 301.1],
    ];
    let prims = vec![
        line(verts[0], verts[1]),
        line(verts[2], verts[3]),
        line(verts[4], verts[5]),
    ];
    for zoom in [6u8, 10, 14, 18, 22] {
        let cfg = TilerConfig {
            max_zoom: zoom,
            cull_units: 0.0,
            ..TilerConfig::default()
        };
        let (tiles, _, world) = tiles_of(&prims, &cfg);
        let budget = cfg.precision(&world, zoom);
        let measured = vertex_error(&tiles, &world, zoom, cfg.extent, &verts);
        eprintln!(
            "MEASURED z{zoom}: unit {:.3e}, quantisation bound {:.3e}, measured vertex error {:.3e} ({:.3} units)",
            budget.tile_unit,
            budget.quantisation * 2f64.sqrt(),
            measured,
            measured / budget.tile_unit
        );
        assert!(
            measured <= budget.quantisation * 2f64.sqrt() * (1.0 + 1e-9),
            "z{zoom}: error {measured} against bound {}",
            budget.quantisation * 2f64.sqrt()
        );
        assert!(
            budget.worst_case() < budget.tile_unit * 2.0,
            "the documented 1.25 unit worst case holds"
        );
    }
}

#[test]
fn curve_deviation_is_measured_and_inside_the_budget() {
    let r = 250.0;
    let circle = Circle::new([500.0, 300.0, 0.0], r, Z).unwrap();
    // Anchor the world with a frame so the circle is not the whole drawing.
    let prims = vec![rect(0.0, 0.0, 1000.0, 600.0), circle.into()];
    for zoom in [2u8, 4, 6, 8] {
        let cfg = TilerConfig {
            max_zoom: zoom,
            ..TilerConfig::default()
        };
        let (tiles, _, world) = tiles_of(&prims, &cfg);
        let budget = cfg.precision(&world, zoom);
        let mut worst = 0.0f64;
        for (&(z, tx, ty), tile) in &tiles.tiles {
            if z != zoom {
                continue;
            }
            for layer in &tile.layers {
                for f in &layer.features {
                    let Geometry::Lines(lines) = &f.geometry else {
                        continue;
                    };
                    for l in lines {
                        // Only runs that belong to the circle: far from the frame.
                        let pts: Vec<[f64; 2]> = l
                            .iter()
                            .map(|p| {
                                world.tile_to_source(
                                    zoom,
                                    tx,
                                    ty,
                                    [f64::from(p[0]), f64::from(p[1])],
                                    cfg.extent,
                                )
                            })
                            .collect();
                        if pts.iter().any(|p| {
                            (p[0] - 500.0).hypot(p[1] - 300.0) < r - 20.0
                                || (p[0] - 500.0).hypot(p[1] - 300.0) > r + 20.0
                        }) {
                            continue;
                        }
                        for w in pts.windows(2) {
                            for k in 0..=8 {
                                let t = f64::from(k) / 8.0;
                                let p = [
                                    w[0][0] + (w[1][0] - w[0][0]) * t,
                                    w[0][1] + (w[1][1] - w[0][1]) * t,
                                ];
                                worst = worst.max(((p[0] - 500.0).hypot(p[1] - 300.0) - r).abs());
                            }
                        }
                    }
                }
            }
        }
        eprintln!(
            "MEASURED z{zoom}: circle deviation {:.3e} drawing units = {:.3} tile units (budget {:.3})",
            worst,
            worst / budget.tile_unit,
            budget.worst_case() / budget.tile_unit
        );
        assert!(worst > 0.0, "the circle was found in the tiles");
        assert!(
            worst <= budget.worst_case(),
            "z{zoom}: deviation {worst} > {}",
            budget.worst_case()
        );
    }
}

#[test]
fn the_tested_max_zoom_is_documented_and_holds() {
    // The documented range: extent 4096, through zoom 22. A short segment at
    // each corner of a 1000 x 600 drawing is the densest thing that fits in a
    // test, and the previous test measured the error at 22.
    let prims = vec![
        line([0.0, 0.0], [3.0, 1.0]),
        line([997.0, 599.0], [1000.0, 600.0]),
    ];
    let cfg = TilerConfig {
        max_zoom: 22,
        cull_units: 0.0,
        ..TilerConfig::default()
    };
    let (_, report, world) = tiles_of(&prims, &cfg);
    let budget = cfg.precision(&world, 22);
    assert!(report.tiles > 0);
    assert!(
        budget.worst_case() < 1.0e-3,
        "a unit at z22 is {} drawing units",
        budget.tile_unit
    );
    assert_eq!(report.precision_at_max_zoom, Some(budget));
    assert!(
        TilerConfig {
            max_zoom: 25,
            ..TilerConfig::default()
        }
        .max_zoom
            > libviprs::vector::tiler::MAX_SUPPORTED_ZOOM
    );
    let mut out = CollectTiles::default();
    assert!(matches!(
        tile_primitives(
            &prims,
            &TilerConfig {
                max_zoom: 25,
                ..TilerConfig::default()
            },
            &mut out
        ),
        Err(TilerError::Config(_))
    ));
}

// ---------------------------------------------------------------------------
// Budgets and bounds
// ---------------------------------------------------------------------------

fn lcg(seed: &mut u64) -> f64 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*seed >> 33) as f64) / f64::from(1u32 << 31)
}

fn dense(n: usize) -> Vec<Primitive> {
    let mut s = 7u64;
    let mut v = vec![rect(0.0, 0.0, 10_000.0, 10_000.0)];
    for _ in 0..n {
        let (x, y) = (lcg(&mut s) * 10_000.0, lcg(&mut s) * 10_000.0);
        v.push(line(
            [x, y],
            [x + lcg(&mut s) * 30.0, y + lcg(&mut s) * 30.0],
        ));
    }
    v
}

#[test]
fn a_dense_drawing_stays_inside_its_point_budget_and_says_what_it_dropped() {
    let prims = dense(60_000);
    let cfg = TilerConfig {
        max_zoom: 4,
        max_points_per_tile: 5_000,
        ..TilerConfig::default()
    };
    let (tiles, report, _) = tiles_of(&prims, &cfg);
    let mut worst = 0usize;
    for tile in tiles.tiles.values() {
        for layer in &tile.layers {
            for f in &layer.features {
                if let Geometry::Lines(l) = &f.geometry {
                    worst = worst.max(l.iter().map(Vec::len).sum());
                }
            }
        }
    }
    assert!(worst <= 5_000, "a tile carried {worst} points");
    assert!(
        report.dropped_for_point_budget > 0,
        "the cap fired and was counted"
    );
    assert!(report.culled_subpixel > 0 || report.dropped_for_point_budget > 0);
}

#[test]
fn a_level_that_would_not_fit_in_memory_is_an_error_not_an_abort() {
    let prims = dense(5_000);
    let cfg = TilerConfig {
        max_zoom: 3,
        max_level_points: 1_000,
        ..TilerConfig::default()
    };
    let mut out = CollectTiles::default();
    assert!(matches!(
        tile_primitives(&prims, &cfg, &mut out),
        Err(TilerError::LevelBudgetExceeded { limit: 1_000, .. })
    ));
}

#[test]
fn oversized_tiles_are_detected_and_can_be_refused() {
    let prims = dense(20_000);
    let cfg = TilerConfig {
        max_zoom: 2,
        max_tile_bytes: 2_000,
        ..TilerConfig::default()
    };
    let (_, report, _) = tiles_of(&prims, &cfg);
    assert!(report.oversized_count > 0);
    assert!(!report.oversized.is_empty());
    assert!(report.largest_tile_bytes > 2_000);

    let strict = TilerConfig {
        oversize: OversizePolicy::Error,
        ..cfg
    };
    let mut out = CollectTiles::default();
    assert!(matches!(
        tile_primitives(&prims, &strict, &mut out),
        Err(TilerError::Oversized { limit: 2_000, .. })
    ));
}

#[test]
fn sub_pixel_geometry_is_culled_and_counted_never_silently_lost() {
    // 2,000 lines under half a drawing unit long in a 10,000-unit world: at
    // zoom 0 and 1 a tile unit is 2.5 and 1.3 drawing units, so every one of
    // them is under half a tile unit.
    let mut s = 3u64;
    let mut prims = vec![rect(0.0, 0.0, 10_000.0, 10_000.0)];
    for _ in 0..2_000 {
        let (x, y) = (lcg(&mut s) * 9_000.0, lcg(&mut s) * 9_000.0);
        prims.push(line([x, y], [x + 0.3, y + 0.2]));
    }
    let (_, report, _) = tiles_of(
        &prims,
        &TilerConfig {
            max_zoom: 1,
            ..TilerConfig::default()
        },
    );
    assert_eq!(report.culled_subpixel, 4_000);
}

#[test]
fn a_huge_arc_is_capped_and_counted_without_hanging() {
    let prims = vec![
        line([0.0, 0.0], [1.0, 1.0]),
        Arc::new([0.0, 0.0, 0.0], 1.0e9, 0.0, 6.0, Z)
            .unwrap()
            .into(),
    ];
    let cfg = TilerConfig {
        max_zoom: 3,
        ..TilerConfig::default()
    };
    let t = std::time::Instant::now();
    let (_, report, _) = tiles_of(&prims, &cfg);
    assert!(t.elapsed().as_secs() < 60);
    assert!(report.drawable == 2);
}

// ---------------------------------------------------------------------------
// Text
// ---------------------------------------------------------------------------

#[test]
fn text_appears_when_it_is_big_enough_and_carries_what_the_style_needs() {
    let (tiles, report, world) = tiles_of(&golden(), &small_config(6));
    assert!(
        report.culled_text > 0,
        "small text is dropped at low zoom, and counted"
    );
    let mut seen_at: Vec<u8> = Vec::new();
    for (&(z, _, _), tile) in &tiles.tiles {
        for layer in tile.layers.iter().filter(|l| l.name == "text") {
            for f in &layer.features {
                let get = |k: &str| {
                    f.properties
                        .iter()
                        .find(|(n, _)| n == k)
                        .map(|(_, v)| v.clone())
                };
                if let Some(libviprs::vector::mvt::Value::String(s)) = get("text")
                    && s == "N"
                {
                    {
                        seen_at.push(z);
                        // Quarter turn counter-clockwise in CAD is 270 clockwise.
                        assert_eq!(
                            get("rotation"),
                            Some(libviprs::vector::mvt::Value::Double(270.0))
                        );
                        assert_eq!(
                            get("height"),
                            Some(libviprs::vector::mvt::Value::Double(1.0))
                        );
                        let h0 = 1.0 / world.side() * 512.0;
                        assert_eq!(get("h0"), Some(libviprs::vector::mvt::Value::Double(h0)));
                    }
                }
            }
        }
    }
    assert!(!seen_at.is_empty());
    assert!(
        seen_at.iter().all(|z| *z >= 2),
        "1-unit text in a 1000-unit drawing needs zoom 2: {seen_at:?}"
    );
    assert_eq!(
        seen_at.len(),
        seen_at.iter().collect::<BTreeSet<_>>().len(),
        "one tile per zoom holds it"
    );
}

// ---------------------------------------------------------------------------
// Determinism, errors and the archive
// ---------------------------------------------------------------------------

#[test]
fn the_same_input_gives_the_same_tiles_and_the_same_archive() {
    let prims = golden();
    let cfg = small_config(5);
    let (a, _, _) = tiles_of(&prims, &cfg);
    let (b, _, _) = tiles_of(&prims, &cfg);
    assert_eq!(a.tiles, b.tiles);

    let dir = std::env::temp_dir().join(format!("tiler1032-det-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    build_archive(&prims, &cfg, dir.join("a.pmtiles"), "d").unwrap();
    build_archive(&prims, &cfg, dir.join("b.pmtiles"), "d").unwrap();
    assert_eq!(
        std::fs::read(dir.join("a.pmtiles")).unwrap(),
        std::fs::read(dir.join("b.pmtiles")).unwrap()
    );
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn nothing_to_draw_and_bad_configuration_are_typed_errors() {
    let mut out = CollectTiles::default();
    assert!(matches!(
        tile_primitives(&[], &small_config(2), &mut out),
        Err(TilerError::NoGeometry)
    ));
    for bad in [
        TilerConfig {
            extent: 0,
            ..TilerConfig::default()
        },
        TilerConfig {
            buffer: 5000,
            ..TilerConfig::default()
        },
        TilerConfig {
            deviation_units: 0.0,
            ..TilerConfig::default()
        },
        TilerConfig {
            deviation_units: f64::NAN,
            ..TilerConfig::default()
        },
        TilerConfig {
            margin: -1.0,
            ..TilerConfig::default()
        },
        TilerConfig {
            max_points_per_tile: 0,
            ..TilerConfig::default()
        },
    ] {
        assert!(
            matches!(
                tile_primitives(&golden(), &bad, &mut out),
                Err(TilerError::Config(_))
            ),
            "{bad:?}"
        );
    }
}

#[test]
fn a_single_point_drawing_still_tiles() {
    let prims = vec![Text::new([5.0, 5.0, 0.0], 2.0, 0.0, "x").unwrap().into()];
    let mut out = CollectTiles::default();
    assert!(tile_primitives(&prims, &small_config(2), &mut out).is_ok());
}

#[test]
fn the_archive_is_a_valid_vector_archive_that_carries_the_transform() {
    let dir = std::env::temp_dir().join(format!("tiler1032-arc-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("g.pmtiles");
    let report = build_archive(&golden(), &small_config(5), &path, "golden").unwrap();
    assert!(report.tiles > 10);

    let bytes = std::fs::read(&path).unwrap();
    let v = libviprs::pmtiles::validate::validate_bytes(&bytes, &Default::default()).unwrap();
    assert!(v.is_valid(), "{:?}", v.findings);

    let archive = VectorArchive::try_open(&path).unwrap();
    let h = archive.pmtiles().header();
    assert_eq!((h.min_zoom, h.max_zoom), (0, 5));
    assert_eq!(h.addressed_tiles_count, report.tiles);
    let meta = archive.pmtiles().metadata().unwrap();
    let t = &meta.extra[TILER_METADATA_KEY];
    assert_eq!(t["extent"], 4096);
    assert!(t["world_side"].as_f64().unwrap() > 1000.0);
    let z0: Tile = archive.tile(0, 0, 0).unwrap().unwrap();
    assert!(z0.layers.iter().any(|l| l.name == "lines"));
    // The drawing's bounds are inside the header's, and the header is not the
    // whole world.
    let (w, s, e, n) = archive.pmtiles().bounds();
    assert!(w > -180.0 && e < 180.0 && s > -85.06 && n < 85.06 && w < e && s < n);
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn a_failed_run_publishes_nothing() {
    let dir = std::env::temp_dir().join(format!("tiler1032-fail-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("f.pmtiles");
    let cfg = TilerConfig {
        max_zoom: 3,
        max_tile_bytes: 10,
        oversize: OversizePolicy::Error,
        ..TilerConfig::default()
    };
    assert!(build_archive(&golden(), &cfg, &path, "x").is_err());
    assert!(!path.exists());
    let _ = std::fs::remove_dir_all(dir);
}
