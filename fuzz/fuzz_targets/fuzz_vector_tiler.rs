#![no_main]

use std::time::Duration;

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use libviprs::cad::{Arc, Circle, Line, Polyline, Primitive, Text};
use libviprs::vector::tiler::{CollectTiles, TilerConfig, tile_primitives};

/// Fuzz the tiler with primitives a drawing could hand it.
///
/// An uploaded DWG reaches the tiler as primitives, and the constructors refuse
/// what is not finite, so what is left to attack is everything they accept:
/// coordinates at either end of the `f64` range, zero and enormous radii,
/// sweeps of a million turns, bulges that are nearly vertical, labels full of
/// MTEXT codes, and polygons with thousands of identical vertices.
///
/// The property is that the tiler returns. An `Err` is a fine answer (a budget
/// or the deadline); a panic, an abort or a hang is the finding. The deadline
/// is what turns a hang into an `Err` here, and is the same deadline a worker
/// sets, so a slow input that is stopped by it is the design working.
#[derive(Debug, Arbitrary)]
enum Shape {
    Line([f64; 4]),
    Circle([f64; 3]),
    Arc([f64; 5]),
    Text {
        at: [f64; 2],
        height: f64,
        angle: f64,
        s: String,
    },
    Poly {
        points: Vec<[f64; 3]>,
        closed: bool,
    },
}

#[derive(Debug, Arbitrary)]
struct Input {
    shapes: Vec<Shape>,
    zoom: u8,
    extent_choice: u8,
    max_text: u16,
}

const Z: [f64; 3] = [0.0, 0.0, 1.0];

fuzz_target!(|input: Input| {
    let mut prims: Vec<Primitive> = Vec::new();
    for s in input.shapes.iter().take(256) {
        let p = match s {
            Shape::Line(v) => Line::new([v[0], v[1], 0.0], [v[2], v[3], 0.0]).map(Primitive::from),
            Shape::Circle(v) => Circle::new([v[0], v[1], 0.0], v[2], Z).map(Primitive::from),
            Shape::Arc(v) => Arc::new([v[0], v[1], 0.0], v[2], v[3], v[4], Z).map(Primitive::from),
            Shape::Text {
                at,
                height,
                angle,
                s,
            } => Text::new([at[0], at[1], 0.0], *height, *angle, s).map(Primitive::from),
            Shape::Poly { points, closed } => {
                let v: Vec<[f64; 3]> = points.iter().take(512).map(|p| [p[0], p[1], 0.0]).collect();
                let b: Vec<f64> = points.iter().take(512).map(|p| p[2]).collect();
                Polyline::new(&v, *closed, Z, &b).map(Primitive::from)
            }
        };
        if let Ok(p) = p {
            prims.push(p);
        }
    }
    let extent = [16u32, 256, 4096][usize::from(input.extent_choice) % 3];
    let cfg = TilerConfig {
        buffer: 8,
        ..TilerConfig::default()
            .with_max_zoom(input.zoom % 9)
            .with_extent(extent)
            .with_max_text_bytes(usize::from(input.max_text))
            .with_max_level_bytes(64 << 20)
            .with_deadline(Duration::from_secs(5))
    };
    let mut out = CollectTiles::default();
    let _ = tile_primitives(&prims, &cfg, &mut out);
});
