//! Tiling patterns of a PDF page read into cells (issue #1578, L3).
//!
//! A `/PatternType 1` pattern becomes a [`PdfTiling`]: its cell box, steps, paint type, matrix and
//! the paths its content stream paints, as primitives in pattern space (curves flattened to 8
//! steps). Only path painting is read. A cell that shows text or an image, paints a shading, or
//! has more primitives or points than PVec's stream 25 holds is not read (`None`), and the caller
//! keeps the flat fill colour and counts it.

use super::pdf_vector::PdfFillRule;
use lopdf::content::Content;
use lopdf::{Document, Object};

/// Primitives a cell holds, and points in all (the limits of PVec's stream 25).
const MAX_PRIMITIVES: usize = 256;
const MAX_POINTS: usize = 8192;

/// One piece of a cell, in pattern space.
#[derive(Debug, Clone, PartialEq)]
pub enum PdfTilingPrimitive {
    /// A filled polygon (not closed again at the end).
    Polygon {
        /// The fill rule.
        rule: PdfFillRule,
        /// The fill colour as sRGB.
        rgb: [u8; 3],
        /// The corners.
        points: Vec<[f64; 2]>,
    },
    /// A stroked polyline.
    Polyline {
        /// The stroke colour as sRGB.
        rgb: [u8; 3],
        /// The line width in pattern units.
        width: f64,
        /// The points.
        points: Vec<[f64; 2]>,
    },
}

/// A tiling pattern: a cell of paths repeated at a step.
#[derive(Debug, Clone, PartialEq)]
pub struct PdfTiling {
    /// `/PaintType 1`: the cell sets its own colours. False for 2, painted in the colour of the fill.
    pub coloured: bool,
    /// `/XStep` and `/YStep`.
    pub steps: [f64; 2],
    /// `/BBox` as `x0 y0 x1 y1`.
    pub cell: [f64; 4],
    /// Pattern space to page space, `[a b c d e f]` in PDF's row-vector convention: `/Matrix`
    /// composed with the page's own map.
    pub matrix: [f64; 6],
    /// The paths of the cell in painting order.
    pub primitives: Vec<PdfTilingPrimitive>,
}

fn number(o: &Object) -> Option<f64> {
    match o {
        Object::Integer(i) => Some(*i as f64),
        Object::Real(f) => Some(f64::from(*f)),
        _ => None,
    }
}

/// `first` and then `next`: a point goes through `first` first.
fn then(first: [f64; 6], next: [f64; 6]) -> [f64; 6] {
    let [a, b, c, d, e, f] = first;
    let [na, nb, nc, nd, ne, nf] = next;
    [
        a * na + b * nc,
        a * nb + b * nd,
        c * na + d * nc,
        c * nb + d * nd,
        e * na + f * nc + ne,
        e * nb + f * nd + nf,
    ]
}

/// The tiling pattern `pattern` (a stream) under `base`, the page's pattern space to page map.
pub(crate) fn read_tiling(doc: &Document, pattern: &Object, base: [f64; 6]) -> Option<PdfTiling> {
    let stream = match pattern {
        Object::Reference(id) => doc.get_object(*id).ok()?.as_stream().ok()?,
        other => other.as_stream().ok()?,
    };
    let dict = &stream.dict;
    let get = |key: &[u8]| -> Option<&Object> {
        match dict.get(key).ok()? {
            Object::Reference(id) => doc.get_object(*id).ok(),
            other => Some(other),
        }
    };
    let numbers = |key: &[u8], n: usize| -> Option<Vec<f64>> {
        let values: Vec<f64> = get(key)?
            .as_array()
            .ok()?
            .iter()
            .map(|o| match o {
                Object::Reference(id) => doc.get_object(*id).ok().and_then(number),
                other => number(other),
            })
            .collect::<Option<_>>()?;
        (values.len() == n && values.iter().all(|v| v.is_finite())).then_some(values)
    };
    if get(b"PatternType").and_then(number) != Some(1.0) {
        return None;
    }
    let coloured = match get(b"PaintType").and_then(number)? as i64 {
        1 => true,
        2 => false,
        _ => return None,
    };
    let bbox = numbers(b"BBox", 4)?;
    let cell = [bbox[0], bbox[1], bbox[2], bbox[3]];
    let steps = [
        get(b"XStep").and_then(number)?,
        get(b"YStep").and_then(number)?,
    ];
    if cell[2] <= cell[0] || cell[3] <= cell[1] || steps.iter().any(|s| !s.is_finite() || *s <= 0.0)
    {
        return None;
    }
    let matrix = match numbers(b"Matrix", 6) {
        Some(m) => then([m[0], m[1], m[2], m[3], m[4], m[5]], base),
        None if dict.has(b"Matrix") => return None,
        None => base,
    };
    let bytes = stream
        .decompressed_content()
        .unwrap_or_else(|_| stream.content.clone());
    let primitives = read_cell(&Content::decode(&bytes).ok()?.operations)?;
    (!primitives.is_empty()).then_some(PdfTiling {
        coloured,
        steps,
        cell,
        matrix,
        primitives,
    })
}

/// The paths a cell's content stream paints. `None` for anything else it paints.
fn read_cell(operations: &[lopdf::content::Operation]) -> Option<Vec<PdfTilingPrimitive>> {
    #[derive(Clone, Copy)]
    struct State {
        ctm: [f64; 6],
        fill: [u8; 3],
        stroke: [u8; 3],
        width: f64,
    }
    let mut state = State {
        ctm: [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        fill: [0, 0, 0],
        stroke: [0, 0, 0],
        width: 1.0,
    };
    let mut saved: Vec<State> = Vec::new();
    let mut subpaths: Vec<(Vec<[f64; 2]>, bool)> = Vec::new();
    let mut out: Vec<PdfTilingPrimitive> = Vec::new();
    let mut points = 0usize;
    let byte = |v: f64| (v.clamp(0.0, 1.0) * 255.0).round() as u8;
    for op in operations {
        let n: Vec<f64> = op.operands.iter().filter_map(number).collect();
        let at = |state: &State, x: f64, y: f64| {
            [
                x * state.ctm[0] + y * state.ctm[2] + state.ctm[4],
                x * state.ctm[1] + y * state.ctm[3] + state.ctm[5],
            ]
        };
        let colour = |n: &[f64]| match n {
            [g] => Some([byte(*g); 3]),
            [r, g, b] => Some([byte(*r), byte(*g), byte(*b)]),
            [c, m, y, k] => {
                let ch = |v: f64| byte((1.0 - v.clamp(0.0, 1.0)) * (1.0 - k.clamp(0.0, 1.0)));
                Some([ch(*c), ch(*m), ch(*y)])
            }
            _ => None,
        };
        match (op.operator.as_str(), n.as_slice()) {
            ("q", _) => saved.push(state),
            ("Q", _) => state = saved.pop()?,
            ("cm", &[a, b, c, d, e, f]) => state.ctm = then([a, b, c, d, e, f], state.ctm),
            ("w", &[w]) => state.width = w,
            ("g" | "rg" | "k", n) => state.fill = colour(n)?,
            ("G" | "RG" | "K", n) => state.stroke = colour(n)?,
            ("m", &[x, y]) => subpaths.push((vec![at(&state, x, y)], false)),
            ("l", &[x, y]) => subpaths.last_mut()?.0.push(at(&state, x, y)),
            ("c" | "v" | "y", n) => {
                let (current, _) = subpaths.last()?;
                let start = *current.last()?;
                let (c1, c2, end) = match (op.operator.as_str(), n) {
                    ("c", &[a, b, c, d, e, f]) => {
                        (at(&state, a, b), at(&state, c, d), at(&state, e, f))
                    }
                    ("v", &[c, d, e, f]) => (start, at(&state, c, d), at(&state, e, f)),
                    ("y", &[a, b, e, f]) => (at(&state, a, b), at(&state, e, f), at(&state, e, f)),
                    _ => return None,
                };
                let path = &mut subpaths.last_mut()?.0;
                for step in 1..=8 {
                    let t = f64::from(step) / 8.0;
                    let u = 1.0 - t;
                    path.push(std::array::from_fn(|i| {
                        u * u * u * start[i]
                            + 3.0 * u * u * t * c1[i]
                            + 3.0 * u * t * t * c2[i]
                            + t * t * t * end[i]
                    }));
                }
            }
            ("re", &[x, y, w, h]) => subpaths.push((
                vec![
                    at(&state, x, y),
                    at(&state, x + w, y),
                    at(&state, x + w, y + h),
                    at(&state, x, y + h),
                ],
                true,
            )),
            ("h", _) => subpaths.last_mut()?.1 = true,
            ("f" | "F" | "f*" | "B" | "B*" | "b" | "b*" | "S" | "s" | "n", _) => {
                let paint = op.operator.as_str();
                let rule = if paint.ends_with('*') {
                    PdfFillRule::EvenOdd
                } else {
                    PdfFillRule::NonZero
                };
                let fills = matches!(paint, "f" | "F" | "f*" | "B" | "B*" | "b" | "b*");
                let strokes = matches!(paint, "B" | "B*" | "b" | "b*" | "S" | "s");
                for (path, closed) in std::mem::take(&mut subpaths) {
                    let mut path = path;
                    path.dedup();
                    let ring = {
                        let mut ring = path.clone();
                        if ring.len() > 1 && ring.first() == ring.last() {
                            ring.pop();
                        }
                        ring
                    };
                    if fills && ring.len() >= 3 {
                        points += ring.len();
                        out.push(PdfTilingPrimitive::Polygon {
                            rule,
                            rgb: state.fill,
                            points: ring,
                        });
                    }
                    if strokes && path.len() >= 2 {
                        if (closed || matches!(paint, "s" | "b" | "b*"))
                            && path.first() != path.last()
                        {
                            path.push(path[0]);
                        }
                        // The width is in cell units, scaled by the cm in force.
                        let scale = (state.ctm[0] * state.ctm[3] - state.ctm[1] * state.ctm[2])
                            .abs()
                            .sqrt();
                        points += path.len();
                        out.push(PdfTilingPrimitive::Polyline {
                            rgb: state.stroke,
                            width: state.width * scale,
                            points: path,
                        });
                    }
                    if out.len() > MAX_PRIMITIVES || points > MAX_POINTS {
                        return None;
                    }
                }
            }
            // Clips, marks and other state are ignored, and so is what is not a path: text, images,
            // shadings and forms change what the cell looks like, so the cell is not read.
            ("BT" | "Do" | "sh" | "BI" | "Tj" | "TJ", _) => return None,
            _ => {}
        }
    }
    if out.iter().any(|p| match p {
        PdfTilingPrimitive::Polygon { points, .. }
        | PdfTilingPrimitive::Polyline { points, .. } => points.len() > 4096,
    }) {
        return None;
    }
    Some(out)
}
