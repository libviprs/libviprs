//! Shadings of a PDF page read into gradients (issue #1578, L2).
//!
//! An axial or radial shading (type 2 or 3) becomes a [`PdfShading`]: its geometry, its extend
//! flags and its colours as stops in sRGB. The colour comes from a PDF function (types 0, 2, 3 and
//! 4, or an array of them) and a colour space (device spaces, `CalGray`, `CalRGB`, `ICCBased` by
//! its component count, `Separation` and `DeviceN` through their tint transform), sampled until a
//! straight line between neighbouring stops is within one level of the function, at most 256
//! stops. The PVec never holds a function (docs/PAINT.md 2.4), so the error accepted is reported
//! in [`PdfShading::max_error`] and nothing is hidden in it.
//!
//! The other shading types (function based, free-form and lattice meshes, Coons and tensor
//! patches) are counted by the caller, not sampled.

use lopdf::{Dictionary, Document, Object};

/// The most stops a gradient holds, which is what PVec's capability 25 allows.
const MAX_STOPS: usize = 256;
/// The largest deviation from a straight line between stops that is accepted, in sRGB levels.
const TOLERANCE: f64 = 1.0;
/// Nesting of functions and colour spaces followed before giving up.
const MAX_DEPTH: u32 = 8;
/// Steps a PostScript calculator function may run per evaluation.
const MAX_CALC_STEPS: u32 = 100_000;
/// The bytes of a sampled function's table that are read.
const MAX_SAMPLE_BYTES: usize = 1 << 20;

/// What a shading paints with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PdfGradientKind {
    /// `/ShadingType 2`.
    Axial,
    /// `/ShadingType 3`.
    Radial,
}

/// One colour stop.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PdfGradientStop {
    /// Position along the gradient, 0 to 1. Two stops at one offset are a hard step.
    pub offset: f64,
    /// The colour at that offset as sRGB.
    pub rgb: [u8; 3],
}

/// An axial or radial shading with its function sampled to stops.
#[derive(Debug, Clone, PartialEq)]
pub struct PdfShading {
    /// Axial or radial.
    pub kind: PdfGradientKind,
    /// `/Coords`: `x0 y0 x1 y1` for axial, `x0 y0 r0 x1 y1 r1` for radial, in gradient space.
    pub coords: Vec<f64>,
    /// `/Extend`: paint before the start, paint after the end.
    pub extend: [bool; 2],
    /// The stops, offsets non-decreasing from 0 to 1 (2 to 256 of them).
    pub stops: Vec<PdfGradientStop>,
    /// Gradient space to page space, `[a b c d e f]` in PDF's row-vector convention: the CTM for
    /// `sh`, the pattern matrix composed with the page's own map for a shading pattern.
    pub matrix: [f64; 6],
    /// `/BBox` in gradient space as `[x0 y0 x1 y1]`, when the dictionary has one.
    pub bbox: Option<[f64; 4]>,
    /// The worst deviation, in sRGB levels, of a point between two stops from the straight line
    /// the stops give, among the points checked while sampling.
    pub max_error: f64,
}

/// What a path is filled with when it is not a flat colour.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PdfFillPattern {
    /// A shading pattern: an index into [`PdfVectorPage::shadings`](super::pdf_vector::PdfVectorPage::shadings).
    Shading(usize),
}

/// An `sh` operator: the shading it painted over the clip in force.
#[derive(Debug, Clone, PartialEq)]
pub struct PdfShadingPaint {
    /// Index into [`PdfVectorPage::shadings`](super::pdf_vector::PdfVectorPage::shadings).
    pub shading: usize,
    /// The `sequence` of every clip-defining entity in force, outermost first.
    pub active_clips: Vec<u64>,
    /// Whether a soft mask was in the graphics state.
    pub soft_masked: bool,
    /// Where it came from. `sequence` is the number of entities painted before the `sh`, so the
    /// paint order against paths and images can be told.
    pub provenance: super::pdf_vector::PdfProvenance,
}

fn number(o: &Object) -> Option<f64> {
    match o {
        Object::Integer(i) => Some(*i as f64),
        Object::Real(f) => Some(f64::from(*f)),
        _ => None,
    }
}

fn deref<'a>(doc: &'a Document, o: &'a Object) -> Option<&'a Object> {
    match o {
        Object::Reference(id) => doc.get_object(*id).ok(),
        other => Some(other),
    }
}

fn numbers(doc: &Document, o: Option<&Object>) -> Option<Vec<f64>> {
    deref(doc, o?)?
        .as_array()
        .ok()?
        .iter()
        .map(|v| deref(doc, v).and_then(number).filter(|x| x.is_finite()))
        .collect()
}

/// The naive CMYK to sRGB map the fill colour already uses, so a gradient and a flat fill of one
/// colour agree.
pub(crate) fn cmyk_to_rgb(c: f64, m: f64, y: f64, k: f64) -> [u8; 3] {
    let unit = |v: f64| v.clamp(0.0, 1.0);
    let ch = |v: f64| ((1.0 - unit(v)) * (1.0 - unit(k)) * 255.0).round() as u8;
    [ch(c), ch(m), ch(y)]
}

// ---------------------------------------------------------------------------------- functions

#[derive(Debug)]
enum Func {
    Sampled {
        domain: [f64; 2],
        encode: [f64; 2],
        decode: Vec<f64>,
        range: Vec<f64>,
        size: usize,
        bits: u32,
        data: Vec<u8>,
    },
    Exponential {
        domain: [f64; 2],
        c0: Vec<f64>,
        c1: Vec<f64>,
        n: f64,
    },
    Stitching {
        domain: [f64; 2],
        funcs: Vec<Func>,
        bounds: Vec<f64>,
        encode: Vec<f64>,
    },
    Calculator {
        domain: Vec<f64>,
        range: Vec<f64>,
        program: Vec<Op>,
    },
    Array(Vec<Func>),
}

#[derive(Debug, Clone)]
enum Op {
    Num(f64),
    Word(String),
    If(Vec<Op>),
    IfElse(Vec<Op>, Vec<Op>),
}

fn lerp(x: f64, x0: f64, x1: f64, y0: f64, y1: f64) -> f64 {
    if x1 == x0 {
        y0
    } else {
        y0 + (x - x0) * (y1 - y0) / (x1 - x0)
    }
}

impl Func {
    fn parse(doc: &Document, obj: &Object, depth: u32) -> Option<Func> {
        if depth > MAX_DEPTH {
            return None;
        }
        let obj = deref(doc, obj)?;
        if let Object::Array(items) = obj {
            let parts = items
                .iter()
                .map(|o| Func::parse(doc, o, depth + 1))
                .collect::<Option<Vec<_>>>()?;
            return (!parts.is_empty()).then_some(Func::Array(parts));
        }
        let (dict, stream) = match obj {
            Object::Dictionary(d) => (d, None),
            Object::Stream(s) => (&s.dict, Some(s)),
            _ => return None,
        };
        let get = |k: &[u8]| numbers(doc, dict.get(k).ok());
        let domain = get(b"Domain")?;
        match dict.get(b"FunctionType").ok().and_then(number)? as i64 {
            0 => {
                let stream = stream?;
                let size = *get(b"Size")?.first()? as usize;
                let bits = dict.get(b"BitsPerSample").ok().and_then(number)? as u32;
                let range = get(b"Range")?;
                let m = range.len() / 2;
                if domain.len() != 2
                    || get(b"Size")?.len() != 1
                    || size == 0
                    || m == 0
                    || !matches!(bits, 1 | 2 | 4 | 8 | 12 | 16 | 24 | 32)
                    || size.saturating_mul(m).saturating_mul(bits as usize) / 8 > MAX_SAMPLE_BYTES
                {
                    return None;
                }
                let encode = get(b"Encode").unwrap_or_else(|| vec![0.0, (size - 1) as f64]);
                let decode = get(b"Decode").unwrap_or_else(|| range.clone());
                if encode.len() != 2 || decode.len() != range.len() {
                    return None;
                }
                let data = stream
                    .decompressed_content()
                    .unwrap_or_else(|_| stream.content.clone());
                if data.len() * 8 < size * m * bits as usize {
                    return None;
                }
                Some(Func::Sampled {
                    domain: [domain[0], domain[1]],
                    encode: [encode[0], encode[1]],
                    decode,
                    range,
                    size,
                    bits,
                    data,
                })
            }
            2 => {
                let c0 = get(b"C0").unwrap_or_else(|| vec![0.0]);
                let c1 = get(b"C1").unwrap_or_else(|| vec![1.0]);
                let n = dict.get(b"N").ok().and_then(number)?;
                (domain.len() == 2 && c0.len() == c1.len() && !c0.is_empty() && n.is_finite()).then(
                    || Func::Exponential {
                        domain: [domain[0], domain[1]],
                        c0,
                        c1,
                        n,
                    },
                )
            }
            3 => {
                let list = deref(doc, dict.get(b"Functions").ok()?)?.as_array().ok()?;
                let funcs = list
                    .iter()
                    .map(|o| Func::parse(doc, o, depth + 1))
                    .collect::<Option<Vec<_>>>()?;
                let bounds = get(b"Bounds")?;
                let encode = get(b"Encode")?;
                (domain.len() == 2
                    && !funcs.is_empty()
                    && bounds.len() + 1 == funcs.len()
                    && encode.len() == 2 * funcs.len())
                .then(|| Func::Stitching {
                    domain: [domain[0], domain[1]],
                    funcs,
                    bounds,
                    encode,
                })
            }
            4 => {
                let stream = stream?;
                let range = get(b"Range")?;
                let text = stream
                    .decompressed_content()
                    .unwrap_or_else(|_| stream.content.clone());
                let program = parse_calculator(&text)?;
                (!domain.is_empty() && domain.len() % 2 == 0 && !range.is_empty()).then_some(
                    Func::Calculator {
                        domain,
                        range,
                        program,
                    },
                )
            }
            _ => None,
        }
    }

    /// The outputs for `input`, clamped to the function's domain and range.
    fn eval(&self, input: &[f64]) -> Option<Vec<f64>> {
        match self {
            Func::Exponential { domain, c0, c1, n } => {
                let x = input
                    .first()?
                    .clamp(domain[0].min(domain[1]), domain[0].max(domain[1]));
                let p = if *n == 1.0 { x } else { x.powf(*n) };
                let out: Vec<f64> = c0.iter().zip(c1).map(|(a, b)| a + p * (b - a)).collect();
                out.iter().all(|v| v.is_finite()).then_some(out)
            }
            Func::Stitching {
                domain,
                funcs,
                bounds,
                encode,
            } => {
                let x = input.first()?.clamp(domain[0], domain[1]);
                let i = bounds.iter().take_while(|b| x >= **b).count();
                let lo = if i == 0 { domain[0] } else { bounds[i - 1] };
                let hi = if i == bounds.len() {
                    domain[1]
                } else {
                    bounds[i]
                };
                let e = lerp(x, lo, hi, encode[2 * i], encode[2 * i + 1]);
                funcs[i].eval(&[e])
            }
            Func::Sampled {
                domain,
                encode,
                decode,
                range,
                size,
                bits,
                data,
            } => {
                let x = input.first()?.clamp(domain[0], domain[1]);
                let e = lerp(x, domain[0], domain[1], encode[0], encode[1])
                    .clamp(0.0, (*size - 1) as f64);
                let (i0, i1) = (e.floor() as usize, e.ceil() as usize);
                let frac = e - e.floor();
                let m = range.len() / 2;
                let max = ((1u64 << bits) - 1) as f64;
                let sample = |i: usize, j: usize| -> f64 {
                    let start = (i * m + j) * *bits as usize;
                    let mut v = 0u64;
                    for b in 0..*bits as usize {
                        let bit = start + b;
                        v = (v << 1) | u64::from((data[bit / 8] >> (7 - bit % 8)) & 1);
                    }
                    v as f64
                };
                Some(
                    (0..m)
                        .map(|j| {
                            let s = sample(i0, j) * (1.0 - frac) + sample(i1, j) * frac;
                            lerp(s, 0.0, max, decode[2 * j], decode[2 * j + 1]).clamp(
                                range[2 * j].min(range[2 * j + 1]),
                                range[2 * j].max(range[2 * j + 1]),
                            )
                        })
                        .collect(),
                )
            }
            Func::Calculator {
                domain,
                range,
                program,
            } => {
                let mut stack: Vec<f64> = input
                    .iter()
                    .zip(domain.chunks(2))
                    .map(|(v, d)| v.clamp(d[0], d[1]))
                    .collect();
                let mut steps = 0;
                run_calculator(program, &mut stack, &mut steps)?;
                let m = range.len() / 2;
                if stack.len() < m {
                    return None;
                }
                let out = &stack[stack.len() - m..];
                Some(
                    out.iter()
                        .zip(range.chunks(2))
                        .map(|(v, r)| v.clamp(r[0], r[1]))
                        .collect(),
                )
            }
            Func::Array(parts) => {
                let mut out = Vec::new();
                for p in parts {
                    out.extend(p.eval(input)?);
                }
                Some(out)
            }
        }
    }

    /// The positions inside the domain where a stitching function changes piece, outermost only.
    fn joints(&self) -> Vec<f64> {
        match self {
            Func::Stitching { bounds, .. } => bounds.clone(),
            Func::Array(parts) => parts.iter().flat_map(Func::joints).collect(),
            _ => Vec::new(),
        }
    }
}

fn parse_calculator(text: &[u8]) -> Option<Vec<Op>> {
    let text = String::from_utf8_lossy(text);
    let spaced = text.replace('{', " { ").replace('}', " } ");
    let mut tokens = spaced.split_whitespace();
    if tokens.next()? != "{" {
        return None;
    }
    fn block<'a>(tokens: &mut impl Iterator<Item = &'a str>, depth: u32) -> Option<Vec<Op>> {
        if depth > MAX_DEPTH {
            return None;
        }
        let mut ops: Vec<Op> = Vec::new();
        let mut pending: Vec<Vec<Op>> = Vec::new();
        loop {
            let t = tokens.next()?;
            match t {
                "}" => return pending.is_empty().then_some(ops),
                "{" => pending.push(block(tokens, depth + 1)?),
                "if" => {
                    let body = pending.pop()?;
                    ops.push(Op::If(body));
                }
                "ifelse" => {
                    let b = pending.pop()?;
                    let a = pending.pop()?;
                    ops.push(Op::IfElse(a, b));
                }
                _ => match t.parse::<f64>() {
                    Ok(v) if v.is_finite() => ops.push(Op::Num(v)),
                    _ => ops.push(Op::Word(t.to_string())),
                },
            }
            if ops.len() > 10_000 {
                return None;
            }
        }
    }
    block(&mut tokens, 0)
}

fn run_calculator(program: &[Op], st: &mut Vec<f64>, steps: &mut u32) -> Option<()> {
    for op in program {
        *steps += 1;
        if *steps > MAX_CALC_STEPS || st.len() > 100 {
            return None;
        }
        match op {
            Op::Num(v) => st.push(*v),
            Op::If(body) => {
                if st.pop()? != 0.0 {
                    run_calculator(body, st, steps)?;
                }
            }
            Op::IfElse(a, b) => {
                let branch = if st.pop()? != 0.0 { a } else { b };
                run_calculator(branch, st, steps)?;
            }
            Op::Word(w) => word(w, st)?,
        }
    }
    Some(())
}

fn word(w: &str, st: &mut Vec<f64>) -> Option<()> {
    let b = |x: bool| if x { 1.0 } else { 0.0 };
    macro_rules! un {
        ($f:expr) => {{
            let a = st.pop()?;
            st.push($f(a));
        }};
    }
    macro_rules! bin {
        ($f:expr) => {{
            let y = st.pop()?;
            let x = st.pop()?;
            st.push($f(x, y));
        }};
    }
    match w {
        "add" => bin!(|x, y| x + y),
        "sub" => bin!(|x, y| x - y),
        "mul" => bin!(|x, y| x * y),
        "div" => bin!(|x: f64, y: f64| x / y),
        "idiv" => bin!(|x: f64, y: f64| (x.trunc() / y.trunc()).trunc()),
        "mod" => bin!(|x: f64, y: f64| x.trunc() % y.trunc()),
        "exp" => bin!(|x: f64, y: f64| x.powf(y)),
        "atan" => bin!(|n: f64, d: f64| {
            let deg = n.atan2(d).to_degrees();
            if deg < 0.0 { deg + 360.0 } else { deg }
        }),
        "neg" => un!(|a: f64| -a),
        "abs" => un!(f64::abs),
        "ceiling" => un!(f64::ceil),
        "floor" => un!(f64::floor),
        "round" => un!(|a: f64| (a + 0.5).floor()),
        "truncate" | "cvi" => un!(f64::trunc),
        "cvr" => {}
        "sqrt" => un!(f64::sqrt),
        "sin" => un!(|a: f64| a.to_radians().sin()),
        "cos" => un!(|a: f64| a.to_radians().cos()),
        "ln" => un!(f64::ln),
        "log" => un!(f64::log10),
        "eq" => bin!(|x, y| b(x == y)),
        "ne" => bin!(|x, y| b(x != y)),
        "gt" => bin!(|x, y| b(x > y)),
        "ge" => bin!(|x, y| b(x >= y)),
        "lt" => bin!(|x, y| b(x < y)),
        "le" => bin!(|x, y| b(x <= y)),
        "and" => bin!(|x: f64, y: f64| ((x as i64) & (y as i64)) as f64),
        "or" => bin!(|x: f64, y: f64| ((x as i64) | (y as i64)) as f64),
        "xor" => bin!(|x: f64, y: f64| ((x as i64) ^ (y as i64)) as f64),
        "not" => un!(|a: f64| if a == 0.0 || a == 1.0 {
            1.0 - a
        } else {
            !(a as i64) as f64
        }),
        "bitshift" => bin!(|x: f64, s: f64| {
            let (x, s) = (x as i64, s as i64);
            (if s >= 0 {
                x << s.min(62)
            } else {
                x >> (-s).min(62)
            }) as f64
        }),
        "true" => st.push(1.0),
        "false" => st.push(0.0),
        "pop" => {
            st.pop()?;
        }
        "exch" => {
            let n = st.len();
            if n < 2 {
                return None;
            }
            st.swap(n - 1, n - 2);
        }
        "dup" => st.push(*st.last()?),
        "copy" => {
            let n = st.pop()? as usize;
            if n > st.len() {
                return None;
            }
            let from = st.len() - n;
            st.extend_from_within(from..);
        }
        "index" => {
            let n = st.pop()? as usize;
            let v = *st.get(st.len().checked_sub(n + 1)?)?;
            st.push(v);
        }
        "roll" => {
            let j = st.pop()? as i64;
            let n = st.pop()? as usize;
            if n > st.len() {
                return None;
            }
            if n > 0 {
                let from = st.len() - n;
                let k = j.rem_euclid(n as i64) as usize;
                st[from..].rotate_right(k);
            }
        }
        _ => return None,
    }
    Some(())
}

// -------------------------------------------------------------------------------- colour spaces

#[derive(Debug)]
enum Space {
    Gray,
    Rgb,
    Cmyk,
    Tint {
        inputs: usize,
        func: Func,
        alternate: Box<Space>,
    },
}

impl Space {
    fn parse(
        doc: &Document,
        obj: &Object,
        resources: Option<&Dictionary>,
        depth: u32,
    ) -> Option<Space> {
        if depth > MAX_DEPTH {
            return None;
        }
        let obj = deref(doc, obj)?;
        match obj {
            Object::Name(n) => match n.as_slice() {
                b"DeviceGray" | b"G" | b"CalGray" => Some(Space::Gray),
                b"DeviceRGB" | b"RGB" | b"CalRGB" => Some(Space::Rgb),
                b"DeviceCMYK" | b"CMYK" => Some(Space::Cmyk),
                other => {
                    let named = deref(doc, resources?.get(b"ColorSpace").ok()?)?
                        .as_dict()
                        .ok()?
                        .get(other)
                        .ok()?;
                    Space::parse(doc, named, resources, depth + 1)
                }
            },
            Object::Array(a) => {
                let family = deref(doc, a.first()?)?.as_name().ok()?;
                match family {
                    b"CalGray" => Some(Space::Gray),
                    b"CalRGB" => Some(Space::Rgb),
                    b"ICCBased" => {
                        let stream = deref(doc, a.get(1)?)?.as_stream().ok()?;
                        match stream.dict.get(b"N").ok().and_then(number)? as i64 {
                            1 => Some(Space::Gray),
                            3 => Some(Space::Rgb),
                            4 => Some(Space::Cmyk),
                            _ => None,
                        }
                    }
                    b"Separation" | b"DeviceN" => {
                        let inputs = if family == b"Separation" {
                            1
                        } else {
                            deref(doc, a.get(1)?)?.as_array().ok()?.len()
                        };
                        let alternate = Space::parse(doc, a.get(2)?, resources, depth + 1)?;
                        let func = Func::parse(doc, a.get(3)?, depth + 1)?;
                        (inputs > 0).then(|| Space::Tint {
                            inputs,
                            func,
                            alternate: Box::new(alternate),
                        })
                    }
                    _ => None,
                }
            }
            _ => None,
        }
    }

    fn components(&self) -> usize {
        match self {
            Space::Gray => 1,
            Space::Rgb => 3,
            Space::Cmyk => 4,
            Space::Tint { inputs, .. } => *inputs,
        }
    }

    fn to_rgb(&self, c: &[f64]) -> Option<[u8; 3]> {
        let byte = |v: f64| (v.clamp(0.0, 1.0) * 255.0).round() as u8;
        match (self, c) {
            (Space::Gray, [g, ..]) => Some([byte(*g); 3]),
            (Space::Rgb, [r, g, b, ..]) => Some([byte(*r), byte(*g), byte(*b)]),
            (Space::Cmyk, [c, m, y, k, ..]) => Some(cmyk_to_rgb(*c, *m, *y, *k)),
            (
                Space::Tint {
                    func, alternate, ..
                },
                _,
            ) => alternate.to_rgb(&func.eval(c)?),
            _ => None,
        }
    }
}

// -------------------------------------------------------------------------------------- sampling

/// A shading dictionary or stream, read. `None` when it is not an axial or radial shading, or
/// when its function or colour space is not one this reads; the caller counts that.
///
/// `matrix` is gradient space to page space.
pub(crate) fn read_shading(
    doc: &Document,
    shading: &Object,
    resources: Option<&Dictionary>,
    matrix: [f64; 6],
) -> Option<PdfShading> {
    let dict = match deref(doc, shading)? {
        Object::Dictionary(d) => d,
        Object::Stream(s) => &s.dict,
        _ => return None,
    };
    let kind = match dict.get(b"ShadingType").ok().and_then(number)? as i64 {
        2 => PdfGradientKind::Axial,
        3 => PdfGradientKind::Radial,
        _ => return None,
    };
    let coords = numbers(doc, dict.get(b"Coords").ok())?;
    if coords.len() != if kind == PdfGradientKind::Axial { 4 } else { 6 } {
        return None;
    }
    let domain = numbers(doc, dict.get(b"Domain").ok()).unwrap_or_else(|| vec![0.0, 1.0]);
    if domain.len() != 2 || domain[0] == domain[1] {
        return None;
    }
    let extend = match dict.get(b"Extend").ok().and_then(|o| deref(doc, o)) {
        Some(Object::Array(a)) if a.len() == 2 => {
            let flag = |o: &Object| matches!(deref(doc, o), Some(Object::Boolean(true)));
            [flag(&a[0]), flag(&a[1])]
        }
        _ => [false, false],
    };
    let space = Space::parse(doc, dict.get(b"ColorSpace").ok()?, resources, 0)?;
    let func = Func::parse(doc, dict.get(b"Function").ok()?, 0)?;
    let bbox = numbers(doc, dict.get(b"BBox").ok())
        .filter(|b| b.len() == 4)
        .map(|b| [b[0], b[1], b[2], b[3]]);

    let colour = |s: f64| -> Option<[u8; 3]> {
        let t = domain[0] + s * (domain[1] - domain[0]);
        let out = func.eval(&[t])?;
        (out.len() >= space.components()).then_some(())?;
        space.to_rgb(&out)
    };
    let (stops, max_error) = sample(&func, &domain, &colour)?;
    Some(PdfShading {
        kind,
        coords,
        extend,
        stops,
        matrix,
        bbox,
        max_error,
    })
}

/// The stops for `colour` over offsets 0 to 1, and the worst deviation accepted.
fn sample(
    func: &Func,
    domain: &[f64],
    colour: &dyn Fn(f64) -> Option<[u8; 3]>,
) -> Option<(Vec<PdfGradientStop>, f64)> {
    // The pieces of a stitching function are sampled one at a time, so a joint is a stop (two,
    // for a hard step) and not something a straight line has to span.
    let mut cuts = vec![0.0];
    let span = domain[1] - domain[0];
    let mut joints: Vec<f64> = func
        .joints()
        .into_iter()
        .map(|j| (j - domain[0]) / span)
        .filter(|s| *s > 0.0 && *s < 1.0)
        .collect();
    joints.sort_by(f64::total_cmp);
    joints.dedup();
    cuts.extend(joints);
    cuts.push(1.0);

    let nudge = 1e-9;
    let mut stops: Vec<PdfGradientStop> = Vec::new();
    let mut worst = 0.0f64;
    for (n, w) in cuts.windows(2).enumerate() {
        let (a, b) = (w[0], w[1]);
        let first = if n == 0 { a } else { a + nudge };
        let last = if n == cuts.len() - 2 { b } else { b - nudge };
        let lo = colour(first)?;
        let hi = colour(last)?;
        let mut piece = vec![(a, lo)];
        let budget = MAX_STOPS.saturating_sub(stops.len() + cuts.len() * 2);
        refine(
            &mut piece,
            (a, lo),
            (b, hi),
            first,
            last,
            colour,
            budget,
            0,
            &mut worst,
        )?;
        piece.push((b, hi));
        for (offset, rgb) in piece {
            if stops
                .last()
                .is_some_and(|s| s.offset == offset && s.rgb == rgb)
            {
                continue;
            }
            stops.push(PdfGradientStop { offset, rgb });
        }
    }
    (stops.len() >= 2 && stops.len() <= MAX_STOPS).then_some((stops, worst))
}

/// Adds the stops between `a` and `b` (exclusive) that keep a straight line within tolerance.
/// `first` and `last` are where the function is read for the two ends, which differ from the
/// offsets only at a joint.
#[allow(clippy::too_many_arguments)]
fn refine(
    out: &mut Vec<(f64, [u8; 3])>,
    a: (f64, [u8; 3]),
    b: (f64, [u8; 3]),
    first: f64,
    last: f64,
    colour: &dyn Fn(f64) -> Option<[u8; 3]>,
    budget: usize,
    depth: u32,
    worst: &mut f64,
) -> Option<()> {
    let mut deviation = 0.0f64;
    for q in [0.25, 0.5, 0.75] {
        let at = first + q * (last - first);
        let got = colour(at)?;
        for (ch, &level) in got.iter().enumerate() {
            let line = f64::from(a.1[ch]) + q * (f64::from(b.1[ch]) - f64::from(a.1[ch]));
            deviation = deviation.max((f64::from(level) - line).abs());
        }
    }
    if deviation <= TOLERANCE || depth >= 12 || out.len() >= budget {
        *worst = worst.max(deviation);
        return Some(());
    }
    let mid_at = (first + last) / 2.0;
    let mid = (a.0 + (b.0 - a.0) / 2.0, colour(mid_at)?);
    refine(out, a, mid, first, mid_at, colour, budget, depth + 1, worst)?;
    out.push(mid);
    refine(out, mid, b, mid_at, last, colour, budget, depth + 1, worst)
}
