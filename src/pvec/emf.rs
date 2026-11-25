//! A bounded player for the enhanced metafile (EMF) preview of an embedded OLE object (issue #1705).
//!
//! An OLE2FRAME in a DWG carries, next to the object, the picture the application cached for display.
//! For a spreadsheet that is an EMF of a table: lines, filled cells and text. [`play`] reads that
//! record stream and gives back the shapes it draws, already mapped from the metafile's device
//! pixels onto the frame's two corners, so a conversion can carry them like any other entity of
//! the sheet. Nothing is rasterised and nothing is read from disk.
//!
//! The player knows the records such a table is made of: header and end, the device context state
//! (background mode, raster op, text alignment, text and background colour, save and restore, clip
//! rectangle and clip region), objects (pen, brush, font, select, delete), `MoveToEx`, `LineTo`,
//! `Rectangle`, a pattern `BitBlt` as a solid fill and `ExtTextOutW` as text. Any other record is
//! counted and named in the drawing's notes, never skipped silently. A record that is malformed, or
//! a preview over a limit, is a typed [`EmfRefusal`] and nothing of it is drawn.
//!
//! The picture's bounds (device pixels, left and right edge both inside) are mapped onto the
//! corners, so the table's outer lines land on them. What it does not carry: pen and text colours
//! and pen widths (a stroke takes the entity's own style), bitmaps, curves, and world transforms.

use std::collections::BTreeMap;

/// The most bytes of a preview the player reads.
const MAX_BYTES: usize = 8 << 20;
/// The most records it plays.
const MAX_RECORDS: u32 = 200_000;
/// The most shapes it carries: a table is a few hundred.
const MAX_SHAPES: usize = 4096;
/// The most object handles it keeps.
const MAX_OBJECTS: usize = 4096;
/// The most device contexts it keeps saved.
const MAX_DEPTH: usize = 64;
/// The most characters of one text run, and of all of them.
const MAX_RUN_CHARS: usize = 4096;
const MAX_TEXT_CHARS: usize = 1 << 17;
/// The most points of one polyline; a longer chain goes on in the next shape.
const MAX_POINTS: usize = 1024;
/// The most notes.
const MAX_NOTES: usize = 64;

/// The em of a face, and where its capitals and baseline sit in it, as for Arial, the face Office sets
/// a table in.
const CAP_HEIGHT: f64 = 0.716;
const ASCENT: f64 = 0.905;
const DESCENT: f64 = 0.212;
/// A positive `lfHeight` is a cell (ascent and descent), not the em.
const CELL_PER_EM: f64 = ASCENT + DESCENT;

/// How the player ended without a drawing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum EmfRefusal {
    /// The bytes are not an enhanced metafile.
    NotAMetafile,
    /// The preview is larger than the player reads.
    TooLarge {
        /// The most bytes it reads.
        limit: usize,
    },
    /// The metafile has more records than the player plays.
    TooManyRecords {
        /// The most records it plays.
        limit: u32,
    },
    /// A record is shorter than its header, is not a whole number of words or runs past the end.
    BadRecord {
        /// The record's place in the stream, counting the header as 0.
        index: u32,
    },
    /// The metafile draws more shapes than the player carries.
    TooManyShapes {
        /// The most shapes it carries.
        limit: usize,
    },
    /// An object handle past the table the player keeps.
    TooManyObjects {
        /// The most handles it keeps.
        limit: usize,
    },
    /// The picture has no width or no height to map onto the corners.
    NoExtent,
    /// More device contexts are saved than the player keeps.
    TooDeep,
}

impl std::fmt::Display for EmfRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotAMetafile => f.write_str("the preview is not an enhanced metafile"),
            Self::TooLarge { limit } => write!(f, "the preview is over {limit} bytes"),
            Self::TooManyRecords { limit } => {
                write!(f, "the preview has more than {limit} records")
            }
            Self::BadRecord { index } => write!(f, "record {index} of the preview is malformed"),
            Self::TooManyShapes { limit } => {
                write!(f, "the preview draws more than {limit} shapes")
            }
            Self::TooManyObjects { limit } => {
                write!(f, "the preview names an object handle past {limit}")
            }
            Self::NoExtent => f.write_str("the preview's picture has no width or height"),
            Self::TooDeep => write!(f, "the preview saves more than {MAX_DEPTH} device contexts"),
        }
    }
}

impl std::error::Error for EmfRefusal {}

/// One thing the preview draws, in the frame's coordinates.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum EmfShape {
    /// An open polyline.
    Line(Vec<[f64; 2]>),
    /// A closed ring filled with a solid colour (red, green, blue, alpha, one byte each).
    Fill {
        /// The corners of the ring.
        ring: Vec<[f64; 2]>,
        /// The colour.
        rgba: u32,
    },
    /// One run of text.
    Text {
        /// Where the baseline starts, or is centred or ends (see `anchor`).
        at: [f64; 2],
        /// The height of a capital.
        height: f64,
        /// The rotation in radians, counter-clockwise.
        rotation: f64,
        /// How far along the run `at` sits: 0 at its start, 0.5 in the middle, 1 at its end.
        anchor: f64,
        /// The characters.
        text: String,
    },
}

/// What a preview draws.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct EmfDrawing {
    /// The shapes, in the order the metafile draws them.
    pub shapes: Vec<EmfShape>,
    /// What was left out, one line each: a record type the player does not know, with its count.
    pub notes: Vec<String>,
}

/// What a selected object is.
#[derive(Clone, Copy)]
enum Object {
    Pen { visible: bool },
    Brush { rgba: Option<u32> },
    Font { em: f64, escapement: f64 },
}

/// The device context the records change.
#[derive(Clone)]
struct Dc {
    pen_visible: bool,
    brush: Option<u32>,
    em: f64,
    escapement: f64,
    align: u32,
    /// The clip rectangle, `[left, top, right, bottom]` in device units.
    clip: Option<[f64; 4]>,
    at: [i32; 2],
}

fn word(rec: &[u8], at: usize) -> Option<u32> {
    Some(u32::from_le_bytes(rec.get(at..at + 4)?.try_into().ok()?))
}

fn int(rec: &[u8], at: usize) -> Option<i32> {
    word(rec, at).map(|w| w as i32)
}

/// A COLORREF (`0x00BBGGRR`) as opaque red, green, blue, alpha.
fn rgba(colorref: u32) -> u32 {
    let [r, g, b, _] = colorref.to_le_bytes();
    u32::from_be_bytes([r, g, b, 255])
}

/// `[left, top, right, bottom]` read at `at`, ordered.
fn rect(rec: &[u8], at: usize) -> Option<[f64; 4]> {
    let v = [
        int(rec, at)?,
        int(rec, at + 4)?,
        int(rec, at + 8)?,
        int(rec, at + 12)?,
    ];
    Some([
        f64::from(v[0].min(v[2])),
        f64::from(v[1].min(v[3])),
        f64::from(v[0].max(v[2])),
        f64::from(v[1].max(v[3])),
    ])
}

fn intersect(a: Option<[f64; 4]>, b: [f64; 4]) -> [f64; 4] {
    a.map_or(b, |a| {
        [
            a[0].max(b[0]),
            a[1].max(b[1]),
            a[2].min(b[2]),
            a[3].min(b[3]),
        ]
    })
}

/// The part of the segment inside `clip` (Liang-Barsky), or the whole segment without one.
fn clipped(a: [f64; 2], b: [f64; 2], clip: Option<[f64; 4]>) -> Option<([f64; 2], [f64; 2])> {
    let Some(c) = clip else { return Some((a, b)) };
    let (mut t0, mut t1) = (0.0_f64, 1.0_f64);
    let d = [b[0] - a[0], b[1] - a[1]];
    for (p, q) in [
        (-d[0], a[0] - c[0]),
        (d[0], c[2] - a[0]),
        (-d[1], a[1] - c[1]),
        (d[1], c[3] - a[1]),
    ] {
        if p == 0.0 {
            if q < 0.0 {
                return None;
            }
        } else {
            let t = q / p;
            if p < 0.0 {
                t0 = t0.max(t);
            } else {
                t1 = t1.min(t);
            }
        }
    }
    (t0 <= t1).then(|| {
        (
            [a[0] + d[0] * t0, a[1] + d[1] * t0],
            [a[0] + d[0] * t1, a[1] + d[1] * t1],
        )
    })
}

/// Plays `bytes` and maps the picture onto the rectangle whose upper-left and lower-right corners
/// are `upper_left` and `lower_right`.
///
/// # Errors
///
/// An [`EmfRefusal`] for a preview that is not an EMF, is malformed, or is over a limit.
pub fn play(
    bytes: &[u8],
    upper_left: [f64; 2],
    lower_right: [f64; 2],
) -> Result<EmfDrawing, EmfRefusal> {
    if bytes.len() > MAX_BYTES {
        return Err(EmfRefusal::TooLarge { limit: MAX_BYTES });
    }
    let head_size = word(bytes, 4).map(|s| s as usize);
    if word(bytes, 0) != Some(1)
        || bytes.get(40..44) != Some(b" EMF")
        || head_size.is_none_or(|s| s < 88 || s % 4 != 0 || s > bytes.len())
    {
        return Err(EmfRefusal::NotAMetafile);
    }
    // The bounds are inclusive on both edges, so a table's outer lines sit on them.
    let bounds = rect(bytes, 8).ok_or(EmfRefusal::NotAMetafile)?;
    let (width, height) = (bounds[2] - bounds[0], bounds[3] - bounds[1]);
    let spread = [
        lower_right[0] - upper_left[0],
        lower_right[1] - upper_left[1],
    ];
    if !(width > 0.0
        && height > 0.0
        && spread.iter().all(|v| v.is_finite())
        && upper_left.iter().all(|v| v.is_finite()))
    {
        return Err(EmfRefusal::NoExtent);
    }
    let scale = [spread[0] / width, spread[1] / height];
    let map = |p: [f64; 2]| {
        [
            upper_left[0] + (p[0] - bounds[0]) * scale[0],
            upper_left[1] + (p[1] - bounds[1]) * scale[1],
        ]
    };

    let mut shapes: Vec<EmfShape> = Vec::new();
    let mut unknown: BTreeMap<u32, u32> = BTreeMap::new();
    let mut bitmaps = 0u32;
    let mut objects: Vec<Option<Object>> = Vec::new();
    let mut saved: Vec<Dc> = Vec::new();
    let mut dc = Dc {
        pen_visible: true,
        brush: Some(rgba(0x00FF_FFFF)),
        em: 16.0,
        escapement: 0.0,
        align: 0,
        clip: None,
        at: [0, 0],
    };
    // The polyline a run of LineTo records is extending, and where it ends in device units.
    let mut open: Option<(usize, [i32; 2])> = None;
    let mut chars = 0usize;

    let (mut off, mut index) = (0usize, 0u32);
    while off < bytes.len() {
        if index >= MAX_RECORDS {
            return Err(EmfRefusal::TooManyRecords { limit: MAX_RECORDS });
        }
        let bad = EmfRefusal::BadRecord { index };
        let (kind, size) = match (word(bytes, off), word(bytes, off + 4)) {
            (Some(k), Some(s)) => (k, s as usize),
            _ => return Err(bad),
        };
        if size < 8 || size % 4 != 0 || size > bytes.len() - off {
            return Err(bad);
        }
        let rec = &bytes[off..off + size];
        off += size;
        index += 1;
        let body = |at: usize| word(rec, 8 + at);
        let ibody = |at: usize| int(rec, 8 + at);
        match kind {
            // The header (checked above) and the end.
            1 if index == 1 => {}
            14 => break,
            // State this player does not need: colours, modes, comments.
            18 | 20 | 24 | 25 | 70 => {}
            22 => dc.align = body(0).ok_or(bad)?,
            33 => {
                if saved.len() >= MAX_DEPTH {
                    return Err(EmfRefusal::TooDeep);
                }
                saved.push(dc.clone());
            }
            34 => {
                let n = ibody(0).ok_or(bad)?;
                for _ in 0..n.unsigned_abs().min(MAX_DEPTH as u32) {
                    if let Some(prev) = saved.pop() {
                        dc = Dc { at: dc.at, ..prev };
                    }
                }
            }
            30 => dc.clip = Some(intersect(dc.clip, rect(rec, 8).ok_or(bad)?)),
            75 => {
                let (cb, mode) = (body(0).ok_or(bad)?, body(4).ok_or(bad)?);
                if cb == 0 {
                    if mode == 5 {
                        dc.clip = None;
                    }
                } else {
                    // The region's bounding box stands for it, which only ever clips less.
                    let r = rect(rec, 8 + 8 + 16).ok_or(bad)?;
                    match mode {
                        5 => dc.clip = Some(r),
                        1 => dc.clip = Some(intersect(dc.clip, r)),
                        _ => *unknown.entry(kind).or_default() += 1,
                    }
                }
            }
            38 | 39 | 82 | 95 => {
                let ih = body(0).ok_or(bad)? as usize;
                if ih >= MAX_OBJECTS {
                    return Err(EmfRefusal::TooManyObjects { limit: MAX_OBJECTS });
                }
                let made = match kind {
                    38 => Object::Pen {
                        visible: body(4).ok_or(bad)? & 0xF != 5,
                    },
                    95 => Object::Pen {
                        visible: body(20).ok_or(bad)? & 0xF != 5,
                    },
                    39 => Object::Brush {
                        rgba: (body(4).ok_or(bad)? == 0)
                            .then(|| body(8))
                            .flatten()
                            .map(rgba),
                    },
                    _ => {
                        let h = ibody(4).ok_or(bad)?;
                        let em = match h {
                            h if h < 0 => f64::from(h.unsigned_abs()),
                            0 => 16.0,
                            h => f64::from(h) / CELL_PER_EM,
                        };
                        Object::Font {
                            em,
                            escapement: f64::from(ibody(12).ok_or(bad)?) / 10.0,
                        }
                    }
                };
                if objects.len() <= ih {
                    objects.resize(ih + 1, None);
                }
                objects[ih] = Some(made);
            }
            37 => {
                let ih = body(0).ok_or(bad)?;
                let picked = if ih & 0x8000_0000 != 0 {
                    // The stock objects: brushes 0 to 5 and pens 6 to 8.
                    match ih & 0x7FFF_FFFF {
                        0 => Some(Object::Brush {
                            rgba: Some(rgba(0x00FF_FFFF)),
                        }),
                        1 => Some(Object::Brush {
                            rgba: Some(rgba(0x00C0_C0C0)),
                        }),
                        2 => Some(Object::Brush {
                            rgba: Some(rgba(0x0080_8080)),
                        }),
                        3 => Some(Object::Brush {
                            rgba: Some(rgba(0x0040_4040)),
                        }),
                        4 => Some(Object::Brush {
                            rgba: Some(rgba(0)),
                        }),
                        5 => Some(Object::Brush { rgba: None }),
                        6 | 7 => Some(Object::Pen { visible: true }),
                        8 => Some(Object::Pen { visible: false }),
                        _ => None,
                    }
                } else {
                    objects.get(ih as usize).copied().flatten()
                };
                match picked {
                    Some(Object::Pen { visible }) => dc.pen_visible = visible,
                    Some(Object::Brush { rgba }) => dc.brush = rgba,
                    Some(Object::Font { em, escapement }) => {
                        dc.em = em;
                        dc.escapement = escapement;
                    }
                    None => {}
                }
            }
            40 => {
                if let Some(slot) = objects.get_mut(body(0).ok_or(bad)? as usize) {
                    *slot = None;
                }
            }
            27 => dc.at = [ibody(0).ok_or(bad)?, ibody(4).ok_or(bad)?],
            54 => {
                let to = [ibody(0).ok_or(bad)?, ibody(4).ok_or(bad)?];
                let from = dc.at;
                dc.at = to;
                if dc.pen_visible
                    && let Some((a, b)) = clipped(
                        [f64::from(from[0]), f64::from(from[1])],
                        [f64::from(to[0]), f64::from(to[1])],
                        dc.clip,
                    )
                {
                    let (a, b) = (map(a), map(b));
                    match open {
                        Some((i, end)) if end == from => {
                            if let EmfShape::Line(points) = &mut shapes[i] {
                                if points.len() < MAX_POINTS && points.last() == Some(&a) {
                                    points.push(b);
                                } else {
                                    open = None;
                                }
                            }
                        }
                        _ => open = None,
                    }
                    if open.is_none() {
                        shapes.push(EmfShape::Line(vec![a, b]));
                        open = Some((shapes.len() - 1, to));
                    } else if let Some((_, end)) = &mut open {
                        *end = to;
                    }
                }
            }
            43 => {
                let r = rect(rec, 8).ok_or(bad)?;
                if let Some(color) = dc.brush
                    && let Some(r) = fill_box(r, dc.clip)
                {
                    shapes.push(fill(r, color, &map));
                }
                if dc.pen_visible {
                    let ring = [
                        [r[0], r[1]],
                        [r[2], r[1]],
                        [r[2], r[3]],
                        [r[0], r[3]],
                        [r[0], r[1]],
                    ];
                    for pair in ring.windows(2) {
                        if let Some((a, b)) = clipped(pair[0], pair[1], dc.clip) {
                            shapes.push(EmfShape::Line(vec![map(a), map(b)]));
                        }
                    }
                }
                open = None;
            }
            76 => {
                let (x, y, w, h) = (
                    ibody(16).ok_or(bad)?,
                    ibody(20).ok_or(bad)?,
                    ibody(24).ok_or(bad)?,
                    ibody(28).ok_or(bad)?,
                );
                let color = match body(32).ok_or(bad)? {
                    0x00F0_0021 => dc.brush,
                    0x0000_0042 => Some(rgba(0)),
                    0x00FF_0062 => Some(rgba(0x00FF_FFFF)),
                    _ => {
                        bitmaps += 1;
                        None
                    }
                };
                let r = [
                    f64::from(x.min(x.saturating_add(w))),
                    f64::from(y.min(y.saturating_add(h))),
                    f64::from(x.max(x.saturating_add(w))),
                    f64::from(y.max(y.saturating_add(h))),
                ];
                if let (Some(color), Some(r)) = (color, fill_box(r, dc.clip)) {
                    shapes.push(fill(r, color, &map));
                }
                open = None;
            }
            84 => {
                let (rx, ry) = (ibody(28).ok_or(bad)?, ibody(32).ok_or(bad)?);
                let (n, at) = (body(36).ok_or(bad)? as usize, body(40).ok_or(bad)? as usize);
                if n > MAX_RUN_CHARS {
                    return Err(bad);
                }
                let raw = at
                    .checked_add(n * 2)
                    .and_then(|end| rec.get(at..end))
                    .ok_or(bad)?;
                chars += n;
                if chars > MAX_TEXT_CHARS {
                    return Err(EmfRefusal::TooManyShapes { limit: MAX_SHAPES });
                }
                let text: String = char::decode_utf16(
                    raw.as_chunks::<2>()
                        .0
                        .iter()
                        .map(|c| u16::from_le_bytes(*c)),
                )
                .map(|c| c.unwrap_or(' '))
                .map(|c| if c.is_control() { ' ' } else { c })
                .collect();
                let inside = dc.clip.is_none_or(|c| {
                    f64::from(rx) >= c[0]
                        && f64::from(rx) <= c[2]
                        && f64::from(ry) >= c[1]
                        && f64::from(ry) <= c[3]
                });
                if inside && !text.trim().is_empty() {
                    // The reference point is the baseline, the top or the bottom of the cell.
                    let down = match dc.align & 0x18 {
                        0x18 => 0.0,
                        0x08 => -DESCENT,
                        _ => ASCENT,
                    } * dc.em;
                    let anchor = match dc.align & 0x6 {
                        0x2 => 1.0,
                        0x6 => 0.5,
                        _ => 0.0,
                    };
                    shapes.push(EmfShape::Text {
                        at: map([f64::from(rx), f64::from(ry) + down]),
                        height: CAP_HEIGHT * dc.em * scale[1].abs(),
                        rotation: dc.escapement.to_radians(),
                        anchor,
                        text,
                    });
                }
            }
            _ => *unknown.entry(kind).or_default() += 1,
        }
        if shapes.len() > MAX_SHAPES {
            return Err(EmfRefusal::TooManyShapes { limit: MAX_SHAPES });
        }
    }
    if index == 0 {
        return Err(EmfRefusal::NotAMetafile);
    }
    let mut notes: Vec<String> = unknown
        .into_iter()
        .map(|(kind, n)| format!("EMF record type {kind} not played ({n})"))
        .collect();
    if bitmaps > 0 {
        notes.push(format!("EMF bitmap blocks not played ({bitmaps})"));
    }
    notes.truncate(MAX_NOTES);
    Ok(EmfDrawing { shapes, notes })
}

/// `r` cut to the clip, or nothing when no area is left.
fn fill_box(r: [f64; 4], clip: Option<[f64; 4]>) -> Option<[f64; 4]> {
    let r = intersect(clip, r);
    (r[2] > r[0] && r[3] > r[1]).then_some(r)
}

fn fill(r: [f64; 4], color: u32, map: &impl Fn([f64; 2]) -> [f64; 2]) -> EmfShape {
    EmfShape::Fill {
        ring: [[r[0], r[1]], [r[2], r[1]], [r[2], r[3]], [r[0], r[3]]]
            .into_iter()
            .map(map)
            .collect(),
        rgba: color,
    }
}
