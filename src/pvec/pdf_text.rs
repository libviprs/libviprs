//! The text a PDF page shows, as runs the PVec text stream can carry (issue #1578, L7).
//!
//! [`TextTracker`] follows the text state (`Tm`, `Tf`, `Tc`, `Tw`, `Tz`, `TL`, `Ts` and the moves)
//! through a content stream and turns each fill-mode `Tj`, `TJ`, `'` and `"` into a [`PdfTextRun`]:
//! the string decoded through `/ToUnicode` (or the byte itself for a simple font without one), the
//! baseline in page space and the cap height. A Type0 font without `/ToUnicode` shows nothing
//! decodable and stays omitted, as `docs/pvec-text-policy.md` says.

use super::pdf_vector::PdfPoint;
use lopdf::{Dictionary, Document, Object};
use std::collections::BTreeMap;
use std::rc::Rc;

type Matrix = [f64; 6];
const IDENTITY: Matrix = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0];

/// `a` applied first, then `b`.
fn then(a: &Matrix, b: &Matrix) -> Matrix {
    [
        a[0] * b[0] + a[1] * b[2],
        a[0] * b[1] + a[1] * b[3],
        a[2] * b[0] + a[3] * b[2],
        a[2] * b[1] + a[3] * b[3],
        a[4] * b[0] + a[5] * b[2] + b[4],
        a[4] * b[1] + a[5] * b[3] + b[5],
    ]
}

fn apply(m: &Matrix, p: PdfPoint) -> PdfPoint {
    [
        m[0] * p[0] + m[2] * p[1] + m[4],
        m[1] * p[0] + m[3] * p[1] + m[5],
    ]
}

/// One text-showing operator, in page space (points, y up, `/Rotate` applied).
#[derive(Debug, Clone, PartialEq)]
pub struct PdfTextRun {
    /// The text shown.
    pub text: String,
    /// The origin of the first glyph on its baseline.
    pub start: PdfPoint,
    /// The end of the baseline: the origin plus the advance.
    pub end: PdfPoint,
    /// The cap height: font size times the scale of the matrices times `/CapHeight` over 1000
    /// (0.7 em when the descriptor has none).
    pub cap_height: f64,
    /// The fill colour in force, as sRGB.
    pub fill_colour: [u8; 3],
    /// The number of entities painted before the run, like [`super::PdfImagePlacement`].
    pub sequence: u64,
}

/// What a font tells the tracker: how to read its codes and how wide they are.
struct Font {
    /// Bytes per code: 2 for an Identity Type0 font, 1 otherwise.
    code_bytes: usize,
    to_unicode: BTreeMap<u32, String>,
    /// Whether codes without a `/ToUnicode` entry are their own Latin-1 character.
    latin1: bool,
    first_char: u32,
    widths: Vec<f64>,
    cid_widths: BTreeMap<u32, f64>,
    default_width: f64,
    cap_height: f64,
}

/// What the interpreter knows around an operator: the matrix from user space to page space, the
/// `Tr` and fill colour in force, and the number of entities painted so far.
pub(crate) struct TextContext {
    pub(crate) ctm: [f64; 6],
    pub(crate) render_mode: i64,
    pub(crate) fill: [u8; 3],
    pub(crate) sequence: u64,
}

/// The text state of a content stream and the runs it has shown.
#[derive(Default)]
pub(crate) struct TextTracker {
    tm: Matrix,
    tlm: Matrix,
    size: f64,
    char_space: f64,
    word_space: f64,
    scale: f64,
    leading: f64,
    rise: f64,
    font: Option<Rc<Font>>,
    cache: BTreeMap<(u32, u16), Rc<Font>>,
    /// The runs shown so far, in content-stream order.
    pub(crate) runs: Vec<PdfTextRun>,
}

impl TextTracker {
    pub(crate) fn new() -> Self {
        Self {
            tm: IDENTITY,
            tlm: IDENTITY,
            scale: 1.0,
            ..Self::default()
        }
    }

    /// Follows one operator.
    pub(crate) fn operator(
        &mut self,
        doc: &Document,
        resources: Option<&Dictionary>,
        operator: &str,
        operands: &[Object],
        context: &TextContext,
    ) {
        let num = |i: usize| operands.get(i).and_then(number);
        match operator {
            "BT" => {
                self.tm = IDENTITY;
                self.tlm = IDENTITY;
            }
            "Tc" => self.char_space = num(0).unwrap_or(0.0),
            "Tw" => self.word_space = num(0).unwrap_or(0.0),
            "Tz" => self.scale = num(0).unwrap_or(100.0) / 100.0,
            "TL" => self.leading = num(0).unwrap_or(0.0),
            "Ts" => self.rise = num(0).unwrap_or(0.0),
            "Tf" => {
                self.size = num(1).unwrap_or(0.0);
                self.font = self.load_font(doc, resources, operands.first());
            }
            "Td" | "TD" => {
                let (tx, ty) = (num(0).unwrap_or(0.0), num(1).unwrap_or(0.0));
                if operator == "TD" {
                    self.leading = -ty;
                }
                self.move_line(tx, ty);
            }
            "Tm" => {
                if operands.len() == 6 {
                    let v: Vec<f64> = operands.iter().filter_map(number).collect();
                    if v.len() == 6 {
                        self.tm = [v[0], v[1], v[2], v[3], v[4], v[5]];
                        self.tlm = self.tm;
                    }
                }
            }
            "T*" => self.move_line(0.0, -self.leading),
            "Tj" | "TJ" | "'" | "\"" => {
                if operator == "'" || operator == "\"" {
                    if operator == "\"" {
                        self.word_space = num(0).unwrap_or(self.word_space);
                        self.char_space = num(1).unwrap_or(self.char_space);
                    }
                    self.move_line(0.0, -self.leading);
                }
                self.show(operator, operands, context);
            }
            _ => {}
        }
    }

    fn move_line(&mut self, tx: f64, ty: f64) {
        self.tlm = then(&[1.0, 0.0, 0.0, 1.0, tx, ty], &self.tlm);
        self.tm = self.tlm;
    }

    fn show(&mut self, operator: &str, operands: &[Object], context: &TextContext) {
        let TextContext {
            ctm,
            render_mode,
            fill,
            sequence,
        } = *context;
        let Some(font) = self.font.clone() else {
            return;
        };
        let mut pieces: Vec<(&[u8], f64)> = Vec::new();
        fn strings(o: &Object) -> Option<&[u8]> {
            match o {
                Object::String(bytes, _) => Some(bytes.as_slice()),
                _ => None,
            }
        }
        match operator {
            "TJ" => {
                let Some(Object::Array(items)) = operands.first() else {
                    return;
                };
                for item in items {
                    match (strings(item), number(item)) {
                        (Some(bytes), _) => pieces.push((bytes, 0.0)),
                        (None, Some(n)) => pieces.push((&[], n)),
                        _ => {}
                    }
                }
            }
            "\"" => pieces.extend(operands.get(2).and_then(strings).map(|b| (b, 0.0))),
            _ => pieces.extend(operands.first().and_then(strings).map(|b| (b, 0.0))),
        }
        let mut text = String::new();
        let mut advance = 0.0;
        let (fs, th) = (self.size, self.scale);
        for (bytes, kern) in pieces {
            advance -= kern / 1000.0 * fs * th;
            for code in bytes.chunks(font.code_bytes) {
                let code = code.iter().fold(0u32, |acc, b| (acc << 8) | u32::from(*b));
                match font.to_unicode.get(&code) {
                    Some(s) => text.push_str(s),
                    None if font.latin1 => {
                        text.extend(char::from_u32(code).filter(|c| !c.is_control()));
                    }
                    None => {}
                }
                let width = font.width(code);
                let space = if font.code_bytes == 1 && code == 32 {
                    self.word_space
                } else {
                    0.0
                };
                advance += (width / 1000.0 * fs + self.char_space + space) * th;
            }
        }
        let visible = render_mode == 0 || render_mode == 2;
        let text = text.trim().to_owned();
        let trm = then(&self.tm, &ctm);
        if visible && !text.is_empty() && fs > 0.0 && advance.is_finite() {
            let start = apply(&trm, [0.0, self.rise]);
            let end = apply(&trm, [advance, self.rise]);
            // Basic arithmetic and `sqrt`, not libm `hypot`, which differs in the last bit by architecture.
            let scale = (trm[2] * trm[2] + trm[3] * trm[3]).sqrt();
            self.runs.push(PdfTextRun {
                text,
                start,
                end,
                cap_height: fs * scale * font.cap_height / 1000.0,
                fill_colour: fill,
                sequence,
            });
        }
        self.tm = then(&[1.0, 0.0, 0.0, 1.0, advance, 0.0], &self.tm);
    }

    fn load_font(
        &mut self,
        doc: &Document,
        resources: Option<&Dictionary>,
        name: Option<&Object>,
    ) -> Option<Rc<Font>> {
        let Some(Object::Name(name)) = name else {
            return None;
        };
        let entry = resources?
            .get(b"Font")
            .ok()
            .and_then(|o| doc.dereference(o).ok())?
            .1
            .as_dict()
            .ok()?
            .get(name)
            .ok()?;
        let key = match entry {
            Object::Reference(id) => Some(*id),
            _ => None,
        };
        if let Some(font) = key.and_then(|k| self.cache.get(&k)) {
            return Some(font.clone());
        }
        let font = Rc::new(Font::read(
            doc,
            doc.dereference(entry).ok()?.1.as_dict().ok()?,
        )?);
        if let Some(key) = key {
            self.cache.insert(key, font.clone());
        }
        Some(font)
    }
}

impl Font {
    fn width(&self, code: u32) -> f64 {
        if self.code_bytes == 2 {
            return self
                .cid_widths
                .get(&code)
                .copied()
                .unwrap_or(self.default_width);
        }
        code.checked_sub(self.first_char)
            .and_then(|i| self.widths.get(i as usize))
            .copied()
            .unwrap_or(self.default_width)
    }

    /// `None` for a font whose text cannot be decoded: a Type0 font without `/ToUnicode`, or a
    /// composite font that is not Identity-encoded.
    fn read(doc: &Document, font: &Dictionary) -> Option<Font> {
        let dict = |o: &Object| doc.dereference(o).ok()?.1.as_dict().ok().cloned();
        let is_type0 = font.get(b"Subtype").ok()?.as_name().ok()? == b"Type0";
        let to_unicode = font
            .get(b"ToUnicode")
            .ok()
            .and_then(|o| doc.dereference(o).ok())
            .and_then(|(_, o)| o.as_stream().ok())
            .and_then(|s| s.decompressed_content().ok())
            .map(|bytes| parse_to_unicode(&bytes))
            .unwrap_or_default();
        let mut out = Font {
            code_bytes: 1,
            to_unicode,
            latin1: !is_type0,
            first_char: 0,
            widths: Vec::new(),
            cid_widths: BTreeMap::new(),
            default_width: 500.0,
            cap_height: 700.0,
        };
        let descriptor_cap = |holder: &Dictionary| {
            holder
                .get(b"FontDescriptor")
                .ok()
                .and_then(dict)
                .and_then(|d| d.get(b"CapHeight").ok().and_then(number))
                .filter(|c| *c > 0.0)
        };
        if is_type0 {
            let encoding = font.get(b"Encoding").ok()?.as_name().ok()?;
            if !encoding.starts_with(b"Identity") || out.to_unicode.is_empty() {
                return None;
            }
            out.code_bytes = 2;
            let cid = font
                .get(b"DescendantFonts")
                .ok()
                .and_then(|d| doc.dereference(d).ok())
                .and_then(|(_, o)| o.as_array().ok())
                .and_then(|a| a.first())
                .and_then(dict)?;
            out.default_width = cid.get(b"DW").ok().and_then(number).unwrap_or(1000.0);
            if let Some(cap) = descriptor_cap(&cid) {
                out.cap_height = cap;
            }
            if let Some(Object::Array(w)) = cid
                .get(b"W")
                .ok()
                .and_then(|o| doc.dereference(o).ok())
                .map(|(_, o)| o)
            {
                read_cid_widths(doc, w, &mut out.cid_widths);
            }
        } else {
            out.first_char = font
                .get(b"FirstChar")
                .ok()
                .and_then(number)
                .map_or(0, |n| n.max(0.0) as u32);
            if let Some(Object::Array(w)) = font
                .get(b"Widths")
                .ok()
                .and_then(|o| doc.dereference(o).ok())
                .map(|(_, o)| o)
            {
                out.widths = w.iter().map(|o| number(o).unwrap_or(0.0)).collect();
            }
            if let Some(d) = font.get(b"FontDescriptor").ok().and_then(dict) {
                out.default_width = d
                    .get(b"MissingWidth")
                    .ok()
                    .and_then(number)
                    .unwrap_or(500.0);
            }
            if let Some(cap) = descriptor_cap(font) {
                out.cap_height = cap;
            }
        }
        Some(out)
    }
}

fn number(o: &Object) -> Option<f64> {
    match o {
        Object::Integer(i) => Some(*i as f64),
        Object::Real(r) => Some(f64::from(*r)),
        _ => None,
    }
}

/// `/W` of a CID font: `c [w1 w2 ...]` and `c_first c_last w`.
fn read_cid_widths(doc: &Document, w: &[Object], out: &mut BTreeMap<u32, f64>) {
    let mut i = 0;
    while i < w.len() {
        let Some(first) = number(&w[i]).map(|n| n as u32) else {
            return;
        };
        match w
            .get(i + 1)
            .and_then(|o| doc.dereference(o).ok())
            .map(|(_, o)| o)
        {
            Some(Object::Array(list)) => {
                for (k, width) in list.iter().enumerate() {
                    if let Some(width) = number(width) {
                        out.insert(first + k as u32, width);
                    }
                }
                i += 2;
            }
            Some(last) => {
                let (Some(last), Some(width)) = (
                    number(last).map(|n| n as u32),
                    w.get(i + 2).and_then(number),
                ) else {
                    return;
                };
                for cid in first..=last.min(first.saturating_add(65_535)) {
                    out.insert(cid, width);
                }
                i += 3;
            }
            None => return,
        }
    }
}

/// The `bfchar` and `bfrange` entries of a `/ToUnicode` CMap.
fn parse_to_unicode(bytes: &[u8]) -> BTreeMap<u32, String> {
    let text = String::from_utf8_lossy(bytes);
    let hex = |t: &str| u32::from_str_radix(t, 16).ok();
    let utf16 = |t: &str| {
        let units: Vec<u16> = (0..t.len() / 4)
            .filter_map(|k| u16::from_str_radix(t.get(k * 4..k * 4 + 4)?, 16).ok())
            .collect();
        String::from_utf16_lossy(&units)
    };
    let mut map = BTreeMap::new();
    for (begin, end, range) in [
        ("beginbfchar", "endbfchar", false),
        ("beginbfrange", "endbfrange", true),
    ] {
        let mut rest = text.as_ref();
        while let Some(at) = rest.find(begin) {
            rest = &rest[at + begin.len()..];
            let stop = rest.find(end).unwrap_or(rest.len());
            // Tokens: the hex strings between `<` and `>` (not always separated by white space),
            // and the brackets of a destination array.
            let mut tokens: Vec<String> = Vec::new();
            let mut chars = rest[..stop].chars();
            while let Some(c) = chars.next() {
                match c {
                    '<' => tokens.push(chars.by_ref().take_while(|c| *c != '>').collect()),
                    '[' | ']' => tokens.push(c.to_string()),
                    _ => {}
                }
            }
            let mut k = 0;
            while k < tokens.len() {
                if !range {
                    let Some(dest) = tokens.get(k + 1) else { break };
                    if let Some(code) = hex(&tokens[k]) {
                        map.insert(code, utf16(dest));
                    }
                    k += 2;
                    continue;
                }
                let (Some(lo), Some(hi), Some(dest)) = (
                    tokens.get(k).and_then(|t| hex(t)),
                    tokens.get(k + 1).and_then(|t| hex(t)),
                    tokens.get(k + 2),
                ) else {
                    break;
                };
                let hi = hi.min(lo.saturating_add(65_535));
                if dest == "[" {
                    let list: Vec<&String> =
                        tokens[k + 3..].iter().take_while(|t| *t != "]").collect();
                    for (code, item) in (lo..=hi).zip(&list) {
                        map.insert(code, utf16(item));
                    }
                    k += 4 + list.len();
                } else {
                    for code in lo..=hi {
                        let mut units = utf16(dest);
                        if let Some(last) = units.pop() {
                            units.extend(char::from_u32(u32::from(last) + (code - lo)));
                        }
                        map.insert(code, units);
                    }
                    k += 3;
                }
            }
            rest = &rest[stop..];
        }
    }
    map
}
