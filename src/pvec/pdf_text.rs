//! The text a PDF page shows, as runs the PVec text stream can carry (issue #1578, L7).
//!
//! [`TextTracker`] follows the text state (`Tm`, `Tf`, `Tc`, `Tw`, `Tz`, `TL`, `Ts` and the moves)
//! through a content stream and turns each fill-mode `Tj`, `TJ`, `'` and `"` into a [`PdfTextRun`]:
//! the string decoded through `/ToUnicode` (or the byte itself for a simple font without one), the
//! baseline in page space and the cap height. A Type0 font without `/ToUnicode` shows nothing
//! decodable and stays omitted, as `docs/pvec-text-policy.md` says. A Type0 font with `/Identity-H` or
//! `/Identity-V` is read through its `/ToUnicode` map, stored or filtered (issue #1687), within the
//! font bounds of [`PdfVectorLimits`].

use super::pdf_vector::{PdfPoint, PdfVectorError, PdfVectorLimits, stream_bytes};
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
    /// The `sequence` of every clip-defining entity in force, outermost first (see
    /// [`super::PdfPathEntity::active_clips`]).
    pub active_clips: Vec<u64>,
}

/// What a font tells the tracker: how to read its codes and how wide they are.
struct Font {
    /// Bytes per code: 2 for an Identity Type0 font, 1 otherwise.
    code_bytes: usize,
    /// Whether the font writes top to bottom (`/Identity-V`): the advance of a code is down the
    /// page, one em by default (`/DW2` and `/W2` are not read).
    vertical: bool,
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
pub(crate) struct TextContext<'a> {
    pub(crate) ctm: [f64; 6],
    pub(crate) render_mode: i64,
    pub(crate) fill: [u8; 3],
    pub(crate) sequence: u64,
    pub(crate) active_clips: &'a [u64],
}

/// The text parameters that `q` saves and `Q` restores (`Tc`, `Tw`, `Tz`, `TL`, `Tf` and its size,
/// `Ts`). `Tm` and the line matrix are not part of the graphics state.
#[derive(Clone)]
pub(crate) struct TextParams {
    size: f64,
    char_space: f64,
    word_space: f64,
    scale: f64,
    leading: f64,
    rise: f64,
    font: Option<Rc<Font>>,
}

impl Default for TextParams {
    fn default() -> Self {
        Self {
            size: 0.0,
            char_space: 0.0,
            word_space: 0.0,
            scale: 1.0,
            leading: 0.0,
            rise: 0.0,
            font: None,
        }
    }
}

/// The text state of a content stream and the runs it has shown.
pub(crate) struct TextTracker {
    tm: Matrix,
    tlm: Matrix,
    pub(crate) params: TextParams,
    cache: BTreeMap<(u32, u16), Rc<Font>>,
    limits: PdfVectorLimits,
    /// Decoded font side data read so far, against `limits.max_font_bytes`.
    font_bytes: u64,
    /// The runs shown so far, in content-stream order.
    pub(crate) runs: Vec<PdfTextRun>,
}

impl TextTracker {
    pub(crate) fn new(limits: &PdfVectorLimits) -> Self {
        Self {
            tm: IDENTITY,
            tlm: IDENTITY,
            params: TextParams::default(),
            cache: BTreeMap::new(),
            limits: *limits,
            font_bytes: 0,
            runs: Vec::new(),
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
    ) -> Result<(), PdfVectorError> {
        let num = |i: usize| operands.get(i).and_then(number);
        match operator {
            "BT" => {
                self.tm = IDENTITY;
                self.tlm = IDENTITY;
            }
            "Tc" => self.params.char_space = num(0).unwrap_or(0.0),
            "Tw" => self.params.word_space = num(0).unwrap_or(0.0),
            "Tz" => self.params.scale = num(0).unwrap_or(100.0) / 100.0,
            "TL" => self.params.leading = num(0).unwrap_or(0.0),
            "Ts" => self.params.rise = num(0).unwrap_or(0.0),
            "Tf" => {
                self.params.size = num(1).unwrap_or(0.0);
                self.params.font = self.load_font(doc, resources, operands.first())?;
            }
            "Td" | "TD" => {
                let (tx, ty) = (num(0).unwrap_or(0.0), num(1).unwrap_or(0.0));
                if operator == "TD" {
                    self.params.leading = -ty;
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
            "T*" => self.move_line(0.0, -self.params.leading),
            "Tj" | "TJ" | "'" | "\"" => {
                if operator == "'" || operator == "\"" {
                    if operator == "\"" {
                        self.params.word_space = num(0).unwrap_or(self.params.word_space);
                        self.params.char_space = num(1).unwrap_or(self.params.char_space);
                    }
                    self.move_line(0.0, -self.params.leading);
                }
                self.show(operator, operands, context);
            }
            _ => {}
        }
        Ok(())
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
            active_clips,
        } = *context;
        let Some(font) = self.params.font.clone() else {
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
        // A vertical font ignores the horizontal scale and advances down the page: one em a code
        // (`/DW2` and `/W2` are not read).
        let vertical = font.vertical;
        let (fs, th) = (
            self.params.size,
            if vertical { 1.0 } else { self.params.scale },
        );
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
                    self.params.word_space
                } else {
                    0.0
                };
                advance += if vertical {
                    -fs + self.params.char_space + space
                } else {
                    (width / 1000.0 * fs + self.params.char_space + space) * th
                };
            }
        }
        let visible = render_mode == 0 || render_mode == 2;
        let text = text.trim().to_owned();
        let trm = then(&self.tm, &ctm);
        if visible && !text.is_empty() && fs > 0.0 && advance.is_finite() {
            let start = apply(&trm, [0.0, self.params.rise]);
            let end = if vertical {
                apply(&trm, [0.0, self.params.rise + advance])
            } else {
                apply(&trm, [advance, self.params.rise])
            };
            // Basic arithmetic and `sqrt`, not libm `hypot`, which differs in the last bit by architecture.
            let scale = (trm[2] * trm[2] + trm[3] * trm[3]).sqrt();
            self.runs.push(PdfTextRun {
                text,
                start,
                end,
                cap_height: fs * scale * font.cap_height / 1000.0,
                fill_colour: fill,
                sequence,
                active_clips: active_clips.to_vec(),
            });
        }
        let (dx, dy) = if vertical {
            (0.0, advance)
        } else {
            (advance, 0.0)
        };
        self.tm = then(&[1.0, 0.0, 0.0, 1.0, dx, dy], &self.tm);
    }

    fn load_font(
        &mut self,
        doc: &Document,
        resources: Option<&Dictionary>,
        name: Option<&Object>,
    ) -> Result<Option<Rc<Font>>, PdfVectorError> {
        let Some(Object::Name(name)) = name else {
            return Ok(None);
        };
        let Some(entry) = resources
            .and_then(|r| r.get(b"Font").ok())
            .and_then(|o| doc.dereference(o).ok())
            .and_then(|(_, o)| o.as_dict().ok())
            .and_then(|f| f.get(name).ok())
        else {
            return Ok(None);
        };
        let key = match entry {
            Object::Reference(id) => Some(*id),
            _ => None,
        };
        if let Some(font) = key.and_then(|k| self.cache.get(&k)) {
            return Ok(Some(font.clone()));
        }
        let Some(dict) = doc
            .dereference(entry)
            .ok()
            .and_then(|(_, o)| o.as_dict().ok())
        else {
            return Ok(None);
        };
        let Some(font) = Font::read(doc, dict, &self.limits, &mut self.font_bytes)? else {
            return Ok(None);
        };
        let font = Rc::new(font);
        if let Some(key) = key {
            self.cache.insert(key, font.clone());
        }
        Ok(Some(font))
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
    /// composite font that is not Identity-encoded. A font past a bound is an error (issue #1687).
    fn read(
        doc: &Document,
        font: &Dictionary,
        limits: &PdfVectorLimits,
        spent: &mut u64,
    ) -> Result<Option<Font>, PdfVectorError> {
        let dict = |o: &Object| doc.dereference(o).ok()?.1.as_dict().ok().cloned();
        let Some(subtype) = font.get(b"Subtype").ok().and_then(|s| s.as_name().ok()) else {
            return Ok(None);
        };
        let is_type0 = subtype == b"Type0";
        let to_unicode = match font.get(b"ToUnicode") {
            Ok(o) => match side_stream(doc, o, limits, spent)? {
                Some(bytes) => parse_cmap(&String::from_utf8_lossy(&bytes))?,
                None => BTreeMap::new(),
            },
            Err(_) => BTreeMap::new(),
        };
        let mut out = Font {
            code_bytes: 1,
            vertical: false,
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
            let Some(encoding) = font.get(b"Encoding").ok().and_then(|e| e.as_name().ok()) else {
                return Ok(None);
            };
            if !encoding.starts_with(b"Identity") || out.to_unicode.is_empty() {
                return Ok(None);
            }
            out.code_bytes = 2;
            // `Identity-V` reads the same codes and writes them top to bottom.
            out.vertical = encoding == b"Identity-V";
            let Some(cid) = font
                .get(b"DescendantFonts")
                .ok()
                .and_then(|d| doc.dereference(d).ok())
                .and_then(|(_, o)| o.as_array().ok())
                .and_then(|a| a.first())
                .and_then(dict)
            else {
                return Ok(None);
            };
            out.default_width = cid.get(b"DW").ok().and_then(number).unwrap_or(1000.0);
            if let Some(cap) = descriptor_cap(&cid) {
                out.cap_height = cap;
            }
            // A `/CIDToGIDMap` only picks glyphs from the program, and codes here are CIDs, so it
            // does not change the text. A stream is still read, to bound what a font may hold.
            if let Some(map) = cid.get(b"CIDToGIDMap").ok().filter(|o| {
                doc.dereference(o)
                    .is_ok_and(|(_, o)| matches!(o, Object::Stream(_)))
            }) && let Some(bytes) = side_stream(doc, map, limits, spent)?
            {
                let glyphs = (bytes.len() as u64).div_ceil(2);
                if glyphs > MAX_CID_GLYPHS {
                    return Err(font_limit("glyphs", MAX_CID_GLYPHS, glyphs));
                }
            }
            if let Some(Object::Array(w)) = cid
                .get(b"W")
                .ok()
                .and_then(|o| doc.dereference(o).ok())
                .map(|(_, o)| o)
            {
                // `/W` of a CID font: `c [w1 w2 ...]` and `c_first c_last w`.
                let mut count = 0u64;
                let too_many = |n: u64| font_limit("cid widths", MAX_CID_WIDTHS, n);
                'widths: {
                    let mut i = 0;
                    while i < w.len() {
                        let Some(first) = number(&w[i]).map(|n| n as u32) else {
                            break 'widths;
                        };
                        match w
                            .get(i + 1)
                            .and_then(|o| doc.dereference(o).ok())
                            .map(|(_, o)| o)
                        {
                            Some(Object::Array(list)) => {
                                count += list.len() as u64;
                                if count > MAX_CID_WIDTHS {
                                    return Err(too_many(count));
                                }
                                for (k, width) in list.iter().enumerate() {
                                    if let Some(width) = number(width) {
                                        out.cid_widths
                                            .insert(first.saturating_add(k as u32), width);
                                    }
                                }
                                i += 2;
                            }
                            Some(last) => {
                                let (Some(last), Some(width)) = (
                                    number(last).map(|n| n as u32),
                                    w.get(i + 2).and_then(number),
                                ) else {
                                    break 'widths;
                                };
                                let last = last.min(first.saturating_add(65_535));
                                count += u64::from(last.saturating_sub(first)) + 1;
                                if count > MAX_CID_WIDTHS {
                                    return Err(too_many(count));
                                }
                                for cid in first..=last {
                                    out.cid_widths.insert(cid, width);
                                }
                                i += 3;
                            }
                            None => break 'widths,
                        }
                    }
                }
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
        Ok(Some(out))
    }
}

/// The most codes a `/ToUnicode` CMap may map, counted per code a `bfchar` or `bfrange` entry
/// sets (a code set twice counts twice). A CID font has at most 65 536 CIDs, so this is twice
/// what a real font needs, and past it the CMap is a refusal.
const MAX_CMAP_ENTRIES: u64 = 131_072;
/// The most widths a CID font's `/W` array may set, counted the same way.
const MAX_CID_WIDTHS: u64 = 131_072;
/// The most glyphs a `/CIDToGIDMap` stream may hold: one 2 byte entry per CID.
const MAX_CID_GLYPHS: u64 = 65_536;

fn font_limit(what: &'static str, limit: u64, observed: u64) -> PdfVectorError {
    PdfVectorError::FontLimitExceeded {
        what,
        limit,
        observed,
    }
}

/// The decoded bytes of a stream that a font points at (`/ToUnicode`, `/CIDToGIDMap`), counted
/// against `max_font_bytes` through `spent`. `None` for what is not a stream or has a filter
/// this cannot decode, which leaves the font without that side data.
fn side_stream(
    doc: &Document,
    object: &Object,
    limits: &PdfVectorLimits,
    spent: &mut u64,
) -> Result<Option<Vec<u8>>, PdfVectorError> {
    let Some(stream) = doc
        .dereference(object)
        .ok()
        .and_then(|(_, o)| o.as_stream().ok())
    else {
        return Ok(None);
    };
    let local = PdfVectorLimits {
        max_content_bytes: limits.max_font_bytes.saturating_sub(*spent),
        ..*limits
    };
    match stream_bytes(stream, 0, &local, &mut 0) {
        Ok(bytes) => {
            *spent = spent.saturating_add(bytes.len() as u64);
            Ok(Some(bytes))
        }
        Err(PdfVectorError::ContentBytesExceeded { observed, .. }) => Err(font_limit(
            "font bytes",
            limits.max_font_bytes,
            spent.saturating_add(observed),
        )),
        Err(_) => Ok(None),
    }
}

/// The `bfchar` and `bfrange` entries of a `/ToUnicode` CMap.
fn parse_cmap(text: &str) -> Result<BTreeMap<u32, String>, PdfVectorError> {
    let hex = |t: &str| u32::from_str_radix(t, 16).ok();
    let utf16 = |t: &str| {
        let units: Vec<u16> = (0..t.len() / 4)
            .filter_map(|k| u16::from_str_radix(t.get(k * 4..k * 4 + 4)?, 16).ok())
            .collect();
        String::from_utf16_lossy(&units)
    };
    let mut map = BTreeMap::new();
    let mut entries = 0u64;
    let too_many = |n: u64| font_limit("cmap entries", MAX_CMAP_ENTRIES, n);
    for (begin, end, range) in [
        ("beginbfchar", "endbfchar", false),
        ("beginbfrange", "endbfrange", true),
    ] {
        let mut rest = text;
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
                        entries += 1;
                        if entries > MAX_CMAP_ENTRIES {
                            return Err(too_many(entries));
                        }
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
                        entries += 1;
                        if entries > MAX_CMAP_ENTRIES {
                            return Err(too_many(entries));
                        }
                        map.insert(code, utf16(item));
                    }
                    k += 4 + list.len();
                } else {
                    // The whole range counts before any of it is built.
                    entries = entries.saturating_add(u64::from(hi.saturating_sub(lo)) + 1);
                    if entries > MAX_CMAP_ENTRIES {
                        return Err(too_many(entries));
                    }
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
    Ok(map)
}

fn number(o: &Object) -> Option<f64> {
    match o {
        Object::Integer(i) => Some(*i as f64),
        Object::Real(r) => Some(f64::from(*r)),
        _ => None,
    }
}
