//! The v1 text capability policy (N5.5, #1084). The decision and its measurements are in
//! `docs/pvec-text-policy.md`, and this module is its executable half.
//!
//! v1 **omits text and reports the counts**. No glyph outline, string, font name or text
//! matrix reaches a geometry block. What this module does is make sure an omitted text
//! entity is still visible: every text-showing PDF operator (`Tj`, `TJ`, `'`, `"`) and every
//! DWG `TEXT` and `MTEXT` entity is counted, and each PDF operator is bucketed by what the
//! font situation would have meant for a later version that preserved the text.
//!
//! Nothing here depends on a font or shaping crate.

use std::collections::BTreeMap;

use lopdf::{Dictionary, Document, Object};

use super::provider::RawEntity;

/// What the font behind a text operator would have meant, had the text been kept. Under v1
/// every one of these is omitted and counted; the variant only picks the report bucket.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[non_exhaustive]
pub enum TextFontOutcome {
    /// A full (not subset) embedded font program. The string is recoverable.
    Embedded,
    /// An embedded subset (`ABCDEF+Name`). The string is recoverable but only the glyphs the
    /// drawing used exist, so substitution is impossible.
    SubsetEmbedded,
    /// No font program in the file, only a name. The string is recoverable, the shapes are not.
    NotEmbedded,
    /// A composite (Type0) font with no `/ToUnicode`. Glyph ids only, the string is not
    /// recoverable even in principle.
    CidNoToUnicode,
    /// The operator named no font that resolves (no `Tf`, or a name missing from `/Font`).
    Unresolved,
    /// A DWG text style whose SHX font is not installed where the drawing is read.
    ShxMissing,
}

impl TextFontOutcome {
    /// Every outcome, in report order.
    pub const ALL: [Self; 6] = [
        Self::Embedded,
        Self::SubsetEmbedded,
        Self::NotEmbedded,
        Self::CidNoToUnicode,
        Self::Unresolved,
        Self::ShxMissing,
    ];

    /// The stable report bucket.
    #[must_use]
    pub const fn code(self) -> &'static str {
        match self {
            Self::Embedded => "TEXT_FONT_EMBEDDED",
            Self::SubsetEmbedded => "TEXT_FONT_SUBSET",
            Self::NotEmbedded => "TEXT_FONT_NOT_EMBEDDED",
            Self::CidNoToUnicode => "TEXT_CID_NO_TOUNICODE",
            Self::Unresolved => "TEXT_FONT_UNRESOLVED",
            Self::ShxMissing => "TEXT_SHX_MISSING",
        }
    }
}

/// The four text-showing operators, counted separately.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PdfTextOperatorCounts {
    /// `Tj`.
    pub tj: u64,
    /// `TJ`.
    pub tj_array: u64,
    /// `'`.
    pub quote: u64,
    /// `"`.
    pub double_quote: u64,
}

impl PdfTextOperatorCounts {
    /// All four.
    #[must_use]
    pub const fn total(&self) -> u64 {
        self.tj + self.tj_array + self.quote + self.double_quote
    }
}

/// DWG text entities, counted by class.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DwgTextCounts {
    /// `TEXT`.
    pub text: u64,
    /// `MTEXT`.
    pub mtext: u64,
}

impl DwgTextCounts {
    /// Both.
    #[must_use]
    pub const fn total(&self) -> u64 {
        self.text + self.mtext
    }
}

/// The text part of a conversion report: what was omitted.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TextReport {
    /// PDF text-showing operators, forms expanded in place.
    pub pdf: PdfTextOperatorCounts,
    /// DWG text entities.
    pub dwg: DwgTextCounts,
    /// Every omitted text item by [`TextFontOutcome`]. Sums to `pdf.total() + dwg.total()`
    /// only when a DWG caller bucketed its entities (see [`TextReport::bucket_dwg`]).
    pub by_outcome: BTreeMap<TextFontOutcome, u64>,
}

impl TextReport {
    /// Items omitted under the v1 policy.
    #[must_use]
    pub const fn omitted(&self) -> u64 {
        self.pdf.total() + self.dwg.total()
    }

    pub(crate) fn record_pdf(&mut self, operator: &str, outcome: TextFontOutcome) {
        let slot = match operator {
            "Tj" => &mut self.pdf.tj,
            "TJ" => &mut self.pdf.tj_array,
            "'" => &mut self.pdf.quote,
            "\"" => &mut self.pdf.double_quote,
            _ => return,
        };
        *slot += 1;
        *self.by_outcome.entry(outcome).or_insert(0) += 1;
    }

    /// Count the `TEXT` and `MTEXT` entities of a provider's output. Other classes are ignored.
    pub fn count_dwg(&mut self, entities: &[RawEntity]) {
        for e in entities {
            if e.provenance.class_name.eq_ignore_ascii_case("TEXT") {
                self.dwg.text += 1;
            } else if e.provenance.class_name.eq_ignore_ascii_case("MTEXT") {
                self.dwg.mtext += 1;
            }
        }
    }

    /// Put `n` already counted DWG text entities in the bucket for their style's SHX font.
    /// The provider boundary does not carry a text style yet, so the caller supplies it.
    pub fn bucket_dwg(&mut self, shx: &str, installed: &[&str], n: u64) {
        *self
            .by_outcome
            .entry(shx_outcome(shx, installed))
            .or_insert(0) += n;
    }
}

/// The outcome for a DWG style font: [`TextFontOutcome::ShxMissing`] unless `font` (compared
/// without case, with or without `.shx`) is in `installed`. A present font is reported as
/// [`TextFontOutcome::Embedded`], meaning "available", since DWG has no embedding.
#[must_use]
pub fn shx_outcome(font: &str, installed: &[&str]) -> TextFontOutcome {
    let norm = |s: &str| {
        let s = s.trim().to_ascii_lowercase();
        s.strip_suffix(".shx").map_or(s.clone(), str::to_string)
    };
    let want = norm(font);
    if installed.iter().any(|i| norm(i) == want) {
        TextFontOutcome::Embedded
    } else {
        TextFontOutcome::ShxMissing
    }
}

/// Classify the font a `Tf` named. `resources` is the stream's `/Resources`.
pub(crate) fn font_outcome(
    doc: &Document,
    resources: Option<&Dictionary>,
    name_operand: Option<&Object>,
) -> TextFontOutcome {
    let Some(Object::Name(name)) = name_operand else {
        return TextFontOutcome::Unresolved;
    };
    let Some(font) = resources
        .and_then(|r| r.get(b"Font").ok())
        .and_then(|o| as_dict(doc, o))
        .and_then(|f| f.get(name).ok())
        .and_then(|o| as_dict(doc, o))
    else {
        return TextFontOutcome::Unresolved;
    };
    let is_type0 = font
        .get(b"Subtype")
        .ok()
        .and_then(|s| s.as_name().ok())
        .is_some_and(|n| n == b"Type0");
    if is_type0 && !font.has(b"ToUnicode") {
        return TextFontOutcome::CidNoToUnicode;
    }
    // A Type0 font keeps its descriptor on the descendant.
    let descendant = font
        .get(b"DescendantFonts")
        .ok()
        .and_then(|d| doc.dereference(d).ok())
        .and_then(|(_, o)| o.as_array().ok())
        .and_then(|a| a.first())
        .and_then(|o| as_dict(doc, o));
    let holder = descendant.unwrap_or(font);
    let embedded = holder
        .get(b"FontDescriptor")
        .ok()
        .and_then(|o| as_dict(doc, o))
        .is_some_and(|d| d.has(b"FontFile") || d.has(b"FontFile2") || d.has(b"FontFile3"));
    if !embedded {
        return TextFontOutcome::NotEmbedded;
    }
    let base = font
        .get(b"BaseFont")
        .ok()
        .and_then(|b| b.as_name().ok())
        .unwrap_or_default();
    if is_subset_tag(base) {
        TextFontOutcome::SubsetEmbedded
    } else {
        TextFontOutcome::Embedded
    }
}

fn as_dict<'a>(doc: &'a Document, o: &'a Object) -> Option<&'a Dictionary> {
    doc.dereference(o).ok()?.1.as_dict().ok()
}

/// `ABCDEF+Name`: six uppercase letters and a plus (PDF 32000-1 9.6.4).
fn is_subset_tag(base_font: &[u8]) -> bool {
    base_font.len() > 7 && base_font[6] == b'+' && base_font[..6].iter().all(u8::is_ascii_uppercase)
}
