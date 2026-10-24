//! The `acadsharp-rs` provider, behind the `dwg-acadsharp` feature.
//!
//! This is the one place in libviprs that names a DWG parser. Everything that
//! leaves it is a [`CadDecoder`], a [`CadDrawing`] or a [`CadError`]: no
//! `acadsharp_rs` type appears in a signature, and the tiler, the MVT encoder
//! and the PMTiles writer never import this module.
//!
//! ```text
//! acadsharp_rs::Document::decode  ->  Item stream
//!       Item::Primitive           ->  cad::Primitive   (validated; a refusal is a Diagnostic)
//!       Item::Warning             ->  cad::Diagnostic  (the code kept, the entity kept)
//!       stream.is_complete()      ->  DecodeReport::mark_complete / mark_truncated
//! ```
//!
//! # Nothing is dropped quietly
//!
//! A warning the native reader raised becomes a [`Diagnostic`] with the same
//! code, so a consumer who knows `HATCH_PATTERN_ONLY` from the provider's own
//! documentation finds it under that number. A primitive the IR's constructors
//! refuse (a zero-length line, say) becomes a diagnostic too, filed from the
//! constructor's own error, rather than vanishing. A stream that ends without
//! the provider's totals agreeing is [`DecodeReport::mark_truncated`], never
//! `mark_complete`.
//!
//! # No .NET at run time
//!
//! The library behind `acadsharp-rs` is a NativeAOT artifact, linked by that
//! crate's build script. Whether one was found is decided when `acadsharp-rs`
//! is built, so a build without an archive still compiles and
//! [`AcadSharpDecoder::new`] answers a [`CadError::Provider`] whose source is
//! the provider's `Unlinked` error instead of the type being missing.
//!
//! # Supported DWG versions
//!
//! [`probe`](CadDecoder::probe) answers from the six byte version stamp and
//! accepts exactly [`SUPPORTED_DWG_VERSIONS`], AC1014 through AC1032. A newer
//! or older stamp is `Ok(false)`, and a drawing the native reader then refuses
//! anyway comes back from [`open`](CadDecoder::open) as
//! [`CadError::UnrecognisedSource`].

use std::io::Read;

use acadsharp_rs as acs;

use super::{
    Arc, CadDecoder, CadDrawing, CadError, CadSource, CadView, Circle, DecodeReport, Diagnostic,
    DiagnosticCode, Ellipse, ItemHandle, Line, Origin, Polygon, Polyline, Primitive, PrimitiveSink,
    Spline, Text,
};

/// What this provider calls itself in [`CadError`].
const NAME: &str = "acadsharp";

/// The DWG version stamps the pinned ACadSharp reads: R14, 2000, 2004, 2007,
/// 2010, 2013 and 2018. They are the first six bytes of every DWG.
pub const SUPPORTED_DWG_VERSIONS: [&[u8; 6]; 7] = [
    b"AC1014", b"AC1015", b"AC1018", b"AC1021", b"AC1024", b"AC1027", b"AC1032",
];

/// Record types from here up are the wire format's forward-probe range, which
/// the native library emits on purpose in every decode. They are not news.
const FORWARD_PROBE_RANGE: u16 = 32512;

/// The bounds the native reader enforces on a drawing it is handed.
///
/// A drawing is untrusted input and the native library is not a sandbox, so
/// the caller picks the ceilings. `None` leaves a field to the library's own
/// default. This is libviprs's own type: it carries no provider type, and it
/// maps onto the provider's limits inside [`AcadSharpDecoder`] only.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ProviderLimits {
    /// The largest input the open call will take, refused before it is read.
    pub max_input_bytes: Option<u64>,
    /// Entities counted across the decode, nested expansion included.
    pub max_entities: Option<u64>,
    /// The longest string a text or warning record may carry.
    pub max_string_bytes: Option<u64>,
    /// Points in one record.
    pub max_polyline_points: Option<u64>,
    /// Nesting depth of an expansion.
    pub max_block_depth: Option<u64>,
    /// Total bytes the decode may emit.
    pub max_output_bytes: Option<u64>,
}

impl ProviderLimits {
    /// The bounds an untrusted upload is read under.
    ///
    /// [`ProviderLimits::default`] sets none, which is right for a drawing you
    /// made and wrong for one a stranger did. Every field is set here: a
    /// quarter of a gigabyte of input, ten million entities, a megabyte per
    /// string, a million points per record, 64 levels of nesting and two
    /// gigabytes of output.
    #[must_use]
    pub const fn hardened() -> Self {
        Self {
            max_input_bytes: Some(256 << 20),
            max_entities: Some(10_000_000),
            max_string_bytes: Some(1 << 20),
            max_polyline_points: Some(1_000_000),
            max_block_depth: Some(64),
            max_output_bytes: Some(2 << 30),
        }
    }

    fn to_native(self) -> acs::Limits {
        let mut limits = acs::Limits::new();
        if let Some(v) = self.max_input_bytes {
            limits = limits.with_max_input_bytes(v);
        }
        if let Some(v) = self.max_entities {
            limits = limits.with_max_entities(v);
        }
        if let Some(v) = self.max_string_bytes {
            limits = limits.with_max_string_bytes(v);
        }
        if let Some(v) = self.max_polyline_points {
            limits = limits.with_max_polyline_points(v);
        }
        if let Some(v) = self.max_block_depth {
            limits = limits.with_max_block_depth(v);
        }
        if let Some(v) = self.max_output_bytes {
            limits = limits.with_max_output_bytes(v);
        }
        limits
    }
}

/// A [`CadDecoder`] that reads DWG through `acadsharp-rs`.
///
/// Not `Send` and not `Sync`, because the provider's handle is neither.
#[derive(Debug)]
pub struct AcadSharpDecoder {
    native: acs::Decoder,
    limits: ProviderLimits,
}

impl AcadSharpDecoder {
    /// Runs the provider's ABI handshake, once.
    ///
    /// # Errors
    ///
    /// [`CadError::Provider`] when the native library was not linked into this
    /// build, or when it was built from a header other than the one
    /// `acadsharp-rs` vendored.
    pub fn new() -> Result<Self, CadError> {
        let native = acs::Decoder::new().map_err(|e| CadError::provider(NAME, e))?;
        Ok(Self {
            native,
            limits: ProviderLimits::default(),
        })
    }

    /// Sets the bounds every drawing opened through this decoder is held to.
    #[must_use]
    pub const fn with_limits(mut self, limits: ProviderLimits) -> Self {
        self.limits = limits;
        self
    }

    /// The version string of the ACadSharp the native library was built from.
    #[must_use]
    pub fn provider_version(&self) -> String {
        self.native.capabilities().acadsharp_version().to_string()
    }
}

/// Whether six bytes are a DWG version stamp this provider reads.
#[must_use]
pub fn is_supported_version(stamp: &[u8]) -> bool {
    SUPPORTED_DWG_VERSIONS.iter().any(|v| v.as_slice() == stamp)
}

impl CadDecoder for AcadSharpDecoder {
    fn probe(&self, source: CadSource<'_>) -> Result<bool, CadError> {
        match source {
            CadSource::Bytes(bytes) => Ok(bytes.get(..6).is_some_and(is_supported_version)),
            CadSource::Path(path) => {
                let mut stamp = [0u8; 6];
                let mut file = std::fs::File::open(path)?;
                // A file shorter than the stamp is not a DWG; it is not an I/O
                // failure either.
                match file.read_exact(&mut stamp) {
                    Ok(()) => Ok(is_supported_version(&stamp)),
                    Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => Ok(false),
                    Err(e) => Err(e.into()),
                }
            }
        }
    }

    fn open<'a>(&'a self, source: CadSource<'a>) -> Result<Box<dyn CadDrawing + 'a>, CadError> {
        let limits = self.limits.to_native();
        let document = match source {
            CadSource::Path(path) => acs::Document::open_path(&self.native, path, &limits),
            CadSource::Bytes(bytes) => acs::Document::open_bytes(&self.native, bytes, &limits),
        }
        .map_err(translate_error)?;
        Ok(Box::new(AcadSharpDrawing { document }))
    }
}

/// Turns the provider's error into the contract's.
///
/// A drawing the reader will not take is [`CadError::UnrecognisedSource`]; an
/// unreadable path is [`CadError::Io`]; every other failure keeps the
/// provider's error as its source, so nothing the native side said is lost.
fn translate_error(error: acs::Error) -> CadError {
    match error {
        acs::Error::UnsupportedFormat { .. } | acs::Error::EmptyInput => {
            CadError::UnrecognisedSource { decoder: NAME }
        }
        other => CadError::provider(NAME, other),
    }
}

/// An open drawing.
struct AcadSharpDrawing {
    document: acs::Document,
}

impl CadDrawing for AcadSharpDrawing {
    fn views(&self) -> Result<Vec<CadView>, CadError> {
        let views = self.document.views().map_err(translate_error)?;
        Ok(views
            .iter()
            .map(|v| {
                CadView::new(
                    v.index(),
                    v.raw_kind(),
                    v.raw_extents(),
                    v.entity_count(),
                    v.name(),
                )
            })
            .collect())
    }

    fn decode(
        &self,
        view: u32,
        sink: &mut dyn PrimitiveSink,
        report: &mut DecodeReport,
    ) -> Result<(), CadError> {
        let mut stream = match self.document.decode(view) {
            Ok(stream) => stream,
            Err(e) => return Err(no_such_view_or(&self.document, view, e)),
        };

        for item in &mut stream {
            let item = item.map_err(|e| {
                report.push(
                    Diagnostic::from_code(
                        DiagnosticCode::DECODE_FAILED,
                        format!("the native reader stopped: {e}"),
                    )
                    .with_view(view),
                );
                translate_error(e)
            })?;
            match item {
                acs::Item::Primitive(primitive) => {
                    forward(primitive, view, sink, report)?;
                }
                acs::Item::Warning(warning) => report.push(warning_to_diagnostic(&warning, view)),
                acs::Item::Unknown { record_type, .. } if record_type < FORWARD_PROBE_RANGE => {
                    report.push(
                        Diagnostic::from_code(
                            DiagnosticCode::READER_NOTIFICATION,
                            format!("the native reader sent a record of type {record_type}, which this build does not know"),
                        )
                        .with_view(view),
                    );
                }
                // Framing records and the forward-probe range: the totals
                // they carry are read through `is_complete` below.
                _ => {}
            }
        }

        if stream.is_complete() {
            report.mark_complete();
            sink.finish()
        } else {
            // The stream ended without its own totals agreeing. That is a
            // short decode, and saying so is the whole point of the report.
            report.mark_truncated(
                "the native stream ended before its totals agreed with the records received",
            );
            Err(CadError::provider(
                NAME,
                "the native stream ended before its totals agreed with the records received",
            ))
        }
    }
}

/// An out-of-range view is the contract's `NoSuchView`, with the count.
fn no_such_view_or(document: &acs::Document, view: u32, error: acs::Error) -> CadError {
    if let Ok(views) = document.view_count()
        && view >= views
    {
        return CadError::NoSuchView { index: view, views };
    }
    translate_error(error)
}

fn handle_of(origin: acs::Origin) -> Origin {
    let mut out = Origin::UNKNOWN.with_expanded_insert(origin.from_expanded_insert);
    if let Some(h) = origin.item_handle {
        out = out.with_handle(ItemHandle::new(h.get()));
    }
    out
}

fn warning_to_diagnostic(warning: &acs::Warning, view: u32) -> Diagnostic {
    let raw = warning.code.get();
    // The provider's band and the contract's codes keep their numbers. A
    // number in the core band belongs to libviprs and a provider must not
    // claim it, so it is moved into the provider band, offset by its own
    // value, rather than impersonating one of ours.
    let code = if !(DiagnosticCode::CORE_RANGE_START..DiagnosticCode::PROVIDER_RANGE_START)
        .contains(&raw)
    {
        DiagnosticCode::new(raw)
    } else {
        DiagnosticCode::provider(raw)
    };
    let mut d = Diagnostic::from_code(code, warning.message.clone()).with_view(view);
    if let Some(h) = warning.entity {
        d = d.with_entity(ItemHandle::new(h.get()));
    }
    d
}

/// Builds the IR primitive, or the error that refused it.
fn convert(primitive: acs::Primitive) -> Result<(Primitive, Option<acs::ItemHandle>), CadError> {
    Ok(match primitive {
        acs::Primitive::Line(p) => (
            Line::new(p.start, p.end)?
                .with_origin(handle_of(p.origin))
                .into(),
            p.origin.item_handle,
        ),
        acs::Primitive::Polyline(p) => (
            Polyline::new(&p.vertices, p.closed, p.normal, &p.bulges)?
                .with_origin(handle_of(p.origin))
                .into(),
            p.origin.item_handle,
        ),
        acs::Primitive::Polygon(p) => (
            Polygon::new(&p.vertices, p.normal, &p.bulges)?
                .with_origin(handle_of(p.origin))
                .into(),
            p.origin.item_handle,
        ),
        acs::Primitive::Arc(p) => (
            Arc::new(p.centre, p.radius, p.start_angle, p.end_angle, p.normal)?
                .with_origin(handle_of(p.origin))
                .into(),
            p.origin.item_handle,
        ),
        acs::Primitive::Circle(p) => (
            Circle::new(p.centre, p.radius, p.normal)?
                .with_origin(handle_of(p.origin))
                .into(),
            p.origin.item_handle,
        ),
        acs::Primitive::Ellipse(p) => (
            Ellipse::new(
                p.centre,
                p.major_axis,
                p.ratio,
                p.start_param,
                p.end_param,
                p.normal,
            )?
            .with_origin(handle_of(p.origin))
            .into(),
            p.origin.item_handle,
        ),
        acs::Primitive::Spline(p) => (
            Spline::new(p.degree, p.flags, &p.knots, &p.controls, &p.weights)?
                .with_origin(handle_of(p.origin))
                .into(),
            p.origin.item_handle,
        ),
        acs::Primitive::Text(p) => (
            Text::new(p.position, p.height, p.rotation, decode_escapes(&p.text))?
                .with_origin(handle_of(p.origin))
                .into(),
            p.origin.item_handle,
        ),
        // `acs::Primitive` is non-exhaustive. A shape this build has no IR
        // for is reported by the caller as a refusal, not skipped.
        _ => {
            return Err(CadError::provider(
                NAME,
                "the native reader sent a primitive kind this build has no IR for",
            ));
        }
    })
}

fn forward(
    primitive: acs::Primitive,
    view: u32,
    sink: &mut dyn PrimitiveSink,
    report: &mut DecodeReport,
) -> Result<(), CadError> {
    match convert(primitive) {
        Ok((primitive, _)) => {
            let kind = primitive.kind();
            sink.primitive(primitive)?;
            // Counted after the sink took it, so a refusal is not counted as
            // delivered.
            report.record(kind);
        }
        Err(e) => {
            // Refused by a constructor: reported, with the code the error maps
            // to, and the decode carries on.
            report.push(e.to_diagnostic().with_view(view));
        }
    }
    Ok(())
}

/// Decodes the transport escapes the native reader hands through, so the string
/// a [`Text`] carries is the one a person reads.
///
/// `\U+XXXX` (MIF) becomes the character, and `%%c`, `%%d` and `%%p` become
/// the diameter, degree and plus-minus signs. Anything else that looks like an
/// escape is left alone for [`Text::new`] to refuse, which reports it: a gap
/// that is visible beats a guess. `\\U+XXXX` is a literal backslash followed by
/// `U+XXXX` and is not touched, which is the same rule `undecoded_escape`
/// applies.
fn decode_escapes(text: &str) -> String {
    let bytes = text.as_bytes();
    let mut out = String::with_capacity(text.len());
    let mut i = 0;
    let mut copied = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'\\' if bytes.get(i + 1) == Some(&b'\\') => {
                i += 2;
            }
            b'\\' if bytes.get(i + 1) == Some(&b'U') && bytes.get(i + 2) == Some(&b'+') => {
                let hex = text.get(i + 3..i + 7);
                let ch = hex
                    .filter(|h| h.bytes().all(|b| b.is_ascii_hexdigit()))
                    .and_then(|h| u32::from_str_radix(h, 16).ok())
                    .and_then(char::from_u32);
                if let Some(ch) = ch {
                    out.push_str(&text[copied..i]);
                    out.push(ch);
                    i += 7;
                    copied = i;
                } else {
                    i += 3;
                }
            }
            b'%' if bytes.get(i + 1) == Some(&b'%') => {
                let ch = match bytes.get(i + 2) {
                    Some(b'c' | b'C') => Some('\u{2205}'),
                    Some(b'd' | b'D') => Some('\u{b0}'),
                    Some(b'p' | b'P') => Some('\u{b1}'),
                    _ => None,
                };
                if let Some(ch) = ch {
                    out.push_str(&text[copied..i]);
                    out.push(ch);
                    i += 3;
                    copied = i;
                } else {
                    // `%%%` is a literal percent; step over all three so the
                    // `%%c` hiding in `%%%c` is not read as a code.
                    i += if bytes.get(i + 2) == Some(&b'%') {
                        3
                    } else {
                        2
                    };
                }
            }
            _ => i += 1,
        }
    }
    out.push_str(&text[copied..]);
    out
}

#[cfg(test)]
mod tests {
    use super::decode_escapes;

    #[test]
    fn mif_and_percent_codes_decode() {
        assert_eq!(decode_escapes(r"90\U+00B0"), "90\u{b0}");
        assert_eq!(decode_escapes("%%c45,6"), "\u{2205}45,6");
        assert_eq!(decode_escapes("%%d %%P"), "\u{b0} \u{b1}");
        assert_eq!(decode_escapes(r"\U+2205\U+0041"), "\u{2205}A");
    }

    #[test]
    fn a_doubled_backslash_and_a_literal_percent_are_left_alone() {
        assert_eq!(decode_escapes(r"\\U+0041"), r"\\U+0041");
        assert_eq!(decode_escapes("%%%c"), "%%%c");
    }

    #[test]
    fn what_cannot_be_decoded_is_left_for_the_constructor_to_refuse() {
        assert_eq!(decode_escapes(r"\U+12"), r"\U+12");
        assert_eq!(decode_escapes(r"\U+D800"), r"\U+D800");
        assert_eq!(decode_escapes("%%127"), "%%127");
        assert_eq!(decode_escapes("plain é text"), "plain é text");
    }
}
