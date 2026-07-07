//! Issue #140: typed errors must not be laundered into strings.
//!
//! Two acceptance criteria are pinned here from an *external* crate (so the
//! public shape of the new variants is part of the contract):
//!
//! 1. A strip-source failure surfaces as a *source-typed* `EngineError::Source`
//!    that preserves the `source()` chain, instead of being stringified into
//!    `SinkError::Other` (which mis-attributes a source failure to storage).
//! 2. A checkpoint-write failure carries a typed `SinkError::Checkpoint`
//!    wrapping the underlying `ResumeError`, so the same failure is no longer
//!    reported as a bare `SinkError::Other("checkpoint: ...")` on one path and
//!    `EngineError::ResumeFailed` on another.

use std::error::Error;

use libviprs::{EngineError, PdfError, ResumeError, SinkError};

#[test]
fn source_variant_preserves_source_chain() {
    let inner = PdfError::Parse("boom in the PDF layer".to_string());
    let err = EngineError::Source(Box::new(inner));

    // The wrapper must be labelled a source error, never a sink error.
    let display = err.to_string();
    assert!(
        !display.contains("sink error"),
        "a source failure must not be labelled a sink error, got: {display}"
    );

    // The typed source chain survives and downcasts back to the PDF error,
    // rather than being flattened into an opaque `String`.
    let source = Error::source(&err).expect("EngineError::Source must expose its source()");
    let pdf = source
        .downcast_ref::<PdfError>()
        .expect("source() must preserve the concrete PdfError, not a stringified copy");
    assert!(matches!(pdf, PdfError::Parse(_)));
}

#[test]
fn checkpoint_sink_error_preserves_resume_error() {
    let resume = ResumeError::SchemaMismatch {
        expected: "1",
        found: "99".to_string(),
    };
    let sink_err = SinkError::Checkpoint(resume);

    // The checkpoint failure must not masquerade as a generic `Other` string.
    let display = sink_err.to_string();
    assert!(
        display.contains("checkpoint"),
        "checkpoint sink error should mention the checkpoint, got: {display}"
    );

    // The underlying ResumeError is preserved as a typed source, not a String.
    let source = Error::source(&sink_err).expect("SinkError::Checkpoint must expose its source()");
    let resume = source
        .downcast_ref::<ResumeError>()
        .expect("source() must preserve the concrete ResumeError");
    assert!(matches!(resume, ResumeError::SchemaMismatch { .. }));
}
