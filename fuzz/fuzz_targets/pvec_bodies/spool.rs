//! The body of `fuzz_pvec_spool` (#1070), shared with `tests/pvec_1070_spool_fuzz.rs`.
//!
//! Stub: the body lands with the implementation commit.

/// What one input did.
#[derive(Debug, PartialEq, Eq)]
pub enum Outcome {
    /// Nothing ran.
    NotRun,
    /// A write finished.
    Written { spilled: bool },
    /// The sink failed part way.
    SinkFailed,
}

pub fn run(_data: &[u8]) -> Outcome {
    Outcome::NotRun
}
