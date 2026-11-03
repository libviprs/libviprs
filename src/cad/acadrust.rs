//! The acadrust DWG provider behind [`CadDecoder`], behind the `dwg-acadrust` feature (issue #1485).
//!
//! The first commit is the shape only: it opens nothing yet, so the tests next to it are red.

use super::{CadDecoder, CadDrawing, CadError, CadSource};

const NAME: &str = "acadrust";

/// A DWG reader that needs nothing installed.
#[derive(Debug, Clone, Copy, Default)]
pub struct AcadrustDecoder {
    max_input_bytes: Option<u64>,
}

impl AcadrustDecoder {
    /// A decoder with no input cap.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            max_input_bytes: None,
        }
    }

    /// A decoder that refuses an input of more than 256 MiB.
    #[must_use]
    pub const fn hardened() -> Self {
        Self {
            max_input_bytes: Some(256 << 20),
        }
    }
}

impl CadDecoder for AcadrustDecoder {
    fn probe(&self, _source: CadSource<'_>) -> Result<bool, CadError> {
        Ok(false)
    }

    fn open<'a>(&'a self, _source: CadSource<'a>) -> Result<Box<dyn CadDrawing + 'a>, CadError> {
        let _ = self.max_input_bytes;
        Err(CadError::provider(NAME, "not written yet"))
    }
}
