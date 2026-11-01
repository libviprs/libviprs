//! One small module per PDF image codec (issue #1362), the way `jp2k.rs`, `jxl.rs` and `avif.rs`
//! do it for the raster path. Each has a `decode` that turns one [`FilterInput`] into samples,
//! and the build's features decide inside the module which body it has. Only
//! [`decode_filter`](super::pdf_image_decode::decode_filter) calls them.
//!
//! [`FilterInput`]: super::pdf_image_decode::FilterInput

pub(super) mod ccitt;
pub(super) mod dct;
pub(super) mod jbig2;
pub(super) mod jpx;
