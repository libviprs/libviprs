//! Verify-mode entry points.
//!
//! Two flavours share one namespace so `EngineBuilder` can route verify
//! runs by source kind without the rest of the crate hard-coding which
//! helper to call:
//!
//! * [`raster_verify`] — walks every level via `downscale_half` against
//!   the full in-memory source raster. Used when the caller has a
//!   `&Raster` and picks `EngineKind::Monolithic`.
//! * [`verify_from_strip_source`](crate::stream_verify::verify_from_strip_source)
//!   — strip-driven verify for pull-based sources or when the caller
//!   explicitly picks `EngineKind::Streaming` / `EngineKind::MapReduce`.
//!
//! Both emit the same `LevelStarted` / `TileCompleted` / `LevelCompleted`
//! / `Finished` event stream so observers see verify runs as first-class.

pub(crate) use crate::engine::raster_verify;
pub(crate) use crate::stream_verify::verify_from_strip_source;
