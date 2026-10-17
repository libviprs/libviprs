//! Vector output: MVT tiles and the PMTiles archive that carries them.
//!
//! ```text
//! cad::Primitive -> (tiler) -> mvt::Tile -> gzip -> pmtiles::Writer -> .pmtiles
//!                                                    pmtiles::Reader -> VectorArchive
//! ```
//!
//! Nothing in here knows what a DWG is. The tiler that turns primitives into
//! [`mvt::Tile`]s lives above this module, and the container reuses the
//! PMTiles writer and reader from EPIC F rather than inventing a second one.
//!
//! # The three seams that round-trip and are still wrong
//!
//! A writer checked only against its own reader passes with all three, so each
//! has a test that does not share the misreading:
//!
//! 1. **Compression is described, not applied.** The PMTiles writer stores
//!    payloads exactly as handed to it and `tile_compression` only labels
//!    them. [`VectorArchiveWriter`] gzips each tile itself and then labels the
//!    archive gzip. Our reader hands bytes back untouched, so uncompressed
//!    protobuf labelled gzip would read back perfectly here and be unreadable
//!    everywhere else.
//! 2. **`vector_layers` is a MUST nothing enforced.** It has no typed field and
//!    lives in [`Metadata::extra`](crate::pmtiles::Metadata::extra). The
//!    writer here builds it from a declared schema, and
//!    [`pmtiles::validate`](crate::pmtiles::validate) now reports an MVT
//!    archive without a valid one.
//! 3. **The raster sink is the wrong door.** `PmTilesSink` encodes a raster.
//!    Vector tiles drive [`pmtiles::writer::Writer`](crate::pmtiles::writer::Writer)
//!    with [`TileType::Mvt`](crate::pmtiles::TileType::Mvt) directly, and
//!    [`TileFormat`](crate::sink::TileFormat) is left alone.

pub mod archive;
pub mod mvt;

pub use archive::{
    FieldType, LayerSchema, VectorArchive, VectorArchiveOptions, VectorArchiveWriter, VectorError,
};
