//! A raster on its way into a PVec (issue #1328, R2.4, ADR 0018 in libviprs/pvec).
//!
//! This is the **raster side** of the hand-off: the georeference composition and the pyramid
//! producer. It lives outside [`crate::pvec`] on purpose, because nothing under `src/pvec/` may
//! refer to the raster-side georeference types (`tests/pvec_units.rs` scans for it): the vector
//! path never rescales or georeferences geometry, and a raster is the one thing that needs a
//! `GeoTransform` to find its place in the drawing's source space.
//!
//! * [`placement`] composes `pixel -> source` from a stored `GeoTransform` and a calibration.
//! * [`pyramid`] cuts and encodes the pyramid and returns the tiles and placement fields.
//! * `container` (feature `pvec-raster`) hands them to PVec's writer, tiles untouched.

pub mod placement;
pub mod pyramid;

#[cfg(feature = "pvec-raster")]
#[cfg_attr(docsrs, doc(cfg(feature = "pvec-raster")))]
pub mod container;
