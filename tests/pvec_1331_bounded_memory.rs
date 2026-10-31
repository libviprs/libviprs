//! Issue #1331: peak resident memory of the PVec sink against the PMTiles sink on the same scan.
//! Ignored by default like `pmtiles_bounded_memory.rs`; the gate runs it. Needs `pvec-raster`.
//!
//! What the PVec sink can promise is limited by the writer it hands tiles to: PVec's writer holds
//! the encoded tiles of the artifact until `finish`, so the sink's peak is the pixel working set
//! the engine was given plus the encoded pyramid, and not the working set alone. This test pins
//! the claim that is true: the PVec sink stays within the PMTiles sink's peak plus the encoded
//! pyramid held by the writer plus its index allowance.

#![cfg(feature = "pvec-raster")]

#[test]
#[ignore = "memory gate: run with --ignored on the gate host"]
fn the_pvec_sink_peaks_within_the_pmtiles_sink_plus_the_encoded_pyramid() {
    panic!("red: not measured yet");
}
