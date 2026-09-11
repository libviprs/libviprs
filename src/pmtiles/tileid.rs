//! The `(z, x, y)` to `u64` TileID mapping, and its inverse.
//!
//! PMTiles addresses every tile in an archive with one 64-bit integer. The ids
//! run zoom by zoom, and within a zoom they follow a Hilbert curve, so tiles
//! that are near each other on the map are near each other in the file. That
//! is the whole reason the format can answer a tile request with two or three
//! ranged reads: a viewer panning around asks for ids that sit close together,
//! and one directory page covers them all.
//!
//! ```text
//!   zoom 0:  id 0                     zoom 1, walked as a "U":
//!   zoom 1:  ids 1..=4                     x=0   x=1
//!   zoom 2:  ids 5..=20               y=0   1     4
//!   zoom 3:  ids 21..=84              y=1   2  -  3
//!   zoom z:  first id is (4^z - 1)/3
//! ```
//!
//! The coordinates are plain ZXY, the slippy-map convention, with row 0 at the
//! top. That is what [`Layout::Xyz`](crate::planner::Layout) already emits, so
//! a [`TileCoord`](crate::planner::TileCoord) maps 1:1 onto `(z, x, y)` with
//! no TMS row inversion anywhere in this crate.
//!
//! # Testing this is harder than implementing it, and that is not a joke
//!
//! The specification does not contain an algorithm for this mapping. It gives
//! one sentence ("a cumulative position on the series of Hilbert curves"), a
//! link to a Wikipedia article that no longer carries the reference code it
//! used to, and a table of seven `(z, x, y) -> id` rows. That is the entire
//! normative source.
//!
//! Seven rows is not enough to pin the convention, and the trap is sharper
//! than it sounds: **three different wrong mappings reproduce every row of the
//! spec's table up to zoom 2**, including plain Morton / Z-order with no
//! Hilbert behaviour at all. They separate only deeper. On the spec's own
//! zoom-12 row they answer 21314735, 19217573 and 20757839 against the correct
//! 19078479.
//!
//! Worse, a round trip cannot notice any of it. The encoder and the decoder
//! share the convention, so a wrong pair round-trips perfectly and agrees with
//! nothing else in the world.
//!
//! So the tests here are pinned to three things that are not each other:
//!
//! * the spec's seven rows, `(12, 3423, 1763) -> 19078479` included, which is
//!   the row that discriminates;
//! * the zoom-2 and zoom-3 orderings, enumerated, where Hilbert and Z-order
//!   visibly diverge from the fifth id onward;
//! * the curve's own invariants: a bijection onto `base(z) ..= base(z+1)-1`,
//!   and consecutive ids landing on 4-adjacent tiles, which no Z-order
//!   implementation satisfies.
//!
//! # Where `u64` runs out
//!
//! Zoom 31 is the last addressable level. Zoom 32 would start at
//! 6148914691236517205 and end past `u64::MAX`, so both directions refuse
//! above [`MAX_ZOOM`] / [`MAX_TILE_ID`] rather than wrapping. The bound itself
//! is representable, which is what makes the check clean: `id > MAX_TILE_ID`
//! needs no arithmetic that could overflow in the check.

use crate::pmtiles::PmTilesError;

/// The highest zoom level a PMTiles v3 TileID can address.
///
/// Not a number the spec states: it falls out of the format meeting `u64`.
/// Zoom `z` holds `4^z` tiles and the ids are cumulative, so zoom 32 would
/// need ids up to `(4^33 - 1)/3 - 1`, which is about 2.4e19 against
/// `u64::MAX`'s 1.8e19. An implementation with a wider integer could go
/// further and would be incompatible with every other implementation.
pub const MAX_ZOOM: u8 = 31;

/// The largest valid TileID, which is the last tile of zoom [`MAX_ZOOM`].
///
/// `(4^32 - 1)/3 - 1`. The first id of zoom 32 is exactly one more than this
/// and still fits in a `u64`, so `id > MAX_TILE_ID` is a total check that
/// needs no overflow handling of its own.
pub const MAX_TILE_ID: u64 = 6_148_914_691_236_517_204;

/// The first TileID of a zoom level, `(4^z - 1)/3`.
///
/// This is the "cumulative" half of the mapping: every tile of every lower
/// zoom is numbered before the first tile of this one.
pub fn first_tileid_of_zoom(zoom: u8) -> Result<u64, PmTilesError> {
    if zoom > MAX_ZOOM {
        return Err(PmTilesError::ZoomOutOfRange {
            zoom,
            max: MAX_ZOOM,
        });
    }
    Ok(base(zoom))
}

/// `(z, x, y)` to TileID.
///
/// Refuses a zoom above [`MAX_ZOOM`] and an `x` or `y` outside the `2^z` grid
/// of its own level. Both refusals are what keeps the arithmetic below safe:
/// with `z <= 31` the largest intermediate is `s * s * 3` at `s = 2^30`, which
/// is under `2^62`, and the sum `base(z) + d` is at most [`MAX_TILE_ID`].
///
/// # Examples
///
/// ```
/// use libviprs::pmtiles::zxy_to_tileid;
///
/// // The spec's own worked row, and the one that separates a real Hilbert
/// // curve from three plausible imitations.
/// assert_eq!(zxy_to_tileid(12, 3423, 1763).unwrap(), 19_078_479);
///
/// // Out of the grid is an error, never a wrapped id.
/// assert!(zxy_to_tileid(3, 8, 0).is_err());
/// ```
pub fn zxy_to_tileid(z: u8, x: u32, y: u32) -> Result<u64, PmTilesError> {
    if z > MAX_ZOOM {
        return Err(PmTilesError::ZoomOutOfRange {
            zoom: z,
            max: MAX_ZOOM,
        });
    }
    let side: u64 = 1u64 << z;
    if u64::from(x) >= side || u64::from(y) >= side {
        return Err(PmTilesError::CoordOutOfRange { z, x, y, side });
    }

    let mut tx = u64::from(x);
    let mut ty = u64::from(y);
    let mut d: u64 = 0;
    let mut s: u64 = side >> 1;
    while s > 0 {
        let rx: u64 = u64::from((tx & s) > 0);
        let ry: u64 = u64::from((ty & s) > 0);
        // The quadrant index, ordered (0,0) -> 0, (0,1) -> 1, (1,1) -> 2,
        // (1,0) -> 3. At zoom 1 the loop runs once and `d` is exactly this,
        // which is what reproduces the spec's four zoom-1 rows.
        d += s * s * ((3 * rx) ^ ry);
        rotate(s, &mut tx, &mut ty, rx, ry);
        s >>= 1;
    }

    Ok(base(z) + d)
}

/// TileID to `(z, x, y)`.
///
/// Refuses an id above [`MAX_TILE_ID`], which is every id no `(z, x, y)`
/// triple can produce.
///
/// # Examples
///
/// ```
/// use libviprs::pmtiles::tileid_to_zxy;
///
/// assert_eq!(tileid_to_zxy(0).unwrap(), (0, 0, 0));
/// assert_eq!(tileid_to_zxy(5).unwrap(), (2, 0, 0));
///
/// // The last tile of zoom 31 lands on (x_max, 0) rather than the far
/// // corner, because that is where a Hilbert curve of odd order ends.
/// assert_eq!(tileid_to_zxy(6_148_914_691_236_517_204).unwrap(), (31, 2_147_483_647, 0));
/// ```
pub fn tileid_to_zxy(id: u64) -> Result<(u8, u32, u32), PmTilesError> {
    if id > MAX_TILE_ID {
        return Err(PmTilesError::TileIdOutOfRange {
            id,
            max: MAX_TILE_ID,
        });
    }

    // Walk up the levels while the id is still past the start of the next one.
    // Bounded by the check above, so this cannot run past zoom 31.
    let mut z: u8 = 0;
    while z < MAX_ZOOM && id >= base(z + 1) {
        z += 1;
    }

    let side: u64 = 1u64 << z;
    let mut t = id - base(z);
    let mut tx: u64 = 0;
    let mut ty: u64 = 0;
    let mut s: u64 = 1;
    while s < side {
        let rx: u64 = 1 & (t >> 1);
        let ry: u64 = 1 & (t ^ rx);
        rotate(s, &mut tx, &mut ty, rx, ry);
        tx += s * rx;
        ty += s * ry;
        t >>= 2;
        s <<= 1;
    }

    // Both are below `2^z <= 2^31`, so the casts cannot truncate.
    Ok((z, tx as u32, ty as u32))
}

/// The first TileID of `zoom`, with no range check.
///
/// `(4^z - 1)/3`, exact in integer arithmetic for every `z <= 31`. Private
/// because it is only sound under that precondition: at `z = 32` the shift is
/// wider than the type, which panics in debug and silently masks in release.
fn base(zoom: u8) -> u64 {
    debug_assert!(zoom <= MAX_ZOOM, "base() is only defined up to zoom 31");
    ((1u64 << (2 * u32::from(zoom))) - 1) / 3
}

/// Rotate and flip a quadrant so the curve joins up with its neighbours.
///
/// This is the entire orientation convention, and it is the part the spec does
/// not state at all. When `ry == 0` (the upper half in `y`): if `rx == 1` (the
/// right quadrant) reflect both coordinates about the sub-square's centre,
/// then in either case swap `x` and `y`. When `ry == 1`, do nothing.
///
/// Dropping either half leaves something that still passes every row of the
/// spec's table up to zoom 2 and is wrong from zoom 3 onward.
fn rotate(n: u64, x: &mut u64, y: &mut u64, rx: u64, ry: u64) {
    if ry == 0 {
        if rx == 1 {
            *x = n - 1 - *x;
            *y = n - 1 - *y;
        }
        std::mem::swap(x, y);
    }
}
