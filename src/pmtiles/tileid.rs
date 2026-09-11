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
///
/// # The mask is not decoration
///
/// `n` here is the size of the *sub*-square, which on the way in is smaller
/// than the coordinates: `zxy_to_tileid` walks `s` down from `n/2` while `x`
/// and `y` keep their full value. So `n - 1 - x` is a subtraction that goes
/// negative, and the reference implementations lean on it wrapping (C `int`
/// overflow, Go's unsigned wrap-around) rather than meaning it. Rust in a
/// debug build panics on that subtraction instead, which is how this was
/// found.
///
/// Masking first gives the same answer without the wrap. Only the bits below
/// `n` are ever read again, because each following step halves `n`, and `n` is
/// a power of two, so `(n - 1) - (x & (n - 1))` agrees with the wrapped
/// `n - 1 - x` on exactly those bits. The 185 oracle rows hold that claim:
/// they run to zoom 31, and a coordinate whose high bits mattered would
/// disagree there.
fn rotate(n: u64, x: &mut u64, y: &mut u64, rx: u64, ry: u64) {
    if ry == 0 {
        if rx == 1 {
            let mask = n - 1;
            *x = mask - (*x & mask);
            *y = mask - (*y & mask);
        }
        std::mem::swap(x, y);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `(z, x, y, tile_id)` measured with `pmtiles.ZxyToID` from go-pmtiles
    /// v1.31.2, commit a3e4951ea6a0477b784c27c1dcbfd9c130878c5a, the
    /// `linux_arm64` release tarball with sha256
    /// `f8bd47e7ea866863489cad588fbaf2f31f42e5821f7a03f009b3769f05801cb1`.
    ///
    /// Nothing in this table was computed here. The oracle lane ran the
    /// reference implementation and wrote the numbers down; they also match a
    /// derivation made independently from the specification text, and a second
    /// one made here from a recursive quadrant construction, which is as much
    /// agreement as three parties who never saw each other's work can produce.
    ///
    /// # The rows that carry the weight are the deep ones
    ///
    /// The structural coordinates (level firsts and lasts, grid corners) agree
    /// under several different mappings, **plain Z-order with no rotation
    /// among them**, so on their own they cannot say which convention PMTiles
    /// uses. The discriminating rows below are mid-to-high zoom, off every
    /// quadrant boundary, with `x` and `y` of differing parity, and four of
    /// them are swapped pairs so a mapping that is accidentally symmetric does
    /// not survive either. `(12, 3423, 1763) -> 19078479` is the row the spec
    /// itself publishes and the one that rules out 21314735, 19217573 and
    /// 20757839.
    const ORACLE_DISCRIMINATORS: &[(u8, u32, u32, u64)] = &[
        (12, 3423, 1763, 19078479), // the row that discriminates between candidate Hilbert conventions: x odd, y odd, mid-zoom, off every quadrant boundary
        (9, 173, 298, 210828),      // z9, x odd and y even
        (9, 306, 91, 324512),       // z9, x even and y odd
        (11, 1337, 642, 5020316),   // z11, x odd and y even
        (11, 908, 1511, 3238970),   // z11, x even and y odd
        (13, 5107, 2884, 79053962), // z13, x odd and y even
        (13, 2884, 5107, 53847284), // z13, the same pair with x and y swapped, which a symmetric mapping would collide
        (15, 20749, 9310, 1284441852), // z15, x odd and y even
        (15, 9310, 20749, 856223324), // z15, the same pair swapped
        (12, 1763, 3423, 12796581), // z12, the discriminating pair swapped
        (14, 11113, 6006, 321880274), // z14, x odd and y even
        (10, 617, 428, 1232398),    // z10, x odd and y even
    ];

    /// The first and last tile of each level the oracle dumped, and a spread of
    /// coordinates that are not round numbers.
    const ORACLE_STRUCTURAL: &[(u8, u32, u32, u64)] = &[
        (0, 0, 0, 0),                             // first tile of z=0
        (0, 0, 0, 0),               // last tile of z=0, coords from IDToZxy(base(z+1)-1)
        (1, 0, 0, 1),               // first tile of z=1
        (1, 1, 0, 4),               // last tile of z=1, coords from IDToZxy(base(z+1)-1)
        (2, 0, 0, 5),               // first tile of z=2
        (2, 3, 0, 20),              // last tile of z=2, coords from IDToZxy(base(z+1)-1)
        (3, 0, 0, 21),              // first tile of z=3
        (3, 7, 0, 84),              // last tile of z=3, coords from IDToZxy(base(z+1)-1)
        (4, 0, 0, 85),              // first tile of z=4
        (4, 15, 0, 340),            // last tile of z=4, coords from IDToZxy(base(z+1)-1)
        (5, 0, 0, 341),             // first tile of z=5
        (5, 31, 0, 1364),           // last tile of z=5, coords from IDToZxy(base(z+1)-1)
        (6, 0, 0, 1365),            // first tile of z=6
        (6, 63, 0, 5460),           // last tile of z=6, coords from IDToZxy(base(z+1)-1)
        (7, 0, 0, 5461),            // first tile of z=7
        (7, 127, 0, 21844),         // last tile of z=7, coords from IDToZxy(base(z+1)-1)
        (8, 0, 0, 21845),           // first tile of z=8
        (8, 255, 0, 87380),         // last tile of z=8, coords from IDToZxy(base(z+1)-1)
        (10, 0, 0, 349525),         // first tile of z=10
        (10, 1023, 0, 1398100),     // last tile of z=10, coords from IDToZxy(base(z+1)-1)
        (12, 0, 0, 5592405),        // first tile of z=12
        (12, 4095, 0, 22369620),    // last tile of z=12, coords from IDToZxy(base(z+1)-1)
        (14, 0, 0, 89478485),       // first tile of z=14
        (14, 16383, 0, 357913940),  // last tile of z=14, coords from IDToZxy(base(z+1)-1)
        (16, 0, 0, 1431655765),     // first tile of z=16
        (16, 65535, 0, 5726623060), // last tile of z=16, coords from IDToZxy(base(z+1)-1)
        (18, 0, 0, 22906492245),    // first tile of z=18
        (18, 262143, 0, 91625968980), // last tile of z=18, coords from IDToZxy(base(z+1)-1)
        (20, 0, 0, 366503875925),   // first tile of z=20
        (20, 1048575, 0, 1466015503700), // last tile of z=20, coords from IDToZxy(base(z+1)-1)
        (31, 0, 0, 1537228672809129301), // first tile of z=31
        (31, 2147483647, 0, 6148914691236517204), // last tile of z=31, coords from IDToZxy(base(z+1)-1)
        (5, 17, 11, 1209),                        // arbitrary
        (6, 37, 23, 4835),                        // arbitrary, both prime
        (7, 73, 101, 14797),                      // arbitrary, both prime
        (9, 291, 177, 309339),                    // arbitrary
        (10, 601, 383, 1238035),                  // arbitrary, 383 is 2^k-1 shaped
        (11, 1279, 809, 4933265),                 // arbitrary
        (12, 1207, 1539, 8633061),                // arbitrary
        (13, 4123, 2871, 79362511),               // arbitrary
        (14, 8171, 5680, 129937594),              // z14 tile over Denver
        (14, 9649, 12321, 241770071),             // z14 tile over Sydney
        (14, 8377, 5449, 317666519),              // z14 tile over Chicago
        (14, 16383, 0, 357913940),                // z14 max x with y=0
        (14, 0, 16383, 178956970),                // z14 x=0 with max y
        (15, 21845, 32767, 1002159035),           // arbitrary at z15, y is the maximum
        (18, 200000, 131072, 73019712853),        // arbitrary at z18, y is a power of two
        (20, 1048575, 1048575, 1099511627775),    // z20 max x, max y
        (20, 524287, 524288, 916259689812),       // z20 straddling the centre seam
    ];

    /// Every tile of zoom 2 in tile id order, from the oracle.
    ///
    /// This is the cheapest discriminator after the deep rows: a Hilbert curve
    /// visits `(0,0), (1,0), (1,1), (0,1)` for ids 5 to 8 while Z-order visits
    /// `(0,0), (0,1), (1,0), (1,1)`, and both agree on the spec's only zoom-2
    /// row.
    const ORACLE_Z2_ORDER: &[(u8, u32, u32, u64)] = &[
        (2, 0, 0, 5),
        (2, 1, 0, 6),
        (2, 1, 1, 7),
        (2, 0, 1, 8),
        (2, 0, 2, 9),
        (2, 0, 3, 10),
        (2, 1, 3, 11),
        (2, 1, 2, 12),
        (2, 2, 2, 13),
        (2, 2, 3, 14),
        (2, 3, 3, 15),
        (2, 3, 2, 16),
        (2, 3, 1, 17),
        (2, 2, 1, 18),
        (2, 2, 0, 19),
        (2, 3, 0, 20),
    ];

    /// What go-pmtiles answers for coordinates that are outside the grid.
    ///
    /// **These are not values to reproduce.** `ZxyToID` masks its inputs
    /// instead of checking them, so it answers a question it was not asked:
    /// `(2, 4, 0)` comes back as tile id 5, which is the `(2, 0, 0)` tile, and
    /// zooms 32, 33 and 63 all come back as 6148914691236517205, the first id
    /// past what a `u64` can address.
    ///
    /// This is the third oracle posture from the campaign's list: the
    /// reference accepts the input and produces garbage, so **refusing is more
    /// faithful than matching**. A pyramid reader handed `(2, 4, 0)` by a URL
    /// router would otherwise serve a real, decodable tile for a coordinate
    /// that does not exist, which is indistinguishable from working.
    const ORACLE_MASKED_OUT_OF_RANGE: &[(u8, u32, u32, u64)] = &[
        (2, 4, 0, 5),                    // x is out of range for z=2 (valid x is 0..3)
        (2, 0, 4, 5),                    // y is out of range for z=2
        (1, 2, 2, 1),                    // both out of range for z=1
        (15, 32771, 21845, 688296049), // x=32771 is out of range for z=15, whose maximum x is 32767
        (1, 0, 2, 1), // y is out of range for z=1; this is one of the tile-fetch probes
        (2, 5, 1, 7), // x is out of range for z=2; this is one of the tile-fetch probes
        (32, 0, 0, 6148914691236517205), // z=32, past what a uint64 tile id can address
        (32, 1, 0, 6148914691236517206), // z=32 with a non-zero x
        (33, 0, 0, 6148914691236517205), // z=33
        (63, 0, 0, 6148914691236517205), // z=63
    ];

    #[test]
    fn tile_ids_match_the_go_pmtiles_oracle() {
        let mut checked = 0;
        for &(z, x, y, want) in ORACLE_DISCRIMINATORS
            .iter()
            .chain(ORACLE_STRUCTURAL)
            .chain(ORACLE_Z2_ORDER)
        {
            assert_eq!(
                zxy_to_tileid(z, x, y).expect("an oracle row is inside the addressable range"),
                want,
                "zxy_to_tileid({z}, {x}, {y})"
            );
            assert_eq!(
                tileid_to_zxy(want).expect("an oracle id decodes"),
                (z, x, y),
                "tileid_to_zxy({want})"
            );
            checked += 1;
        }
        // The positive control. Three `chain`ed slices are three chances to
        // iterate nothing, and a loop that runs zero times passes every
        // assertion inside it.
        assert_eq!(
            checked,
            ORACLE_DISCRIMINATORS.len() + ORACLE_STRUCTURAL.len() + ORACLE_Z2_ORDER.len(),
            "the sweep did not visit every oracle row"
        );
        assert!(checked >= 70, "only {checked} oracle rows were checked");
    }

    #[test]
    fn zoom_2_is_walked_in_hilbert_order_not_z_order() {
        // Ids 5, 6, 7, 8 are the four tiles of the first quadrant, and the
        // order they are visited in is what separates the two curves.
        let visited: Vec<(u32, u32)> = (5..=8)
            .map(|id| {
                let (_, x, y) = tileid_to_zxy(id).unwrap();
                (x, y)
            })
            .collect();
        assert_eq!(
            visited,
            vec![(0, 0), (1, 0), (1, 1), (0, 1)],
            "this is Z-order, not the Hilbert order the oracle measured"
        );

        // And the whole level, in order, against the oracle.
        for (index, &(z, x, y, id)) in ORACLE_Z2_ORDER.iter().enumerate() {
            assert_eq!(
                id,
                5 + index as u64,
                "the oracle's zoom-2 rows are in id order"
            );
            assert_eq!(tileid_to_zxy(id).unwrap(), (z, x, y));
        }
    }

    #[test]
    fn swapping_x_and_y_changes_the_tile_id() {
        // The oracle's swapped pairs. A mapping that is symmetric in x and y
        // round-trips perfectly and fails every one of these.
        let pairs = [
            ((13u8, 5107u32, 2884u32), (13u8, 2884u32, 5107u32)),
            ((15, 20749, 9310), (15, 9310, 20749)),
            ((12, 3423, 1763), (12, 1763, 3423)),
        ];
        for ((za, xa, ya), (zb, xb, yb)) in pairs {
            let a = zxy_to_tileid(za, xa, ya).unwrap();
            let b = zxy_to_tileid(zb, xb, yb).unwrap();
            assert_ne!(a, b, "({za},{xa},{ya}) and ({zb},{xb},{yb}) share an id");
        }
    }

    #[test]
    fn out_of_grid_coordinates_are_refused_where_go_pmtiles_masks_them() {
        for &(z, x, y, masked) in ORACLE_MASKED_OUT_OF_RANGE {
            let got = zxy_to_tileid(z, x, y);
            assert!(
                matches!(
                    got,
                    Err(PmTilesError::ZoomOutOfRange { .. } | PmTilesError::CoordOutOfRange { .. })
                ),
                "({z}, {x}, {y}) must be refused, not answered; go-pmtiles masks it to {masked}"
            );
        }

        // The positive control, and it is the whole point of this test: the
        // in-range neighbour of each masked row still works, so "everything is
        // refused" cannot pass here.
        assert_eq!(zxy_to_tileid(2, 3, 0).unwrap(), 20);
        assert_eq!(zxy_to_tileid(1, 1, 1).unwrap(), 3);
        assert_eq!(zxy_to_tileid(15, 21845, 32767).unwrap(), 1_002_159_035);
        assert_eq!(zxy_to_tileid(31, 0, 0).unwrap(), 1_537_228_672_809_129_301);
    }

    #[test]
    fn consecutive_ids_on_one_level_land_on_adjacent_tiles() {
        // The defining property of a Hilbert curve, and one no Z-order
        // implementation has: step by one id and you step to a tile that
        // shares an edge. It is an invariant rather than a vector, so it holds
        // the mapping even where the oracle table has no row.
        let mut steps = 0;
        for z in 1..=7u8 {
            let first = first_tileid_of_zoom(z).unwrap();
            let count = 1u64 << (2 * u32::from(z));
            for id in first..first + count - 1 {
                let (_, x0, y0) = tileid_to_zxy(id).unwrap();
                let (_, x1, y1) = tileid_to_zxy(id + 1).unwrap();
                let distance = x0.abs_diff(x1) + y0.abs_diff(y1);
                assert_eq!(distance, 1, "ids {id} and {} are not neighbours", id + 1);
                steps += 1;
            }
        }
        // 4 + 16 + ... + 16384, less one step per level.
        assert_eq!(
            steps,
            21844 - 7,
            "the adjacency sweep did not cover every level"
        );
    }

    #[test]
    fn every_level_is_a_bijection_onto_its_id_range() {
        for z in 0..=6u8 {
            let side = 1u32 << z;
            let first = first_tileid_of_zoom(z).unwrap();
            let mut seen = std::collections::BTreeSet::new();
            for y in 0..side {
                for x in 0..side {
                    let id = zxy_to_tileid(z, x, y).unwrap();
                    assert!(
                        (first..first + u64::from(side) * u64::from(side)).contains(&id),
                        "({z}, {x}, {y}) -> {id} is outside zoom {z}'s id range"
                    );
                    assert!(seen.insert(id), "id {id} is produced twice at zoom {z}");
                }
            }
            assert_eq!(seen.len(), (side as usize) * (side as usize));
        }
    }

    #[test]
    fn level_bases_are_the_closed_form_and_the_ceiling_is_where_u64_ends() {
        // (4^z - 1)/3, checked against the oracle's own level_bases rows.
        for (z, want) in [
            (0u8, 0u64),
            (1, 1),
            (2, 5),
            (3, 21),
            (12, 5_592_405),
            (20, 366_503_875_925),
            (31, 1_537_228_672_809_129_301),
        ] {
            assert_eq!(first_tileid_of_zoom(z).unwrap(), want);
            assert_eq!(zxy_to_tileid(z, 0, 0).unwrap(), want);
        }

        // `MAX_TILE_ID` derived a second way, in `u128` where zoom 32's base is
        // not a special case at all: it is `(4^32 - 1)/3 - 1`.
        let base_of_zoom_32 = (((1u128 << 64) - 1) / 3) as u64;
        assert_eq!(MAX_TILE_ID, base_of_zoom_32 - 1);
        // And it is what go-pmtiles saturates to for z=32, measured.
        assert_eq!(base_of_zoom_32, 6_148_914_691_236_517_205);

        assert_eq!(tileid_to_zxy(MAX_TILE_ID).unwrap(), (31, 2_147_483_647, 0));
        assert!(matches!(
            tileid_to_zxy(base_of_zoom_32),
            Err(PmTilesError::TileIdOutOfRange { .. })
        ));
        assert!(matches!(
            tileid_to_zxy(u64::MAX),
            Err(PmTilesError::TileIdOutOfRange { .. })
        ));
        assert!(matches!(
            first_tileid_of_zoom(32),
            Err(PmTilesError::ZoomOutOfRange { .. })
        ));
    }
}
