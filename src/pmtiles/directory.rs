//! Directory entries, and the column-oriented form they are stored in.
//!
//! A directory is a sorted list of entries, and an entry says one of two
//! things: where a tile lives, or where a leaf directory that knows more
//! lives. There is no type tag; `run_length == 0` is the only discriminant.
//!
//! ```text
//!   Entry { tile_id: 5, offset: 1337, length: 42, run_length: 1 }
//!     -> tile 5 is at bytes 1337..1379 of the tile data section
//!   Entry { tile_id: 5, offset: 1337, length: 42, run_length: 3 }
//!     -> tiles 5, 6 and 7 are all that same blob
//!   Entry { tile_id: 5, offset: 1337, length: 42, run_length: 0 }
//!     -> a leaf directory starting at tile 5 is at bytes 1337..1379
//!        of the *leaf directories* section
//! ```
//!
//! The base an offset is relative to comes from the **entry's kind, not from
//! the directory it was found in**. A tile entry six levels deep in a leaf
//! chain is still relative to `tile_data_offset`. The two tempting mistakes
//! are making it relative to the leaf's own start or to
//! `leaf_directories_offset`, and a writer and reader that make the same wrong
//! choice round-trip perfectly.
//!
//! # The storage form is columns, not records
//!
//! An encoded directory is five parts in order: the entry count, then all `n`
//! tile ids, then all `n` run lengths, then all `n` lengths, then all `n`
//! offsets. Every one of them is a varint. Two of the columns are not stored
//! literally:
//!
//! * **tile ids are deltas.** Entry `i` stores `tile_id[i] - tile_id[i-1]`,
//!   with a notional `tile_id[-1] = 0`, so the first entry stores its own id.
//!   The deltas are unsigned, which is what forces entries to ascend.
//! * **offsets use a contiguous shorthand.** An entry whose offset is exactly
//!   where the previous entry ended stores `0`; anything else stores
//!   `offset + 1`. **Entry 0 never uses the shorthand**, even when its offset
//!   is 0, because it has no previous entry: a clustered archive's first entry
//!   encodes as the varint `01`.
//!
//! That last rule is load-bearing in the decoder, because a literal `0` at
//! index 0 would be decoded as `value - 1` and **underflow a `u64` to
//! 18446744073709551615 in a release build rather than panicking**. It is
//! refused by name as [`PmTilesError::ContiguousOffsetAtFirstEntry`].
//!
//! The whole encoded buffer is then compressed with the header's internal
//! compression, and each leaf is compressed individually rather than the leaf
//! section being compressed as one blob. That is the caller's job:
//! [`serialize_entries`] and [`deserialize_entries`] work on plain bytes so
//! the compression stays one decision made in one place.

use crate::pmtiles::PmTilesError;
use crate::pmtiles::tileid::MAX_TILE_ID;
use crate::pmtiles::varint::{VarintCursor, encode_uvarint, uvarint_len};

/// One directory entry.
///
/// # Why this one is not `#[non_exhaustive]`
///
/// Same reason as [`Header`](crate::pmtiles::Header): the shape is a wire
/// format with four fields and no room for a fifth, so there is no future
/// field for the attribute to protect a caller from, and leaving it
/// exhaustive is what lets a writer outside this crate build one with a struct
/// literal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Entry {
    /// The tile this entry starts at, or the first tile of the leaf directory
    /// it points at.
    pub tile_id: u64,
    /// Where the blob starts, relative to the tile data section for a tile
    /// entry and to the leaf directories section for a leaf pointer.
    pub offset: u64,
    /// How many bytes the blob occupies, **as stored**. Always greater than
    /// zero; the spec says so twice.
    pub length: u32,
    /// How many consecutive tile ids this entry covers, or 0 to mark it a
    /// pointer to a leaf directory.
    pub run_length: u32,
}

impl Entry {
    /// Whether this entry points at a leaf directory rather than a tile.
    pub fn is_leaf(&self) -> bool {
        self.run_length == 0
    }

    /// Whether `tile_id` falls inside this entry's run.
    ///
    /// # This is the check that separates a hit from a miss
    ///
    /// A directory lookup finds the **largest entry whose `tile_id` is at most
    /// the one being looked for**, which means a lookup for a tile that is not
    /// in the archive lands on the preceding entry rather than finding
    /// nothing. Landing on an entry is not a hit. Without this check a reader
    /// returns the previous tile's bytes for every hole in the archive, and
    /// because those bytes are a real, decodable tile it looks like it is
    /// working. A round-trip test that only asks for tiles it wrote cannot see
    /// it.
    ///
    /// The spec never writes this down, because it never writes the lookup
    /// down at all. It follows from a run of `r` covering
    /// `tile_id ..= tile_id + r - 1`.
    ///
    /// A leaf pointer contains nothing: its run length is 0, so this is always
    /// false for one, and a caller that lands on a leaf pointer should follow
    /// it rather than ask this.
    pub fn run_contains(&self, tile_id: u64) -> bool {
        if self.run_length == 0 || tile_id < self.tile_id {
            return false;
        }
        // `checked_add` rather than a comparison the other way round: a
        // hostile entry can carry a tile id near `u64::MAX` and a long run,
        // and the sum is what wraps.
        match self.tile_id.checked_add(u64::from(self.run_length)) {
            Some(end) => tile_id < end,
            // The run runs off the end of the id space, so it covers every id
            // at or above its start, and `tile_id >= self.tile_id` already.
            None => true,
        }
    }

    /// The last tile id this entry covers, or `None` for a leaf pointer.
    pub fn last_tile_id(&self) -> Option<u64> {
        if self.run_length == 0 {
            return None;
        }
        Some(self.tile_id.saturating_add(u64::from(self.run_length - 1)))
    }
}

/// Append one tile to `entries`, collapsing it into the previous entry's run
/// where the format allows.
///
/// This is the run-length half of a writer. A run may absorb the new tile only
/// when all three hold:
///
/// * the ids are consecutive, `tile_id == previous.tile_id + previous.run_length`;
/// * the payload is the same blob, meaning the same offset **and** the same
///   length. Dedupe is what makes this common: a pyramid of mostly blank tiles
///   points thousands of ids at one stored blob;
/// * the previous entry is a tile entry, not a leaf pointer.
///
/// Entries must arrive in ascending id order, which is the order the format
/// requires and the only order the delta encoding can express.
///
/// # Examples
///
/// ```
/// use libviprs::pmtiles::directory::push_entry;
///
/// let mut entries = Vec::new();
/// push_entry(&mut entries, 5, 1337, 42).unwrap();
/// push_entry(&mut entries, 6, 1337, 42).unwrap();   // same blob, next id
/// push_entry(&mut entries, 7, 4096, 11).unwrap();   // a different blob
///
/// assert_eq!(entries.len(), 2);
/// assert_eq!(entries[0].run_length, 2);
/// assert!(entries[0].run_contains(6));
/// assert!(!entries[0].run_contains(7));
/// ```
pub fn push_entry(
    entries: &mut Vec<Entry>,
    tile_id: u64,
    offset: u64,
    length: u32,
) -> Result<(), PmTilesError> {
    if length == 0 {
        return Err(PmTilesError::ZeroLengthEntry {
            index: entries.len(),
        });
    }
    if tile_id > MAX_TILE_ID {
        return Err(PmTilesError::TileIdOutOfRange {
            id: tile_id,
            max: MAX_TILE_ID,
        });
    }

    if let Some(last) = entries.last_mut() {
        // The first id a new entry may claim: one past the end of the previous
        // run, or one past a leaf pointer's own id since a leaf covers nothing.
        let run_end =
            last.tile_id
                .checked_add(u64::from(last.run_length))
                .ok_or(PmTilesError::Overflow {
                    what: "the end of the previous run",
                })?;
        let first_free = if last.is_leaf() {
            last.tile_id.checked_add(1).ok_or(PmTilesError::Overflow {
                what: "the id after a leaf pointer",
            })?
        } else {
            run_end
        };
        if tile_id < first_free {
            return Err(PmTilesError::NonAscendingEntry {
                index: entries.len(),
            });
        }
        let extends_the_run =
            !last.is_leaf() && tile_id == run_end && last.offset == offset && last.length == length;
        if extends_the_run {
            last.run_length = last
                .run_length
                .checked_add(1)
                .ok_or(PmTilesError::Overflow {
                    what: "a run length",
                })?;
            return Ok(());
        }
    }

    entries.push(Entry {
        tile_id,
        offset,
        length,
        run_length: 1,
    });
    Ok(())
}

/// Serialise a directory into its column-oriented form, uncompressed.
///
/// The caller compresses the result with the header's internal compression.
///
/// Refuses, in this order: an empty directory (the spec makes a non-empty one
/// a `MUST`, and a directory exists to point at something), an entry length of
/// zero, a tile id past the addressable range, and entries that do not
/// strictly ascend. The last one is not a style preference: the tile id column
/// holds unsigned deltas, so a repeated or decreasing id is a value the format
/// cannot express, and writing one would produce a file whose decode does not
/// round-trip.
pub fn serialize_entries(entries: &[Entry]) -> Result<Vec<u8>, PmTilesError> {
    if entries.is_empty() {
        return Err(PmTilesError::EmptyDirectory);
    }

    // A guess, not a bound. A worst-case entry is four ten-byte varints, but
    // no real archive has one: measured across the three go-pmtiles goldens,
    // an entry costs four to seven bytes. Twelve is comfortably over that and
    // well under 40, so the common case takes no regrowth and the rare one
    // lets `Vec` do its job.
    let mut out = Vec::with_capacity(uvarint_len(entries.len() as u64) + entries.len() * 12);
    encode_uvarint(entries.len() as u64, &mut out);

    let mut last_id: u64 = 0;
    for (index, entry) in entries.iter().enumerate() {
        if entry.length == 0 {
            return Err(PmTilesError::ZeroLengthEntry { index });
        }
        if entry.tile_id > MAX_TILE_ID {
            return Err(PmTilesError::TileIdOutOfRange {
                id: entry.tile_id,
                max: MAX_TILE_ID,
            });
        }
        if index > 0 && entry.tile_id <= last_id {
            return Err(PmTilesError::NonAscendingEntry { index });
        }
        encode_uvarint(entry.tile_id - last_id, &mut out);
        last_id = entry.tile_id;
    }

    for entry in entries {
        encode_uvarint(u64::from(entry.run_length), &mut out);
    }

    for entry in entries {
        encode_uvarint(u64::from(entry.length), &mut out);
    }

    // `next_byte` is where the previous entry ended, and it advances on every
    // entry whichever branch was taken. A decoder that only advanced it on the
    // long form would drift the moment two shorthand entries met.
    let mut next_byte: u64 = 0;
    for (index, entry) in entries.iter().enumerate() {
        if index > 0 && entry.offset == next_byte {
            encode_uvarint(0, &mut out);
        } else {
            let shifted = entry.offset.checked_add(1).ok_or(PmTilesError::Overflow {
                what: "an entry offset",
            })?;
            encode_uvarint(shifted, &mut out);
        }
        next_byte =
            entry
                .offset
                .checked_add(u64::from(entry.length))
                .ok_or(PmTilesError::Overflow {
                    what: "the end of an entry",
                })?;
    }

    Ok(out)
}

/// Parse a directory from its column-oriented form, already decompressed.
///
/// Every refusal here exists because the bytes are untrusted:
///
/// * a claimed entry count of zero, or one larger than a quarter of the bytes
///   remaining. Each of the four columns needs at least one byte per entry, so
///   no honest directory claims more entries than that, and the bound is what
///   stops a ten-byte input asking for a ten-million-entry allocation;
/// * a tile id delta of zero after the first entry, or an accumulated id past
///   [`MAX_TILE_ID`];
/// * a length of zero, or a length or run length too large for its field;
/// * the contiguous-offset shorthand at index 0, which would underflow;
/// * bytes left over after the fourth column. The format has no extension
///   mechanism, so trailing bytes mean this buffer is not the directory it
///   claims to be. That is the one refusal here that is strictness rather than
///   safety, and it is the first thing to relax if it ever costs
///   interoperability.
pub fn deserialize_entries(bytes: &[u8]) -> Result<Vec<Entry>, PmTilesError> {
    let mut cursor = VarintCursor::new(bytes);

    let claimed = cursor.next_uvarint()?;
    if claimed == 0 {
        return Err(PmTilesError::EmptyDirectory);
    }
    let remaining = cursor.remaining();
    if claimed > (remaining / 4) as u64 {
        return Err(PmTilesError::DirectoryTooManyEntries { claimed, remaining });
    }
    // The cast is sound because `claimed` is at most a quarter of a slice
    // length, which is a `usize` by construction.
    let count = claimed as usize;

    let mut entries: Vec<Entry> = Vec::new();
    entries
        .try_reserve_exact(count)
        .map_err(|_| std::io::Error::from(std::io::ErrorKind::OutOfMemory))?;

    let mut last_id: u64 = 0;
    for index in 0..count {
        let delta = cursor.next_uvarint()?;
        if index > 0 && delta == 0 {
            return Err(PmTilesError::NonAscendingEntry { index });
        }
        last_id = last_id.checked_add(delta).ok_or(PmTilesError::Overflow {
            what: "a tile id delta",
        })?;
        if last_id > MAX_TILE_ID {
            return Err(PmTilesError::TileIdOutOfRange {
                id: last_id,
                max: MAX_TILE_ID,
            });
        }
        entries.push(Entry {
            tile_id: last_id,
            ..Entry::default()
        });
    }

    for (index, entry) in entries.iter_mut().enumerate() {
        let run_length = cursor.next_uvarint()?;
        entry.run_length =
            u32::try_from(run_length).map_err(|_| PmTilesError::EntryFieldTooLarge {
                index,
                field: "run length",
                value: run_length,
            })?;
    }

    for (index, entry) in entries.iter_mut().enumerate() {
        let length = cursor.next_uvarint()?;
        if length == 0 {
            return Err(PmTilesError::ZeroLengthEntry { index });
        }
        entry.length = u32::try_from(length).map_err(|_| PmTilesError::EntryFieldTooLarge {
            index,
            field: "length",
            value: length,
        })?;
    }

    for index in 0..count {
        let value = cursor.next_uvarint()?;
        if value == 0 {
            if index == 0 {
                return Err(PmTilesError::ContiguousOffsetAtFirstEntry);
            }
            let previous = entries[index - 1];
            entries[index].offset = previous
                .offset
                .checked_add(u64::from(previous.length))
                .ok_or(PmTilesError::Overflow {
                    what: "a contiguous entry offset",
                })?;
        } else {
            entries[index].offset = value - 1;
        }
    }

    let left_over = cursor.remaining();
    if left_over > 0 {
        return Err(PmTilesError::TrailingDirectoryBytes {
            remaining: left_over,
        });
    }

    Ok(entries)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The root directory of `dupes-z0z3.pmtiles`, decompressed, as
    /// go-pmtiles v1.31.2 wrote it (commit
    /// a3e4951ea6a0477b784c27c1dcbfd9c130878c5a, archive sha256
    /// `bfc9db4c6ce6a04194e02b3d4815814adb05209f1aaba8591e4e1332f6e56a27`).
    ///
    /// The oracle lane confirmed that go-pmtiles' own `SerializeEntries`
    /// reproduces these bytes exactly from the entries it decoded, so this is
    /// a byte-for-byte target for a serializer and not only a fixture for a
    /// parser.
    ///
    /// It is the interesting golden rather than the simple one: 67 entries
    /// covering 85 addressed tiles across zooms 0 to 3, with deliberate
    /// duplicate tiles, so it exercises run lengths above 1, repeated offsets,
    /// and a **backwards** offset jump from 74 to 0 where a deduplicated tile
    /// points back at an earlier blob.
    const ORACLE_DIRECTORY_HEX: &str = concat!(
        "4300010410010101010101010101010101010101010101010101010101010101",
        "0101010101010101010101010101010101010101010101010101010101010101",
        "0101010101041001010101010101010101010101010101010101010101010101",
        "0101010101010101010101010101010101010101010101010101010101010101",
        "010101010101014a4a4a4a4a4a4a4a4a4a4a4a4a4a4a4a4a4a4a4a4a4a4a4a4a",
        "4a4a4a4a4a4a4a4a4a4a4a4a4a4a4a4a4a4a4a4a4a4a4a4a4a4a4a4a4a4a4a4a",
        "4a4a4a4a4a4a4a4a4a4a01000195010000000000000000000000000000000000",
        "000000000000000000009501ad110000000000000000000000009501ef180000",
        "0000000000000000009501e71f00000000000000",
    );

    /// Header counts go-pmtiles reports for the same archive, which the entry
    /// list has to agree with.
    const ORACLE_ADDRESSED_TILES: u64 = 85;
    const ORACLE_TILE_ENTRIES: usize = 67;
    const ORACLE_TILE_CONTENTS: usize = 63;

    fn oracle_directory_bytes() -> Vec<u8> {
        let hex = ORACLE_DIRECTORY_HEX;
        assert_eq!(hex.len() % 2, 0, "the golden hex is not whole bytes");
        (0..hex.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).expect("golden hex"))
            .collect()
    }

    #[test]
    fn the_oracle_s_directory_decodes_to_the_entries_go_pmtiles_reports() {
        let bytes = oracle_directory_bytes();
        let entries = deserialize_entries(&bytes).expect("a real archive's root directory");

        // The three header counts, which come from go-pmtiles rather than from
        // this decode, so they cross-check the entry list from outside.
        assert_eq!(entries.len(), ORACLE_TILE_ENTRIES);
        let addressed: u64 = entries.iter().map(|e| u64::from(e.run_length)).sum();
        assert_eq!(addressed, ORACLE_ADDRESSED_TILES);
        let distinct: std::collections::BTreeSet<u64> = entries.iter().map(|e| e.offset).collect();
        assert_eq!(distinct.len(), ORACLE_TILE_CONTENTS);

        // The first four entries verbatim. The second covers a run of four and
        // the third a run of sixteen, which is what a pyramid of identical
        // tiles looks like.
        let want: &[(u64, u64, u32, u32)] = &[
            (0, 0, 74, 1),
            (1, 74, 74, 4),
            (5, 0, 74, 16),
            (21, 148, 74, 1),
        ];
        for (index, &(tile_id, offset, length, run_length)) in want.iter().enumerate() {
            assert_eq!(
                entries[index],
                Entry {
                    tile_id,
                    offset,
                    length,
                    run_length
                },
                "entry {index}"
            );
        }

        // Deduplication shows up in a directory two different ways, and a
        // reader that handles one and not the other looks correct on most
        // archives. Identical tiles at consecutive ids collapse into a run;
        // identical tiles that are *not* consecutive stay separate entries
        // pointing at the same offset. This golden has both, deliberately:
        // runs of 4 and 16, and offset 148 shared by four entries that are
        // nowhere near each other (3, 31, 45 and 58).
        assert_eq!(entries.iter().filter(|e| e.offset == 148).count(), 4);
        let shared: Vec<usize> = entries
            .iter()
            .enumerate()
            .filter(|(_, e)| e.offset == 148)
            .map(|(i, _)| i)
            .collect();
        assert_eq!(shared, vec![3, 31, 45, 58]);
        let runs: std::collections::BTreeSet<u32> = entries.iter().map(|e| e.run_length).collect();
        assert_eq!(runs, [1u32, 4, 16].into_iter().collect());

        // The backwards offset jump: entry 2 points at offset 0 while entry 1
        // sits at 74. A signed or zigzagged offset column would have to show
        // itself here and does not, because the column stores `offset + 1`
        // outright rather than a delta.
        assert!(entries[2].offset < entries[1].offset);

        // Strictly ascending ids, which the unsigned delta encoding forces.
        for pair in entries.windows(2) {
            assert!(pair[0].tile_id < pair[1].tile_id, "ids are not ascending");
        }

        // And what this crate serialises from those entries is the same bytes.
        assert_eq!(serialize_entries(&entries).unwrap(), bytes);
    }

    #[test]
    fn the_column_order_and_the_two_shorthands_are_what_the_spec_says() {
        // Hand-computed from the spec's five-part layout, so a swapped pair of
        // columns fails here with a readable diff rather than only failing the
        // 276-byte golden above.
        let entries = vec![
            Entry {
                tile_id: 0,
                offset: 0,
                length: 10,
                run_length: 1,
            },
            // Contiguous with the previous entry, so the offset column stores 0.
            Entry {
                tile_id: 1,
                offset: 10,
                length: 20,
                run_length: 2,
            },
            // A gap, so the offset column stores offset + 1. A delta of 4 on
            // the id column, and a leaf pointer's run length of 0.
            Entry {
                tile_id: 5,
                offset: 100,
                length: 30,
                run_length: 0,
            },
        ];
        let want: &[u8] = &[
            0x03, // three entries
            0x00, 0x01, 0x04, // tile id deltas: 0, 1, 4
            0x01, 0x02, 0x00, // run lengths: 1, 2, 0
            0x0A, 0x14, 0x1E, // lengths: 10, 20, 30
            0x01, 0x00, 0x65, // offsets: 0+1, contiguous, 100+1
        ];
        assert_eq!(serialize_entries(&entries).unwrap(), want);
        assert_eq!(deserialize_entries(want).unwrap(), entries);
    }

    #[test]
    fn the_first_entry_never_uses_the_contiguous_shorthand() {
        // Its offset is 0 and the shorthand byte is also 0, so an encoder that
        // dropped the `index > 0` guard would write a `0` that decodes as an
        // underflow. Every clustered archive has exactly this entry.
        let entries = vec![Entry {
            tile_id: 0,
            offset: 0,
            length: 5,
            run_length: 1,
        }];
        let bytes = serialize_entries(&entries).unwrap();
        assert_eq!(bytes, vec![0x01, 0x00, 0x01, 0x05, 0x01]);
        assert_eq!(deserialize_entries(&bytes).unwrap(), entries);

        // And the decoder refuses the malformed spelling by name rather than
        // underflowing a u64 to 18446744073709551615, which is what
        // `value - 1` does in a release build.
        let malformed: &[u8] = &[0x01, 0x00, 0x01, 0x05, 0x00];
        assert!(matches!(
            deserialize_entries(malformed),
            Err(PmTilesError::ContiguousOffsetAtFirstEntry)
        ));
    }

    #[test]
    fn a_malformed_directory_is_refused_field_by_field() {
        // Every case starts from the same valid three-entry directory and
        // changes one thing, so each refusal is attributable.
        let valid = serialize_entries(&[
            Entry {
                tile_id: 1,
                offset: 0,
                length: 10,
                run_length: 1,
            },
            Entry {
                tile_id: 2,
                offset: 10,
                length: 10,
                run_length: 1,
            },
            Entry {
                tile_id: 9,
                offset: 40,
                length: 10,
                run_length: 1,
            },
        ])
        .unwrap();
        assert!(
            deserialize_entries(&valid).is_ok(),
            "the fixture must be valid"
        );

        // A claimed count of zero.
        assert!(matches!(
            deserialize_entries(&[0x00]),
            Err(PmTilesError::EmptyDirectory)
        ));

        // A count no honest directory could fill: four columns need at least
        // one byte per entry each.
        assert!(matches!(
            deserialize_entries(&[0x7F, 0x01, 0x01, 0x01, 0x01]),
            Err(PmTilesError::DirectoryTooManyEntries { claimed: 127, .. })
        ));

        // A repeated tile id, spelled as a zero delta on the second entry.
        let mut duplicate = valid.clone();
        duplicate[2] = 0x00;
        assert!(matches!(
            deserialize_entries(&duplicate),
            Err(PmTilesError::NonAscendingEntry { index: 1 })
        ));

        // A zero length, which the spec forbids twice and which would make the
        // contiguous shorthand ambiguous.
        let mut zero_length = valid.clone();
        zero_length[7] = 0x00;
        assert!(matches!(
            deserialize_entries(&zero_length),
            Err(PmTilesError::ZeroLengthEntry { .. })
        ));

        // Trailing bytes after the fourth column.
        let mut trailing = valid.clone();
        trailing.push(0x00);
        assert!(matches!(
            deserialize_entries(&trailing),
            Err(PmTilesError::TrailingDirectoryBytes { remaining: 1 })
        ));

        // A column that runs off the end.
        assert!(deserialize_entries(&valid[..valid.len() - 1]).is_err());

        // A tile id past the addressable range, built directly rather than by
        // corrupting a byte: the delta is the id itself on the first entry.
        let mut too_far = Vec::new();
        crate::pmtiles::varint::encode_uvarint(1, &mut too_far);
        crate::pmtiles::varint::encode_uvarint(MAX_TILE_ID + 1, &mut too_far);
        crate::pmtiles::varint::encode_uvarint(1, &mut too_far);
        crate::pmtiles::varint::encode_uvarint(10, &mut too_far);
        crate::pmtiles::varint::encode_uvarint(1, &mut too_far);
        assert!(matches!(
            deserialize_entries(&too_far),
            Err(PmTilesError::TileIdOutOfRange { .. })
        ));

        // A length that does not fit the u32 this crate models it with.
        let mut huge_length = Vec::new();
        crate::pmtiles::varint::encode_uvarint(1, &mut huge_length);
        crate::pmtiles::varint::encode_uvarint(7, &mut huge_length);
        crate::pmtiles::varint::encode_uvarint(1, &mut huge_length);
        crate::pmtiles::varint::encode_uvarint(u64::from(u32::MAX) + 1, &mut huge_length);
        crate::pmtiles::varint::encode_uvarint(1, &mut huge_length);
        assert!(matches!(
            deserialize_entries(&huge_length),
            Err(PmTilesError::EntryFieldTooLarge {
                field: "length",
                ..
            })
        ));
    }

    #[test]
    fn a_directory_this_crate_will_not_write() {
        // The writer side refuses the same shapes the reader does, so a
        // libviprs archive cannot contain one.
        assert!(matches!(
            serialize_entries(&[]),
            Err(PmTilesError::EmptyDirectory)
        ));
        assert!(matches!(
            serialize_entries(&[Entry {
                tile_id: 1,
                offset: 0,
                length: 0,
                run_length: 1
            }]),
            Err(PmTilesError::ZeroLengthEntry { index: 0 })
        ));
        assert!(matches!(
            serialize_entries(&[
                Entry {
                    tile_id: 5,
                    offset: 0,
                    length: 1,
                    run_length: 1
                },
                Entry {
                    tile_id: 5,
                    offset: 1,
                    length: 1,
                    run_length: 1
                },
            ]),
            Err(PmTilesError::NonAscendingEntry { index: 1 })
        ));
        assert!(matches!(
            serialize_entries(&[Entry {
                tile_id: MAX_TILE_ID + 1,
                offset: 0,
                length: 1,
                run_length: 1
            }]),
            Err(PmTilesError::TileIdOutOfRange { .. })
        ));
        // An offset one below the ceiling would encode as `offset + 1` and
        // wrap. The arithmetic is checked rather than trusted.
        assert!(matches!(
            serialize_entries(&[Entry {
                tile_id: 1,
                offset: u64::MAX,
                length: 1,
                run_length: 1
            }]),
            Err(PmTilesError::Overflow { .. })
        ));

        // The positive control: the valid neighbour of each of those writes.
        assert!(
            serialize_entries(&[
                Entry {
                    tile_id: 5,
                    offset: 0,
                    length: 1,
                    run_length: 1
                },
                Entry {
                    tile_id: 6,
                    offset: 1,
                    length: 1,
                    run_length: 1
                },
            ])
            .is_ok()
        );
        assert!(
            serialize_entries(&[Entry {
                tile_id: MAX_TILE_ID,
                offset: 0,
                length: 1,
                run_length: 1
            }])
            .is_ok()
        );
    }

    #[test]
    fn a_run_absorbs_the_next_tile_only_when_all_three_conditions_hold() {
        let mut entries = Vec::new();
        push_entry(&mut entries, 5, 1337, 42).unwrap();
        push_entry(&mut entries, 6, 1337, 42).unwrap(); // same blob, next id
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].run_length, 2);

        // A gap in the ids starts a new entry even with the same blob.
        push_entry(&mut entries, 9, 1337, 42).unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[1].run_length, 1);

        // A different offset starts a new entry even with consecutive ids.
        push_entry(&mut entries, 10, 4096, 42).unwrap();
        assert_eq!(entries.len(), 3);

        // So does a different length at the same offset, which is the case a
        // comparison on the offset alone would fold together.
        push_entry(&mut entries, 11, 4096, 11).unwrap();
        assert_eq!(entries.len(), 4);

        // A leaf pointer is never extended.
        let mut with_leaf = vec![Entry {
            tile_id: 5,
            offset: 0,
            length: 9,
            run_length: 0,
        }];
        push_entry(&mut with_leaf, 6, 0, 9).unwrap();
        assert_eq!(with_leaf.len(), 2);
        assert_eq!(with_leaf[0].run_length, 0);

        // Out-of-order and zero-length are refused rather than accepted into a
        // directory nothing can encode.
        let mut refused = Vec::new();
        push_entry(&mut refused, 10, 0, 5).unwrap();
        assert!(matches!(
            push_entry(&mut refused, 10, 5, 5),
            Err(PmTilesError::NonAscendingEntry { .. })
        ));
        assert!(matches!(
            push_entry(&mut refused, 11, 5, 0),
            Err(PmTilesError::ZeroLengthEntry { .. })
        ));
        // The positive control: the next valid id, pointing at the same blob,
        // still goes in and extends the run.
        push_entry(&mut refused, 11, 0, 5).unwrap();
        assert_eq!(refused.len(), 1);
        assert_eq!(refused[0].run_length, 2);

        // What came out is a directory that serialises.
        assert!(serialize_entries(&entries).is_ok());
    }

    #[test]
    fn landing_on_an_entry_is_not_the_same_as_being_inside_its_run() {
        let run = Entry {
            tile_id: 5,
            offset: 0,
            length: 9,
            run_length: 3,
        };
        assert!(!run.run_contains(4), "below the run");
        assert!(run.run_contains(5));
        assert!(run.run_contains(6));
        assert!(run.run_contains(7));
        // This is the one that matters. A lookup for tile 8 lands on this
        // entry because it is the largest with a tile id at or below 8, and a
        // reader without the run-end check returns tile 5's bytes for it.
        assert!(!run.run_contains(8), "past the end of the run");
        assert_eq!(run.last_tile_id(), Some(7));

        // A leaf pointer contains nothing; it is followed, not read.
        let leaf = Entry {
            tile_id: 5,
            offset: 0,
            length: 9,
            run_length: 0,
        };
        assert!(leaf.is_leaf());
        assert!(!leaf.run_contains(5));
        assert_eq!(leaf.last_tile_id(), None);

        // A hostile entry whose run runs off the end of the id space. The sum
        // wraps in a release build, and a wrapped end is below every id, so
        // the naive comparison answers "not contained" for everything.
        let hostile = Entry {
            tile_id: u64::MAX - 1,
            offset: 0,
            length: 9,
            run_length: u32::MAX,
        };
        assert!(hostile.run_contains(u64::MAX));
        assert!(hostile.run_contains(u64::MAX - 1));
        assert!(!hostile.run_contains(u64::MAX - 2));
        assert_eq!(hostile.last_tile_id(), Some(u64::MAX));
    }

    #[test]
    fn a_leaf_pointer_survives_the_round_trip() {
        // Round-tripping a run length of 0 is the whole leaf mechanism: it is
        // the only discriminant between a tile entry and a pointer, so a
        // codec that normalised it to 1 would turn every leaf into a tile.
        let entries = vec![
            Entry {
                tile_id: 0,
                offset: 0,
                length: 100,
                run_length: 5,
            },
            Entry {
                tile_id: 5,
                offset: 100,
                length: 250,
                run_length: 0,
            },
            Entry {
                tile_id: 21,
                offset: 350,
                length: 90,
                run_length: 0,
            },
        ];
        let back = deserialize_entries(&serialize_entries(&entries).unwrap()).unwrap();
        assert_eq!(back, entries);
        assert!(back[1].is_leaf());
        assert!(back[2].is_leaf());
    }

    #[test]
    fn a_large_archive_s_offsets_survive_past_four_gigabytes() {
        // The arithmetic that silently does not work is the 32-bit one, so
        // this puts every offset above 2^32 and asks for them back.
        let base: u64 = 5_000_000_000;
        let entries = vec![
            Entry {
                tile_id: 1,
                offset: base,
                length: 4096,
                run_length: 1,
            },
            Entry {
                tile_id: 2,
                offset: base + 4096,
                length: 8192,
                run_length: 1,
            },
            Entry {
                tile_id: 3,
                offset: 12_000_000_000,
                length: 1,
                run_length: 1,
            },
        ];
        let back = deserialize_entries(&serialize_entries(&entries).unwrap()).unwrap();
        assert_eq!(back, entries);
        assert_eq!(back[1].offset, 5_000_004_096);
        assert_eq!(back[2].offset, 12_000_000_000);
    }
}
