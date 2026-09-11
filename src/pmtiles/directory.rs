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
        let run_end = last
            .tile_id
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
        let extends_the_run = !last.is_leaf()
            && tile_id == run_end
            && last.offset == offset
            && last.length == length;
        if extends_the_run {
            last.run_length = last.run_length.checked_add(1).ok_or(PmTilesError::Overflow {
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

    // Four columns of at most ten bytes each, plus the count. Over-reserving
    // by a factor of a few is cheaper than growing a directory buffer while
    // writing four passes over it.
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
            let shifted = entry
                .offset
                .checked_add(1)
                .ok_or(PmTilesError::Overflow { what: "an entry offset" })?;
            encode_uvarint(shifted, &mut out);
        }
        next_byte = entry
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
        last_id = last_id
            .checked_add(delta)
            .ok_or(PmTilesError::Overflow { what: "a tile id delta" })?;
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

    for index in 0..count {
        let run_length = cursor.next_uvarint()?;
        entries[index].run_length =
            u32::try_from(run_length).map_err(|_| PmTilesError::EntryFieldTooLarge {
                index,
                field: "run length",
                value: run_length,
            })?;
    }

    for index in 0..count {
        let length = cursor.next_uvarint()?;
        if length == 0 {
            return Err(PmTilesError::ZeroLengthEntry { index });
        }
        entries[index].length =
            u32::try_from(length).map_err(|_| PmTilesError::EntryFieldTooLarge {
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
