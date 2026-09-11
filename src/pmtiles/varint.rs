//! Bounded LEB128 varints, the only number format inside a PMTiles directory.
//!
//! Every value in a directory, the entry count included, is a Protocol Buffers
//! base-128 varint: the top bit of each byte says whether another byte
//! follows, the low seven bits are payload, and the payload groups run
//! least-significant first. That is what the spec means by calling them
//! "little-endian variable-width integers"; it is about the order of the 7-bit
//! groups, not about byte order within a fixed-width field.
//!
//! A `u64` takes between one and ten bytes.
//!
//! # There is no zigzag here, and reaching for one is a real mistake
//!
//! The tile id column is delta-encoded, and delta encoding usually implies
//! signed deltas, which usually implies zigzag. Not here. PMTiles v3 uses no
//! zigzag anywhere: the word does not occur in the specification, and every
//! column holds a value that cannot be negative, because directory entries
//! ascend by construction. A writer that zigzags the tile id deltas produces
//! a file that decodes to `5, 21, 35` where it meant `5, 42, 69`, which is
//! wrong in a way that still looks like a plausible list of ascending ids.
//!
//! So this module has an unsigned pair and nothing else. If a signed column
//! ever appears, it needs its own transform and its own tests, not a helper
//! sitting here unused waiting to be misapplied.
//!
//! # Bounded means two separate refusals
//!
//! Both matter and they catch different inputs:
//!
//! * **an eleventh byte.** No `u64` needs one, so a value still continuing
//!   after ten bytes is malformed however small it looks.
//! * **a tenth byte other than `0x01`.** Ten bytes carry `10 * 7 = 70` payload
//!   bits, so the tenth byte's payload lands at bit positions 63 and above.
//!   Only a tenth byte of exactly `0x01` keeps the value inside 64 bits.
//!
//! Neither is in the specification. Both follow from every field the varints
//! encode being 64 bits wide, and from the rule that an archive is untrusted
//! input.
//!
//! # Non-canonical encodings are accepted on the way in
//!
//! `0x80 0x00` is an overlong spelling of zero. Nothing in the format forbids
//! it, no writer emits it, and refusing it would mean refusing an archive that
//! every other implementation reads, so [`decode_uvarint`] accepts it and
//! answers 0. [`encode_uvarint`] only ever emits the shortest form, so
//! decoding and re-encoding normalises. Worth knowing for a property test:
//! `decode(encode(v)) == v` holds for every `v`, while `encode(decode(b)) == b`
//! holds only for canonical `b`.

use crate::pmtiles::PmTilesError;

/// The most bytes a `u64` varint can occupy.
pub const MAX_UVARINT_LEN: usize = 10;

/// Append `value` to `out` in the shortest LEB128 form.
///
/// # Examples
///
/// ```
/// use libviprs::pmtiles::varint::encode_uvarint;
///
/// let mut out = Vec::new();
/// encode_uvarint(300, &mut out);
/// assert_eq!(out, vec![0xAC, 0x02]);
/// ```
pub fn encode_uvarint(mut value: u64, out: &mut Vec<u8>) {
    while value >= 0x80 {
        out.push((value as u8) | 0x80);
        value >>= 7;
    }
    out.push(value as u8);
}

/// How many bytes [`encode_uvarint`] would push for `value`.
///
/// Used to size a directory buffer before building it, and to check a claimed
/// entry count against the bytes actually available.
pub fn uvarint_len(value: u64) -> usize {
    let mut len = 1;
    let mut value = value >> 7;
    while value > 0 {
        len += 1;
        value >>= 7;
    }
    len
}

/// Decode one varint from the front of `bytes`.
///
/// Returns the value and how many bytes it consumed, so a caller sweeping a
/// column advances by the second element rather than guessing. `offset` is
/// only used to make the error message point at a position in the enclosing
/// buffer; pass the absolute position of `bytes` within it.
///
/// # Examples
///
/// ```
/// use libviprs::pmtiles::varint::decode_uvarint;
///
/// // Trailing bytes are left alone: the decoder stops at the first byte
/// // without a continuation bit and says how far it got.
/// let (value, used) = decode_uvarint(&[0xAC, 0x02, 0xFF], 0).unwrap();
/// assert_eq!((value, used), (300, 2));
///
/// // Ten bytes is the ceiling, and the tenth may only be 0x01.
/// assert!(decode_uvarint(&[0xFF; 11], 0).is_err());
/// ```
pub fn decode_uvarint(bytes: &[u8], offset: usize) -> Result<(u64, usize), PmTilesError> {
    let mut value: u64 = 0;
    let mut shift: u32 = 0;

    for (index, &byte) in bytes.iter().take(MAX_UVARINT_LEN).enumerate() {
        if byte < 0x80 {
            // The last byte of a ten-byte varint carries bits 63 and up, so
            // anything but a bare 1 there has overflowed a u64.
            if index == MAX_UVARINT_LEN - 1 && byte > 1 {
                return Err(PmTilesError::VarintOverflow { offset });
            }
            return Ok((value | (u64::from(byte) << shift), index + 1));
        }
        value |= u64::from(byte & 0x7F) << shift;
        shift += 7;
    }

    // Either the buffer ran out with a continuation bit still set, or ten
    // bytes went by and an eleventh was being asked for. They are different
    // failures and a caller debugging a corrupt archive wants to know which.
    if bytes.len() < MAX_UVARINT_LEN {
        Err(PmTilesError::TruncatedVarint { offset })
    } else {
        Err(PmTilesError::VarintOverflow { offset })
    }
}

/// A cursor over a buffer of varints.
///
/// A directory is four columns of `n` varints each, read in order, and every
/// read has to advance a position and bounds-check the rest. Doing that by
/// hand at each of the four columns is where an off-by-one lives, so the
/// cursor owns it.
pub(crate) struct VarintCursor<'a> {
    bytes: &'a [u8],
    position: usize,
}

impl<'a> VarintCursor<'a> {
    pub(crate) fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, position: 0 }
    }

    /// Read the next varint and advance past it.
    pub(crate) fn next_uvarint(&mut self) -> Result<u64, PmTilesError> {
        let (value, used) = decode_uvarint(&self.bytes[self.position..], self.position)?;
        self.position += used;
        Ok(value)
    }

    /// Bytes not yet consumed.
    pub(crate) fn remaining(&self) -> usize {
        self.bytes.len() - self.position
    }
}
