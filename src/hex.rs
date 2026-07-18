//! Internal lowercase-hex encoding helper.
//!
//! Consolidates the byte-slice → lowercase hex string encoder that used to be
//! copied into `dedupe.rs`, `manifest.rs`, and `sink.rs` (issue #302). Kept as
//! a hand-rolled encoder — deliberately no external `hex` crate — so on-disk
//! manifests and checksums stay byte-identical to the pre-consolidation output
//! (same lowercase alphabet, fixed two chars per byte, empty input → "").

/// Encode a byte slice as a lowercase hex string: two characters per byte, no
/// separators, using the alphabet `0123456789abcdef`. Empty input yields an
/// empty string.
pub(crate) fn hex_lower(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut s = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        s.push(HEX[(b >> 4) as usize] as char);
        s.push(HEX[(b & 0x0f) as usize] as char);
    }
    s
}

#[cfg(test)]
mod tests {
    use super::hex_lower;

    #[test]
    fn known_vector_deadbeef() {
        assert_eq!(hex_lower(&[0xde, 0xad, 0xbe, 0xef]), "deadbeef");
    }

    #[test]
    fn empty_input_is_empty_string() {
        assert_eq!(hex_lower(&[]), "");
    }

    #[test]
    fn single_byte_is_zero_padded_to_width_two() {
        assert_eq!(hex_lower(&[0x00]), "00");
        assert_eq!(hex_lower(&[0x05]), "05");
        assert_eq!(hex_lower(&[0x0f]), "0f");
        assert_eq!(hex_lower(&[0xff]), "ff");
    }

    #[test]
    fn round_trips_every_byte_value() {
        // The output must be lowercase, exactly two chars wide, and parse back
        // to the exact input byte for every possible value — i.e. the encoding
        // is losslessly invertible.
        for b in 0u8..=u8::MAX {
            let s = hex_lower(&[b]);
            assert_eq!(s.len(), 2, "byte {b:#04x} did not encode to two chars");
            assert!(
                s.chars()
                    .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()),
                "byte {b:#04x} encoded to non-lowercase-hex {s:?}",
            );
            assert_eq!(u8::from_str_radix(&s, 16).unwrap(), b);
        }
    }

    #[test]
    fn multi_byte_round_trip() {
        let input: Vec<u8> = (0u8..=63).collect();
        let s = hex_lower(&input);
        assert_eq!(s.len(), input.len() * 2);
        let decoded: Vec<u8> = (0..s.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
            .collect();
        assert_eq!(decoded, input);
    }
}
