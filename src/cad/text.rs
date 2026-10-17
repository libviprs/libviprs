//! The string a person reads, out of the string a drawing stores.
//!
//! [`Text`](crate::cad::Text) carries decoded characters, but MTEXT stores
//! *formatting* in the same string: `\P` ends a paragraph, `{\fArial|b1;bold}`
//! switches font for a group, `\S1#4;` stacks a fraction. Those are content in
//! the sense that they are not transport, and a viewer that drew them verbatim
//! would put `\H1x;` on the sheet. [`display_text`] reads them.
//!
//! It keeps what changes *which characters are shown*, and says what it let go:
//!
//! * `\P` and `\X` become a newline, `\~` a space, `\\`, `\{` and `\}` the
//!   character they escape, and `{` and `}` grouping vanish;
//! * `\S` stacks become `n/d` (for `/` and `#`) or `n d` (for `^`);
//! * a font change (`\f` or `\F`) is recorded in [`DisplayText::fonts`], because
//!   the output carries one font and a drawing that asked for another has been
//!   substituted, which the caller reports;
//! * every other code is dropped and counted in
//!   [`DisplayText::ignored`], by its letter, so a caller can say that heights
//!   or widths were not honoured instead of pretending they were.
//!
//! Anything this does not recognise is kept as the character after the
//! backslash and counted as ignored, which is how AutoCAD treats an unknown
//! escape. Nothing here panics on any input and the output is never longer than
//! the input.

use std::collections::BTreeMap;

/// A string as it is displayed, and what was set aside to get there.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct DisplayText {
    /// The characters to draw. Lines are separated by `\n`.
    pub text: String,
    /// Font names the text asked for, in first-use order, without duplicates.
    pub fonts: Vec<String>,
    /// Formatting codes that were dropped, by their letter, with a count.
    ///
    /// Toggles (`L`, `l`, `O`, `o`, `K`, `k`) and colour and alignment codes
    /// (`C`, `c`, `A`) are in here too: the output has no underline or colour,
    /// and saying so is the point.
    pub ignored: BTreeMap<char, u64>,
}

impl DisplayText {
    /// Whether anything is left to draw.
    #[must_use]
    pub fn is_blank(&self) -> bool {
        self.text.trim().is_empty()
    }
}

/// Reads MTEXT formatting out of `content`.
///
/// ```
/// use libviprs::cad::display_text;
///
/// let d = display_text(r"4'-0{\H1x;\S1#4;}""");
/// assert_eq!(d.text, "4'-0 1/4\"".replace(' ', ""));
/// assert_eq!(d.ignored.get(&'H'), Some(&1));
///
/// let d = display_text(r"Line one\PLine {\fArial|b1;two}");
/// assert_eq!(d.text, "Line one\nLine two");
/// assert_eq!(d.fonts, ["Arial"]);
/// ```
#[must_use]
pub fn display_text(content: &str) -> DisplayText {
    let mut out = DisplayText::default();
    let mut chars = content.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '{' | '}' => {}
            '\\' => {
                let Some(code) = chars.next() else { break };
                match code {
                    '\\' | '{' | '}' => out.text.push(code),
                    'P' | 'X' => out.text.push('\n'),
                    '~' => out.text.push(' '),
                    'L' | 'l' | 'O' | 'o' | 'K' | 'k' => bump(&mut out, code),
                    'f' | 'F' => {
                        let arg = until_semicolon(&mut chars);
                        // `\fArial|b0|i0|c0|p34;` : the name is the first field.
                        let name = arg.split('|').next().unwrap_or("").trim();
                        if !name.is_empty() && !out.fonts.iter().any(|f| f == name) {
                            out.fonts.push(name.to_owned());
                        }
                    }
                    'S' => {
                        let arg = until_semicolon(&mut chars);
                        out.text.push_str(&stack(&arg));
                    }
                    'U' if chars.peek() == Some(&'+') => {
                        // `\U+XXXX` reaches here only if nothing upstream
                        // decoded it. Decode it rather than draw the escape.
                        let mut look = chars.clone();
                        look.next();
                        let hex: String = look.by_ref().take(4).collect();
                        match (hex.len() == 4)
                            .then(|| u32::from_str_radix(&hex, 16).ok())
                            .flatten()
                            .and_then(char::from_u32)
                        {
                            Some(ch) => {
                                out.text.push(ch);
                                chars = look;
                            }
                            None => bump(&mut out, 'U'),
                        }
                    }
                    // Codes with an argument that ends at a semicolon.
                    'C' | 'c' | 'H' | 'T' | 'Q' | 'W' | 'A' | 'p' => {
                        let _ = until_semicolon(&mut chars);
                        bump(&mut out, code);
                    }
                    other => {
                        out.text.push(other);
                        bump(&mut out, other);
                    }
                }
            }
            other => out.text.push(other),
        }
    }
    out
}

fn bump(out: &mut DisplayText, code: char) {
    *out.ignored.entry(code).or_default() += 1;
}

/// Takes characters up to and including the next `;`, returning those before
/// it. A code with no terminator takes the rest of the string, which is as
/// much as can be said of one.
fn until_semicolon(chars: &mut std::iter::Peekable<std::str::Chars<'_>>) -> String {
    let mut s = String::new();
    for c in chars.by_ref() {
        if c == ';' {
            break;
        }
        s.push(c);
    }
    s
}

/// `1#4`, `1/4` and `1^2` as text.
fn stack(arg: &str) -> String {
    if let Some((n, d)) = arg.split_once('^') {
        return format!("{n} {d}");
    }
    if let Some((n, d)) = arg.split_once('#').or_else(|| arg.split_once('/')) {
        return format!("{n}/{d}");
    }
    arg.to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_text_passes_through() {
        let d = display_text("Hello this is a single line text");
        assert_eq!(d.text, "Hello this is a single line text");
        assert!(d.fonts.is_empty() && d.ignored.is_empty());
    }

    #[test]
    fn paragraphs_and_escapes() {
        assert_eq!(display_text(r"a\Pb\Xc").text, "a\nb\nc");
        assert_eq!(display_text(r"50\~mm").text, "50 mm");
        assert_eq!(display_text(r"\\ \{ \}").text, r"\ { }");
        assert_eq!(display_text("{grouped}").text, "grouped");
    }

    #[test]
    fn the_real_drawing_cases() {
        // Both strings are from sample_AC1018.dwg.
        let d = display_text(r#"4'-0{\H1x;\S1#4;}""#);
        assert_eq!(d.text, "4'-01/4\"");
        assert_eq!(d.ignored.get(&'H'), Some(&1));
        let d = display_text(r"\A1;7.0711");
        assert_eq!(d.text, "7.0711");
        assert_eq!(d.ignored.get(&'A'), Some(&1));
    }

    #[test]
    fn stacks_in_each_form() {
        assert_eq!(display_text(r"\S1/2;").text, "1/2");
        assert_eq!(display_text(r"\S1#2;").text, "1/2");
        assert_eq!(display_text(r"\S+0.1^-0.2;").text, "+0.1 -0.2");
        assert_eq!(display_text(r"\Sabc;").text, "abc");
    }

    #[test]
    fn fonts_are_named_once_and_not_drawn() {
        let d = display_text(
            r"{\fArial|b1|i0|c0|p34;A}{\fTimes New Roman;B}{\fArial|b0;C}\Fromans.shx;D",
        );
        assert_eq!(d.text, "ABCD");
        assert_eq!(d.fonts, ["Arial", "Times New Roman", "romans.shx"]);
    }

    #[test]
    fn dropped_codes_are_counted_by_letter() {
        let d = display_text(r"\H2x;a\W0.8;b\T1.5;c\Q15;d\C1;e\LU\lx\O\o\K\k");
        assert_eq!(d.text, "abcdeUx");
        for (k, n) in [
            ('H', 1),
            ('W', 1),
            ('T', 1),
            ('Q', 1),
            ('C', 1),
            ('L', 1),
            ('l', 1),
            ('O', 1),
            ('o', 1),
            ('K', 1),
            ('k', 1),
        ] {
            assert_eq!(d.ignored.get(&k), Some(&n), "{k}");
        }
    }

    #[test]
    fn an_unknown_escape_keeps_its_character_and_is_counted() {
        let d = display_text(r"a\zb");
        assert_eq!(d.text, "azb");
        assert_eq!(d.ignored.get(&'z'), Some(&1));
    }

    #[test]
    fn a_leftover_unicode_escape_is_decoded() {
        assert_eq!(display_text(r"90\U+00B0").text, "90\u{b0}");
        // Not four hex digits: counted, not guessed.
        let d = display_text(r"\U+12");
        assert_eq!(d.ignored.get(&'U'), Some(&1));
    }

    #[test]
    fn hostile_input_neither_panics_nor_grows() {
        for s in [
            r"\",
            r"\f",
            r"\S",
            r"{{{{",
            r"\f|||;;;",
            "\\\\\\\\\\",
            r"\U+",
            r"\U+D800",
        ] {
            let d = display_text(s);
            assert!(d.text.len() <= s.len() + 8, "{s:?} -> {:?}", d.text);
        }
        let long = "\\H1;".repeat(100_000);
        assert!(display_text(&long).text.is_empty());
        let many = "{".repeat(1_000_000);
        assert!(display_text(&many).text.is_empty());
    }

    #[test]
    fn blank_text_is_noticed() {
        assert!(display_text(r"\A1;").is_blank());
        assert!(display_text("  \\P ").is_blank());
        assert!(!display_text("x").is_blank());
    }
}
