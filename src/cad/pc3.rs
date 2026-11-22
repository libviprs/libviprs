//! PC3 plot configuration files (issue #1634).
//!
//! A JSON PC3 is a first line `PIAFILEVERSION_3.0,json` ending in CRLF, then
//! a JSON document written by JsonCpp's styled writer: keys sorted bytewise,
//! one space per indent level, ` : ` between a key and its value, an object or
//! array value on the line after its key, LF line endings and no trailing
//! newline, reals at 17 significant digits with `.0` on a whole value.
//!
//! The reader keeps every scalar's text exactly as the file spells it, so a
//! file that is read and written comes back byte for byte, and a value that
//! was not edited keeps its spelling when another value is. Sections nobody
//! models (`custom`, `io`, `res_color_mem`, anything unknown) ride through
//! untouched. `serde_json` cannot do this without `arbitrary_precision`, and
//! the layout is JsonCpp's own either way, so the tokenizer is written here.
//!
//! `io.pathname` is carried and never opened. The device identity in `meta`
//! is never rewritten: the file keeps its own device.
//!
//! The older binary kinds (`PC3VER1`, `PIAFILEVERSION_1.0`) are recognised by
//! their prefix, stored verbatim and refused for editing.

use std::fmt;

/// Largest PC3 accepted, in bytes.
pub const MAX_PC3_BYTES: usize = 1024 * 1024;
/// Deepest JSON nesting accepted.
pub const MAX_PC3_DEPTH: usize = 32;

const HEADER_JSON: &str = "PIAFILEVERSION_3.0,json";

/// Why a PC3 was refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Pc3Error {
    /// The file is over [`MAX_PC3_BYTES`].
    TooLarge(usize),
    /// The JSON nests deeper than [`MAX_PC3_DEPTH`].
    TooDeep,
    /// Not a PC3 this reader knows.
    Unrecognised,
    /// The JSON does not parse: byte offset and what was wrong.
    Syntax(usize, &'static str),
    /// A binary PC3 is kept verbatim and cannot be edited.
    BinaryNotEditable(&'static str),
    /// The document lacks a section a typed accessor needs.
    Missing(&'static str),
    /// A paper value is not a finite number, or the printable area computed from the values
    /// overflows. A file can spell one (`1e999`), so an edit refuses it instead of writing it.
    NotFinite(&'static str),
}

impl fmt::Display for Pc3Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Pc3Error::TooLarge(n) => {
                write!(f, "PC3 is {n} bytes, over the {MAX_PC3_BYTES}-byte limit")
            }
            Pc3Error::TooDeep => write!(f, "PC3 nests deeper than {MAX_PC3_DEPTH} levels"),
            Pc3Error::Unrecognised => write!(f, "not a PC3 file"),
            Pc3Error::Syntax(at, what) => write!(f, "PC3 JSON, byte {at}: {what}"),
            Pc3Error::BinaryNotEditable(reason) => write!(f, "{reason}"),
            Pc3Error::Missing(what) => write!(f, "PC3 has no {what}"),
            Pc3Error::NotFinite(what) => write!(f, "PC3 {what} is not a finite number"),
        }
    }
}

impl std::error::Error for Pc3Error {}

/// A PC3 of either kind.
#[derive(Debug, Clone)]
pub enum Pc3File {
    /// The JSON kind, which can be read and edited.
    Json(Pc3),
    /// A binary kind, stored as it came.
    Binary(Pc3Binary),
}

/// A binary PC3 (`PC3VER1` or `PIAFILEVERSION_1.0`), held verbatim.
#[derive(Debug, Clone)]
pub struct Pc3Binary {
    /// The format's name.
    pub format: &'static str,
    /// The file as read.
    pub bytes: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq)]
enum Value {
    Null,
    Bool(bool),
    /// A number, spelled as the file spells it.
    Number(String),
    /// A string, between its quotes and still escaped.
    Str(String),
    Array(Vec<Value>),
    /// Members in file order; the writer sorts them.
    Object(Vec<(String, Value)>),
}

/// A JSON PC3.
#[derive(Debug, Clone)]
pub struct Pc3 {
    /// The first line as read, line ending included.
    header: String,
    root: Value,
    /// The document as read, so a value put back to what the file held gets
    /// the file's own spelling back.
    baseline: Value,
    /// Whatever followed the document (whitespace), as read.
    trailer: String,
}

/// The paper half of a PC3, in millimetres.
#[derive(Debug, Clone, PartialEq)]
pub struct Pc3Media {
    /// The paper's name, e.g. `ANSI_B_(11.00_x_17.00_Inches)`.
    pub paper_name: String,
    /// Media width and height.
    pub size_mm: [f64; 2],
    /// Printable rectangle: lower left x and y, upper right x and y.
    pub printable_mm: [f64; 4],
    /// Landscape mode of the paper size.
    pub landscape: bool,
    /// The device's default orientation (`dm_orientation`).
    pub orientation: i64,
}

/// Which device a PC3 names. Never rewritten.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Pc3Device {
    /// `localized_family_name`.
    pub family: String,
    /// `localized_model_name`.
    pub model: String,
}

impl Pc3File {
    /// Read a PC3 of either kind.
    pub fn parse(bytes: &[u8]) -> Result<Self, Pc3Error> {
        for format in ["PC3VER1", "PIAFILEVERSION_1.0"] {
            if bytes.starts_with(format.as_bytes()) {
                return Ok(Pc3File::Binary(Pc3Binary {
                    format,
                    bytes: bytes.to_vec(),
                }));
            }
        }
        Pc3::parse(bytes).map(Pc3File::Json)
    }

    /// The file's bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        match self {
            Pc3File::Json(pc3) => pc3.to_bytes(),
            Pc3File::Binary(binary) => binary.bytes.clone(),
        }
    }

    /// The editable document, or the reason there is none.
    pub fn editable(&mut self) -> Result<&mut Pc3, Pc3Error> {
        match self {
            Pc3File::Json(pc3) => Ok(pc3),
            Pc3File::Binary(_) => Err(Pc3Error::BinaryNotEditable(
                "this PC3 is in the older binary format, which is kept as it is and cannot be edited",
            )),
        }
    }
}

impl Pc3 {
    /// Read a JSON PC3.
    pub fn parse(bytes: &[u8]) -> Result<Self, Pc3Error> {
        if bytes.len() > MAX_PC3_BYTES {
            return Err(Pc3Error::TooLarge(bytes.len()));
        }
        if !bytes.starts_with(HEADER_JSON.as_bytes()) {
            return Err(Pc3Error::Unrecognised);
        }
        let text = std::str::from_utf8(bytes)
            .map_err(|e| Pc3Error::Syntax(e.valid_up_to(), "the document is not UTF-8"))?;
        let body_at = text.find('\n').map_or(text.len(), |i| i + 1);
        let mut reader = Reader { text, at: body_at };
        let root = reader.value(1)?;
        let value_end = reader.at;
        reader.skip_space();
        if reader.at != text.len() {
            return Err(Pc3Error::Syntax(reader.at, "text after the document"));
        }
        Ok(Pc3 {
            header: text[..body_at].to_string(),
            baseline: root.clone(),
            root,
            trailer: text[value_end..].to_string(),
        })
    }

    /// The file's bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = self.header.clone();
        write_value(&self.root, 0, &mut out);
        out.push_str(&self.trailer);
        out.into_bytes()
    }

    /// The raw text of a scalar at `path`, as the file spells it. Strings
    /// come back between no quotes and still escaped.
    pub fn scalar(&self, path: &[&str]) -> Option<&str> {
        match at(&self.root, path)? {
            Value::Number(text) | Value::Str(text) => Some(text),
            Value::Bool(true) => Some("true"),
            Value::Bool(false) => Some("false"),
            Value::Null => Some("null"),
            Value::Array(_) | Value::Object(_) => None,
        }
    }

    /// The paper half.
    pub fn media(&self) -> Result<Pc3Media, Pc3Error> {
        let real = |path: &[&str]| -> Result<f64, Pc3Error> {
            self.scalar(path)
                .and_then(|text| text.parse().ok())
                .ok_or(Pc3Error::Missing("media value"))
        };
        let desc = ["data", "media", "size", "media_description"];
        let bound = |key: &str| real(&[desc[0], desc[1], desc[2], desc[3], "media_bounds", key]);
        let printable = |key: &str| real(&["data", "media", key]);
        Ok(Pc3Media {
            paper_name: unescape(
                self.scalar(&["data", "media", "size", "name"])
                    .ok_or(Pc3Error::Missing("paper name"))?,
            ),
            size_mm: [bound("urx")?, bound("ury")?],
            printable_mm: [
                printable("actual_printable_bounds_llx")?,
                printable("actual_printable_bounds_lly")?,
                printable("actual_printable_bounds_urx")?,
                printable("actual_printable_bounds_ury")?,
            ],
            landscape: self.scalar(&["data", "media", "size", "landscape_mode"]) == Some("true"),
            orientation: self
                .scalar(&["data", "media", "dm_orientation"])
                .and_then(|text| text.parse().ok())
                .ok_or(Pc3Error::Missing("dm_orientation"))?,
        })
    }

    /// Change the paper half. A value that did not change keeps the text the
    /// file spells it with; a new real is written at 17 significant digits.
    pub fn set_media(&mut self, media: &Pc3Media) -> Result<(), Pc3Error> {
        let now = self.media()?;
        let p = media.printable_mm;
        if !media.size_mm.iter().all(|v| v.is_finite()) {
            return Err(Pc3Error::NotFinite("paper size"));
        }
        if !p.iter().all(|v| v.is_finite()) {
            return Err(Pc3Error::NotFinite("printable bound"));
        }
        if !((p[2] - p[0]) * (p[3] - p[1])).is_finite() {
            return Err(Pc3Error::NotFinite("printable area"));
        }
        let (m, md) = (
            ["data", "media"],
            ["data", "media", "size", "media_description"],
        );
        if now.paper_name != media.paper_name {
            self.put(
                &[m[0], m[1], "size", "name"],
                Value::Str(escape(&media.paper_name)),
            )?;
        }
        for (i, key) in ["urx", "ury"].into_iter().enumerate() {
            if now.size_mm[i] != media.size_mm[i] {
                let path = [md[0], md[1], md[2], md[3], "media_bounds", key];
                self.put(&path, real_value(media.size_mm[i]))?;
            }
        }
        let keys = ["llx", "lly", "urx", "ury"];
        let mut printable_changed = false;
        for (i, key) in keys.into_iter().enumerate() {
            if now.printable_mm[i] != media.printable_mm[i] {
                printable_changed = true;
                let value = real_value(media.printable_mm[i]);
                self.put(
                    &[m[0], m[1], &format!("actual_printable_bounds_{key}")],
                    value.clone(),
                )?;
                self.put(
                    &[
                        md[0],
                        md[1],
                        md[2],
                        md[3],
                        &format!("printable_bounds_{key}"),
                    ],
                    value,
                )?;
            }
        }
        if printable_changed {
            let area_path = [md[0], md[1], md[2], md[3], "printable_area"];
            let held = |key: &str| {
                at(&self.baseline, &["data", "media", key]).and_then(|v| match v {
                    Value::Number(text) => text.parse::<f64>().ok(),
                    _ => None,
                })
            };
            let back_to_file = ["llx", "lly", "urx", "ury"]
                .iter()
                .zip(p)
                .all(|(key, now)| held(&format!("actual_printable_bounds_{key}")) == Some(now));
            let area = match at(&self.baseline, &area_path) {
                Some(original) if back_to_file => original.clone(),
                _ => real_value((p[2] - p[0]) * (p[3] - p[1])),
            };
            self.put(&area_path, area)?;
        }
        if now.landscape != media.landscape {
            self.put(
                &[m[0], m[1], "size", "landscape_mode"],
                Value::Bool(media.landscape),
            )?;
        }
        if now.orientation != media.orientation {
            let value = Value::Number(media.orientation.to_string());
            self.put(&[m[0], m[1], "dm_orientation"], value)?;
        }
        Ok(())
    }

    /// The device the file names.
    pub fn device(&self) -> Option<Pc3Device> {
        Some(Pc3Device {
            family: unescape(self.scalar(&["data", "meta", "localized_family_name"])?),
            model: unescape(self.scalar(&["data", "meta", "localized_model_name"])?),
        })
    }

    /// Set the value at `path`. A value equal to the one the file held there
    /// goes back in as the file spelled it.
    fn put(&mut self, path: &[&str], value: Value) -> Result<(), Pc3Error> {
        let value = match at(&self.baseline, path) {
            Some(original) if same_value(original, &value) => original.clone(),
            _ => value,
        };
        let mut node = &mut self.root;
        for key in path {
            let Value::Object(members) = node else {
                return Err(Pc3Error::Missing("object on the path"));
            };
            node = members
                .iter_mut()
                .find(|(k, _)| k == key)
                .map(|(_, v)| v)
                .ok_or(Pc3Error::Missing("value on the path"))?;
        }
        *node = value;
        Ok(())
    }
}

/// Equal as values: numbers by what they parse to, not by spelling.
fn same_value(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Number(x), Value::Number(y)) => match (x.parse::<f64>(), y.parse::<f64>()) {
            (Ok(x), Ok(y)) => x == y,
            _ => x == y,
        },
        _ => a == b,
    }
}

fn at<'a>(mut node: &'a Value, path: &[&str]) -> Option<&'a Value> {
    for key in path {
        let Value::Object(members) = node else {
            return None;
        };
        node = &members.iter().find(|(k, _)| k == key)?.1;
    }
    Some(node)
}

// ── reading ──────────────────────────────────────────────────────────────

struct Reader<'a> {
    text: &'a str,
    at: usize,
}

impl Reader<'_> {
    fn skip_space(&mut self) {
        while matches!(
            self.text.as_bytes().get(self.at),
            Some(b' ' | b'\t' | b'\r' | b'\n')
        ) {
            self.at += 1;
        }
    }

    fn eat(&mut self, byte: u8, what: &'static str) -> Result<(), Pc3Error> {
        self.skip_space();
        if self.text.as_bytes().get(self.at) == Some(&byte) {
            self.at += 1;
            Ok(())
        } else {
            Err(Pc3Error::Syntax(self.at, what))
        }
    }

    fn value(&mut self, depth: usize) -> Result<Value, Pc3Error> {
        if depth > MAX_PC3_DEPTH {
            return Err(Pc3Error::TooDeep);
        }
        self.skip_space();
        let bytes = self.text.as_bytes();
        match bytes.get(self.at) {
            None => Err(Pc3Error::Syntax(self.at, "the document ends early")),
            Some(b'{') => {
                self.at += 1;
                let mut members = Vec::new();
                self.skip_space();
                if bytes.get(self.at) == Some(&b'}') {
                    self.at += 1;
                    return Ok(Value::Object(members));
                }
                loop {
                    self.skip_space();
                    let key = self.string()?;
                    self.eat(b':', "expected ':'")?;
                    members.push((key, self.value(depth + 1)?));
                    self.skip_space();
                    match bytes.get(self.at) {
                        Some(b',') => self.at += 1,
                        Some(b'}') => {
                            self.at += 1;
                            return Ok(Value::Object(members));
                        }
                        _ => return Err(Pc3Error::Syntax(self.at, "expected ',' or '}'")),
                    }
                }
            }
            Some(b'[') => {
                self.at += 1;
                let mut items = Vec::new();
                self.skip_space();
                if bytes.get(self.at) == Some(&b']') {
                    self.at += 1;
                    return Ok(Value::Array(items));
                }
                loop {
                    items.push(self.value(depth + 1)?);
                    self.skip_space();
                    match bytes.get(self.at) {
                        Some(b',') => self.at += 1,
                        Some(b']') => {
                            self.at += 1;
                            return Ok(Value::Array(items));
                        }
                        _ => return Err(Pc3Error::Syntax(self.at, "expected ',' or ']'")),
                    }
                }
            }
            Some(b'"') => Ok(Value::Str(self.string()?)),
            Some(_) => {
                let rest = &self.text[self.at..];
                for (word, value) in [
                    ("true", Value::Bool(true)),
                    ("false", Value::Bool(false)),
                    ("null", Value::Null),
                ] {
                    if rest.starts_with(word) {
                        self.at += word.len();
                        return Ok(value);
                    }
                }
                let len = rest
                    .bytes()
                    .take_while(|b| matches!(b, b'0'..=b'9' | b'-' | b'+' | b'.' | b'e' | b'E'))
                    .count();
                if len == 0 {
                    return Err(Pc3Error::Syntax(self.at, "not a JSON value"));
                }
                self.at += len;
                Ok(Value::Number(rest[..len].to_string()))
            }
        }
    }

    /// A string's text between its quotes, escapes left as written.
    fn string(&mut self) -> Result<String, Pc3Error> {
        let bytes = self.text.as_bytes();
        if bytes.get(self.at) != Some(&b'"') {
            return Err(Pc3Error::Syntax(self.at, "expected a string"));
        }
        let start = self.at + 1;
        let mut i = start;
        while let Some(&b) = bytes.get(i) {
            match b {
                b'"' => {
                    self.at = i + 1;
                    return Ok(self.text[start..i].to_string());
                }
                b'\\' => i += 2,
                _ => i += 1,
            }
        }
        Err(Pc3Error::Syntax(start, "unterminated string"))
    }
}

// ── writing ──────────────────────────────────────────────────────────────

/// A real the way JsonCpp writes it: `%.17g`, then `.0` on a whole value.
fn real_value(value: f64) -> Value {
    Value::Number(format_real(value))
}

fn format_real(value: f64) -> String {
    if value == 0.0 {
        return "0.0".to_string();
    }
    // 17 significant digits, exactly rounded: d.dddddddddddddddde±x
    let sci = format!("{value:.16e}");
    let (mantissa, exp) = sci.split_once('e').expect("exponent form");
    let exp: i32 = exp.parse().expect("exponent");
    let negative = mantissa.starts_with('-');
    let digits: String = mantissa.chars().filter(char::is_ascii_digit).collect();
    let mut out = String::new();
    if negative {
        out.push('-');
    }
    if (-5..17).contains(&exp) {
        // fixed notation, trailing zeros dropped
        if exp >= 0 {
            let int_len = exp as usize + 1;
            out.push_str(&digits[..int_len]);
            let frac = digits[int_len..].trim_end_matches('0');
            if !frac.is_empty() {
                out.push('.');
                out.push_str(frac);
            }
        } else {
            out.push_str("0.");
            out.push_str(&"0".repeat((-exp - 1) as usize));
            out.push_str(digits.trim_end_matches('0'));
        }
    } else {
        out.push_str(&digits[..1]);
        let frac = digits[1..].trim_end_matches('0');
        if !frac.is_empty() {
            out.push('.');
            out.push_str(frac);
        }
        out.push_str(&format!(
            "e{}{:02}",
            if exp < 0 { '-' } else { '+' },
            exp.abs()
        ));
    }
    if !out.contains(['.', 'e']) {
        out.push_str(".0");
    }
    out
}

/// JsonCpp keeps an array of scalars on one line while it fits in 74 columns.
fn array_is_multiline(items: &[Value]) -> bool {
    let flat: usize = items.iter().map(|v| scalar_text(v).len()).sum();
    items.iter().any(is_container) || 4 + items.len().saturating_sub(1) * 2 + flat > 74
}

/// Whether a value starts on its own line, after its key.
fn starts_on_new_line(value: &Value) -> bool {
    match value {
        Value::Object(members) => !members.is_empty(),
        Value::Array(items) => !items.is_empty() && array_is_multiline(items),
        _ => false,
    }
}

fn is_container(value: &Value) -> bool {
    matches!(value, Value::Object(m) if !m.is_empty())
        || matches!(value, Value::Array(a) if !a.is_empty())
}

fn scalar_text(value: &Value) -> String {
    match value {
        Value::Null => "null".into(),
        Value::Bool(b) => b.to_string(),
        Value::Number(n) => n.clone(),
        Value::Str(s) => format!("\"{s}\""),
        Value::Array(_) => "[]".into(),
        Value::Object(_) => "{}".into(),
    }
}

/// JsonCpp's styled layout. `indent` is the level of the line the value
/// starts on, and the caller has already written whatever precedes it.
fn write_value(value: &Value, indent: usize, out: &mut String) {
    match value {
        Value::Object(members) if !members.is_empty() => {
            out.push_str("{\n");
            let mut sorted: Vec<&(String, Value)> = members.iter().collect();
            sorted.sort_by(|a, b| a.0.as_bytes().cmp(b.0.as_bytes()));
            for (i, (key, child)) in sorted.iter().enumerate() {
                out.push_str(&" ".repeat(indent + 1));
                out.push_str(&format!("\"{key}\" : "));
                if starts_on_new_line(child) {
                    out.push('\n');
                    out.push_str(&" ".repeat(indent + 1));
                }
                write_value(child, indent + 1, out);
                if i + 1 < sorted.len() {
                    out.push(',');
                }
                out.push('\n');
            }
            out.push_str(&" ".repeat(indent));
            out.push('}');
        }
        Value::Array(items) if !items.is_empty() => {
            // JsonCpp keeps an array of scalars on one line while it fits
            if array_is_multiline(items) {
                out.push_str("[\n");
                for (i, child) in items.iter().enumerate() {
                    out.push_str(&" ".repeat(indent + 1));
                    write_value(child, indent + 1, out);
                    if i + 1 < items.len() {
                        out.push(',');
                    }
                    out.push('\n');
                }
                out.push_str(&" ".repeat(indent));
                out.push(']');
            } else {
                let flat: Vec<String> = items.iter().map(scalar_text).collect();
                out.push_str(&format!("[ {} ]", flat.join(", ")));
            }
        }
        other => out.push_str(&scalar_text(other)),
    }
}

/// Escape a string for the file: quote, backslash and control characters.
fn escape(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for c in text.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

/// The text of an escaped string; an escape it does not know stays as is.
fn unescape(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut chars = text.chars();
    while let Some(c) = chars.next() {
        if c != '\\' {
            out.push(c);
            continue;
        }
        match chars.next() {
            Some('n') => out.push('\n'),
            Some('r') => out.push('\r'),
            Some('t') => out.push('\t'),
            Some('b') => out.push('\u{8}'),
            Some('f') => out.push('\u{c}'),
            Some('u') => {
                let hex: String = chars.by_ref().take(4).collect();
                match u32::from_str_radix(&hex, 16).ok().and_then(char::from_u32) {
                    Some(c) => out.push(c),
                    None => {
                        out.push_str("\\u");
                        out.push_str(&hex);
                    }
                }
            }
            Some(c) => out.push(c),
            None => out.push('\\'),
        }
    }
    out
}
