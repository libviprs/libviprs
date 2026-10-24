//! The measurements behind `docs/pvec-text-policy.md` (N5.5, #1084).
//!
//! `cargo run --release --example text_policy_measure -- <file.pdf|file.dwg>...`
//!
//! Per PDF it prints page-1 text-showing operator counts (the adapter's own counters, forms
//! expanded), the font-outcome buckets, and the size each option would add to the artifact,
//! in bytes. A `.dwg` is read through the acadrust provider when the `dwg-acadrust` feature is
//! on, and its `TEXT` and `MTEXT` entities are counted.
//!
//! The size model is an estimate and says so: it prices a text item from the page's own
//! top-level content stream (forms are not walked for bytes), so the byte columns are a lower
//! bound where text sits in forms.
//!
//! * omit: 0 bytes.
//! * preserve: per operator, the string bytes, a 48 byte text matrix (six `f64`), 8 bytes of
//!   size, and the font name length.
//! * outline: per glyph (one string byte is one glyph, an overestimate for CID fonts), the
//!   average Bezier control points of the bundled Vera face over printable ASCII, at 16 bytes
//!   a point, plus a 48 byte placement.
//! * paths: the artifact's existing path data priced the same way (16 bytes a point), so the
//!   deltas have a denominator.

use ab_glyph::{Font, FontRef, OutlineCurve};
use libviprs::pvec::pdf_vector::{PdfPathSegment, PdfVectorLimits, extract_page_paths};
use lopdf::{Document, Object};

fn vera_points_per_glyph() -> f64 {
    let font = FontRef::try_from_slice(include_bytes!("../fonts/Vera.ttf")).unwrap();
    let (mut pts, mut n) = (0usize, 0usize);
    for c in ' '..='~' {
        let Some(o) = font.outline(font.glyph_id(c)) else {
            continue;
        };
        n += 1;
        for curve in &o.curves {
            pts += match curve {
                OutlineCurve::Line(..) => 1,
                // A quadratic is promoted to a cubic on the way into canonical geometry.
                OutlineCurve::Quad(..) | OutlineCurve::Cubic(..) => 3,
            };
        }
    }
    pts as f64 / n as f64
}

fn top_level_text(path: &std::path::Path) -> (u64, u64, u64) {
    // (operators, string bytes, font-name bytes) from page 1's own streams.
    let Ok(doc) = Document::load(path) else {
        return (0, 0, 0);
    };
    let Some((_, id)) = doc.get_pages().into_iter().next() else {
        return (0, 0, 0);
    };
    let Ok(bytes) = doc.get_page_content(id) else {
        return (0, 0, 0);
    };
    let Ok(content) = lopdf::content::Content::decode(&bytes) else {
        return (0, 0, 0);
    };
    let (mut ops, mut sbytes, mut fname) = (0, 0, 0);
    let mut cur_font = 0u64;
    for op in &content.operations {
        match op.operator.as_str() {
            "Tf" => {
                if let Some(Object::Name(n)) = op.operands.first() {
                    cur_font = n.len() as u64;
                }
            }
            "Tj" | "TJ" | "'" | "\"" => {
                ops += 1;
                fname += cur_font;
                let mut add = |o: &Object| {
                    if let Object::String(s, _) = o {
                        sbytes += s.len() as u64;
                    }
                };
                for o in &op.operands {
                    match o {
                        Object::Array(a) => a.iter().for_each(&mut add),
                        other => add(other),
                    }
                }
            }
            _ => {}
        }
    }
    (ops, sbytes, fname)
}

fn pdf(path: &std::path::Path, ppg: f64) {
    let page = match extract_page_paths(path, 1, &PdfVectorLimits::default()) {
        Ok(p) => p,
        Err(e) => {
            println!("{}\tERROR\t{e}", path.display());
            return;
        }
    };
    let t = &page.text;
    let pts: u64 = page
        .entities
        .iter()
        .flat_map(|e| &e.segments)
        .map(|s| match s {
            PdfPathSegment::CurveTo { .. } => 3,
            PdfPathSegment::Close => 0,
            _ => 1,
        })
        .sum();
    let (top_ops, sbytes, fname) = top_level_text(path);
    let preserve = sbytes + top_ops * 56 + fname;
    let outline = (sbytes as f64 * (ppg * 16.0 + 48.0)) as u64;
    let buckets: Vec<String> = t
        .by_outcome
        .iter()
        .map(|(k, v)| format!("{}={v}", k.code()))
        .collect();
    println!(
        "{}\tpaths={}\tpath_bytes={}\tTj={}\tTJ={}\t'={}\t\"={}\ttotal={}\ttop_level_ops={top_ops}\tstring_bytes={sbytes}\tpreserve_bytes={preserve}\toutline_bytes={outline}\t{}",
        path.file_name().unwrap().to_string_lossy(),
        page.entities.len(),
        pts * 16,
        t.pdf.tj,
        t.pdf.tj_array,
        t.pdf.quote,
        t.pdf.double_quote,
        t.omitted(),
        buckets.join(",")
    );
}

#[cfg(feature = "dwg-acadrust")]
fn dwg(path: &std::path::Path) {
    use libviprs::cad::CadSource;
    use libviprs::pvec::dwg_acadrust::AcadrustProvider;
    use libviprs::pvec::text_policy::TextReport;
    use libviprs::pvec::{DwgProvider, EntityScope};
    let p = match AcadrustProvider::open(CadSource::Path(path)) {
        Ok(p) => p,
        Err(e) => {
            println!("{}\tERROR\t{e}", path.display());
            return;
        }
    };
    let (mut all, mut other) = (Vec::new(), 0usize);
    for l in p.layouts().unwrap() {
        match p
            .entities(&EntityScope::Layout(l.id))
            .and_then(|mut c| c.collect_all())
        {
            Ok(v) => all.extend(v),
            Err(_) => other += 1,
        }
    }
    let mut r = TextReport::default();
    r.count_dwg(&all);
    println!(
        "{}\tentities={}\tTEXT={}\tMTEXT={}\tlayouts_unreadable={other}",
        path.file_name().unwrap().to_string_lossy(),
        all.len(),
        r.dwg.text,
        r.dwg.mtext
    );
}

fn main() {
    let ppg = vera_points_per_glyph();
    println!("# vera_avg_cubic_points_per_glyph={ppg:.2}");
    for a in std::env::args().skip(1) {
        let p = std::path::PathBuf::from(&a);
        if a.to_ascii_lowercase().ends_with(".dwg") {
            #[cfg(feature = "dwg-acadrust")]
            dwg(&p);
            #[cfg(not(feature = "dwg-acadrust"))]
            println!("{a}\tskipped (needs --features dwg-acadrust)");
        } else {
            pdf(&p, ppg);
        }
    }
}
