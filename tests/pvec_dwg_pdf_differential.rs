//! Issue #1405 (N5.7b, slice of #1086 and #1302): the DWG to PDF differential.
//!
//! This measures a loss profile on purpose. The DWG is the reference, the PDF is what a real exporter
//! (LibreDWG `dwg2SVG`, then `rsvg-convert`, committed as static files) made of it, and the suite
//! never asserts that the two are equal: it asserts that the measured difference is exactly the
//! recorded expected loss in `expected_loss.toml`, with a written reason per entry. A fixture with no
//! such file fails, an unlisted difference fails, and so does an entry that no longer matches.
//!
//! The pairs are committed (`tests/fixtures/pvec/pdf_loss/<id>/source.dwg` and `exported.pdf`, with
//! `exporter.txt` naming the exporter, versions and options), so the gate needs no Docker. They are
//! `REAL_WORLD_DIFFERENTIAL` at best and never `KNOWN_TRUTH_EXACT`. Axes: primitive counts per class,
//! the N3.5 buckets of each side, and the extent aspect with the page origin removed and `/Rotate`
//! composed (`PdfVectorPage` does both). The rendered-image axis needs pdfium and is #1406.
//!
//! Needs `--features "dwg-acadrust pvec-normalized"`.
#![cfg(all(feature = "dwg-acadrust", feature = "pvec-normalized"))]

use std::collections::BTreeMap;
use std::path::PathBuf;

use libviprs::cad::CadSource;
use libviprs::pvec::bounds::{BoundsMethod, curve_bounds};
use libviprs::pvec::dwg_acadrust::AcadrustProvider;
use libviprs::pvec::dwg_basic::{self, BasicKind, SegmentKind};
use libviprs::pvec::flatten::{BlockResolver, FlattenBounds, LineageStore};
use libviprs::pvec::pdf_bezier::normalize_page;
use libviprs::pvec::pdf_clip::{AnalyticBounds, ExactBoundsSource};
use libviprs::pvec::provenance::NoProvenance;
use libviprs::pvec::{
    DwgProvider, EntityScope, LayoutKind, PdfVectorLimits, extract_page_paths_from_bytes,
};
use pvec_geometry::Curve;

const FIXTURES: [&str; 6] = [
    "line-001",
    "circle-001",
    "bulge-arc-001",
    "ellipse-001",
    "polyline-001",
    "polyline-001-rot90",
];

fn dir(id: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/pvec/pdf_loss")
        .join(id)
}

/// Measured pairs, metric to (DWG side, PDF side), both already formatted.
type Measured = BTreeMap<String, (String, String)>;

fn fmt(v: f64) -> String {
    format!("{v:.3}")
}

/// The DWG of a pair through acadrust, flattened. The pairs are flat drawings.
fn flatten_dwg(id: &str) -> libviprs::pvec::flatten::Flattened {
    let provider = AcadrustProvider::open(CadSource::Path(&dir(id).join("source.dwg"))).unwrap();
    let layout = provider
        .layouts()
        .unwrap()
        .into_iter()
        .find(|l| l.kind == LayoutKind::Model)
        .unwrap();
    BlockResolver::new(&provider, FlattenBounds::default())
        .flatten(
            &EntityScope::Layout(layout.id),
            id,
            &layout.name,
            &mut NoProvenance,
            &mut LineageStore::new(),
        )
        .unwrap()
}

fn measure(id: &str) -> Measured {
    let mut m = Measured::new();
    let mut put = |k: &str, d: String, p: String| {
        m.insert(k.to_string(), (d, p));
    };

    // DWG side: acadrust, flattened, each basic entity through PVec's exact bounds.
    let flat = flatten_dwg(id);
    let (mut d_line, mut d_curve) = (0u64, 0u64);
    let mut d_buckets: BTreeMap<&str, u64> = BTreeMap::new();
    let mut d_box: Option<[f64; 4]> = None;
    #[cfg(feature = "pdfium")]
    let mut d_curves: Vec<Curve> = Vec::new();
    for leaf in &flat.entities {
        assert_eq!(
            leaf.matrix(),
            [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            "{id}: the pairs are flat drawings, so source space is block space"
        );
        match dwg_basic::extract(&leaf.entity) {
            Ok(b) => {
                *d_buckets.entry(b.classification.class().tag()).or_insert(0) += 1;
                match b.kind {
                    BasicKind::Line => d_line += 1,
                    BasicKind::Polyline => {
                        for s in &b.segments {
                            if matches!(s, SegmentKind::Straight) {
                                d_line += 1;
                            } else {
                                d_curve += 1;
                            }
                        }
                    }
                    _ => d_curve += 1,
                }
                #[cfg(feature = "pdfium")]
                d_curves.push(b.curve.clone());
                let (bb, method) = curve_bounds(&b.curve).unwrap();
                assert_eq!(method, BoundsMethod::Exact, "{id}");
                let (lo, hi) = (bb.min(), bb.max());
                d_box = Some(match d_box {
                    None => [lo.x, lo.y, hi.x, hi.y],
                    Some(c) => [
                        c[0].min(lo.x),
                        c[1].min(lo.y),
                        c[2].max(hi.x),
                        c[3].max(hi.y),
                    ],
                });
            }
            Err(_) => *d_buckets.entry("Unsupported").or_insert(0) += 1,
        }
    }
    let d_box = d_box.expect("a drawn entity");

    // PDF side.
    let bytes = std::fs::read(dir(id).join("exported.pdf")).unwrap();
    let page = extract_page_paths_from_bytes(&bytes, 1, &PdfVectorLimits::default()).unwrap();
    let (mut p_line, mut p_curve) = (0u64, 0u64);
    let mut p_buckets: BTreeMap<&str, u64> = BTreeMap::new();
    for path in normalize_page(&page) {
        for e in &path.entities {
            *p_buckets.entry(e.classification.class().tag()).or_insert(0) += 1;
            match e.geometry.geometry() {
                Some(Curve::Bezier(_)) => p_curve += 1,
                Some(_) => p_line += 1,
                None => {}
            }
        }
    }
    let mut p_box: Option<[f64; 4]> = None;
    let mut outside = 0u64;
    for e in &page.entities {
        if let Some(b) = AnalyticBounds.exact_bounds(e) {
            p_box = Some(match p_box {
                None => [b.min[0], b.min[1], b.max[0], b.max[1]],
                Some(c) => [
                    c[0].min(b.min[0]),
                    c[1].min(b.min[1]),
                    c[2].max(b.max[0]),
                    c[3].max(b.max[1]),
                ],
            });
            if b.min[0] < 0.0 || b.min[1] < 0.0 || b.max[0] > page.width || b.max[1] > page.height {
                outside += 1;
            }
        }
    }
    let p_box = p_box.expect("a painted path");

    put("count.line", d_line.to_string(), p_line.to_string());
    put("count.curve", d_curve.to_string(), p_curve.to_string());
    for tag in ["SupportedExact", "SupportedEquivalent", "Unsupported"] {
        put(
            &format!("bucket.{tag}"),
            d_buckets.get(tag).copied().unwrap_or(0).to_string(),
            p_buckets.get(tag).copied().unwrap_or(0).to_string(),
        );
    }
    // Extents with the PDF's origin removed and /Rotate composed, both as width over height.
    let aspect = |b: [f64; 4]| fmt((b[2] - b[0]) / (b[3] - b[1]));
    put("extent.aspect", aspect(d_box), aspect(p_box));
    put("extent.outside_page", "0".into(), outside.to_string());
    #[cfg(feature = "pdfium")]
    {
        let r = raster_axis::measure(
            &dir(id).join("exported.pdf"),
            &d_curves,
            page.rotation_degrees,
        );
        for (k, d, p) in r {
            m.insert(k, (d, p));
        }
    }
    m
}

/// The rendered-image axis (issue #1406): pdfium renders the exported PDF, and the DWG's own curves are
/// drawn by a small point-sampling rasteriser into the same pixel frame. libviprs has no rasteriser for
/// DWG-derived PVec, so this one lives in the test and draws one-pixel lines only.
///
/// The frame is the PDF's ink box (the pixels pdfium painted dark), and the DWG's extent is mapped onto
/// it with the page's `/Rotate` composed, so the two are compared shape against shape: a stretch the
/// exporter applied shows in `raster.ink_aspect`, and what is painted where the other side has nothing
/// shows in the two `outside` fractions. Nothing here claims the sides are equal.
#[cfg(feature = "pdfium")]
mod raster_axis {
    use std::path::Path;

    use pvec_geometry::{Curve, Curve2d};

    /// DPI of the pdfium render, recorded in every `expected_loss.toml` under `[raster]`.
    pub const DPI: u32 = 96;
    /// A pixel is ink when its darkest colour channel is below this.
    pub const INK_BELOW: u8 = 128;
    /// A pixel counts as covered by the other side when ink lies within this many pixels.
    pub const TOLERANCE_PX: i64 = 3;
    const SAMPLES: usize = 2048;

    struct Ink {
        w: usize,
        h: usize,
        on: Vec<bool>,
    }

    fn pdf_ink(pdf: &Path) -> Ink {
        let r = libviprs::pdf::render_page_pdfium(pdf, 1, DPI).unwrap();
        let (w, h) = (r.width() as usize, r.height() as usize);
        let ch = r.format().channels();
        let colour = ch.min(3);
        let dark = |x: usize, y: usize| {
            let px = &r.data()[(y * w + x) * ch..][..ch];
            px[..colour].iter().min().copied().unwrap_or(255) < INK_BELOW
        };
        let (mut x0, mut y0, mut x1, mut y1) = (w, h, 0, 0);
        for y in 0..h {
            for x in 0..w {
                if dark(x, y) {
                    (x0, y0, x1, y1) = (x0.min(x), y0.min(y), x1.max(x), y1.max(y));
                }
            }
        }
        assert!(x0 <= x1, "{}: pdfium painted nothing", pdf.display());
        let (cw, chh) = (x1 - x0 + 1, y1 - y0 + 1);
        let mut on = vec![false; cw * chh];
        for y in 0..chh {
            for x in 0..cw {
                on[y * cw + x] = dark(x0 + x, y0 + y);
            }
        }
        Ink { w: cw, h: chh, on }
    }

    fn dwg_ink(curves: &[Curve], rotation: i64, w: usize, h: usize) -> Ink {
        let polylines: Vec<Vec<(f64, f64)>> = curves
            .iter()
            .map(|c| {
                let d = c.domain();
                (0..=SAMPLES)
                    .map(|i| {
                        let t = d.lo() + (d.hi() - d.lo()) * i as f64 / SAMPLES as f64;
                        let p = c.eval(t).unwrap();
                        (p.x, p.y)
                    })
                    .collect()
            })
            .collect();
        let all = polylines.iter().flatten();
        let lo = all
            .clone()
            .fold((f64::MAX, f64::MAX), |a, p| (a.0.min(p.0), a.1.min(p.1)));
        let hi = all.fold((f64::MIN, f64::MIN), |a, p| (a.0.max(p.0), a.1.max(p.1)));
        let unit = |v: f64, lo: f64, hi: f64| if hi > lo { (v - lo) / (hi - lo) } else { 0.5 };
        let to_px = |p: &(f64, f64)| {
            let (u, v) = (unit(p.0, lo.0, hi.0), unit(p.1, lo.1, hi.1));
            // /Rotate is clockwise, y up here.
            let (u, v) = match rotation.rem_euclid(360) {
                90 => (v, 1.0 - u),
                180 => (1.0 - u, 1.0 - v),
                270 => (1.0 - v, u),
                _ => (u, v),
            };
            (u * (w - 1) as f64, (1.0 - v) * (h - 1) as f64)
        };
        let mut on = vec![false; w * h];
        for line in &polylines {
            for pair in line.windows(2) {
                let (a, b) = (to_px(&pair[0]), to_px(&pair[1]));
                let steps = ((b.0 - a.0).abs().max((b.1 - a.1).abs()) * 2.0)
                    .ceil()
                    .max(1.0) as usize;
                for k in 0..=steps {
                    let f = k as f64 / steps as f64;
                    let (x, y) = (a.0 + (b.0 - a.0) * f, a.1 + (b.1 - a.1) * f);
                    on[(y.round() as usize).min(h - 1) * w + (x.round() as usize).min(w - 1)] =
                        true;
                }
            }
        }
        Ink { w, h, on }
    }

    /// The fraction of `a`'s ink with no ink of `b` within [`TOLERANCE_PX`].
    fn outside(a: &Ink, b: &Ink) -> f64 {
        let (mut total, mut far) = (0u64, 0u64);
        for y in 0..a.h {
            for x in 0..a.w {
                if !a.on[y * a.w + x] {
                    continue;
                }
                total += 1;
                let near = (-TOLERANCE_PX..=TOLERANCE_PX).any(|dy| {
                    (-TOLERANCE_PX..=TOLERANCE_PX).any(|dx| {
                        let (nx, ny) = (x as i64 + dx, y as i64 + dy);
                        nx >= 0
                            && ny >= 0
                            && (nx as usize) < b.w
                            && (ny as usize) < b.h
                            && b.on[ny as usize * b.w + nx as usize]
                    })
                });
                far += u64::from(!near);
            }
        }
        far as f64 / total.max(1) as f64
    }

    /// `(metric, dwg side, pdf side)` for the raster axis of one pair.
    pub fn measure(pdf: &Path, curves: &[Curve], rotation: i64) -> Vec<(String, (String, String))> {
        let pdf_side = pdf_ink(pdf);
        let dwg_side = dwg_ink(curves, rotation, pdf_side.w, pdf_side.h);
        let f2 = |v: f64| format!("{v:.2}");
        vec![
            (
                "raster.ink_aspect".into(),
                // The DWG's own sampled extent in the displayed orientation, against what pdfium painted.
                (
                    f2(dwg_aspect(curves, rotation)),
                    f2(pdf_side.w as f64 / pdf_side.h as f64),
                ),
            ),
            (
                "raster.ink_outside_other".into(),
                (
                    f2(outside(&dwg_side, &pdf_side)),
                    f2(outside(&pdf_side, &dwg_side)),
                ),
            ),
        ]
    }

    fn dwg_aspect(curves: &[Curve], rotation: i64) -> f64 {
        let (mut lo, mut hi) = ((f64::MAX, f64::MAX), (f64::MIN, f64::MIN));
        for c in curves {
            let d = c.domain();
            for i in 0..=SAMPLES {
                let p = c
                    .eval(d.lo() + (d.hi() - d.lo()) * i as f64 / SAMPLES as f64)
                    .unwrap();
                (lo, hi) = (
                    (lo.0.min(p.x), lo.1.min(p.y)),
                    (hi.0.max(p.x), hi.1.max(p.y)),
                );
            }
        }
        let (w, h) = (hi.0 - lo.0, hi.1 - lo.1);
        // Pixel extent of a drawn extent is its length plus the one-pixel line itself, ignored here.
        if rotation.rem_euclid(180) == 90 {
            h / w
        } else {
            w / h
        }
    }
}

#[derive(Debug)]
struct Loss {
    evidence: String,
    entries: Vec<(String, String, String, String)>,
}

fn load_loss(id: &str) -> Loss {
    let text = std::fs::read_to_string(dir(id).join("expected_loss.toml"))
        .unwrap_or_else(|e| panic!("{id}: expected_loss.toml is required: {e}"));
    let doc: toml::Table = text.parse().unwrap();
    let evidence = doc["evidence"].as_str().unwrap().to_string();
    let entries = doc
        .get("loss")
        .and_then(|l| l.as_array())
        .map(|a| {
            a.iter()
                .map(|t| {
                    let g = |k: &str| {
                        t[k].as_str()
                            .unwrap_or_else(|| panic!("{id}: {k}"))
                            .to_string()
                    };
                    (g("metric"), g("dwg"), g("pdf"), g("reason"))
                })
                .collect()
        })
        .unwrap_or_default();
    // Without pdfium the raster axis is not measured, so its entries are not checked either.
    let entries = entries
        .into_iter()
        .filter(|e: &(String, String, String, String)| {
            cfg!(feature = "pdfium") || !e.0.starts_with("raster.")
        })
        .collect();
    Loss { evidence, entries }
}

/// The failures of `measured` against `loss`: unlisted differences, wrong values, stale entries and
/// entries with no written reason.
fn check(measured: &Measured, loss: &Loss) -> Vec<String> {
    let mut out = Vec::new();
    for (k, (d, p)) in measured {
        let listed = loss.entries.iter().find(|e| &e.0 == k);
        match (d == p, listed) {
            (true, None) => {}
            (true, Some(_)) => out.push(format!("stale entry {k}: the sides now agree ({d})")),
            (false, None) => out.push(format!("unlisted difference {k}: dwg {d}, pdf {p}")),
            (false, Some(e)) => {
                if (&e.1, &e.2) != (d, p) {
                    out.push(format!(
                        "{k}: recorded dwg {} pdf {}, measured dwg {d} pdf {p}",
                        e.1, e.2
                    ));
                }
                if e.3.trim().is_empty() {
                    out.push(format!("{k}: no written reason"));
                }
            }
        }
    }
    for e in &loss.entries {
        if !measured.contains_key(&e.0) {
            out.push(format!("entry {} names no metric", e.0));
        }
    }
    out
}

#[test]
#[cfg_attr(miri, ignore)]
fn every_pair_measures_exactly_its_recorded_loss() {
    for id in FIXTURES {
        let measured = measure(id);
        if std::env::var_os("PDF_LOSS_PRINT").is_some() {
            for (k, (d, p)) in &measured {
                if d != p {
                    println!("{id}\t{k}\tdwg={d}\tpdf={p}");
                }
            }
        }
        let loss = load_loss(id);
        let failures = check(&measured, &loss);
        assert!(failures.is_empty(), "{id}: {failures:#?}");
    }
}

#[test]
#[cfg_attr(miri, ignore)]
fn every_fixture_has_a_loss_file_with_at_least_one_written_reason() {
    for id in FIXTURES {
        let loss = load_loss(id);
        assert!(
            !loss.entries.is_empty(),
            "{id}: a real exporter always loses something"
        );
        assert!(loss.entries.iter().all(|e| !e.3.trim().is_empty()), "{id}");
    }
}

#[test]
#[cfg_attr(miri, ignore)]
fn an_injected_extra_difference_fails_the_run() {
    let id = "line-001";
    let mut measured = measure(id);
    let loss = load_loss(id);
    assert!(check(&measured, &loss).is_empty());
    measured.get_mut("count.line").unwrap().1.push('9');
    assert!(!check(&measured, &loss).is_empty());
}

#[test]
#[cfg_attr(miri, ignore)]
fn a_loss_entry_that_no_longer_matches_fails_the_run() {
    let id = "circle-001";
    let measured = measure(id);
    let mut loss = load_loss(id);
    loss.entries
        .push(("count.line".into(), "1".into(), "1".into(), "stale".into()));
    assert!(!check(&measured, &loss).is_empty());
}

#[test]
#[cfg_attr(miri, ignore)]
fn the_rotated_pair_swaps_the_pdf_extent_and_a_rotation_blind_comparison_would_fail() {
    let plain = measure("polyline-001");
    let rotated = measure("polyline-001-rot90");
    let (d_plain, p_plain) = &plain["extent.aspect"];
    let (d_rot, p_rot) = &rotated["extent.aspect"];
    assert_eq!(
        d_plain, d_rot,
        "the DWG side does not change with the page rotation"
    );
    let (a, b): (f64, f64) = (p_plain.parse().unwrap(), p_rot.parse().unwrap());
    assert!(
        (a * b - 1.0).abs() < 2e-3,
        "a quarter turn inverts the aspect: {a} and {b}"
    );
    assert_ne!(p_plain, p_rot);
    // Compared as the PDF drew them without composing /Rotate, the rotated pair would read as the
    // unrotated one; the recorded loss for the rotated pair is the composed value.
    let loss = load_loss("polyline-001-rot90");
    let recorded = loss
        .entries
        .iter()
        .find(|e| e.0 == "extent.aspect")
        .map(|e| e.2.clone());
    assert_eq!(recorded.as_deref(), Some(p_rot.as_str()));
}

#[test]
#[cfg_attr(miri, ignore)]
fn the_pairs_are_real_world_differential_and_nothing_labels_them_known_truth() {
    for id in FIXTURES {
        let loss = load_loss(id);
        assert_eq!(loss.evidence, "REAL_WORLD_DIFFERENTIAL", "{id}");
        for f in ["expected_loss.toml", "exporter.txt"] {
            let text = std::fs::read_to_string(dir(id).join(f)).unwrap_or_default();
            assert!(!text.contains("KNOWN_TRUTH_EXACT"), "{id}/{f}");
        }
        let exporter = std::fs::read_to_string(dir(id).join("exporter.txt")).unwrap();
        for needle in [
            "LibreDWG dwg2SVG",
            "rsvg-convert",
            "options:",
            "platform: linux/amd64 native",
        ] {
            assert!(
                exporter.contains(needle),
                "{id}: exporter.txt lacks {needle}"
            );
        }
    }
}

#[test]
fn the_page_names_the_exporter_and_every_pair() {
    // Also what keeps `docs/pvec-dwg-pdf-differential.md` from being an inert doc to the pre-push gate.
    let doc = include_str!("../docs/pvec-dwg-pdf-differential.md");
    for id in FIXTURES {
        assert!(
            doc.contains(id) || id.ends_with("-rot90"),
            "the page does not mention {id}"
        );
    }
    for needle in [
        "dwg2SVG",
        "rsvg-convert",
        "REAL_WORLD_DIFFERENTIAL",
        "expected_loss.toml",
    ] {
        assert!(doc.contains(needle), "the page lost {needle}");
    }
}
