//! Issue #1395 (N2.7/N4.8, slice of #1079 and #1058): the provider differential harness.
//!
//! Needs `--features "dwg-acadrust pvec-normalized"`. The second provider is a test double that
//! wraps acadrust and changes one thing, because acadrust is the only provider compiled in and a
//! harness that has never seen two providers disagree has not been tested.
#![cfg(all(feature = "dwg-acadrust", feature = "pvec-normalized"))]

use std::collections::BTreeMap;
use std::path::PathBuf;

use libviprs::cad::{CadSource, ItemHandle};
use libviprs::pvec::differential::{Deltas, Fixture, Matrix, Verdict};
use libviprs::pvec::dwg_acadrust::AcadrustProvider;
use libviprs::pvec::{
    DwgProvider, EntityCursor, EntityFacts, EntityScope, LayerState, LayoutState, ProviderCaps,
    ProviderError, RawEntity, RawGeometry,
};

const REV: &str = "r2-open-test";

fn fixtures() -> Vec<Fixture> {
    ["vc_layers", "vc_nested"]
        .into_iter()
        .map(|id| Fixture {
            id: id.to_string(),
            path: PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("tests/fixtures/pvec/provider")
                .join(format!("{id}.dwg")),
            evidence: "SYNTHETIC".to_string(),
        })
        .collect()
}

/// acadrust under another name, optionally changing what it reports.
struct Twin {
    inner: AcadrustProvider,
    mutate: fn(&mut RawEntity),
}

struct MutatedCursor<'a> {
    inner: Box<dyn EntityCursor + 'a>,
    mutate: fn(&mut RawEntity),
}

impl EntityCursor for MutatedCursor<'_> {
    fn next_entity(&mut self) -> Result<Option<RawEntity>, ProviderError> {
        let mut e = self.inner.next_entity()?;
        if let Some(e) = e.as_mut() {
            (self.mutate)(e);
        }
        Ok(e)
    }
}

fn keep(_: &mut RawEntity) {}

impl DwgProvider for Twin {
    fn probe(source: CadSource<'_>) -> Result<bool, ProviderError> {
        AcadrustProvider::probe(source)
    }
    fn open(source: CadSource<'_>) -> Result<Self, ProviderError> {
        Ok(Twin {
            inner: AcadrustProvider::open(source)?,
            mutate: keep,
        })
    }
    fn name(&self) -> &'static str {
        "twin"
    }
    fn caps(&self) -> &ProviderCaps {
        self.inner.caps()
    }
    fn layouts(&self) -> Result<Vec<LayoutState>, ProviderError> {
        self.inner.layouts()
    }
    fn layers(&self) -> Result<Vec<LayerState>, ProviderError> {
        self.inner.layers()
    }
    fn entity_facts(
        &self,
        scope: &EntityScope,
    ) -> Result<BTreeMap<ItemHandle, EntityFacts>, ProviderError> {
        self.inner.entity_facts(scope)
    }
    fn entities<'a>(
        &'a self,
        scope: &EntityScope,
    ) -> Result<Box<dyn EntityCursor + 'a>, ProviderError> {
        Ok(Box::new(MutatedCursor {
            inner: self.inner.entities(scope)?,
            mutate: self.mutate,
        }))
    }
}

/// `Twin` with a mutation, as its own provider type so the harness can name it.
macro_rules! twin {
    ($name:ident, $label:literal, $mutate:expr) => {
        struct $name(Twin);
        impl DwgProvider for $name {
            fn probe(source: CadSource<'_>) -> Result<bool, ProviderError> {
                AcadrustProvider::probe(source)
            }
            fn open(source: CadSource<'_>) -> Result<Self, ProviderError> {
                Ok($name(Twin {
                    inner: AcadrustProvider::open(source)?,
                    mutate: $mutate,
                }))
            }
            fn name(&self) -> &'static str {
                $label
            }
            fn caps(&self) -> &ProviderCaps {
                self.0.caps()
            }
            fn layouts(&self) -> Result<Vec<LayoutState>, ProviderError> {
                self.0.layouts()
            }
            fn layers(&self) -> Result<Vec<LayerState>, ProviderError> {
                self.0.layers()
            }
            fn entities<'a>(
                &'a self,
                scope: &EntityScope,
            ) -> Result<Box<dyn EntityCursor + 'a>, ProviderError> {
                self.0.entities(scope)
            }
        }
    };
}

fn nudge_line(e: &mut RawEntity) {
    if let RawGeometry::Line { end, .. } = &mut e.geometry {
        end[0] += 1e-9;
    }
}

fn unmap_lines(e: &mut RawEntity) {
    if matches!(e.geometry, RawGeometry::Line { .. }) {
        e.geometry = RawGeometry::Unmapped {
            reason: "a provider that does not map lines".into(),
        };
    }
}

fn relayer(e: &mut RawEntity) {
    e.provenance.layer = "ELSEWHERE".into();
}

twin!(Same, "twin", keep);
twin!(Nudged, "nudged", nudge_line);
twin!(Unmapper, "unmapper", unmap_lines);
twin!(Relayered, "relayered", relayer);

fn two<B: DwgProvider>() -> Matrix {
    let mut m = Matrix::new(REV);
    m.run::<AcadrustProvider>(&fixtures()).unwrap();
    m.run::<B>(&fixtures()).unwrap();
    m
}

fn none() -> Deltas {
    Deltas::parse("").unwrap()
}

#[test]
fn one_provider_is_partial_names_the_missing_one_and_is_not_a_pass() {
    let mut m = Matrix::new(REV);
    m.run::<AcadrustProvider>(&fixtures()).unwrap();
    m.absent(
        "acadsharp-rs",
        "no DwgProvider adapter yet (libviprs-dep#161)",
    );
    let v = m.verdict(&none());
    assert!(matches!(v, Verdict::Partial { .. }), "{v:?}");
    assert!(!v.passed());
    let r = m.report(&none());
    assert!(r.contains("PARTIAL"), "{r}");
    assert!(r.contains("acadsharp-rs"), "{r}");
    assert!(
        r.contains("no differential comparison was performed"),
        "{r}"
    );
}

#[test]
fn a_run_with_no_second_provider_declared_is_still_partial() {
    let mut m = Matrix::new(REV);
    m.run::<AcadrustProvider>(&fixtures()).unwrap();
    assert!(!m.verdict(&none()).passed());
}

#[test]
fn two_providers_that_agree_pass_as_dual_oracle_equivalent_and_never_as_truth() {
    let m = two::<Same>();
    let v = m.verdict(&none());
    assert!(v.passed(), "{v:?}\n{}", m.report(&none()));
    let r = m.report(&none());
    assert!(r.contains("DUAL_ORACLE_EQUIVALENT"), "{r}");
    assert!(
        !r.contains("KNOWN_TRUTH_EXACT"),
        "agreement is never source truth: {r}"
    );
}

#[test]
fn a_fixtures_own_evidence_class_is_reported_next_to_the_agreement_not_replaced_by_it() {
    let mut f = fixtures();
    f[0].evidence = "KNOWN_TRUTH_EXACT".to_string();
    let mut m = Matrix::new(REV);
    m.run::<AcadrustProvider>(&f).unwrap();
    m.run::<Same>(&f).unwrap();
    let r = m.report(&none());
    assert!(r.contains("evidence=KNOWN_TRUTH_EXACT"), "{r}");
    assert!(r.contains("agreement=DUAL_ORACLE_EQUIVALENT"), "{r}");
    assert!(r.contains("evidence=SYNTHETIC"), "{r}");
}

#[test]
fn a_one_field_divergence_fails_unless_it_is_listed_with_a_reason() {
    let m = two::<Nudged>();
    let diffs = m.differences();
    assert!(!diffs.is_empty(), "the nudge must be seen");
    assert!(diffs.iter().all(|d| d.field == "param:end.x"), "{diffs:?}");
    assert!(
        diffs
            .iter()
            .all(|d| !d.entity.is_empty() && !d.fixture.is_empty())
    );
    let v = m.verdict(&none());
    assert!(matches!(v, Verdict::Failed { .. }), "{v:?}");
    assert!(!v.passed());

    let mut toml = String::new();
    for d in &diffs {
        toml.push_str(&format!(
            "[[delta]]\nfixture = \"{}\"\nentity = \"{}\"\nfield = \"{}\"\nreason = \"nudged by the test double\"\n",
            d.fixture, d.entity, d.field
        ));
    }
    let listed = Deltas::parse(&toml).unwrap();
    assert!(m.verdict(&listed).passed(), "{:?}", m.verdict(&listed));
}

#[test]
fn a_listed_difference_that_no_longer_happens_is_a_failure_too() {
    let m = two::<Same>();
    let stale = Deltas::parse(
        "[[delta]]\nfixture = \"vc_nested\"\nentity = \"/#1\"\nfield = \"layer\"\nreason = \"gone\"\n",
    )
    .unwrap();
    let v = m.verdict(&stale);
    assert!(
        matches!(v, Verdict::Failed { ref stale, .. } if stale.len() == 1),
        "{v:?}"
    );
}

#[test]
fn a_difference_listed_for_another_field_does_not_excuse_this_one() {
    let m = two::<Nudged>();
    let d = &m.differences()[0];
    let wrong = Deltas::parse(&format!(
        "[[delta]]\nfixture = \"{}\"\nentity = \"{}\"\nfield = \"layer\"\nreason = \"wrong field\"\n",
        d.fixture, d.entity
    ))
    .unwrap();
    assert!(!m.verdict(&wrong).passed());
}

#[test]
fn buckets_are_compared_per_entity_not_in_aggregate() {
    let m = two::<Unmapper>();
    let diffs = m.differences();
    let bucket: Vec<_> = diffs.iter().filter(|d| d.field == "bucket").collect();
    assert!(!bucket.is_empty(), "{diffs:?}");
    assert!(
        bucket
            .iter()
            .any(|d| d.left == "SupportedExact" && d.right == "Unsupported"),
        "{bucket:?}"
    );
    assert!(!m.verdict(&none()).passed());
}

#[test]
fn the_layer_is_a_compared_field() {
    let m = two::<Relayered>();
    assert!(m.differences().iter().any(|d| d.field == "layer"));
}

#[test]
fn every_result_line_names_the_platform_and_the_corpus_revision() {
    let m = two::<Same>();
    let r = m.report(&none());
    let platform = format!("{}-{}", std::env::consts::ARCH, std::env::consts::OS);
    let lines: Vec<_> = r.lines().filter(|l| l.starts_with("result ")).collect();
    assert!(!lines.is_empty(), "{r}");
    for l in lines {
        assert!(l.contains(&format!("platform={platform}")), "{l}");
        assert!(l.contains(&format!("corpus={REV}")), "{l}");
    }
}

#[test]
fn the_report_is_the_same_text_on_a_second_run() {
    let a = two::<Same>().report(&none());
    let b = two::<Same>().report(&none());
    assert_eq!(a, b);
}

#[test]
fn deltas_need_a_reason_and_refuse_unknown_keys_and_duplicates() {
    let entry = |extra: &str| {
        format!("[[delta]]\nfixture = \"f\"\nentity = \"/#1\"\nfield = \"layer\"\n{extra}")
    };
    assert!(Deltas::parse(&entry("reason = \"why\"\n")).is_ok());
    assert!(Deltas::parse(&entry("")).is_err(), "no reason");
    assert!(
        Deltas::parse(&entry("reason = \"  \"\n")).is_err(),
        "blank reason"
    );
    assert!(
        Deltas::parse(&entry("reason = \"x\"\ncolour = \"red\"\n")).is_err(),
        "unknown key"
    );
    let twice = format!("{}{}", entry("reason = \"a\"\n"), entry("reason = \"b\"\n"));
    assert!(Deltas::parse(&twice).is_err(), "duplicate entry");
}

#[test]
fn the_shipped_deltas_file_parses() {
    let text = std::fs::read_to_string(
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/pvec/provider_deltas.toml"),
    )
    .unwrap();
    Deltas::parse(&text).unwrap();
}

#[test]
fn the_rows_carry_params_bucket_and_pvec_type_for_the_acadrust_baseline() {
    let mut m = Matrix::new(REV);
    m.run::<AcadrustProvider>(&fixtures()).unwrap();
    let rows = m
        .rows("acadrust", "vc_layers")
        .expect("a run for vc_layers");
    assert!(!rows.is_empty());
    let line = rows
        .iter()
        .find(|r| r.class_name == "LINE")
        .expect("vc_layers has lines");
    assert_eq!(line.bucket, "SupportedExact");
    assert_eq!(line.pvec_type, "trimmed_curve(line)");
    assert!(line.params.iter().any(|(n, _)| n == "end.x"));
    assert!(line.params.iter().any(|(n, _)| n == "matrix.tx"));
}
