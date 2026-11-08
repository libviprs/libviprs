//! Issue #1048 audit: the acceptance lines the slice tests did not pin. Every error variant a real
//! drawing can reach keeps its originating error as `source()`, and a cancel that fires as soon as
//! the source is open is honoured before the view profile is applied. Needs `--features
//! pvec-convert`.

#![cfg(feature = "pvec-convert")]

use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use libviprs::cad::CadSource;
use libviprs::pvec::{SourceLimits, SourceUnit, SpaceSelection, ViewProfile};
use libviprs::pvec_convert::{PvecConversion, PvecConvertError};
use libviprs::{CancelToken, EngineEvent, EngineObserver};

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/pvec/provider")
        .join(name)
}

fn scratch(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("libviprs-1048-{tag}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&d);
    fs::create_dir_all(&d).unwrap();
    d
}

fn layers<'a>(src: &'a Path, dest: &'a Path) -> PvecConversion<'a> {
    PvecConversion::new(CadSource::Path(src), dest).unit(SourceUnit::Meters)
}

/// The error and every `source()` under it, outermost first.
fn depth(e: &dyn Error) -> usize {
    let (mut n, mut cur) = (1, e.source());
    while let Some(s) = cur {
        n += 1;
        cur = s.source();
    }
    n
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1048_every_wrapping_variant_a_drawing_can_reach_keeps_its_source() {
    let dir = scratch("chain");
    let dest = dir.join("o.pvec");
    let src = fixture("vc_layers.dwg");
    let nested = fixture("vc_nested.dwg");

    let mut paper = ViewProfile::default_profile();
    paper.space = SpaceSelection::PaperLayout("Layout1".into());
    let view = layers(&src, &dest)
        .view_profile(paper)
        .convert()
        .unwrap_err();
    assert!(matches!(view, PvecConvertError::View(_)), "{view:?}");

    let mut shallow = SourceLimits::default();
    shallow.max_block_depth = 1;
    let flatten = layers(&nested, &dest)
        .limits(shallow)
        .convert()
        .unwrap_err();
    assert!(
        matches!(flatten, PvecConvertError::Flatten(_)),
        "{flatten:?}"
    );

    let as_dir = dir.join("is-a-dir");
    fs::create_dir_all(&as_dir).unwrap();
    let publish = layers(&src, &as_dir).convert().unwrap_err();
    assert!(
        matches!(publish, PvecConvertError::Publish(_)),
        "{publish:?}"
    );

    for e in [&view, &flatten, &publish] {
        assert!(depth(e) >= 2, "{e:?} lost its source");
    }
    let _ = fs::remove_dir_all(&dir);
}

struct CancelOnOpen {
    token: CancelToken,
    heard: Mutex<Vec<EngineEvent>>,
}

impl EngineObserver for CancelOnOpen {
    fn on_event(&self, event: EngineEvent) {
        if matches!(event, EngineEvent::ConversionSourceOpened { .. }) {
            self.token.cancel();
        }
        self.heard.lock().unwrap().push(event);
    }
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1048_a_cancel_fired_as_the_source_opens_stops_before_the_view_is_applied() {
    let dir = scratch("open");
    let dest = dir.join("o.pvec");
    let src = fixture("vc_layers.dwg");
    let token = CancelToken::new();
    let observer = Arc::new(CancelOnOpen {
        token: token.clone(),
        heard: Mutex::new(Vec::new()),
    });
    let err = layers(&src, &dest)
        .observer(observer.clone())
        .cancel(token)
        .convert()
        .unwrap_err();
    assert!(matches!(err, PvecConvertError::Cancelled), "{err:?}");
    let heard = observer.heard.lock().unwrap();
    assert!(
        !heard
            .iter()
            .any(|e| matches!(e, EngineEvent::ConversionViewResolved { .. })),
        "the view was applied after the cancel: {heard:?}"
    );
    assert_eq!(fs::read_dir(&dir).unwrap().count(), 0);
    let _ = fs::remove_dir_all(&dir);
}
