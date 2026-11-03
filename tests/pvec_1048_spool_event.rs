//! Issue #1048 (N7.4): a conversion that spills tells its observer so. Needs `--features
//! pvec-convert`.

#![cfg(feature = "pvec-convert")]

use std::fs;
use std::path::PathBuf;
use std::sync::Arc;

use libviprs::cad::CadSource;
use libviprs::pvec::SourceUnit;
use libviprs::pvec::writer::SpillConfig;
use libviprs::pvec_convert::PvecConversion;
use libviprs::{CollectingObserver, EngineEvent};

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1048_a_spilled_conversion_reports_the_spill_and_a_memory_one_does_not() {
    let dir = std::env::temp_dir().join(format!("libviprs-1048-spool-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    let root = dir.join("root");
    fs::create_dir_all(&root).unwrap();
    let src = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/pvec/provider/vc_layers.dwg");

    let memory = Arc::new(CollectingObserver::new());
    PvecConversion::new(CadSource::Path(&src), &dir.join("m.pvec"))
        .unit(SourceUnit::Meters)
        .observer(memory.clone())
        .convert()
        .unwrap();
    assert!(
        !memory
            .events()
            .iter()
            .any(|e| matches!(e, EngineEvent::ConversionSpoolSpilled { .. }))
    );

    let spilled = Arc::new(CollectingObserver::new());
    PvecConversion::new(CadSource::Path(&src), &dir.join("s.pvec"))
        .unit(SourceUnit::Meters)
        .spill(SpillConfig { root, budget: 4096 })
        .observer(spilled.clone())
        .convert()
        .unwrap();
    let events = spilled.events();
    let at = events
        .iter()
        .position(|e| matches!(e, EngineEvent::ConversionSpoolSpilled { bytes } if *bytes > 0))
        .expect("a spool event with the bytes streamed");
    let written = events
        .iter()
        .position(|e| matches!(e, EngineEvent::ConversionArtifactWritten { .. }))
        .unwrap();
    assert!(
        at < written,
        "the spill is reported before the artifact is sealed"
    );
    let _ = fs::remove_dir_all(&dir);
}
