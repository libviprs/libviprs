//! The conversion's tracked heap, split by stage (N8.4a, #1515, slice of #1481).
//!
//! Needs the counting allocator from `pvec_alloc.rs` installed as the global one. Two passes, both
//! through public calls only, so `src/` carries nothing for this:
//!
//! * pass A opens the provider, applies the view profile and flattens the model space by hand, keeping
//!   each result alive, so `open`, `view` and `flatten` are cumulative: what is held after each;
//! * pass B runs `PvecConversion::convert` with an observer that reads the allocator at the events it
//!   already emits, which gives `classify` (up to the first write event: flatten, view, records and
//!   rows all alive), `write` (up to `ConversionArtifactWritten`) and `report` (rows, report JSON and
//!   publish, up to `ConversionPublished`).
//!
//! Every figure is bytes above the heap the pass started from. `held` is the live heap when the
//! stage ended and `peak` the most it reached during the stage, so a stage that frees what it
//! consumed shows a `held` under its `peak`.

#![allow(dead_code)] // each suite uses its own part

use std::path::Path;
use std::sync::{Arc, Mutex};

use libviprs::cad::CadSource;
use libviprs::observe::{EngineEvent, EngineObserver};
use libviprs::pvec::SourceUnit;
use libviprs::pvec::dwg_acadrust::AcadrustProvider;
use libviprs::pvec::flatten::{BlockResolver, FlattenBounds, LineageStore};
use libviprs::pvec::provenance::NoProvenance;
use libviprs::pvec::view_profile::ViewProfile;
use libviprs::pvec::{DwgProvider, EntityScope, LayoutKind};
use libviprs::pvec_convert::PvecConversion;
use serde_json::{Map, Value, json};

use super::pvec_alloc;

/// Stage names in the order they are reported.
pub const STAGES: [&str; 6] = ["open", "view", "flatten", "classify", "write", "report"];

/// Cuts the run into stages: each [`cut`](Self::cut) closes the stage that just ended.
struct Cuts {
    base: u64,
    /// Where the live heap stood, above `base`, when the stage began.
    from: u64,
    mark: pvec_alloc::Mark,
    out: Map<String, Value>,
}

impl Cuts {
    fn new() -> Self {
        let base = pvec_alloc::live();
        Self {
            base,
            from: 0,
            mark: pvec_alloc::mark(),
            out: Map::new(),
        }
    }

    fn cut(&mut self, stage: &str) {
        let peak = self.from + pvec_alloc::since(self.mark).0;
        self.mark = pvec_alloc::mark();
        let held = pvec_alloc::live().saturating_sub(self.base);
        self.from = held;
        // A stage that ends where the last one did and allocated nothing has a peak equal to what it
        // held; never report a peak under the held figure.
        self.out.insert(
            stage.to_owned(),
            json!({"held": held, "peak": peak.max(held)}),
        );
    }
}

/// Pass B's observer: the first write event closes `classify`.
struct Watch {
    cuts: Mutex<Cuts>,
}

impl EngineObserver for Watch {
    fn on_event(&self, event: EngineEvent) {
        let mut c = self.cuts.lock().unwrap();
        let seen = |c: &Cuts, s: &str| c.out.contains_key(s);
        match event {
            EngineEvent::ConversionSourceOpened { .. } => c.cut("b_open"),
            EngineEvent::ConversionViewResolved { .. } => c.cut("b_view"),
            EngineEvent::ConversionChunkWritten { .. }
            | EngineEvent::ConversionSpoolSpilled { .. }
                if !seen(&c, "classify") =>
            {
                c.cut("classify")
            }
            EngineEvent::ConversionArtifactWritten { .. } => {
                if !seen(&c, "classify") {
                    c.cut("classify");
                }
                c.cut("write");
            }
            EngineEvent::ConversionPublished => c.cut("report"),
            _ => {}
        }
    }
}

/// What the stages of one conversion of `bytes` hold, as `{stage: {held, peak}}` plus
/// `tracked_peak` (the largest stage peak) and `entities`.
pub fn measure(bytes: &[u8], dest: &Path, unit: SourceUnit) -> Value {
    let mut a = Cuts::new();
    {
        let provider = AcadrustProvider::open(CadSource::Bytes(bytes)).expect("opens");
        a.cut("open");
        let profile = ViewProfile::default_profile();
        let view = profile.apply(&provider).expect("applies the view profile");
        a.cut("view");
        let layout = provider
            .layouts()
            .expect("layouts")
            .into_iter()
            .find(|l| l.kind == LayoutKind::Model)
            .expect("model space");
        let flat = BlockResolver::new(&provider, FlattenBounds::from_profile(&profile))
            .flatten(
                &EntityScope::Layout(layout.id),
                "stages",
                &layout.name,
                &mut NoProvenance,
                &mut LineageStore::new(),
            )
            .expect("flattens");
        a.cut("flatten");
        std::hint::black_box((&view, &flat));
    }
    let watch = Arc::new(Watch {
        cuts: Mutex::new(Cuts::new()),
    });
    let out = PvecConversion::new(CadSource::Bytes(bytes), dest)
        .unit(unit)
        .observer(watch.clone())
        .convert()
        .expect("converts");
    let b = watch.cuts.lock().unwrap().out.clone();
    let mut stages = Map::new();
    for name in STAGES {
        let v = a
            .out
            .get(name)
            .or_else(|| b.get(name))
            .unwrap_or_else(|| panic!("no {name}"));
        stages.insert(name.to_owned(), v.clone());
    }
    let top = stages
        .values()
        .map(|v| v["peak"].as_u64().unwrap())
        .max()
        .unwrap();
    json!({"stages": stages, "tracked_peak": top, "entities": out.report.totals.encountered})
}

/// What opening `bytes` alone holds and peaks at: the provider's document and nothing after it.
pub fn open_only(bytes: &[u8]) -> (u64, u64) {
    let base = pvec_alloc::live();
    let m = pvec_alloc::mark();
    let provider = AcadrustProvider::open(CadSource::Bytes(bytes)).expect("opens");
    let held = pvec_alloc::live().saturating_sub(base);
    let peak = pvec_alloc::since(m).0;
    drop(provider);
    (held, peak.max(held))
}
