//! `ResumeMode::Verify` against a PMTiles archive (issue #1122).
//!
//! Verify used to be refused by name for this sink, because the only verify
//! the engine had was `raster_verify`, which stats `plan.tile_path(coord)`
//! under a checkpoint root. That is a loose-file tree walk and a single-file
//! archive has no seam to enter it through. The archive path is a sibling
//! rather than a replacement: `raster_verify` still owns the tree, and a sink
//! that can open its own output for reading gets routed through
//! `PyramidReader` instead.
//!
//! # What a verify over an archive has to check, and why one check is not
//! enough
//!
//! The obvious implementation is `reader.tile(coord).is_some()` for every
//! planned coordinate, and it is wrong in three separate ways that all look
//! green:
//!
//! * a PMTiles directory entry carries a `run_length` over Hilbert-ordered id
//!   space, so one entry can answer `Some` for a stretch of coordinates the
//!   writer never visited. An archive of a single blank tile with one long run
//!   answers `Some` at every coordinate in its range;
//! * a zero-length payload is `Some(vec![])`, which is not a decodable tile;
//! * a plan **narrower** than the archive resolves every coordinate it asks
//!   about and goes green while extra zoom levels sit in the file. Nothing in
//!   either backend caught that direction before this.
//!
//! So the tests below drive four separate properties through the public engine
//! seam: the archive's self-description against the plan, the sweep, the
//! addressed-tile count in **both** directions, and a positive control on how
//! many coordinates the sweep actually resolved. The count assertions are the
//! ones that matter most: a verify over an empty coordinate set agrees most
//! perfectly of all.
//!
//! # The plan pairs, spelled out
//!
//! Several cells here verify an archive against a plan it was not written
//! with, and each pair is chosen so that exactly one property differs. An Xyz
//! plan halves to 1x1, so its level count is fixed by the longest side, which
//! is what makes a pair with a matching level range but a different tile count
//! constructible at all:
//!
//! | source | tile | levels | tiles |
//! |--------|------|--------|-------|
//! | 512x512 | 256 | 0..=9  | 13 |
//! | 512x256 | 256 | 0..=9  | 11 |
//! | 256x256 | 256 | 0..=8  | 9  |
//! | 256x256 | 512 | 0..=8  | 9  |
//!
//! The first two share a level range, a tile size, a layout and an encoding
//! and differ only in how many tiles they address. The last two share
//! everything including the tile count and differ only in tile size.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use libviprs::planner::{Layout, PyramidPlan, PyramidPlanner, TileCoord};
use libviprs::pyramid_reader::{PmTilesPyramidReader, PyramidReadError, PyramidReader};
use libviprs::resume::{ResumeMode, ResumePolicy};
use libviprs::sink::SinkError;
use libviprs::sink_pmtiles::PmTilesSink;
use libviprs::{
    CollectingObserver, EngineBuilder, EngineError, EngineEvent, EngineKind, FsSink, PixelFormat,
    Raster,
};

// ---------------------------------------------------------------------------
// Sources and plans
// ---------------------------------------------------------------------------

/// A raster where no two tiles can come out the same, so every archive entry
/// has `run_length == 1` and an absent tile is genuinely absent.
fn gradient(w: u32, h: u32) -> Raster {
    let mut data = vec![0u8; w as usize * h as usize * 3];
    for y in 0..h {
        for x in 0..w {
            let off = (y as usize * w as usize + x as usize) * 3;
            data[off] = (x % 251) as u8;
            data[off + 1] = (y % 241) as u8;
            data[off + 2] = ((x * 7 + y * 13) % 239) as u8;
        }
    }
    Raster::new(w, h, PixelFormat::Rgb8, data).expect("a gradient raster is well formed")
}

/// A raster where every tile comes out identical, which collapses the archive
/// to one stored payload and the directory to the longest runs the writer can
/// emit. This is the shape the `is_some()` trap lives in.
fn uniform(w: u32, h: u32) -> Raster {
    Raster::new(
        w,
        h,
        PixelFormat::Rgb8,
        vec![0xf0; w as usize * h as usize * 3],
    )
    .expect("a uniform raster is well formed")
}

fn plan_for(w: u32, h: u32, tile: u32) -> PyramidPlan {
    PyramidPlanner::new(w, h, tile, 0, Layout::Xyz)
        .expect("an Xyz plan over a positive source is valid")
        .plan()
}

// ---------------------------------------------------------------------------
// Running the two halves
// ---------------------------------------------------------------------------

/// Write `src` into a published archive at `path` and release the sink.
fn write_archive(path: &Path, plan: &PyramidPlan, src: &Raster) {
    let sink = PmTilesSink::builder(path)
        .plan(plan.clone())
        .build()
        .expect("the archive sink builds for an Overwrite run");
    EngineBuilder::new(src, plan.clone(), sink)
        .with_engine(EngineKind::Monolithic)
        .run()
        .expect("the archive run succeeds");
    assert!(path.is_file(), "the run published {}", path.display());
}

/// Verify the archive at `path` against `plan`, reporting the engine's answer
/// and the events it emitted.
fn verify_archive(
    path: &Path,
    plan: &PyramidPlan,
    src: &Raster,
) -> (
    Result<libviprs::EngineResult, EngineError>,
    Arc<CollectingObserver>,
) {
    let observer = Arc::new(CollectingObserver::new());
    let sink = PmTilesSink::builder(path)
        .plan(plan.clone())
        .resume_mode(ResumeMode::Verify)
        .build()
        .expect("a Verify-mode archive sink builds");
    let result = EngineBuilder::new(src, plan.clone(), sink)
        .with_engine(EngineKind::Monolithic)
        .with_observer_arc(observer.clone() as Arc<dyn libviprs::EngineObserver>)
        .with_resume(ResumePolicy::verify())
        .run();
    (result, observer)
}

/// How many coordinates the verify actually resolved.
///
/// `TileCompleted` is emitted once a coordinate has been found and accepted,
/// so counting the events is counting resolutions rather than attempts. That
/// is the whole point of the positive control: a sweep that visited nothing
/// and a sweep that resolved everything both return `Ok`.
fn resolved(observer: &CollectingObserver) -> usize {
    observer
        .events()
        .iter()
        .filter(|e| matches!(e, EngineEvent::TileCompleted { .. }))
        .count()
}

/// The event stream with everything that cannot be compared across two runs
/// (the per-event timestamp) dropped.
fn transcript(events: &[EngineEvent]) -> Vec<String> {
    events
        .iter()
        .map(|event| match event {
            EngineEvent::LevelStarted {
                level,
                width,
                height,
                tile_count,
            } => format!("LevelStarted {level} {width}x{height} {tile_count}"),
            EngineEvent::TileCompleted { coord, .. } => {
                format!("TileCompleted {}/{}/{}", coord.level, coord.col, coord.row)
            }
            EngineEvent::LevelCompleted {
                level,
                tiles_produced,
            } => format!("LevelCompleted {level} {tiles_produced}"),
            EngineEvent::Finished {
                total_tiles,
                levels,
            } => format!("Finished {total_tiles} {levels}"),
            other => format!("{other:?}"),
        })
        .collect()
}

/// Every name directly under `dir`, sorted.
fn listing(dir: &Path) -> Vec<String> {
    let mut names: Vec<String> = std::fs::read_dir(dir)
        .expect("the directory is readable")
        .map(|entry| {
            entry
                .expect("the entry is readable")
                .file_name()
                .to_string_lossy()
                .into_owned()
        })
        .collect();
    names.sort();
    names
}

/// The sidecar directory the sink takes its advisory lock in.
fn job_dir(archive: &Path) -> PathBuf {
    let mut name = archive.to_path_buf().into_os_string();
    name.push(".job");
    PathBuf::from(name)
}

// ---------------------------------------------------------------------------
// The archive verifies, and the sweep is not empty
// ---------------------------------------------------------------------------

/// A freshly written archive verifies against the plan that wrote it, writes
/// nothing, and resolves every planned coordinate.
///
/// The last clause is the positive control and it is not decoration. Verify
/// returns `Ok` for a sweep that resolved all thirteen coordinates and for a
/// sweep that resolved none, so without a count this cell would stay green for
/// an implementation that walks an empty iterator.
#[test]
#[cfg_attr(miri, ignore)]
fn a_freshly_written_archive_verifies_and_the_sweep_resolved_every_coordinate() {
    let dir = tempfile::tempdir().expect("tempdir");
    let plan = plan_for(512, 512, 256);
    let src = gradient(512, 512);
    let archive = dir.path().join("fresh.pmtiles");
    write_archive(&archive, &plan, &src);

    let (result, observer) = verify_archive(&archive, &plan, &src);
    let result = result.expect("a good archive verifies against its own plan");

    assert_eq!(
        result.tiles_produced, 0,
        "Verify is read-only, so it produced no tiles"
    );
    assert_eq!(
        result.levels_processed as usize,
        plan.levels.len(),
        "every level was visited"
    );

    let planned = plan.tile_coords().count();
    assert!(planned > 0, "the plan itself has coordinates to check");
    assert_eq!(
        resolved(&observer),
        planned,
        "the sweep resolved {} of the plan's {planned} coordinates; a verify \
         that agreed about nothing agrees most perfectly of all",
        resolved(&observer)
    );
}

/// The same, over an archive whose every tile is identical.
///
/// This is the archive the `is_some()` trap is built on: one stored payload,
/// and directory entries whose `run_length` covers long stretches of
/// Hilbert-ordered id space at once. It has to verify green, and the sweep has
/// to have genuinely resolved every coordinate rather than short-circuited on
/// a run.
#[test]
#[cfg_attr(miri, ignore)]
fn an_archive_collapsed_into_runs_still_resolves_every_coordinate() {
    let dir = tempfile::tempdir().expect("tempdir");
    let plan = plan_for(512, 512, 256);
    let src = uniform(512, 512);
    let archive = dir.path().join("runs.pmtiles");
    write_archive(&archive, &plan, &src);

    let header = *PmTilesPyramidReader::try_open(&archive)
        .expect("the archive opens")
        .reader()
        .header();
    assert!(
        header.tile_contents_count < header.addressed_tiles_count,
        "the control on the control: a uniform source has to collapse into \
         fewer stored payloads ({}) than addressed tiles ({}), otherwise this \
         cell is testing the same archive as the gradient one",
        header.tile_contents_count,
        header.addressed_tiles_count
    );

    let (result, observer) = verify_archive(&archive, &plan, &src);
    result.expect("an archive of one repeated tile is still a valid archive");
    assert_eq!(
        resolved(&observer),
        plan.tile_coords().count(),
        "every planned coordinate was resolved"
    );
}

// ---------------------------------------------------------------------------
// The four refusals
// ---------------------------------------------------------------------------

/// An archive missing coordinates the plan names is refused, and the refusal
/// says which coordinate.
///
/// The archive is written from a 512x256 source and verified against the
/// 512x512 plan, so the two agree about the level range, the tile size, the
/// layout and the encoding, and disagree only about the top level's grid.
/// The source is a gradient so that every entry has `run_length == 1`: in this
/// pair the absence is a real absence, and the case a run could paper over is
/// the sibling cell below, where no per-coordinate sweep can help at all.
#[test]
#[cfg_attr(miri, ignore)]
fn an_archive_short_of_the_plan_is_refused_and_names_the_missing_coordinate() {
    let dir = tempfile::tempdir().expect("tempdir");
    let narrow = plan_for(512, 256, 256);
    let wide = plan_for(512, 512, 256);
    let archive = dir.path().join("short.pmtiles");
    write_archive(&archive, &narrow, &gradient(512, 256));

    let (result, _observer) = verify_archive(&archive, &wide, &gradient(512, 512));
    let err = result.expect_err("an archive short of the plan cannot verify");
    let message = err.to_string();

    // The top level is 2x2 in the plan and 2x1 in the archive, and levels run
    // top-down, so the first coordinate the sweep cannot resolve is the second
    // row of the top level.
    let first_missing = TileCoord {
        level: 9,
        col: 0,
        row: 1,
    };
    assert!(
        message.contains(&format!("{first_missing:?}")),
        "the refusal must name the first coordinate it could not resolve \
         ({first_missing:?}), got: {message}"
    );
    // A verify that had simply been handed a directory to walk would report
    // the FIRST coordinate of the top level instead, which this archive does
    // have. Naming that one is the half-supported shape, not a verify.
    let present = TileCoord {
        level: 9,
        col: 0,
        row: 0,
    };
    assert!(
        !message.contains(&format!("{present:?}")),
        "{present:?} is in the archive, so a refusal naming it means the verify \
         never looked in the archive at all: {message}"
    );
}

/// An archive that addresses MORE tiles than the plan is refused, even though
/// every coordinate the plan names resolves.
///
/// Nothing caught this direction in either backend before #1122, and no
/// per-coordinate sweep ever can: the sweep only asks about coordinates the
/// plan has, so extra zoom levels and extra columns sitting in the file are
/// invisible to it. The first assertion below is the positive control that
/// makes the refusal meaningful, because it proves the sweep would have gone
/// green.
#[test]
#[cfg_attr(miri, ignore)]
fn an_archive_wider_than_the_plan_is_refused_although_every_coordinate_resolves() {
    let dir = tempfile::tempdir().expect("tempdir");
    let wide = plan_for(512, 512, 256);
    let narrow = plan_for(512, 256, 256);
    let archive = dir.path().join("wide.pmtiles");
    write_archive(&archive, &wide, &gradient(512, 512));

    // The control: every coordinate the narrow plan asks about is in there.
    let reader = PmTilesPyramidReader::try_open(&archive).expect("the archive opens");
    let mut probed = 0usize;
    for coord in narrow.tile_coords() {
        assert!(
            reader
                .tile(coord)
                .expect("reading a tile from a sound archive")
                .is_some(),
            "{coord:?} is missing, so this cell is no longer about the archive \
             being wider than the plan"
        );
        probed += 1;
    }
    assert_eq!(
        probed,
        narrow.tile_coords().count(),
        "the control itself probed every coordinate"
    );

    let (result, _observer) = verify_archive(&archive, &narrow, &gradient(512, 256));
    let err = result.expect_err("an archive holding tiles the plan does not name cannot verify");
    let message = err.to_string();
    assert!(
        message.contains(&wide.tile_coords().count().to_string())
            && message.contains(&narrow.tile_coords().count().to_string()),
        "the refusal must name both counts ({} in the archive, {} in the plan), \
         got: {message}",
        wide.tile_coords().count(),
        narrow.tile_coords().count()
    );
}

/// A structurally damaged archive is refused as damaged, not as missing a
/// tile.
///
/// Two separate damages, because they are caught in two different places and
/// a test carrying only the first proves nothing about the walk.
///
/// * Lopping the last 64 bytes off breaks the section bounds, and that is
///   caught when the archive is opened, before any walk starts.
/// * Rewriting `addressed_tiles_count` in the header breaks nothing a reader
///   would notice: the archive parses, every directory decodes and every tile
///   still reads. The only thing wrong with it is that the header's own count
///   disagrees with what the runs cover, and the structural walk is the only
///   thing that would ever look.
#[test]
#[cfg_attr(miri, ignore)]
fn a_damaged_archive_is_refused_as_damaged_rather_than_as_a_missing_tile() {
    let dir = tempfile::tempdir().expect("tempdir");
    let plan = plan_for(512, 512, 256);
    let src = gradient(512, 512);
    let good_path = dir.path().join("good.pmtiles");
    write_archive(&good_path, &plan, &src);
    let good = std::fs::read(&good_path).expect("read the archive back");
    assert!(
        good.len() > 64,
        "lopping 64 bytes off a {}-byte archive is a deletion, not a truncation",
        good.len()
    );

    let truncated = dir.path().join("truncated.pmtiles");
    std::fs::write(&truncated, &good[..good.len() - 64]).expect("write the truncated archive");
    let (result, _observer) = verify_archive(&truncated, &plan, &src);
    let err = result.expect_err("a truncated archive cannot verify");
    // Named rather than asserted absent. "It did not say missing tile" is
    // satisfied by any message at all, including one about a file that could
    // not be opened for a reason nobody looked at; the variant says the
    // refusal came back through the reader seam, carrying the archive's own
    // typed complaint.
    assert!(
        matches!(
            err,
            EngineError::Sink(SinkError::PyramidRead(PyramidReadError::PmTiles(_)))
        ),
        "a truncated archive is damaged, not short of a tile, got {err:?}"
    );
    assert!(
        !err.to_string().contains("missing tile"),
        "and it must not read as a missing tile either: {err}"
    );

    // `addressed_tiles_count` is a u64 at offset 72 of the 127-byte header.
    const OFF_ADDRESSED_TILES: usize = 72;
    let mut miscounted = good.clone();
    let claimed = u64::from_le_bytes(
        miscounted[OFF_ADDRESSED_TILES..OFF_ADDRESSED_TILES + 8]
            .try_into()
            .expect("eight bytes"),
    );
    assert_eq!(
        claimed,
        plan.tile_coords().count() as u64,
        "offset {OFF_ADDRESSED_TILES} is not addressed_tiles_count any more, so \
         the corruption below is aimed at the wrong field"
    );
    miscounted[OFF_ADDRESSED_TILES..OFF_ADDRESSED_TILES + 8]
        .copy_from_slice(&(claimed + 1).to_le_bytes());
    let lying = dir.path().join("lying.pmtiles");
    std::fs::write(&lying, &miscounted).expect("write the miscounted archive");

    // The control: every tile still reads, so nothing but the structural walk
    // can refuse this one.
    let reader =
        PmTilesPyramidReader::try_open(&lying).expect("the miscounted archive still opens");
    for coord in plan.tile_coords() {
        assert!(
            reader.tile(coord).expect("the tile reads").is_some(),
            "{coord:?} stopped reading, so this is no longer a walk-only defect"
        );
    }

    let (result, _observer) = verify_archive(&lying, &plan, &src);
    let err = result.expect_err("an archive whose header miscounts its own tiles cannot verify");
    assert!(
        matches!(
            err,
            EngineError::Sink(SinkError::PyramidRead(
                PyramidReadError::StructuralDefects { .. }
            ))
        ),
        "every tile is present, so the only thing that can refuse this is the \
         structural walk, got {err:?}"
    );
    assert!(
        err.to_string().contains("addressed_tiles_count"),
        "the refusal must name the field that does not add up, got: {err}"
    );
}

/// An archive whose recorded generation disagrees with the plan is refused.
///
/// This is the archive's stand-in for `verify_checkpoint_contract`, and it is
/// the check an implementation drops first, because the sweep passes without
/// it: a 256-pixel tile at `0/0/0` is a perfectly readable tile whatever the
/// plan meant by a tile. Written at `tile_size: 256`, verified against a
/// 512 plan over the same source, so the level range and the tile count match
/// and the tile size is the only thing left to disagree about.
#[test]
#[cfg_attr(miri, ignore)]
fn an_archive_generated_at_another_tile_size_is_refused() {
    let dir = tempfile::tempdir().expect("tempdir");
    let written = plan_for(256, 256, 256);
    let asked = plan_for(256, 256, 512);
    assert_eq!(
        (written.levels.len(), written.tile_coords().count()),
        (asked.levels.len(), asked.tile_coords().count()),
        "the pair has to agree about levels and counts, or this cell is about \
         one of the other checks"
    );

    let archive = dir.path().join("tilesize.pmtiles");
    write_archive(&archive, &written, &gradient(256, 256));

    let (result, _observer) = verify_archive(&archive, &asked, &gradient(256, 256));
    let message = result
        .expect_err("an archive generated at another tile size cannot verify")
        .to_string();
    assert!(
        message.contains("256") && message.contains("512"),
        "the refusal must name both tile sizes, got: {message}"
    );
}

// ---------------------------------------------------------------------------
// Verify is read-only
// ---------------------------------------------------------------------------

/// A verify run changes not one byte of the archive and leaves no sidecar
/// behind.
///
/// Letting `Verify` past the builder's gate means a verify run now takes the
/// advisory run lock, which creates `<archive>.job` for the life of the sink.
/// That is correct (two jobs aimed at one archive still must not overlap) and
/// it is new, so the removal is pinned here rather than assumed.
#[test]
#[cfg_attr(miri, ignore)]
fn a_verify_run_leaves_the_archive_and_its_directory_exactly_as_it_found_them() {
    let dir = tempfile::tempdir().expect("tempdir");
    let plan = plan_for(512, 512, 256);
    let src = gradient(512, 512);
    let archive = dir.path().join("readonly.pmtiles");
    write_archive(&archive, &plan, &src);

    let before = std::fs::read(&archive).expect("read the archive");
    let before_listing = listing(dir.path());

    let (result, _observer) = verify_archive(&archive, &plan, &src);
    result.expect("the archive verifies");

    let after = std::fs::read(&archive).expect("read the archive again");
    assert!(
        before == after,
        "verify rewrote the archive: {} bytes before, {} after",
        before.len(),
        after.len()
    );
    assert_eq!(
        before_listing,
        listing(dir.path()),
        "verify left something behind in the archive's directory"
    );
    assert!(
        !job_dir(&archive).exists(),
        "the run lock's sidecar {} outlived the sink that took it",
        job_dir(&archive).display()
    );
}

/// The archive path and the tree path emit the same events for the same
/// pyramid.
///
/// Verify runs are first-class for observers, and `raster_verify` has always
/// emitted `LevelStarted` / `TileCompleted` / `LevelCompleted` / `Finished`.
/// A second verify path that counted levels up instead of down, or emitted a
/// different `tile_count`, would be a progress bar that behaves differently
/// depending on where the tiles went.
#[test]
#[cfg_attr(miri, ignore)]
fn an_archive_verify_emits_the_same_events_as_a_tree_verify() {
    let dir = tempfile::tempdir().expect("tempdir");
    let plan = plan_for(512, 512, 256);
    let src = gradient(512, 512);

    let archive = dir.path().join("events.pmtiles");
    write_archive(&archive, &plan, &src);
    let tree = dir.path().join("tree");
    EngineBuilder::new(&src, plan.clone(), FsSink::new(&tree, plan.clone()))
        .with_engine(EngineKind::Monolithic)
        .run()
        .expect("the tree run succeeds");

    let tree_observer = Arc::new(CollectingObserver::new());
    EngineBuilder::new(&src, plan.clone(), FsSink::new(&tree, plan.clone()))
        .with_engine(EngineKind::Monolithic)
        .with_observer_arc(tree_observer.clone() as Arc<dyn libviprs::EngineObserver>)
        .with_resume(ResumePolicy::verify())
        .run()
        .expect("the tree verifies");

    let (result, archive_observer) = verify_archive(&archive, &plan, &src);
    result.expect("the archive verifies");

    let from_tree = transcript(&tree_observer.events());
    let from_archive = transcript(&archive_observer.events());
    assert!(
        !from_tree.is_empty(),
        "the tree verify emitted nothing, so there is no transcript to match"
    );
    assert_eq!(
        from_archive, from_tree,
        "the two verify paths describe the same run differently"
    );
}

// ---------------------------------------------------------------------------
// The negative control
// ---------------------------------------------------------------------------

/// Resume is still refused by name, and refused at the tile the engine would
/// have skipped.
///
/// #1122 is Verify only. The writer's staging is not reconstructible from a
/// checkpoint, so a resumed run would publish an archive with every pre-crash
/// tile silently absent, and that refusal has to survive the gate being
/// widened for Verify. If this cell ever goes green the change went too far.
#[test]
#[cfg_attr(miri, ignore)]
fn resume_against_an_archive_is_still_refused() {
    let dir = tempfile::tempdir().expect("tempdir");
    let plan = plan_for(256, 256, 256);

    let built = PmTilesSink::builder(dir.path().join("resume.pmtiles"))
        .plan(plan.clone())
        .resume_mode(ResumeMode::Resume)
        .build();
    match built {
        Err(SinkError::UnsupportedResumeMode {
            mode: ResumeMode::Resume,
        }) => {}
        other => panic!("Resume must still be refused by name, got {other:?}"),
    }

    // And the control beside it: the same builder, the same archive, Verify.
    PmTilesSink::builder(dir.path().join("verify.pmtiles"))
        .plan(plan)
        .resume_mode(ResumeMode::Verify)
        .build()
        .expect("Verify is the half that opened up");
}
