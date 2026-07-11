//! Round-trip tests for the `serde` cargo feature (issue #67).
//!
//! The feature gates `Serialize` / `Deserialize` derives on the wire/config
//! types so an out-of-process caller (an external worker layer plugged in via
//! `WorkExecutor`) can reconstruct a `PyramidPlan` / `EngineConfig` from a
//! JSON envelope. These tests pin two properties:
//!
//! * value round-trip: serialize → deserialize reproduces the original value
//!   for every gated type, including a real multi-level plan produced by
//!   `PyramidPlanner::plan` (not a hand-built toy);
//! * runtime-handle hygiene: `EngineConfig::cancel` is process-local state
//!   and must be skipped, never serialized, and always `None` after
//!   deserialization.

#![cfg(feature = "serde")]

use libviprs::{
    BlankTileStrategy, CancelToken, DedupeStrategy, EngineConfig, FailurePolicy, Layout,
    PyramidPlan, PyramidPlanner, RetryPolicy, TileCoord, TileRect, WorkerId,
};
use std::time::Duration;

/// A realistic plan: non-square image, overlap, centring, multiple levels.
fn sample_plan() -> PyramidPlan {
    PyramidPlanner::new(4097, 2731, 254, 1, Layout::DeepZoom)
        .expect("valid planner parameters")
        .with_centre(true)
        .plan()
}

#[test]
fn pyramid_plan_round_trips_through_json() {
    let plan = sample_plan();
    assert!(plan.levels.len() > 1, "sample plan should be multi-level");

    let json = serde_json::to_string(&plan).expect("serialize PyramidPlan");
    let back: PyramidPlan = serde_json::from_str(&json).expect("deserialize PyramidPlan");

    assert_eq!(back, plan);
}

#[test]
fn plan_leaf_types_round_trip_through_json() {
    let coord = TileCoord {
        level: 7,
        col: 3,
        row: 11,
    };
    let json = serde_json::to_string(&coord).expect("serialize TileCoord");
    let back: TileCoord = serde_json::from_str(&json).expect("deserialize TileCoord");
    assert_eq!(back, coord);

    let rect = TileRect {
        x: 253,
        y: 507,
        width: 256,
        height: 129,
    };
    let json = serde_json::to_string(&rect).expect("serialize TileRect");
    let back: TileRect = serde_json::from_str(&json).expect("deserialize TileRect");
    assert_eq!(back, rect);

    for layout in [Layout::DeepZoom, Layout::Xyz, Layout::Google] {
        let json = serde_json::to_string(&layout).expect("serialize Layout");
        let back: Layout = serde_json::from_str(&json).expect("deserialize Layout");
        assert_eq!(back, layout);
    }
}

#[test]
fn engine_config_round_trips_through_json() {
    let mut config = EngineConfig::default()
        .with_concurrency(4)
        .with_buffer_size(128)
        .with_blank_tile_strategy(BlankTileStrategy::PlaceholderWithTolerance {
            max_channel_delta: 3,
        })
        .with_checkpoint_every(50)
        .with_dedupe_strategy(DedupeStrategy::Blanks)
        .with_source_content_hash("blake3:00ff");
    config.background_rgb = [0, 128, 255];

    let json = serde_json::to_string(&config).expect("serialize EngineConfig");
    let back: EngineConfig = serde_json::from_str(&json).expect("deserialize EngineConfig");

    assert_eq!(back.concurrency, config.concurrency);
    assert_eq!(back.buffer_size, config.buffer_size);
    assert_eq!(back.background_rgb, config.background_rgb);
    assert_eq!(back.blank_tile_strategy, config.blank_tile_strategy);
    assert_eq!(back.failure_policy, config.failure_policy);
    assert_eq!(back.checkpoint_every, config.checkpoint_every);
    assert_eq!(back.dedupe_strategy, config.dedupe_strategy);
    assert_eq!(back.checkpoint_root, config.checkpoint_root);
    assert_eq!(back.source_content_hash, config.source_content_hash);
    assert!(back.cancel.is_none());
}

#[test]
fn engine_config_cancel_token_is_never_serialized() {
    let mut config = EngineConfig::default();
    config.cancel = Some(CancelToken::new());

    let json = serde_json::to_string(&config).expect("serialize EngineConfig");
    assert!(
        !json.contains("cancel"),
        "cancel token is a process-local handle and must stay off the wire: {json}"
    );

    let back: EngineConfig = serde_json::from_str(&json).expect("deserialize EngineConfig");
    assert!(back.cancel.is_none(), "deserialized cancel must be None");
}

#[test]
fn failure_and_retry_policies_round_trip_through_json() {
    let retry = RetryPolicy::new(5, Duration::from_millis(75))
        .with_multiplier(1.5)
        .with_max_backoff(Duration::from_secs(2))
        .with_jitter(false);
    let json = serde_json::to_string(&retry).expect("serialize RetryPolicy");
    let back: RetryPolicy = serde_json::from_str(&json).expect("deserialize RetryPolicy");
    assert_eq!(back, retry);

    for policy in [
        FailurePolicy::FailFast,
        FailurePolicy::RetryThenFail(retry.clone()),
        FailurePolicy::RetryThenSkip(retry),
    ] {
        let json = serde_json::to_string(&policy).expect("serialize FailurePolicy");
        let back: FailurePolicy = serde_json::from_str(&json).expect("deserialize FailurePolicy");
        assert_eq!(back, policy);
    }
}

#[test]
fn dedupe_strategy_and_worker_id_round_trip_through_json() {
    for strategy in [
        DedupeStrategy::None,
        DedupeStrategy::Blanks,
        DedupeStrategy::All {
            algo: libviprs::ChecksumAlgo::Sha256,
        },
    ] {
        let json = serde_json::to_string(&strategy).expect("serialize DedupeStrategy");
        let back: DedupeStrategy = serde_json::from_str(&json).expect("deserialize DedupeStrategy");
        assert_eq!(back, strategy);
    }

    let worker = WorkerId::new("pool-3/agent-11");
    let json = serde_json::to_string(&worker).expect("serialize WorkerId");
    let back: WorkerId = serde_json::from_str(&json).expect("deserialize WorkerId");
    assert_eq!(back, worker);
}
