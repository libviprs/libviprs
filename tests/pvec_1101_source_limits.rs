//! Issue #1101 (N3.6): source validation and resource limits.
//!
//! One test per acceptance criterion. A limit that trips must be a typed error naming the limit
//! and the observed value, never a panic and never a truncated result.

use libviprs::pvec::{
    ArrayGrid, Limit, LimitError, ProvenanceStub, RawEntity, RawGeometry, RawVertex, SourceFact,
    SourceGuard, SourceLimits,
};

fn stub(layer: &str) -> ProvenanceStub {
    ProvenanceStub {
        handle: None,
        class_name: "TEST".to_string(),
        block_path: Default::default(),
        layer: layer.to_string(),
    }
}

fn entity(geometry: RawGeometry) -> RawEntity {
    RawEntity {
        style: libviprs::pvec::EntityStyle::absent(),
        geometry,
        provenance: stub("0"),
    }
}

fn line() -> RawEntity {
    entity(RawGeometry::Line {
        start: [0.0; 3],
        end: [1.0, 1.0, 0.0],
    })
}

fn polyline(n: usize) -> RawEntity {
    entity(RawGeometry::Polyline {
        vertices: vec![
            RawVertex {
                x: 0.0,
                y: 0.0,
                bulge: 0.0
            };
            n
        ],
        closed: false,
        elevation: 0.0,
        normal: [0.0, 0.0, 1.0],
    })
}

fn spline(ctrl: usize, knots: usize, weights: usize) -> RawEntity {
    entity(RawGeometry::Spline {
        degree: 3,
        control_points: vec![[0.0; 3]; ctrl],
        knots: SourceFact::Present(vec![0.0; knots]),
        weights: SourceFact::Present(vec![1.0; weights]),
        periodic: SourceFact::Absent,
        closed: false,
    })
}

fn grid(rows: u32, columns: u32) -> ArrayGrid {
    ArrayGrid {
        rows,
        columns,
        row_spacing: 1.0,
        column_spacing: 1.0,
    }
}

fn tight() -> SourceLimits {
    let mut l = SourceLimits::default();
    l.max_block_depth = 3;
    l.max_entities = 5;
    l.max_polyline_vertices = 10;
    l.max_spline_control_points = 10;
    l.max_spline_knots = 14;
    l.max_insert_grid = 100;
    l.max_expanded_instances = 1000;
    l.max_allocation_bytes = 1 << 20;
    l.max_name_bytes = 8;
    l
}

fn exceeded(e: LimitError) -> (Limit, u64, u64) {
    match e {
        LimitError::Exceeded {
            limit,
            observed,
            ceiling,
        } => (limit, observed, ceiling),
        other => panic!("expected Exceeded, got {other:?}"),
    }
}

#[test]
fn defaults_are_the_documented_values() {
    let d = SourceLimits::default();
    assert_eq!(d.max_block_depth, 64);
    assert_eq!(d.max_entities, 20_000_000);
    assert_eq!(d.max_polyline_vertices, 1 << 20);
    assert_eq!(d.max_spline_control_points, 1 << 20);
    assert_eq!(d.max_spline_knots, (1 << 20) + 64);
    assert_eq!(d.max_insert_grid, 1_000_000);
    assert_eq!(d.max_expanded_instances, 10_000_000);
    assert_eq!(d.max_allocation_bytes, 2 << 30);
    assert_eq!(d.max_name_bytes, 1020);
}

#[test]
fn block_depth_breach_names_the_limit_and_the_value() {
    let mut g = SourceGuard::new(tight());
    for n in ["A", "B", "C"] {
        g.enter_block(n, ArrayGrid::SINGLE).unwrap();
    }
    let e = g.enter_block("D", ArrayGrid::SINGLE).unwrap_err();
    assert_eq!(exceeded(e.clone()), (Limit::BlockDepth, 4, 3));
    assert!(e.to_string().contains("max_block_depth"), "{e}");
    assert!(e.to_string().contains('4'), "{e}");
    // Nothing was pushed for the refused entry.
    assert_eq!(g.depth(), 3);
}

#[test]
fn entity_count_breach() {
    let mut g = SourceGuard::new(tight());
    for _ in 0..5 {
        g.check_entity(&line()).unwrap();
    }
    let e = g.check_entity(&line()).unwrap_err();
    assert_eq!(exceeded(e), (Limit::Entities, 6, 5));
}

#[test]
fn polyline_vertex_breach() {
    let mut g = SourceGuard::new(tight());
    g.check_entity(&polyline(10)).unwrap();
    let e = g.check_entity(&polyline(11)).unwrap_err();
    assert_eq!(exceeded(e), (Limit::PolylineVertices, 11, 10));
}

#[test]
fn spline_control_point_knot_and_weight_breaches() {
    let mut g = SourceGuard::new(tight());
    g.check_entity(&spline(10, 14, 10)).unwrap();
    let e = g.check_entity(&spline(11, 4, 0)).unwrap_err();
    assert_eq!(exceeded(e), (Limit::SplineControlPoints, 11, 10));
    let e = g.check_entity(&spline(4, 15, 0)).unwrap_err();
    assert_eq!(exceeded(e), (Limit::SplineKnots, 15, 14));
    let e = g.check_entity(&spline(4, 8, 15)).unwrap_err();
    assert_eq!(exceeded(e), (Limit::SplineWeights, 15, 14));
}

#[test]
fn insert_grid_breach() {
    let mut g = SourceGuard::new(tight());
    g.enter_block("OK", grid(10, 10)).unwrap();
    g.exit_block();
    let e = g.enter_block("BIG", grid(101, 1)).unwrap_err();
    assert_eq!(exceeded(e), (Limit::InsertGrid, 101, 100));
    // The huge file-format grid does not overflow anything on the way.
    let e = g.enter_block("HUGE", grid(32767, 32767)).unwrap_err();
    assert_eq!(exceeded(e), (Limit::InsertGrid, 32767 * 32767, 100));
}

#[test]
fn nesting_that_is_legal_at_every_level_is_bounded_together() {
    // Each INSERT is 10 x 10 = 100, under the grid ceiling, and the depth is 2, under the
    // depth ceiling, but placing the inner grid in every instance of the outer one is
    // 100 + 100 * 100 = 10100 instances, over the 1000 ceiling.
    let mut l = tight();
    l.max_block_depth = 8;
    let mut g = SourceGuard::new(l);
    g.enter_block("A", grid(10, 10)).unwrap();
    let e = g.enter_block("B", grid(10, 10)).unwrap_err();
    let (limit, observed, ceiling) = exceeded(e);
    assert_eq!(limit, Limit::ExpandedInstances);
    assert_eq!((observed, ceiling), (10100, 1000));
}

#[test]
fn expanded_instances_saturate_instead_of_overflowing() {
    let mut l = SourceLimits::raised();
    l.max_expanded_instances = u64::MAX - 1;
    let mut g = SourceGuard::new(l);
    let mut last = Ok(());
    for i in 0..40 {
        last = g.enter_block(&format!("B{i}"), grid(u32::MAX, u32::MAX));
        if last.is_err() {
            break;
        }
    }
    let (limit, _, _) = exceeded(last.unwrap_err());
    assert_eq!(limit, Limit::ExpandedInstances);
}

#[test]
fn allocation_ceiling_is_cumulative_across_entities() {
    let mut l = SourceLimits::default();
    l.max_allocation_bytes = 100_000;
    let mut g = SourceGuard::new(l);
    let mut n = 0;
    let err = loop {
        match g.check_entity(&polyline(1000)) {
            Ok(()) => n += 1,
            Err(e) => break e,
        }
        assert!(n < 1000, "the ceiling never tripped");
    };
    let (limit, observed, ceiling) = exceeded(err);
    assert_eq!(limit, Limit::AllocationBytes);
    assert_eq!(ceiling, 100_000);
    assert!(observed > ceiling);
    assert!(n >= 1, "a 1000 vertex polyline fits in 100 kB");
}

#[test]
fn explicit_charge_counts_against_the_same_ceiling() {
    let mut l = SourceLimits::default();
    l.max_allocation_bytes = 100;
    let mut g = SourceGuard::new(l);
    g.charge(60).unwrap();
    let e = g.charge(60).unwrap_err();
    assert_eq!(exceeded(e), (Limit::AllocationBytes, 120, 100));
    assert!(g.charge(u64::MAX).is_err(), "no overflow on a huge charge");
}

#[test]
fn name_lengths_are_checked_for_layer_block_and_class() {
    let g = SourceGuard::new(tight());
    g.check_name("12345678").unwrap();
    let e = g.check_name("123456789").unwrap_err();
    assert_eq!(exceeded(e), (Limit::NameBytes, 9, 8));

    let mut g = SourceGuard::new(tight());
    let e = g
        .enter_block("a_very_long_block_name", ArrayGrid::SINGLE)
        .unwrap_err();
    assert_eq!(exceeded(e).0, Limit::NameBytes);

    let mut ent = line();
    ent.provenance.layer = "a_very_long_layer_name".to_string();
    let e = g.check_entity(&ent).unwrap_err();
    assert_eq!(exceeded(e).0, Limit::NameBytes);

    let mut ent = line();
    ent.provenance.class_name = "a_very_long_class_name".to_string();
    assert_eq!(
        exceeded(g.check_entity(&ent).unwrap_err()).0,
        Limit::NameBytes
    );
}

#[test]
fn self_referential_block_ends_in_a_cycle_diagnostic_not_the_depth_limit() {
    // Depth is effectively unlimited, so only the detector can stop this.
    let mut g = SourceGuard::new(SourceLimits::raised());
    g.enter_block("LOOP", ArrayGrid::SINGLE).unwrap();
    let e = g.enter_block("LOOP", ArrayGrid::SINGLE).unwrap_err();
    match &e {
        LimitError::BlockCycle { block, chain } => {
            assert_eq!(block, "LOOP");
            assert_eq!(chain, &["LOOP", "LOOP"]);
        }
        other => panic!("expected BlockCycle, got {other:?}"),
    }
    assert!(e.to_string().contains("LOOP -> LOOP"), "{e}");
}

#[test]
fn indirect_cycle_is_found_and_names_the_chain() {
    let mut g = SourceGuard::new(SourceLimits::raised());
    for n in ["A", "B", "C"] {
        g.enter_block(n, ArrayGrid::SINGLE).unwrap();
    }
    let e = g.enter_block("b", ArrayGrid::SINGLE).unwrap_err();
    match e {
        LimitError::BlockCycle { chain, .. } => {
            // Block names are case-insensitive in a drawing, and the chain starts at the
            // first occurrence.
            assert_eq!(chain, ["B", "C", "b"]);
        }
        other => panic!("expected BlockCycle, got {other:?}"),
    }
}

#[test]
fn a_block_may_be_entered_twice_when_it_is_not_nested_in_itself() {
    let mut g = SourceGuard::new(SourceLimits::default());
    g.enter_block("A", ArrayGrid::SINGLE).unwrap();
    g.exit_block();
    g.enter_block("A", ArrayGrid::SINGLE).unwrap();
    g.enter_block("B", ArrayGrid::SINGLE).unwrap();
    g.exit_block();
    g.enter_block("C", ArrayGrid::SINGLE).unwrap();
    assert_eq!(g.depth(), 2);
}

#[test]
fn exit_with_nothing_entered_is_harmless() {
    let mut g = SourceGuard::new(SourceLimits::default());
    g.exit_block();
    assert_eq!(g.depth(), 0);
}

#[test]
fn a_breach_is_an_error_never_a_silent_pass() {
    // The entity that trips a limit is not counted as accepted: asking again fails again.
    let mut g = SourceGuard::new(tight());
    assert!(g.check_entity(&polyline(11)).is_err());
    assert!(g.check_entity(&polyline(11)).is_err());
}

fn walk(limits: SourceLimits) -> Vec<String> {
    let mut g = SourceGuard::new(limits);
    let mut out = Vec::new();
    g.enter_block("SYM", grid(2, 3)).unwrap();
    for e in [line(), polyline(40), spline(8, 12, 8)] {
        g.check_entity(&e).unwrap();
        out.push(format!("{:?}", e.geometry));
    }
    g.exit_block();
    out.push(format!(
        "{} {} {}",
        g.entities_seen(),
        g.expanded_instances(),
        g.depth()
    ));
    out
}

#[test]
fn input_inside_every_limit_is_unaffected_by_them() {
    let defaults = walk(SourceLimits::default());
    let raised = walk(SourceLimits::raised());
    assert_eq!(defaults, raised);
    assert_eq!(defaults.last().unwrap(), "3 6 0");
}

#[test]
fn counters_report_what_was_seen() {
    let mut g = SourceGuard::new(SourceLimits::default());
    assert_eq!(g.allocated_bytes(), 0);
    g.check_entity(&polyline(100)).unwrap();
    assert_eq!(g.entities_seen(), 1);
    assert!(g.allocated_bytes() >= 100 * std::mem::size_of::<RawVertex>() as u64);
}

#[test]
fn limits_are_a_plain_copyable_value_a_caller_can_override() {
    let mut l = SourceLimits::default();
    l.max_entities = 7;
    let copy = l;
    assert_eq!(copy.max_entities, 7);
    assert_ne!(copy, SourceLimits::default());
    assert_eq!(SourceGuard::new(copy).limits().max_entities, 7);
}
