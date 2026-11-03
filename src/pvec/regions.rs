//! Derived room and corridor regions (issue #1093, N6.9).
//!
//! Walls in, faces out: the empty space the walls enclose is labelled room or corridor by
//! documented rules, every label says it is a guess, and nothing here changes the source geometry.

use std::collections::BTreeMap;

use pvec_capabilities::Registry;
use pvec_facade::EntityId as PvecEntityId;
use pvec_facade::components::{
    Origin, RegionKind, RegionRecord, SemanticClassStream, SourceTag, Stream,
};
use thiserror::Error;

use super::ordering::EntityId;

/// Distance by basic arithmetic and a correctly rounded `sqrt`: libm `hypot` differs in the last bit between
/// architectures, and these lengths decide which regions exist (`tests/pvec_1471_reproducibility.rs`).
fn dist(a: [f64; 2], b: [f64; 2]) -> f64 {
    let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
    (dx * dx + dy * dy).sqrt()
}

/// One wall segment, by the entity it came from.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WallEdge {
    /// The wall entity.
    pub entity: EntityId,
    /// Its start.
    pub start: [f64; 2],
    /// Its end.
    pub end: [f64; 2],
}

/// A text or block-attribute anchor that may name the room it sits in.
#[derive(Debug, Clone, PartialEq)]
pub struct RoomLabel {
    /// Where it sits.
    pub at: [f64; 2],
    /// What it says.
    pub text: String,
}

/// A room the drawing states itself: a closed boundary polyline, or a tagged block.
#[derive(Debug, Clone, PartialEq)]
pub struct ExplicitRoom {
    /// The boundary polygon, not repeating the first point.
    pub polygon: Vec<[f64; 2]>,
    /// The entities that draw it, in ring order.
    pub boundary: Vec<EntityId>,
    /// Its name, when it has one.
    pub name: Option<String>,
}

/// The knobs of the derivation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RegionConfig {
    /// The widest gap between two wall ends that is closed, in drawing units.
    pub gap_tolerance: f64,
    /// A face whose short side is at most this is a corridor candidate.
    pub corridor_max_width: f64,
    /// ... and whose long side is at least this many times the short one.
    pub corridor_min_aspect: f64,
    /// The most faces the step builds before it refuses.
    pub max_faces: usize,
    /// Keep faces no rule labels, as regions with no kind. Off omits them.
    pub keep_unlabelled: bool,
}

/// What a region boundary refers to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundaryRef {
    /// A source wall entity.
    Source(EntityId),
    /// The `n`th gap-closing segment of [`Regions::derived`].
    Derived(usize),
}

/// One derived region.
#[derive(Debug, Clone, PartialEq)]
pub struct Region {
    /// Room or corridor, `None` for a face no rule labels.
    pub kind: Option<RegionKind>,
    /// Inferred with a confidence, or explicit.
    pub origin: Origin,
    /// The ring, in order.
    pub boundary: Vec<BoundaryRef>,
    /// A name found inside the region.
    pub label: Option<String>,
}

/// A segment created by closing a gap. Never source geometry.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DerivedSegment {
    /// Start.
    pub start: [f64; 2],
    /// End.
    pub end: [f64; 2],
}

/// Counts for the conversion report.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RegionReport {
    /// Rooms found.
    pub rooms: u64,
    /// Corridors found.
    pub corridors: u64,
    /// Gaps closed.
    pub gaps_closed: u64,
    /// Faces no rule labelled.
    pub unlabelled: u64,
}

/// The result of the step.
#[derive(Debug, Clone, PartialEq)]
pub struct Regions {
    /// The regions.
    pub regions: Vec<Region>,
    /// The gap-closing segments, flagged derived by being here and not in the source geometry.
    pub derived: Vec<DerivedSegment>,
    /// The counts.
    pub report: RegionReport,
}

/// The most distinct wall ends the step takes, and the most close pairs of loose ends it keeps.
/// Finding the node for an end and pairing loose ends both scan every node, so this bounds the
/// work at about the square of it and the memory at its pairs; a plan past it is refused with
/// `regions.node-limit` rather than left to run.
pub const MAX_NODES: usize = 20_000;

/// PVec's bound on the entities in one region ring, repeated because the facade does not export it.
const MAX_RING: usize = 4_096;

/// Why the step refused.
#[derive(Debug, Error, PartialEq, Eq)]
#[non_exhaustive]
pub enum RegionsError {
    /// More faces than [`RegionConfig::max_faces`].
    #[error("the walls enclose more than {limit} faces")]
    TooManyFaces {
        /// The limit.
        limit: usize,
    },
    /// More wall ends than [`MAX_NODES`], or more close pairs of loose ends than that.
    #[error("the walls have more than {limit} distinct ends")]
    TooManyNodes {
        /// The limit.
        limit: usize,
    },
    /// A tolerance or limit that is not a usable number.
    #[error("bad region configuration: {0}")]
    BadConfig(&'static str),
}

impl RegionsError {
    /// The stable reason code.
    #[must_use]
    pub fn reason_code(&self) -> &'static str {
        match self {
            RegionsError::TooManyFaces { .. } => "regions.face-limit",
            RegionsError::TooManyNodes { .. } => "regions.node-limit",
            RegionsError::BadConfig(_) => "regions.bad-config",
        }
    }
}

type P = [f64; 2];

/// One undirected edge of the wall graph.
struct Edge {
    a: usize,
    b: usize,
    what: BoundaryRef,
}

/// Builds the regions the walls enclose.
///
/// Wall ends within a hair of each other are one node. A wall end with nothing else on it (a door
/// jamb) is joined to the nearest other loose end within [`RegionConfig::gap_tolerance`], and the
/// joining segment is a [`DerivedSegment`]. Walls that stay loose are dropped, the faces of what is
/// left are traced, and the face around each connected piece is discarded. A face is a corridor
/// when its smallest enclosing rectangle is at most [`RegionConfig::corridor_max_width`] across and
/// [`RegionConfig::corridor_min_aspect`] times longer than wide, and a room otherwise. Its
/// confidence is 0.5 plus half the margin by which it clears that test, so a face right at a
/// threshold scores about 0.5 and a clear case near 1. Walls must meet at their ends: a wall that
/// passes through a junction is not split, and an obstacle inside a room (a column) is not a hole.
///
/// # Errors
///
/// [`RegionsError`].
pub fn derive_regions(
    walls: &[WallEdge],
    labels: &[RoomLabel],
    explicit: &[ExplicitRoom],
    config: &RegionConfig,
) -> Result<Regions, RegionsError> {
    let good = |v: f64| v.is_finite() && v >= 0.0;
    if !good(config.gap_tolerance) {
        return Err(RegionsError::BadConfig("gap_tolerance"));
    }
    if !good(config.corridor_max_width) || !good(config.corridor_min_aspect) {
        return Err(RegionsError::BadConfig("corridor thresholds"));
    }
    /// Index of the node at `p`, adding it when no node is within `snap`.
    fn node_at(nodes: &mut Vec<P>, p: P, snap: f64) -> usize {
        if let Some(i) = nodes.iter().position(|q| dist(p, *q) <= snap) {
            return i;
        }
        nodes.push(p);
        nodes.len() - 1
    }

    fn signed_area(poly: &[P]) -> f64 {
        let n = poly.len();
        (0..n)
            .map(|i| {
                let (p, q) = (poly[i], poly[(i + 1) % n]);
                p[0] * q[1] - q[0] * p[1]
            })
            .sum::<f64>()
            / 2.0
    }

    fn inside(poly: &[P], pt: P) -> bool {
        let n = poly.len();
        let mut c = false;
        for i in 0..n {
            let (p, q) = (poly[i], poly[(i + 1) % n]);
            if (p[1] > pt[1]) != (q[1] > pt[1])
                && pt[0] < (q[0] - p[0]) * (pt[1] - p[1]) / (q[1] - p[1]) + p[0]
            {
                c = !c;
            }
        }
        c
    }

    /// The side lengths of the smallest rectangle, aligned to one polygon edge, that holds `poly`.
    fn min_rect(poly: &[P]) -> (f64, f64) {
        let n = poly.len();
        let mut best = (f64::INFINITY, 0.0, 0.0);
        for i in 0..n {
            let (p, q) = (poly[i], poly[(i + 1) % n]);
            let len = dist(p, q);
            if len == 0.0 {
                continue;
            }
            let (ux, uy) = ((q[0] - p[0]) / len, (q[1] - p[1]) / len);
            let (mut lo, mut hi, mut lo_v, mut hi_v) = (f64::MAX, f64::MIN, f64::MAX, f64::MIN);
            for v in poly {
                let (dx, dy) = (v[0] - p[0], v[1] - p[1]);
                let (s, t) = (dx * ux + dy * uy, -dx * uy + dy * ux);
                lo = lo.min(s);
                hi = hi.max(s);
                lo_v = lo_v.min(t);
                hi_v = hi_v.max(t);
            }
            let (w, h) = (hi - lo, hi_v - lo_v);
            if w * h < best.0 {
                best = (w * h, w, h);
            }
        }
        (best.1.min(best.2), best.1.max(best.2))
    }

    /// Whether `pt` lies on the polygon's outline.
    fn on_ring(poly: &[P], pt: P) -> bool {
        let n = poly.len();
        (0..n).any(|i| {
            let (p, q) = (poly[i], poly[(i + 1) % n]);
            let cross = (q[0] - p[0]) * (pt[1] - p[1]) - (q[1] - p[1]) * (pt[0] - p[0]);
            let len = dist(p, q);
            len > 0.0
                && cross.abs() <= 1e-9 * len
                && pt[0] >= p[0].min(q[0]) - 1e-9
                && pt[0] <= p[0].max(q[0]) + 1e-9
                && pt[1] >= p[1].min(q[1]) - 1e-9
                && pt[1] <= p[1].max(q[1]) + 1e-9
        })
    }

    let snap = 1e-9_f64.max(config.gap_tolerance * 1e-6);
    let mut nodes: Vec<P> = Vec::new();
    let mut edges: Vec<Edge> = Vec::new();
    for w in walls {
        let (a, b) = (
            node_at(&mut nodes, w.start, snap),
            node_at(&mut nodes, w.end, snap),
        );
        if nodes.len() > MAX_NODES {
            return Err(RegionsError::TooManyNodes { limit: MAX_NODES });
        }
        if a != b {
            edges.push(Edge {
                a,
                b,
                what: BoundaryRef::Source(w.entity),
            });
        }
    }
    // Close gaps between loose ends, nearest pair first, each end used once.
    let mut degree = vec![0usize; nodes.len()];
    for e in &edges {
        degree[e.a] += 1;
        degree[e.b] += 1;
    }
    let loose: Vec<usize> = (0..nodes.len()).filter(|i| degree[*i] == 1).collect();
    let mut pairs: Vec<(f64, usize, usize)> = Vec::new();
    for (k, &i) in loose.iter().enumerate() {
        for &j in &loose[k + 1..] {
            let d = dist(nodes[i], nodes[j]);
            if d <= config.gap_tolerance {
                pairs.push((d, i, j));
                if pairs.len() > MAX_NODES {
                    return Err(RegionsError::TooManyNodes { limit: MAX_NODES });
                }
            }
        }
    }
    pairs.sort_by(|x, y| x.0.total_cmp(&y.0).then(x.1.cmp(&y.1)).then(x.2.cmp(&y.2)));
    let mut used = vec![false; nodes.len()];
    let mut derived = Vec::new();
    for (_, i, j) in pairs {
        if used[i] || used[j] {
            continue;
        }
        used[i] = true;
        used[j] = true;
        edges.push(Edge {
            a: i,
            b: j,
            what: BoundaryRef::Derived(derived.len()),
        });
        derived.push(DerivedSegment {
            start: nodes[i],
            end: nodes[j],
        });
    }
    // Drop what still dangles, repeatedly: a dead-end wall encloses nothing.
    loop {
        let mut degree = vec![0usize; nodes.len()];
        for e in &edges {
            degree[e.a] += 1;
            degree[e.b] += 1;
        }
        let before = edges.len();
        edges.retain(|e| degree[e.a] > 1 && degree[e.b] > 1);
        if edges.len() == before {
            break;
        }
    }
    // Half-edges 2k (a to b) and 2k+1 (b to a), sorted by angle around each node.
    let origin = |h: usize| {
        if h.is_multiple_of(2) {
            edges[h / 2].a
        } else {
            edges[h / 2].b
        }
    };
    let target = |h: usize| {
        if h.is_multiple_of(2) {
            edges[h / 2].b
        } else {
            edges[h / 2].a
        }
    };
    let angle = |h: usize| {
        let (p, q) = (nodes[origin(h)], nodes[target(h)]);
        (q[1] - p[1]).atan2(q[0] - p[0])
    };
    let mut around: Vec<Vec<usize>> = vec![Vec::new(); nodes.len()];
    for h in 0..edges.len() * 2 {
        around[origin(h)].push(h);
    }
    for list in &mut around {
        list.sort_by(|x, y| angle(*x).total_cmp(&angle(*y)));
    }
    // The next half-edge of a face keeps the face on its left: at the target, turn to the
    // neighbour just clockwise of the reverse edge.
    let next = |h: usize| {
        let twin = h ^ 1;
        let list = &around[target(h)];
        let at = list.iter().position(|x| *x == twin).unwrap_or(0);
        list[(at + list.len() - 1) % list.len()]
    };
    let mut seen = vec![false; edges.len() * 2];
    let mut faces: Vec<(Vec<usize>, Vec<P>)> = Vec::new();
    for start in 0..edges.len() * 2 {
        if seen[start] {
            continue;
        }
        let mut ring = Vec::new();
        let mut h = start;
        while !seen[h] {
            seen[h] = true;
            ring.push(h);
            h = next(h);
        }
        let poly: Vec<P> = ring.iter().map(|h| nodes[origin(*h)]).collect();
        if signed_area(&poly) > 0.0 {
            if faces.len() >= config.max_faces {
                return Err(RegionsError::TooManyFaces {
                    limit: config.max_faces,
                });
            }
            faces.push((ring, poly));
        }
    }
    let mut report = RegionReport {
        gaps_closed: derived.len() as u64,
        ..RegionReport::default()
    };
    let mut regions = Vec::new();
    // A face is the explicit room when every midpoint of its edges is in or on the explicit polygon.
    let explicit_hit = |poly: &[P]| {
        explicit.iter().any(|x| {
            poly.iter().zip(poly.iter().cycle().skip(1)).all(|(p, q)| {
                let m = [(p[0] + q[0]) / 2.0, (p[1] + q[1]) / 2.0];
                inside(&x.polygon, m) || on_ring(&x.polygon, m)
            })
        })
    };
    for (ring, poly) in &faces {
        if explicit_hit(poly) {
            continue;
        }
        let area = signed_area(poly);
        let (short, long) = min_rect(poly);
        let corridor = short > 0.0
            && short <= config.corridor_max_width
            && long / short >= config.corridor_min_aspect;
        let kind = if area > 0.0 && short > 0.0 {
            Some(if corridor {
                RegionKind::Corridor
            } else {
                RegionKind::Room
            })
        } else {
            None
        };
        let margin = if corridor {
            ((long / short) / config.corridor_min_aspect.max(f64::MIN_POSITIVE) - 1.0).min(1.0)
        } else if config.corridor_max_width > 0.0 {
            ((short / config.corridor_max_width) - 1.0).clamp(0.0, 1.0)
        } else {
            1.0
        };
        let label = labels
            .iter()
            .find(|l| inside(poly, l.at))
            .map(|l| l.text.clone());
        match kind {
            Some(RegionKind::Corridor) => report.corridors += 1,
            Some(RegionKind::Room) => report.rooms += 1,
            _ => {
                report.unlabelled += 1;
                if !config.keep_unlabelled {
                    continue;
                }
            }
        }
        regions.push(Region {
            kind,
            origin: Origin {
                source: SourceTag::Inferred,
                confidence: Some((0.5 + margin.max(0.0) / 2.0).clamp(0.0, 1.0)),
            },
            boundary: ring.iter().map(|h| edges[h / 2].what).collect(),
            label,
        });
    }
    for x in explicit {
        report.rooms += 1;
        regions.push(Region {
            kind: Some(RegionKind::Room),
            origin: Origin {
                source: SourceTag::Explicit,
                confidence: None,
            },
            boundary: x.boundary.iter().map(|e| BoundaryRef::Source(*e)).collect(),
            label: x.name.clone(),
        });
    }
    Ok(Regions {
        regions,
        derived,
        report,
    })
}

/// A region left out of the semantic class stream, and why.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DroppedRegion {
    /// Its position in [`Regions::regions`].
    pub region: usize,
    /// `unlabelled` (no kind), `derived-boundary`, `unwritten-boundary`, `no-boundary` or
    /// `ring-too-long`.
    pub reason: &'static str,
}

/// The semantic class stream and the regions it could not hold.
#[derive(Debug)]
pub struct SemanticStream {
    /// The stream, or `None` when no region is left to write.
    pub stream: Option<Stream>,
    /// Every region left out, for the conversion report.
    pub dropped: Vec<DroppedRegion>,
}

/// The semantic class stream for `regions`: PVec entity `i` is `source_ids[i]`.
///
/// A region is left out, and listed in [`SemanticStream::dropped`], when it has no kind (the
/// unlabelled ones [`RegionConfig::keep_unlabelled`] keeps in `regions` are not a class PVec can
/// name), when its ring is empty or longer than PVec's 4096 references, when it points at an
/// entity that was not written, or when it runs along a [`BoundaryRef::Derived`] segment: derived
/// segments are not written as geometry, so a reference to one would name an entity the artifact
/// does not have. When nothing is left no stream is emitted.
///
/// # Errors
///
/// A message when PVec's encoder refuses the payload.
pub fn semantic_stream(
    regions: &Regions,
    source_ids: &[EntityId],
) -> Result<SemanticStream, String> {
    let pos: BTreeMap<EntityId, usize> = source_ids
        .iter()
        .enumerate()
        .map(|(n, e)| (*e, n))
        .collect();
    let mut dropped = Vec::new();
    let mut records = Vec::new();
    'regions: for (n, r) in regions.regions.iter().enumerate() {
        let mut skip = |reason| dropped.push(DroppedRegion { region: n, reason });
        let Some(kind) = r.kind else {
            skip("unlabelled");
            continue;
        };
        if r.boundary.is_empty() {
            skip("no-boundary");
            continue;
        }
        if r.boundary.len() > MAX_RING {
            skip("ring-too-long");
            continue;
        }
        let mut outer = Vec::with_capacity(r.boundary.len());
        for b in &r.boundary {
            match b {
                BoundaryRef::Source(e) => {
                    let id = pos
                        .get(e)
                        .and_then(|p| u32::try_from(*p).ok())
                        .and_then(|p| PvecEntityId::new(p).ok());
                    let Some(id) = id else {
                        skip("unwritten-boundary");
                        continue 'regions;
                    };
                    outer.push(id);
                }
                BoundaryRef::Derived(_) => {
                    skip("derived-boundary");
                    continue 'regions;
                }
            }
        }
        records.push(RegionRecord {
            kind,
            origin: r.origin,
            outer,
            holes: Vec::new(),
        });
    }
    if records.is_empty() {
        return Ok(SemanticStream {
            stream: None,
            dropped,
        });
    }
    let stream = SemanticClassStream {
        elements: Vec::new(),
        regions: records,
    }
    .into_stream(&Registry::v1())
    .map_err(|e| e.to_string())?;
    Ok(SemanticStream {
        stream: Some(stream),
        dropped,
    })
}
