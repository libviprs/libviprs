//! The debug and inspection export (issue #1103, N3.8; format in `docs/normalized-dump.md`).
//!
//! A human-readable, line-oriented dump of [`Normalized2DGeometry`] records. It is produced at the
//! normalization stage, before and independently of any PVec serialization, so a conversion that
//! dies while writing a `.pvec` still leaves a dump of everything normalized up to that point:
//! [`NormalizedDump`] writes each entity into its sink the moment it is pushed.
//!
//! This is inspection output. Nothing reads it back, and nothing here should grow a loader: a
//! second serialization of the geometry is exactly what the export is not for.
//!
//! Numbers print through `f64`'s shortest round-trip rendering, so every printed value reparses to
//! the identical bits. The entity ID is the file-local ID N3.7 assigns when the caller passes it
//! ([`NormalizedDump::push_with_id`]), and otherwise the position in push order.

use std::fmt;

use pvec_geometry::{Curve, Point2};

use super::normalized::Normalized2DGeometry;
use super::ordering::EntityId;
use super::provenance::{Provenance, ProvenanceStore};

/// The format version on the dump's first line.
pub const DUMP_VERSION: u32 = 1;

/// A streaming writer over normalized entities.
///
/// Every [`push`](Self::push) writes straight into the sink, so dropping the writer without
/// [`finish`](Self::finish) (a failed conversion) leaves a dump with no end line and every entity
/// that was pushed before the failure.
#[derive(Debug)]
pub struct NormalizedDump<W: fmt::Write> {
    out: W,
    count: u64,
}

impl<W: fmt::Write> NormalizedDump<W> {
    /// Starts a dump by writing the header.
    pub fn new(mut out: W) -> Result<Self, fmt::Error> {
        writeln!(out, "# libviprs normalized dump v{DUMP_VERSION}")?;
        writeln!(
            out,
            "# inspection output, not an interchange format, nothing reads it back"
        )?;
        Ok(NormalizedDump { out, count: 0 })
    }

    /// Writes one entity, with its provenance when the caller has it, and returns its ID. The ID
    /// is the position in push order; use [`push_with_id`](Self::push_with_id) to print the
    /// file-local ID N3.7 assigned instead.
    pub fn push(
        &mut self,
        entity: &Normalized2DGeometry,
        provenance: Option<&Provenance>,
    ) -> Result<u64, fmt::Error> {
        self.write_entity(self.count, entity, provenance)
    }

    /// Like [`push`](Self::push), printing the file-local ID from `order_entities` (N3.7), so a
    /// dump line and a conversion report line name the same entity.
    pub fn push_with_id(
        &mut self,
        id: EntityId,
        entity: &Normalized2DGeometry,
        provenance: Option<&Provenance>,
    ) -> Result<u64, fmt::Error> {
        self.write_entity(id.get(), entity, provenance)
    }

    fn write_entity(
        &mut self,
        id: u64,
        entity: &Normalized2DGeometry,
        provenance: Option<&Provenance>,
    ) -> Result<u64, fmt::Error> {
        let kind = entity.geometry().map_or("none", kind_tag);
        writeln!(self.out, "entity {id} {kind} {}", entity.fidelity().tag())?;
        self.out.write_str("  geometry ")?;
        match entity.geometry() {
            Some(c) => write_curve(&mut self.out, c)?,
            None => self.out.write_str("none")?,
        }
        self.out.write_char('\n')?;
        match entity.source_bounds() {
            Some(b) => {
                self.out.write_str("  bounds min=")?;
                write_point(&mut self.out, b.min())?;
                self.out.write_str(" max=")?;
                write_point(&mut self.out, b.max())?;
                self.out.write_char('\n')?;
            }
            None => self.out.write_str("  bounds none\n")?,
        }
        if let Some(p) = provenance {
            self.out.write_str("  provenance ")?;
            p.write_report(&mut self.out)?;
            self.out.write_char('\n')?;
        }
        self.count += 1;
        Ok(id)
    }

    /// Like [`push`](Self::push), resolving the entity's provenance handle against `store`. A
    /// record with no handle, or a handle the store did not issue, prints no provenance line.
    pub fn push_stored(
        &mut self,
        entity: &Normalized2DGeometry,
        store: &ProvenanceStore,
    ) -> Result<u64, fmt::Error> {
        self.push(entity, entity.provenance().and_then(|h| store.get(h)))
    }

    /// How many entities have been written.
    pub fn entities_written(&self) -> u64 {
        self.count
    }

    /// Writes the end line and hands the sink back.
    pub fn finish(mut self) -> Result<W, fmt::Error> {
        writeln!(self.out, "end entities={}", self.count)?;
        Ok(self.out)
    }
}

/// Dumps `entities` into a string, a one-call entry point for tests and debugging.
///
/// # Panics
/// Never: writing into a `String` does not fail.
pub fn dump_to_string<'a, I>(entities: I) -> String
where
    I: IntoIterator<Item = (&'a Normalized2DGeometry, Option<&'a Provenance>)>,
{
    let mut d = NormalizedDump::new(String::new()).expect("a String sink cannot fail");
    for (e, p) in entities {
        d.push(e, p).expect("a String sink cannot fail");
    }
    d.finish().expect("a String sink cannot fail")
}

fn kind_tag(c: &Curve) -> &'static str {
    match c {
        Curve::Line(_) => "line",
        Curve::Circle(_) => "circle",
        Curve::Ellipse(_) => "ellipse",
        Curve::Polyline(_) => "polyline",
        Curve::Trimmed(_) => "trimmed",
        Curve::Composite(_) => "composite",
        Curve::Bezier(_) => "bezier",
        Curve::BSpline(_) => "bspline",
        Curve::RationalBSpline(_) => "rational-bspline",
        _ => "other",
    }
}

fn write_point<W: fmt::Write>(w: &mut W, p: Point2) -> fmt::Result {
    write!(w, "({:?},{:?})", p.x, p.y)
}

fn write_points<W: fmt::Write>(w: &mut W, key: &str, ps: &[Point2]) -> fmt::Result {
    write!(w, " {key}=[")?;
    for (i, p) in ps.iter().enumerate() {
        if i > 0 {
            w.write_char(' ')?;
        }
        write_point(w, *p)?;
    }
    w.write_char(']')
}

fn write_floats<W: fmt::Write>(w: &mut W, key: &str, vs: &[f64]) -> fmt::Result {
    write!(w, " {key}=[")?;
    for (i, v) in vs.iter().enumerate() {
        if i > 0 {
            w.write_char(' ')?;
        }
        write!(w, "{v:?}")?;
    }
    w.write_char(']')
}

/// One curve on one line, parameters as PVec holds them.
fn write_curve<W: fmt::Write>(w: &mut W, c: &Curve) -> fmt::Result {
    w.write_str(kind_tag(c))?;
    match c {
        Curve::Line(l) => {
            w.write_str(" origin=")?;
            write_point(w, l.origin())?;
            write!(
                w,
                " direction=({:?},{:?})",
                l.direction().x,
                l.direction().y
            )
        }
        Curve::Circle(c) => {
            let p = c.placement();
            w.write_str(" origin=")?;
            write_point(w, p.origin())?;
            write!(w, " x_axis=({:?},{:?})", p.x_axis().x, p.x_axis().y)?;
            write!(w, " radius={:?}", c.radius())
        }
        Curve::Ellipse(e) => {
            let p = e.placement();
            w.write_str(" origin=")?;
            write_point(w, p.origin())?;
            write!(w, " x_axis=({:?},{:?})", p.x_axis().x, p.x_axis().y)?;
            write!(
                w,
                " semi_axis_1={:?} semi_axis_2={:?}",
                e.semi_axis_1(),
                e.semi_axis_2()
            )
        }
        Curve::Polyline(p) => write_points(w, "vertices", p.vertices()),
        Curve::Trimmed(t) => {
            let trim = t.trim();
            write!(
                w,
                " start={:?} span={:?} reversed={} basis=(",
                trim.start, trim.span, trim.reversed
            )?;
            write_curve(w, t.basis())?;
            w.write_char(')')
        }
        Curve::Composite(c) => {
            write!(w, " closed={} segments=[", c.is_declared_closed())?;
            for (i, s) in c.segments().iter().enumerate() {
                if i > 0 {
                    w.write_str("; ")?;
                }
                write_curve(w, s)?;
            }
            w.write_char(']')
        }
        Curve::Bezier(b) => {
            write!(
                w,
                " degree={} closed={}",
                b.degree(),
                b.is_declared_closed()
            )?;
            write_points(w, "control_points", b.control_points())
        }
        Curve::BSpline(s) => {
            write!(w, " degree={}", s.degree())?;
            write_points(w, "control_points", s.control_points())?;
            write_floats(w, "knots", s.knots())
        }
        Curve::RationalBSpline(s) => {
            write!(w, " degree={}", s.degree())?;
            write_points(w, "control_points", s.control_points())?;
            write_floats(w, "knots", s.knots())?;
            write_floats(w, "weights", s.weights())
        }
        _ => Ok(()),
    }
}
