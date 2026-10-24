use acadrust::entities::hatch::{BoundaryEdge, BoundaryPath, PolylineEdge};
use acadrust::entities::*;
use acadrust::tables::Layer;
use acadrust::types::{DxfVersion, Vector2, Vector3};
use acadrust::{CadDocument, DwgReader, DwgWriter, DxfWriter};
use std::collections::BTreeMap;

fn put(doc: &mut CadDocument, only: Option<&str>, layer: &str, mut e: EntityType) {
    if only.map_or(false, |c| c != class(&e)) {
        return;
    }
    e.common_mut().layer = layer.into();
    doc.add_entity(e).expect("add_entity");
}

fn build(only: Option<&str>) -> CadDocument {
    let mut doc = CadDocument::with_version(DxfVersion::AC1032);
    for (n, c) in [("WALLS", 1i16), ("ARCS", 3), ("ANNO", 5)] {
        let mut l = Layer::new(n);
        l.color = acadrust::types::Color::from_index(c);
        doc.layers.add(l).unwrap();
    }
    // lines
    for i in 0..6 {
        let y = i as f64 * 10.0;
        put(&mut doc, only, "WALLS", EntityType::Line(Line::from_coords(0.0, y, 0.0, 100.0, y + 5.0, 0.0)));
    }
    // lwpolylines: 2 closed rectangles (4 vertices) and 2 open (5 vertices)
    for k in 0..2 {
        let o = 120.0 + k as f64 * 40.0;
        let mut p = LwPolyline::from_points(vec![
            Vector2::new(o, 0.0),
            Vector2::new(o + 30.0, 0.0),
            Vector2::new(o + 30.0, 20.0),
            Vector2::new(o, 20.0),
        ]);
        p.is_closed = true;
        put(&mut doc, only, "WALLS", EntityType::LwPolyline(p));
    }
    for k in 0..2 {
        let o = k as f64 * 30.0;
        put(
            &mut doc,
            only,
            "WALLS",
            EntityType::LwPolyline(LwPolyline::from_points(vec![
                Vector2::new(o, 80.0),
                Vector2::new(o + 10.0, 90.0),
                Vector2::new(o + 20.0, 80.0),
                Vector2::new(o + 25.0, 95.0),
                Vector2::new(o + 28.0, 82.0),
            ])),
        );
    }
    // arcs
    for i in 0..5 {
        put(
            &mut doc,
            only,
            "ARCS",
            EntityType::Arc(Arc::from_coords(
                20.0 + i as f64 * 30.0,
                140.0,
                0.0,
                10.0 + i as f64,
                0.2 * i as f64,
                std::f64::consts::PI * (0.5 + 0.3 * i as f64),
            )),
        );
    }
    // circles
    for i in 0..4 {
        put(&mut doc, only, "ARCS", EntityType::Circle(Circle::from_coords(20.0 + i as f64 * 30.0, 180.0, 0.0, 8.0 + i as f64)));
    }
    // ellipses: 2 full, 1 partial
    for i in 0..3 {
        let mut e = Ellipse::from_center_axes(Vector3::new(30.0 + i as f64 * 50.0, 220.0, 0.0), Vector3::new(20.0, 5.0 * i as f64, 0.0), 0.5);
        if i == 2 {
            e.start_parameter = 0.0;
            e.end_parameter = std::f64::consts::PI;
        }
        put(&mut doc, only, "ARCS", EntityType::Ellipse(e));
    }
    // splines
    for i in 0..3 {
        let o = i as f64 * 50.0;
        put(
            &mut doc,
            only,
            "ARCS",
            EntityType::Spline(Spline::from_control_points(
                3,
                vec![
                    Vector3::new(o, 260.0, 0.0),
                    Vector3::new(o + 10.0, 280.0, 0.0),
                    Vector3::new(o + 20.0, 250.0, 0.0),
                    Vector3::new(o + 30.0, 280.0, 0.0),
                    Vector3::new(o + 40.0, 260.0, 0.0),
                ],
            )),
        );
    }
    // text
    for (i, s) in ["ROOM 101", "CORRIDOR", "STAIR A", "NOTE 7"].iter().enumerate() {
        let mut t = Text::with_value(*s, Vector3::new(5.0 + i as f64 * 40.0, 300.0, 0.0));
        t.height = 5.0;
        put(&mut doc, only, "ANNO", EntityType::Text(t));
    }
    // solid hatches
    for i in 0..2 {
        let o = 200.0 + i as f64 * 40.0;
        let mut h = Hatch::solid();
        let mut path = BoundaryPath::new();
        path.add_edge(BoundaryEdge::Polyline(PolylineEdge::new(
            vec![Vector2::new(o, 100.0), Vector2::new(o + 30.0, 100.0), Vector2::new(o + 30.0, 130.0), Vector2::new(o, 130.0)],
            true,
        )));
        h.add_path(path);
        put(&mut doc, only, "WALLS", EntityType::Hatch(h));
    }
    doc
}

fn class(e: &EntityType) -> &'static str {
    match e {
        EntityType::Line(_) => "line",
        EntityType::LwPolyline(_) | EntityType::Polyline2D(_) | EntityType::Polyline3D(_) => "polyline",
        EntityType::Arc(_) => "arc",
        EntityType::Circle(_) => "circle",
        EntityType::Ellipse(_) => "ellipse",
        EntityType::Spline(_) => "spline",
        EntityType::Text(_) | EntityType::MText(_) => "text",
        EntityType::Hatch(_) => "hatch",
        _ => "other",
    }
}

fn main() {
    let out = std::env::args().nth(1).expect("usage: pair-gen <outdir>");
    let dwg = format!("{out}/pair.dwg");
    let dxf = format!("{out}/pair.dxf");
    DwgWriter::write_to_file(&dwg, &build(None)).expect("write dwg");
    // One DWG and DXF per class, so the PDF side can be measured per class and not attributed by guesswork.
    for c in ["line", "polyline", "arc", "circle", "ellipse", "spline", "text", "hatch"] {
        let p = format!("{out}/class_{c}.dwg");
        DwgWriter::write_to_file(&p, &build(Some(c))).expect("write class dwg");
        let b = DwgReader::from_file(&p).expect("open class dwg").read().expect("read class dwg");
        DxfWriter::new(&b).write_to_file(format!("{out}/class_{c}.dxf")).expect("write class dxf");
    }
    // The DXF is made from the DWG as read back, so the PDF is an export of the DWG's content.
    let back = DwgReader::from_file(&dwg).expect("open dwg").read().expect("read dwg");
    DxfWriter::new(&back).write_to_file(&dxf).expect("write dxf");
    let mut n: BTreeMap<&str, usize> = BTreeMap::new();
    for c in ["line", "polyline", "arc", "circle", "ellipse", "spline", "text", "hatch", "other"] {
        n.insert(c, 0);
    }
    let mut layers: BTreeMap<String, usize> = BTreeMap::new();
    for e in back.entities() {
        *n.get_mut(class(e)).unwrap() += 1;
        *layers.entry(e.common().layer.clone()).or_default() += 1;
    }
    println!("fixture\tkey\tvalue");
    for (c, v) in &n {
        println!("pair.dwg\tdwg_{c}\t{v}");
    }
    for (l, v) in &layers {
        println!("pair.dwg\tdwg_layer_{l}\t{v}");
    }
    let bytes = std::fs::read(&dwg).unwrap();
    println!("pair.dwg\tdwg_version\t{}", String::from_utf8_lossy(&bytes[..6]));
}
