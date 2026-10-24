//! What acadrust exposes about layer state, block names and viewports, on the
//! same drawings the .NET probe reads. Prints one line per fact, in a shape that
//! diffs against `vcprobe describe`.
use acadrust::{DwgReader, EntityType};

fn main() {
    let path = std::env::args().nth(1).expect("usage: vc-acadrust-probe <dwg>");
    let mut reader = DwgReader::from_file(&path).expect("open");
    let doc = reader.read().expect("read");

    println!("layers {}", doc.layers.len());
    for l in doc.layers.iter() {
        println!(
            "  layer name={} off={} frozen={} frozenNewVp={} visible={}",
            l.name, l.is_off(), l.is_frozen(), l.flags.frozen_in_new_viewport, l.is_visible()
        );
    }

    println!("model_space");
    for e in doc.model_space_entities() {
        let c = e.common();
        let layer_visible = doc.layers.get(&c.layer).map(|l| l.is_visible());
        let kind = format!("{e:?}");
        let kind = kind.split(|ch| ch == '(' || ch == ' ' || ch == '{').next().unwrap_or("?");
        match e {
            EntityType::Insert(i) => println!(
                "  {kind} handle={:X} layer={} layer_visible={:?} block_name={}",
                c.handle.value(), c.layer, layer_visible, i.block_name
            ),
            _ => println!(
                "  {kind} handle={:X} layer={} layer_visible={:?}",
                c.handle.value(), c.layer, layer_visible
            ),
        }
    }

    println!("blocks");
    for b in doc.block_records.iter() {
        println!(
            "  block name={} anonymous={} xref={} xrefPath={} entities={}",
            b.name, b.flags.anonymous, b.flags.is_xref, b.xref_path, b.entity_handles.len()
        );
        for h in &b.entity_handles {
            if let Some(EntityType::Viewport(v)) = doc.get_entity(*h) {
                println!(
                    "    viewport handle={:X} id={} center=({},{}) w={} h={} viewCenter=({},{}) viewHeight={} twist={} frozenInVp={} clipBoundary={:X}",
                    v.common.handle.value(), v.id, v.center.x, v.center.y, v.width, v.height,
                    v.view_center.x, v.view_center.y, v.view_height, v.twist_angle,
                    v.frozen_layers.len(), v.clip_boundary_handle.value()
                );
            }
        }
    }
}
