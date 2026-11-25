//! Issue #1705, the hostile blob: an OLE2FRAME whose compound file is crafted to make a reader
//! allocate or loop (a DIFAT that points at itself, a FAT chain that loops, a stream that declares
//! 4 GiB). acadrust's compound-file reader had no bound on these (libviprs/acadrust#75, fixed in the
//! fork's #76), so a few KB in a drawing cost hundreds of MB. The frame has to come out as a row that
//! says it is not drawn, within a small peak of live heap. Needs `--features pvec-convert`.

#![cfg(feature = "pvec-convert")]

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use acadrust::compound_file::BinaryRecord;
use acadrust::entities::{Line, Ole2Frame, OleFrameEnvelope, OleObjectType};
use acadrust::objects::ObjectType;
use acadrust::types::Vector3;
use acadrust::{CadDocument, DwgWriter, EntityType};
use libviprs::cad::CadSource;
use libviprs::pvec::ViewProfile;
use libviprs::pvec_convert::PvecConversion;

struct Counting;
static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

// SAFETY: every call is forwarded to the system allocator unchanged; only counters are added.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        let p = unsafe { System.alloc(l) };
        if !p.is_null() {
            let now = LIVE.fetch_add(l.size(), Ordering::Relaxed) + l.size();
            PEAK.fetch_max(now, Ordering::Relaxed);
        }
        p
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        LIVE.fetch_sub(l.size(), Ordering::Relaxed);
        unsafe { System.dealloc(p, l) }
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, new: usize) -> *mut u8 {
        let q = unsafe { System.realloc(p, l, new) };
        if !q.is_null() {
            if new >= l.size() {
                let now = LIVE.fetch_add(new - l.size(), Ordering::Relaxed) + new - l.size();
                PEAK.fetch_max(now, Ordering::Relaxed);
            } else {
                LIVE.fetch_sub(l.size() - new, Ordering::Relaxed);
            }
        }
        q
    }
}

/// The peak counter is global, so the two tests take turns.
static TURN: Mutex<()> = Mutex::new(());

#[global_allocator]
static ALLOC: Counting = Counting;

fn put(buf: &mut [u8], at: usize, v: &[u8]) {
    buf[at..at + v.len()].copy_from_slice(v);
}

/// A compound file of two sectors (a FAT and a directory) that is hostile three ways: the header's
/// DIFAT chain starts at the FAT sector whose last entry points back at itself, the directory's
/// FAT chain loops, and its stream declares 4 GiB with a start sector past the file.
fn hostile_cfb() -> Vec<u8> {
    let mut b = vec![0u8; 512 + 1024];
    put(&mut b, 0, &[0xD0, 0xCF, 0x11, 0xE0, 0xA1, 0xB1, 0x1A, 0xE1]);
    put(&mut b, 0x18, &0x003Eu16.to_le_bytes());
    put(&mut b, 0x1A, &3u16.to_le_bytes());
    put(&mut b, 0x1C, &0xFFFEu16.to_le_bytes());
    put(&mut b, 0x1E, &9u16.to_le_bytes());
    put(&mut b, 0x20, &6u16.to_le_bytes());
    put(&mut b, 0x2C, &1u32.to_le_bytes());
    put(&mut b, 0x30, &1u32.to_le_bytes());
    put(&mut b, 0x38, &4096u32.to_le_bytes());
    put(&mut b, 0x3C, &0xFFFF_FFFEu32.to_le_bytes());
    put(&mut b, 0x44, &0u32.to_le_bytes());
    put(&mut b, 0x48, &4096u32.to_le_bytes());
    for i in 0..109 {
        put(
            &mut b,
            0x4C + i * 4,
            &(if i == 0 { 0u32 } else { 0xFFFF_FFFF }).to_le_bytes(),
        );
    }
    // Sector 0: the FAT. Sector 1 chains to itself, and the last entry (the DIFAT's next pointer)
    // names sector 0 again.
    for i in 0..128 {
        put(&mut b, 512 + i * 4, &0xFFFF_FFFFu32.to_le_bytes());
    }
    put(&mut b, 512, &0xFFFF_FFFDu32.to_le_bytes());
    put(&mut b, 512 + 4, &1u32.to_le_bytes());
    put(&mut b, 512 + 127 * 4, &0u32.to_le_bytes());
    // Sector 1: the directory. A root, and a stream that declares 4 GiB.
    let dir = 512 + 512;
    let name = |at: usize, s: &str| {
        let units: Vec<u8> = s.encode_utf16().flat_map(u16::to_le_bytes).collect();
        (at, units)
    };
    let (n0, u0) = name(dir, "Root Entry");
    put(&mut b, n0, &u0);
    put(&mut b, dir + 64, &(u0.len() as u16 + 2).to_le_bytes());
    b[dir + 66] = 5;
    put(&mut b, dir + 76, &1u32.to_le_bytes());
    put(&mut b, dir + 116, &1u32.to_le_bytes());
    let e1 = dir + 128;
    let (n1, u1) = name(e1, "CONTENTS");
    put(&mut b, n1, &u1);
    put(&mut b, e1 + 64, &(u1.len() as u16 + 2).to_le_bytes());
    b[e1 + 66] = 2;
    put(&mut b, e1 + 116, &1u32.to_le_bytes());
    put(&mut b, e1 + 120, &0xFFFF_FFFFu32.to_le_bytes());
    b
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1705_the_readers_own_peak_on_a_hostile_compound_file_is_small() {
    let _turn = TURN.lock().unwrap_or_else(|e| e.into_inner());
    // The same bytes straight into the reader the provider calls, so a drawing that carries them
    // cannot make `presentation()` allocate what the header declares.
    let blob = hostile_cfb();
    let before = LIVE.load(Ordering::Relaxed);
    PEAK.store(before, Ordering::Relaxed);
    let _ = acadrust::entities::extract_presentation(&blob);
    let peak = PEAK.load(Ordering::Relaxed).saturating_sub(before);
    assert!(
        peak < 64 << 20,
        "{} bytes of compound file took {peak} bytes of live heap",
        blob.len()
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1705_a_hostile_compound_file_is_a_row_and_a_small_peak() {
    let _turn = TURN.lock().unwrap_or_else(|e| e.into_inner());
    let mut ole = Ole2Frame::new();
    ole.upper_left_corner = Vector3::new(100.0, 200.0, 0.0);
    ole.lower_right_corner = Vector3::new(150.0, 175.0, 0.0);
    ole.ole_object_type = OleObjectType::Embedded;
    ole.is_paper_space = true;
    ole.envelope = OleFrameEnvelope::Legacy {
        records: BinaryRecord::split(&hostile_cfb(), 4096),
    };
    let mut doc = CadDocument::new();
    let layout = doc.add_layout("Sheet A").expect("layout");
    let Some(ObjectType::Layout(l)) = doc.objects.get_mut(&layout) else {
        panic!("the layout object");
    };
    l.paper_size = "ISO_A3_(420.00_x_297.00_MM)".into();
    l.paper_width = 420.0;
    l.paper_height = 297.0;
    l.plot_paper_units = 1;
    l.plot_type = 5;
    l.plot_scale_numerator = 1.0;
    l.plot_scale_denominator = 1.0;
    doc.add_entity_to_layout(
        EntityType::Line(Line::from_coords(10.0, 10.0, 0.0, 60.0, 10.0, 0.0)),
        "Sheet A",
    )
    .expect("a line");
    doc.add_entity_to_layout(EntityType::Ole2Frame(Box::new(ole)), "Sheet A")
        .expect("the frame");
    let bytes = DwgWriter::write_to_vec(&doc).expect("write");

    let dir = std::env::temp_dir().join(format!("libviprs-1705-cfb-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let before = LIVE.load(Ordering::Relaxed);
    PEAK.store(before, Ordering::Relaxed);
    let out = PvecConversion::new(CadSource::Bytes(&bytes), &dir.join("sheet.pvec"))
        .view_profile(ViewProfile::print_profile("Sheet A"))
        .ole_previews(true)
        .convert();
    let peak = PEAK.load(Ordering::Relaxed).saturating_sub(before);
    let out = out.expect("a hostile preview never fails the sheet");
    assert!(
        peak < 64 << 20,
        "a few KB of compound file took {peak} bytes of live heap"
    );
    assert!(
        out.report
            .entities
            .iter()
            .filter(|r| r.source_class == "OLE2FRAME")
            .flat_map(|r| r.warnings.iter())
            .any(|w| w.starts_with("OLE2FRAME: not drawn")),
        "the frame has a row that says it is not drawn"
    );
}
