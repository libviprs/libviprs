//! Issue #1673: what the fork's own `PlotStyleTable::resolve_lineweight` does with the styles that name
//! no list entry: 0 is "use the object's lineweight" and a number past the list is too. The three
//! cases that name an entry (styles 1, 2 and the last) were ignored here because the fork reads one
//! entry too far (acadrust#73). A conversion no longer calls that function for them (#1692, it reads
//! the list itself), so they are tested through the conversion in `tests/pvec_1692_ctb_lineweight.rs`.

#![cfg(feature = "pvec-convert")]

use libviprs::cad::plot_files::PlotStyleTable;

/// A table whose lineweight list is 0.10, 0.20, 0.30, 0.40 mm, and whose styles for ACI 1 to 4 pick
/// the numbers 1 to 4. The scale factor is off.
fn table() -> PlotStyleTable {
    let mut t = PlotStyleTable::identity("synthetic.ctb");
    t.lineweights = vec![0.10, 0.20, 0.30, 0.40];
    t.apply_factor = false;
    for aci in 1..=4u8 {
        t.aci_entries[usize::from(aci)].lineweight = aci;
    }
    t
}

#[test]
fn issue_1673_zero_still_means_the_objects_own_lineweight() {
    let mut t = table();
    t.aci_entries[5].lineweight = 0;
    assert_eq!(t.resolve_lineweight(5), None);
}

#[test]
fn issue_1673_a_number_past_the_table_is_the_objects_lineweight() {
    let mut t = table();
    t.aci_entries[6].lineweight = 5;
    assert_eq!(t.resolve_lineweight(6), None);
}
