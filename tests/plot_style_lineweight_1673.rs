//! Issue #1673: the lineweight a CTB style names. In the plot style editor the lineweight list
//! reads "Use object lineweight" first and then the custom lineweight table from its first entry,
//! so a style's `lineweight` is one more than the table index it picks: 0 is "use the object's",
//! 1 is the table's first entry (index 0), 2 its second.
//!
//! `PlotStyleTable::resolve_lineweight` reads the entry at the style's number itself, one entry too
//! far, so these fail against it. They call it the way `PvecConversion` does, on tables that are
//! built here, with a custom lineweight table whose entries are all different.

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
fn issue_1673_style_lineweight_one_is_the_first_entry_of_the_custom_table() {
    assert_eq!(table().resolve_lineweight(1), Some(0.10));
}

#[test]
fn issue_1673_style_lineweight_two_is_the_second_entry() {
    assert_eq!(table().resolve_lineweight(2), Some(0.20));
}

#[test]
fn issue_1673_the_last_entry_is_reachable_by_its_number() {
    // 4 names the fourth entry. Read as an index it is past the end and the style falls back
    // to the object's lineweight without saying so.
    assert_eq!(table().resolve_lineweight(4), Some(0.40));
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
