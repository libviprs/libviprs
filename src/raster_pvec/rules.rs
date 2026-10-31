//! The rules a producer follows to put a raster into a PVec (issue #1333, RL2.2).
//!
//! Skeleton: the type is final, the rows come in the next commit.

/// One producer rule: where it comes from, so a reader can find the sentence that asks for it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProducerRule {
    /// A short stable name.
    pub id: &'static str,
    /// What the producer does.
    pub rule: &'static str,
    /// The ADR 0018 item or `RASTER-NN` / `LIMIT-NN` / `NP-NN` code it comes from.
    pub source: &'static str,
}

/// Every rule, in the order the page lists them.
pub const RASTER_PRODUCER_RULES: &[ProducerRule] = &[];

/// The rules as a Markdown table, which is the block `docs/pvec-raster.md` carries.
#[must_use]
pub fn render_rules_table() -> String {
    String::new()
}
