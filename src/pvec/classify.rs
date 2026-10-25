//! Wall and column classification from layer and block rules (N4.9, #1233).
//!
//! A drawing already says what most of its walls and columns are: they sit on
//! labelled layers (`A-WALL*`, `S-COL*`) or are inserts of named blocks. An
//! [`ElementRules`] profile maps those names to [`ElementClass::Wall`] or
//! [`ElementClass::Column`], and [`ElementRules::classify`] gives every
//! [`RawEntity`] exactly one class. There is no geometry guessing here, and
//! geometry is never touched: the classifier reads a `&RawEntity` and returns a
//! [`Classification`] next to it.
//!
//! # Order
//!
//! The order is fixed, so one entity and one profile always give one answer:
//!
//! 1. **Block rules**, innermost block first. The blocks an entity can be
//!    "inside" are its own INSERT's block (when the entity is an INSERT) and then
//!    the blocks of its [`InstancePath`] from the innermost step outwards. For
//!    each block, the profile's block rules are tried in list order and the first
//!    match decides. A column block inside a wall unit is therefore a column, and
//!    a column block inserted on a wall layer is a column too.
//! 2. **Layer rules**, in list order, against the entity's own layer name.
//! 3. Otherwise [`ElementClass::Unclassified`].
//!
//! Layer and block names are matched as globs (`*` and `?`), anchored at both
//! ends and case-insensitively, because DWG symbol-table names are
//! case-insensitive.
//!
//! # What this is not
//!
//! [`ElementClass`] answers "what is it in the building". The N3.5 conversion
//! classification answers "can it be converted", and the two do not share a type.
//! An entity no rule matches is `Unclassified`: it is counted, never dropped and
//! never defaulted to a wall.
//!
//! Colour and lineweight rules need those facts on the provider boundary, which
//! does not carry them yet (#1264).

use super::provider::{InstancePath, RawEntity, RawGeometry, SourceColour};
use crate::cad::ItemHandle;

/// What an entity is, as far as the rules can tell.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[non_exhaustive]
pub enum ElementClass {
    /// A wall.
    Wall,
    /// A column.
    Column,
    /// No rule matched. Never a default for "probably a wall".
    Unclassified,
}

/// Which kind of rule decided a class: the source tag the semantic class stream
/// records (`layer-rule`, `block-rule`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ClassSource {
    /// A layer-name rule.
    LayerRule,
    /// A block-name rule.
    BlockRule,
    /// A colour rule.
    ColourRule,
    /// A lineweight rule.
    LineweightRule,
}

impl ClassSource {
    /// The tag as the stream spells it.
    #[must_use]
    pub const fn tag(self) -> &'static str {
        match self {
            ClassSource::LayerRule => "layer-rule",
            ClassSource::BlockRule => "block-rule",
            ClassSource::ColourRule => "colour-rule",
            ClassSource::LineweightRule => "lineweight-rule",
        }
    }
}

/// What a rule matches against.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum RuleMatch {
    /// A layer-name glob.
    Layer(String),
    /// A block-name glob.
    Block(String),
    /// An exact colour, as [`EntityStyle::colour`](super::provider::EntityStyle::colour)
    /// reports it (#1264).
    Colour(SourceColour),
    /// An exact lineweight in hundredths of a millimetre (#1264).
    Lineweight(u16),
}

/// One rule: a pattern and the class it assigns.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ElementRule {
    /// What it matches.
    pub matcher: RuleMatch,
    /// The class it assigns. A rule that names `Unclassified` never matches
    /// anything: "not classified" is the absence of a rule, not a rule.
    pub class: ElementClass,
}

/// The ordered rule list a caller supplies (or takes from
/// [`ElementRules::default_profile`]).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ElementRules {
    rules: Vec<ElementRule>,
}

/// The class of one entity and where it came from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Classification {
    /// The class.
    pub class: ElementClass,
    /// The kind of rule that decided it, `None` for `Unclassified`.
    pub source: Option<ClassSource>,
    /// Index of the deciding rule in the profile, `None` for `Unclassified`.
    pub rule_index: Option<usize>,
    /// The entity's handle, when it has one.
    pub handle: Option<ItemHandle>,
    /// The INSERT chain the entity was reached through, kept for N4.4's lineage.
    pub block_path: InstancePath,
}

/// Counts by class, for the conversion report.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ClassificationReport {
    /// Entities classified `Wall`.
    pub walls: usize,
    /// Entities classified `Column`.
    pub columns: usize,
    /// Entities no rule matched.
    pub unclassified: usize,
}

impl ClassificationReport {
    /// Every entity counted.
    #[must_use]
    pub const fn total(&self) -> usize {
        self.walls + self.columns + self.unclassified
    }
}

/// Case-insensitive, anchored glob with `*` (any run, including empty) and `?`
/// (exactly one character). Iterative with one backtrack point, so a hostile
/// pattern cannot go exponential.
fn glob_match(pattern: &str, text: &str) -> bool {
    let p: Vec<char> = pattern.chars().flat_map(char::to_lowercase).collect();
    let t: Vec<char> = text.chars().flat_map(char::to_lowercase).collect();
    let (mut pi, mut ti) = (0usize, 0usize);
    let mut star: Option<(usize, usize)> = None;
    while ti < t.len() {
        if pi < p.len() && (p[pi] == '?' || p[pi] == t[ti]) && p[pi] != '*' {
            pi += 1;
            ti += 1;
        } else if pi < p.len() && p[pi] == '*' {
            star = Some((pi, ti));
            pi += 1;
        } else if let Some((sp, st)) = star {
            pi = sp + 1;
            ti = st + 1;
            star = Some((sp, st + 1));
        } else {
            return false;
        }
    }
    while pi < p.len() && p[pi] == '*' {
        pi += 1;
    }
    pi == p.len()
}

impl ElementRules {
    /// An empty profile: everything is `Unclassified`.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// The default profile for common conventions, which a caller replaces by
    /// building their own with [`ElementRules::new`] and
    /// [`ElementRules::with_rule`].
    ///
    /// | Match | Class |
    /// |---|---|
    /// | layer `A-WALL*` | wall |
    /// | layer `S-WALL*` | wall |
    /// | layer `A-COL*` | column |
    /// | layer `S-COL*` | column |
    /// | block `COLUMN*` | column |
    ///
    /// The `A-` and `S-` prefixes are the AIA / US National CAD Standard
    /// discipline designators for architecture and structure. Block rules are
    /// tried before layer rules whatever the list order, see the module docs.
    #[must_use]
    pub fn default_profile() -> Self {
        Self::new()
            .with_rule(RuleMatch::Layer("A-WALL*".into()), ElementClass::Wall)
            .with_rule(RuleMatch::Layer("S-WALL*".into()), ElementClass::Wall)
            .with_rule(RuleMatch::Layer("A-COL*".into()), ElementClass::Column)
            .with_rule(RuleMatch::Layer("S-COL*".into()), ElementClass::Column)
            .with_rule(RuleMatch::Block("COLUMN*".into()), ElementClass::Column)
    }

    /// Appends a rule. Earlier rules win over later ones within the same kind.
    #[must_use]
    pub fn with_rule(mut self, matcher: RuleMatch, class: ElementClass) -> Self {
        self.rules.push(ElementRule { matcher, class });
        self
    }

    /// The rules in order.
    #[must_use]
    pub fn rules(&self) -> &[ElementRule] {
        &self.rules
    }

    fn first_block_rule(&self, block: &str) -> Option<(usize, ElementClass)> {
        self.rules
            .iter()
            .enumerate()
            .find_map(|(i, r)| match &r.matcher {
                RuleMatch::Block(g)
                    if r.class != ElementClass::Unclassified && glob_match(g, block) =>
                {
                    Some((i, r.class))
                }
                _ => None,
            })
    }

    fn first_layer_rule(&self, layer: &str) -> Option<(usize, ElementClass)> {
        self.rules
            .iter()
            .enumerate()
            .find_map(|(i, r)| match &r.matcher {
                RuleMatch::Layer(g)
                    if r.class != ElementClass::Unclassified && glob_match(g, layer) =>
                {
                    Some((i, r.class))
                }
                _ => None,
            })
    }

    /// Classifies one entity. Pure: the entity is not changed and the same input
    /// and profile always give the same result.
    #[must_use]
    pub fn classify(&self, entity: &RawEntity) -> Classification {
        let prov = &entity.provenance;
        let own_block = match &entity.geometry {
            RawGeometry::Insert(b) => Some(b.block_name.as_str()),
            _ => None,
        };
        let enclosing = prov
            .block_path
            .steps()
            .iter()
            .rev()
            .map(|s| s.block_name.as_str());
        let hit = own_block
            .into_iter()
            .chain(enclosing)
            .find_map(|b| self.first_block_rule(b))
            .map(|(i, c)| (i, c, ClassSource::BlockRule))
            .or_else(|| {
                self.first_layer_rule(&prov.layer)
                    .map(|(i, c)| (i, c, ClassSource::LayerRule))
            });
        let (class, source, rule_index) = match hit {
            Some((i, c, s)) => (c, Some(s), Some(i)),
            None => (ElementClass::Unclassified, None, None),
        };
        Classification {
            class,
            source,
            rule_index,
            handle: prov.handle,
            block_path: prov.block_path.clone(),
        }
    }

    /// Classifies a stream of entities in order, one result each, and counts them.
    pub fn classify_all<'a, I>(&self, entities: I) -> (Vec<Classification>, ClassificationReport)
    where
        I: IntoIterator<Item = &'a RawEntity>,
    {
        let mut report = ClassificationReport::default();
        let out = entities
            .into_iter()
            .map(|e| {
                let c = self.classify(e);
                match c.class {
                    ElementClass::Wall => report.walls += 1,
                    ElementClass::Column => report.columns += 1,
                    ElementClass::Unclassified => report.unclassified += 1,
                }
                c
            })
            .collect();
        (out, report)
    }
}

#[cfg(test)]
mod tests {
    use super::glob_match;

    #[test]
    fn globs() {
        assert!(glob_match("A-WALL*", "A-WALL"));
        assert!(glob_match("A-WALL*", "a-wall-full"));
        assert!(!glob_match("A-WALL*", "XA-WALL"));
        assert!(glob_match("*COL*", "S-COLS"));
        assert!(glob_match("S-C?L", "S-COL"));
        assert!(!glob_match("S-C?L", "S-COOL"));
        assert!(glob_match("*", ""));
        assert!(glob_match("a*b*c", "aXbYc"));
        assert!(!glob_match("a*b*c", "aXbY"));
        assert!(!glob_match("", "x"));
        // Many stars against a long miss stays fast (no exponential backtracking).
        assert!(!glob_match("*a*a*a*a*a*a*a*a*b", &"a".repeat(200)));
    }
}
