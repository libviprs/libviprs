//! Wall and column classification from layer and block rules (N4.9, #1233).
//!
//! STUB: the types and signatures only, so the red tests compile and fail on
//! their assertions. The implementation lands in the next commit.

use super::provider::{InstancePath, RawEntity};

/// What an entity is, as far as the rules can tell.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[non_exhaustive]
pub enum ElementClass {
    /// A wall.
    Wall,
    /// A column.
    Column,
    /// No rule matched.
    Unclassified,
}

/// Which kind of rule decided a class.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ClassSource {
    /// A layer-name rule.
    LayerRule,
    /// A block-name rule.
    BlockRule,
}

/// What a rule matches against.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum RuleMatch {
    /// A layer-name glob.
    Layer(String),
    /// A block-name glob.
    Block(String),
}

/// One rule: a pattern and the class it assigns.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ElementRule {
    /// What it matches.
    pub matcher: RuleMatch,
    /// The class it assigns (`Wall` or `Column`).
    pub class: ElementClass,
}

/// The ordered rule list.
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
    /// Index of the deciding rule in the profile.
    pub rule_index: Option<usize>,
    /// The entity's handle, when it has one.
    pub handle: Option<crate::cad::ItemHandle>,
    /// The INSERT chain the entity was reached through.
    pub block_path: InstancePath,
}

/// Counts by class.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ClassificationReport {
    /// Entities classified `Wall`.
    pub walls: usize,
    /// Entities classified `Column`.
    pub columns: usize,
    /// Entities no rule matched.
    pub unclassified: usize,
}

impl ElementRules {
    /// An empty profile: everything is `Unclassified`.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// The documented default profile.
    #[must_use]
    pub fn default_profile() -> Self {
        Self::default()
    }

    /// Appends a rule.
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

    /// Classifies one entity.
    #[must_use]
    pub fn classify(&self, entity: &RawEntity) -> Classification {
        Classification {
            class: ElementClass::Unclassified,
            source: None,
            rule_index: None,
            handle: entity.provenance.handle,
            block_path: entity.provenance.block_path.clone(),
        }
    }

    /// Classifies a stream of entities, in order, and counts the result.
    pub fn classify_all<'a, I>(&self, entities: I) -> (Vec<Classification>, ClassificationReport)
    where
        I: IntoIterator<Item = &'a RawEntity>,
    {
        let out: Vec<Classification> = entities.into_iter().map(|e| self.classify(e)).collect();
        let mut report = ClassificationReport::default();
        report.unclassified = out.len();
        (out, report)
    }
}
