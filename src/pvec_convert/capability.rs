//! The allowed-capability gate and the three conversion profiles (issue #1414, N7.2).
//!
//! A caller says before the run which PVec capabilities it may use, and an entity that needs one
//! outside that set is refused instead of reshaped. The identifiers are PVec's own
//! ([`pvec_facade::CapabilityId`], re-exported from its registry): libviprs declares no
//! capability vocabulary, it only looks up which identifier each normalized kind needs in
//! [`NORMALIZED_TO_PVEC`](crate::pvec::mapping::NORMALIZED_TO_PVEC).
//!
//! # Policies
//!
//! * [`CapabilityPolicy::RejectInput`]: the first entity outside the set aborts the run with
//!   [`PvecConvertError::CapabilityNotAllowed`](super::PvecConvertError::CapabilityNotAllowed),
//!   naming the capability and the entity, and nothing is published.
//! * [`CapabilityPolicy::RejectEntity`]: the entity is left out of the artifact and counted in the
//!   report as `RequiresPVecCapability`, reason `capability-not-allowed`.
//! * [`CapabilityPolicy::Downgrade`]: the entity is mapped to a weaker supported representation
//!   and the report records it. No weaker representation exists in the v1 mapping table that is
//!   not a tessellation or a split of one entity into several, and the epic forbids both, so
//!   today every downgrade resolves to the `RejectEntity` outcome with the warning
//!   `downgrade-unavailable`. The policy is kept so a later representation needs no API change.
//!
//! No policy turns a curve into canonical PVec geometry it was not.
//!
//! # Profiles
//!
//! | profile | capabilities | entities | policy |
//! |---|---|---|---|
//! | `Strict` | those of the structurally exact rows | `SupportedExact` only | `RejectInput` |
//! | `Default` | every row of the mapping table | exact and equivalent | `RejectEntity` |
//! | `Permissive` | every row of the mapping table | exact and equivalent | `Downgrade` |

use std::collections::BTreeSet;

use pvec_facade::CapabilityId;

use crate::pvec::mapping::{ConformanceClass, NORMALIZED_TO_PVEC};

/// What happens to an entity that needs a capability outside the allowed set.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum CapabilityPolicy {
    /// The first such entity aborts the run.
    RejectInput,
    /// The entity is omitted, counted and reported.
    RejectEntity,
    /// The entity is mapped to a weaker supported representation and reported. See the module
    /// docs: none exists yet, so this currently records `downgrade-unavailable` and omits it.
    Downgrade,
}

/// A named capability set and policy, written down once.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[non_exhaustive]
pub enum ConversionProfile {
    /// Only entities classified `SupportedExact`; an out-of-set entity aborts the run.
    Strict,
    /// Every capability the mapping table names; an out-of-set entity is left out and counted.
    #[default]
    Default,
    /// As `Default`, with a downgrade attempted before an entity is left out.
    Permissive,
}

impl ConversionProfile {
    /// Every profile, in the order the docs list them.
    pub const ALL: [ConversionProfile; 3] = [
        ConversionProfile::Strict,
        ConversionProfile::Default,
        ConversionProfile::Permissive,
    ];

    /// The profile as the report spells it.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            ConversionProfile::Strict => "Strict",
            ConversionProfile::Default => "Default",
            ConversionProfile::Permissive => "Permissive",
        }
    }

    /// The capabilities this profile allows.
    #[must_use]
    pub fn capabilities(self) -> BTreeSet<CapabilityId> {
        NORMALIZED_TO_PVEC
            .iter()
            .filter(|r| {
                self != ConversionProfile::Strict
                    || r.conformance == ConformanceClass::StructuralExactness
            })
            .map(|r| CapabilityId::new(u32::from(r.capability.id)))
            .collect()
    }

    /// Whether only `SupportedExact` entities pass.
    #[must_use]
    pub const fn exact_only(self) -> bool {
        matches!(self, ConversionProfile::Strict)
    }

    /// The policy this profile applies to an out-of-set entity.
    #[must_use]
    pub const fn policy(self) -> CapabilityPolicy {
        match self {
            ConversionProfile::Strict => CapabilityPolicy::RejectInput,
            ConversionProfile::Default => CapabilityPolicy::RejectEntity,
            ConversionProfile::Permissive => CapabilityPolicy::Downgrade,
        }
    }
}
