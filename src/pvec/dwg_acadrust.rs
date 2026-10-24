//! Red stub for N4.1: the acadrust provider, not written yet.

use crate::cad::CadSource;
use crate::pvec::{
    DwgProvider, EntityCursor, EntityScope, LayerState, LayoutState, ProviderCaps, ProviderError,
};

/// The first provider.
pub struct AcadrustProvider {
    caps: ProviderCaps,
}

impl AcadrustProvider {
    /// Every block definition's name, in the block table's order.
    #[must_use]
    pub fn block_names(&self) -> Vec<String> {
        unimplemented!("N4.1 red stub")
    }
}

impl DwgProvider for AcadrustProvider {
    fn probe(_source: CadSource<'_>) -> Result<bool, ProviderError> {
        unimplemented!("N4.1 red stub")
    }
    fn open(_source: CadSource<'_>) -> Result<Self, ProviderError> {
        unimplemented!("N4.1 red stub")
    }
    fn name(&self) -> &'static str {
        unimplemented!("N4.1 red stub")
    }
    fn caps(&self) -> &ProviderCaps {
        &self.caps
    }
    fn layouts(&self) -> Result<Vec<LayoutState>, ProviderError> {
        unimplemented!("N4.1 red stub")
    }
    fn layers(&self) -> Result<Vec<LayerState>, ProviderError> {
        unimplemented!("N4.1 red stub")
    }
    fn entities<'a>(
        &'a self,
        _scope: &EntityScope,
    ) -> Result<Box<dyn EntityCursor + 'a>, ProviderError> {
        unimplemented!("N4.1 red stub")
    }
}
