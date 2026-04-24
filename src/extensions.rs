//! Typed extension map for pipeline-level context that does not fit one of
//! the first-class builder slots ([`TileSink`](crate::sink::TileSink),
//! [`EngineObserver`](crate::observe::EngineObserver),
//! [`StripSource`](crate::streaming::StripSource)).
//!
//! The shape mirrors [`http::Extensions`]: a `TypeId`-keyed map of
//! `Box<dyn Any + Send + Sync>`. One slot per `TypeId`; re-inserting a type
//! overwrites the previous value. Values must be `Send + Sync + 'static` so
//! the map itself is trivially `Send + Sync`.
//!
//! libviprs itself reads **zero** extensions as of this release — the hatch
//! is deliberately inert. It exists so third-party crates can stash shared
//! context (metrics recorders, tracing spans, custom config blobs) through
//! [`EngineBuilder::with_extension`](crate::EngineBuilder::with_extension)
//! and retrieve it from a custom `EngineObserver` or the rest of the
//! pipeline without a semver bump to libviprs. When libviprs grows a
//! feature that needs cross-cutting context (e.g. an optional `metrics`
//! integration), that feature's code becomes the first internal reader.
//!
//! # Example
//!
//! ```ignore
//! use std::sync::Arc;
//! use libviprs::extensions::Extensions;
//!
//! struct MyContext { counter: std::sync::atomic::AtomicU64 }
//!
//! let mut ext = Extensions::new();
//! ext.insert(Arc::new(MyContext { counter: 0.into() }));
//!
//! let pulled: &Arc<MyContext> = ext.get().unwrap();
//! ```

use std::any::{Any, TypeId};
use std::collections::HashMap;

type BoxedAny = Box<dyn Any + Send + Sync>;

/// Typed map of `Send + Sync + 'static` values keyed by [`TypeId`].
///
/// One slot per type; [`Extensions::insert`] returns the previous value if
/// one was present. The internal [`HashMap`] is lazily allocated so an
/// [`EngineBuilder`](crate::EngineBuilder) that never sets an extension
/// pays no heap cost.
#[derive(Default)]
pub struct Extensions {
    map: Option<HashMap<TypeId, BoxedAny>>,
}

impl std::fmt::Debug for Extensions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Extensions")
            .field("len", &self.len())
            .finish_non_exhaustive()
    }
}

impl Extensions {
    /// Construct an empty extension map. No allocation until the first
    /// [`Extensions::insert`] call.
    pub fn new() -> Self {
        Self { map: None }
    }

    /// Insert `value`, returning the previous value for the same `T` if one
    /// was present.
    pub fn insert<T: Send + Sync + 'static>(&mut self, value: T) -> Option<T> {
        let map = self.map.get_or_insert_with(HashMap::new);
        map.insert(TypeId::of::<T>(), Box::new(value))
            .and_then(|boxed| boxed.downcast::<T>().ok().map(|b| *b))
    }

    /// Retrieve a shared reference to the stored value of type `T`, if any.
    pub fn get<T: Send + Sync + 'static>(&self) -> Option<&T> {
        self.map
            .as_ref()?
            .get(&TypeId::of::<T>())
            .and_then(|b| b.downcast_ref::<T>())
    }

    /// Retrieve a unique reference to the stored value of type `T`, if any.
    pub fn get_mut<T: Send + Sync + 'static>(&mut self) -> Option<&mut T> {
        self.map
            .as_mut()?
            .get_mut(&TypeId::of::<T>())
            .and_then(|b| b.downcast_mut::<T>())
    }

    /// Remove and return the stored value of type `T`, if any.
    pub fn remove<T: Send + Sync + 'static>(&mut self) -> Option<T> {
        self.map
            .as_mut()?
            .remove(&TypeId::of::<T>())
            .and_then(|boxed| boxed.downcast::<T>().ok().map(|b| *b))
    }

    /// Number of stored extensions.
    pub fn len(&self) -> usize {
        self.map.as_ref().map_or(0, |m| m.len())
    }

    /// Whether the map currently holds zero extensions.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Drop every stored extension.
    pub fn clear(&mut self) {
        if let Some(m) = self.map.as_mut() {
            m.clear();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, PartialEq)]
    struct Marker(&'static str);

    #[test]
    fn roundtrip() {
        let mut ext = Extensions::new();
        assert!(ext.is_empty());
        ext.insert(Marker("hi"));
        assert_eq!(ext.len(), 1);
        assert_eq!(ext.get::<Marker>(), Some(&Marker("hi")));
    }

    #[test]
    fn overwrites_same_type() {
        let mut ext = Extensions::new();
        assert_eq!(ext.insert(Marker("first")), None);
        assert_eq!(ext.insert(Marker("second")), Some(Marker("first")));
        assert_eq!(ext.get::<Marker>(), Some(&Marker("second")));
    }

    #[test]
    fn different_types_coexist() {
        #[derive(Debug, PartialEq)]
        struct Alpha(u32);
        #[derive(Debug, PartialEq)]
        struct Beta(String);

        let mut ext = Extensions::new();
        ext.insert(Alpha(1));
        ext.insert(Beta("b".into()));
        assert_eq!(ext.get::<Alpha>(), Some(&Alpha(1)));
        assert_eq!(ext.get::<Beta>(), Some(&Beta("b".into())));
        assert_eq!(ext.len(), 2);
    }

    #[test]
    fn remove_returns_ownership() {
        let mut ext = Extensions::new();
        ext.insert(Marker("x"));
        assert_eq!(ext.remove::<Marker>(), Some(Marker("x")));
        assert!(ext.get::<Marker>().is_none());
    }

    #[test]
    fn get_mut_is_mutable() {
        let mut ext = Extensions::new();
        ext.insert(vec![1u32, 2, 3]);
        ext.get_mut::<Vec<u32>>().unwrap().push(4);
        assert_eq!(ext.get::<Vec<u32>>().unwrap(), &vec![1, 2, 3, 4]);
    }

    #[test]
    fn send_sync_bound_holds() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Extensions>();
    }
}
