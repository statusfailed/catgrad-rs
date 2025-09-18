//! # Categories with *definitions*
//!
//! Allows extending a set of ops with some additional definitions.
//! For example, extend the set of ops `{add, negate}` with `sub := λ x y . add(x, negate(y))`.
//!
//! Formally, given an symmetric monoidal theory `T = (Σ₀, Σ₁, Σ₂)`.
//! Def(T) is the theory presented by `(Σ₀, Σ₁ + K, Σ₂ + R)` where:
//! - `K` is a set of adjoined operations - (e.g. 'sub')
//! - `R` is a set of rewrites expanding each adjoined name to its definition, e.g., `sub ⇝  x y ↦  add(x, negate(y))`.

use open_hypergraphs::lax::{
    OpenHypergraph,
    functor::{Functor, define_map_arrow},
};

/// A set of operations Arr extended with some additional operations in the set K.
/// `Def<K, Arr> ~= Σ₁ + K`
#[derive(Clone, PartialEq)]
pub enum Def<K, Arr> {
    Def(K),
    Arr(Arr),
}

/// Given an identity-on-objects functor F : C → D,
/// lift it to one Def(C) → Def(D).
#[derive(Clone, PartialEq)]
pub struct Lift<F> {
    functor: F,
}

impl<F: Functor<O, A1, O, A2> + Clone, K: Clone, O: Clone + PartialEq, A1: Clone, A2: Clone>
    Functor<O, Def<K, A1>, O, Def<K, A2>> for Lift<F>
{
    fn map_object(&self, o: &O) -> impl ExactSizeIterator<Item = O> {
        std::iter::once(o.clone())
    }

    fn map_operation(
        &self,
        a: &Def<K, A1>,
        source: &[O],
        target: &[O],
    ) -> OpenHypergraph<O, Def<K, A2>> {
        match a {
            Def::Def(k) => {
                OpenHypergraph::singleton(Def::Def(k.clone()), source.to_vec(), target.to_vec())
            }
            Def::Arr(a) => {
                let g = self.functor.map_operation(a, source, target);
                // TODO: replace with open-hypergraphs util once available
                crate::util::map_nodes_and_edges(g, |n| n, |e| Def::Arr(e))
            }
        }
    }

    fn map_arrow(&self, f: &OpenHypergraph<O, Def<K, A1>>) -> OpenHypergraph<O, Def<K, A2>> {
        define_map_arrow(self, f)
    }
}

/// Apply a functor lifted to a category of definitions.
pub fn lift<F: Functor<O, T, O, U> + Clone, K: Clone, O: Clone + PartialEq, T: Clone, U: Clone>(
    functor: F,
    term: OpenHypergraph<O, Def<K, T>>,
) -> OpenHypergraph<O, Def<K, U>> {
    Lift { functor }.map_arrow(&term)
}
