//! Trait helpers for creating the stdlib and defining models/layers/etc.
use crate::category::lang::*;
use crate::util::build_typed;

use open_hypergraphs::lax::var;

////////////////////////////////////////////////////////////////////////////////
// Generic interface

/// A type implementing [`Def`] defines a *typed term* with additional metadata.
/// Analogous to a `nn.Module` in PyTorch.
/// In total, a `T: Def` defines:
///
/// 1. A *unique global name* (`path`)
/// 2. A *type* (`ty`)
/// 3. A *definition* (`inline`) - an open hypergraph representing
///
/// Note that definitions have fixed arity/coarity, but types can vary.
pub trait Def<const A: usize, const B: usize> {
    /// The *type* of this definition, used to construct a [`TypedTerm`]
    fn ty(&self) -> ([Type; A], [Type; B]);

    /// Unique global name in the stdlib/environment
    fn path(&self) -> Path;

    /// The *definition* of this term, as a function which mutably inlines it into the provided
    /// Builder.
    fn def(&self, builder: &Builder, args: [Var; A]) -> [Var; B];

    ////////////////////////////////////////
    // Derived functions

    /// The *sort* of this type are the *object labels* of the sources/targets of its definition.
    fn sort(&self) -> ([Object; A], [Object; B]) {
        let (v1, v2) = self.ty();
        (v1.map(to_sort), v2.map(to_sort))
    }

    /// alias for `def` which is clearer to use in context, e.g. Sigmoid::inline();
    fn inline(&self, builder: &Builder, args: [Var; A]) -> [Var; B] {
        self.def(builder, args)
    }

    /// Create a single `Definition` operation in the graph with name `self.path()`.
    fn op(&self, builder: &Builder, args: [Var; A]) -> [Var; B] {
        let result_types = self.sort().1.to_vec();
        var::operation(
            builder,
            &args,
            result_types,
            Operation::Definition(self.path()),
        )
        .try_into()
        .unwrap() // guaranteed to work: size fixed by result_types
    }

    /// Construct a standalone OpenHypergraph for this definition
    /// Returns None when `self.inline` returned vars with sorts different to those declared in
    /// `self.ty`.
    fn term(&self) -> Option<TypedTerm> {
        let (source_type, target_type) = self.ty();
        let source_object = source_type.clone().map(to_sort);

        // TODO: err handling
        let term = build_typed(source_object, |builder, args| {
            self.inline(builder, args).to_vec()
        })
        .unwrap();

        use open_hypergraphs::category::*; // TODO: remove this trait import when it's auto-exported by OpenHypergraph
        if term.target() != target_type.clone().map(to_sort) {
            None
        } else {
            Some(TypedTerm {
                term,
                source_type: source_type.to_vec(),
                target_type: target_type.to_vec(),
            })
        }
    }
}

/// Get the corresponding [`Object`] (sort) for a given [`Type`]
// TODO: move?
fn to_sort(value: Type) -> Object {
    use crate::check::Value;
    match value {
        Value::Type(_) => Object::NdArrayType,
        Value::Shape(_) => Object::Shape,
        Value::Nat(_) => Object::Nat,
        Value::Dtype(_) => Object::Dtype,
        Value::Tensor(_) => Object::Tensor,
    }
}

////////////////////////////////////////////////////////////////////////////////
// "Function" definitions just return a single Var- this makes it easier to call them.

/// A FnDef is a "Function Definition": a `Def` with a single output var.
/// The `call` method is a helper for getting the single output of self.op.
pub trait FnDef<const N: usize>: Def<N, 1> {
    /// Like [`Def::op`] for coarity 1.
    fn call(&self, builder: &Builder, args: [Var; N]) -> Var {
        let [r] = self.op(builder, args);
        r
    }

    // TODO: call_inline?
}

impl<const N: usize, T: Def<N, 1>> FnDef<N> for T {}
