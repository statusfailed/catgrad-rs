use crate::category::lang::*;
use crate::util::build_typed;

use std::cell::RefCell;
use std::rc::Rc;

use open_hypergraphs::lax::{var, *};

////////////////////////////////////////////////////////////////////////////////
// Generic interface

// A Definition of arity A and coarity B
// TODO: do we want to just have a struct with a closure in it instead?
// TODO: Can we read a Def like this from disk? (probably not: has rust code in it!)
// TODO: can we write a function which *writes a dyn Def to disk* for all A, B?
pub trait Def<const A: usize, const B: usize> {
    ////////////////////////////////////////
    // User-provided

    /// TODO: make this return [`Value`] instead, figure out sort from that.
    fn ty(&self) -> ([Type; A], [Type; B]);
    fn path(&self) -> Path; // unique global name of this op

    /// Construct this definition by mutably *inlining it* into the provided OpenHypergraph
    fn inline(
        &self,
        builder: &Rc<RefCell<OpenHypergraph<Object, Operation>>>,
        args: [Var; A],
    ) -> [Var; B];

    ////////////////////////////////////////
    // Derived functions (TODO: move outside trait?)

    // Sort, derived from ty.
    fn sort(&self) -> ([Object; A], [Object; B]) {
        let (v1, v2) = self.ty();
        (v1.map(to_sort), v2.map(to_sort))
    }

    /// Create a single `Definition` operation in the graph with name `self.path()`.
    fn op(
        &self,
        builder: &Rc<RefCell<OpenHypergraph<Object, Operation>>>,
        args: [Var; A],
    ) -> [Var; B] {
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

    /// Construct a "standalone" OpenHypergraph for this definition
    fn term(&self) -> TypedTerm {
        let (source_type, target_type) = self.ty();
        let source_object = source_type.clone().map(to_sort);

        // TODO: err handling
        let term = build_typed(source_object, |builder, args| {
            self.inline(builder, args).to_vec()
        })
        .unwrap();

        TypedTerm {
            term,
            source_type: source_type.to_vec(),
            target_type: target_type.to_vec(),
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
    fn call(
        &self,
        builder: &Rc<RefCell<OpenHypergraph<Object, Operation>>>,
        args: [Var; N],
    ) -> Var {
        let [r] = self.op(builder, args);
        r
    }
}

impl<const N: usize, T: Def<N, 1>> FnDef<N> for T {}
