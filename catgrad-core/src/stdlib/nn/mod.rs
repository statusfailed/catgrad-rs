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
fn to_sort(value: Type) -> Object {
    todo!()
}

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

////////////////////////////////////////////////////////////////////////////////
// instances

pub struct Sigmoid;

impl Def<1, 1> for Sigmoid {
    // Type maps
    fn ty(&self) -> ([Type; 1], [Type; 1]) {
        // TODO: allow any dtype; cast constants in exp.
        use crate::check::*;
        let ty = Value::Tensor(TypeExpr::NdArrayType(NdArrayType {
            dtype: DtypeExpr::Constant(Dtype::F32),
            shape: ShapeExpr::Var(0),
        }));
        ([ty.clone()], [ty])
    }

    // Name of the op
    fn path(&self) -> Path {
        path(vec!["nn", "sigmoid"])
    }

    // def
    fn inline(
        &self,
        graph: &Rc<RefCell<OpenHypergraph<Object, Operation>>>,
        [x]: [Var; 1],
    ) -> [Var; 1] {
        let c1 = constant_f32(graph, 1.0);
        let s = shape(graph, x.clone());
        let c1 = broadcast(graph, c1, s);

        let r = c1.clone() / (c1 + Exp.call(graph, [-x]));
        [r]
    }
}

////////////////////////////////////////
// Exp

pub struct Exp;

impl Def<1, 1> for Exp {
    // Type maps
    fn ty(&self) -> ([Type; 1], [Type; 1]) {
        use crate::check::*;
        let ty = Value::Tensor(TypeExpr::NdArrayType(NdArrayType {
            dtype: DtypeExpr::Constant(Dtype::F32),
            shape: ShapeExpr::Var(0),
        }));
        ([ty.clone()], [ty])
    }

    // Name of the op
    fn path(&self) -> Path {
        path(vec!["nn", "exp"])
    }

    // def
    fn inline(
        &self,
        graph: &Rc<RefCell<OpenHypergraph<Object, Operation>>>,
        [x]: [Var; 1],
    ) -> [Var; 1] {
        let e = constant_f32(graph, std::f32::consts::E);
        let s = shape(graph, x.clone());
        let e = broadcast(graph, e, s);
        [pow(graph, e, x)]
    }
}
