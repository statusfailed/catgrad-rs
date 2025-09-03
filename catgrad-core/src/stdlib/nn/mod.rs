use crate::category::lang::*;
use crate::stdlib::types::*;

use std::cell::RefCell;
use std::rc::Rc;

use open_hypergraphs::lax::*;

////////////////////////////////////////
// Sigmoid

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
