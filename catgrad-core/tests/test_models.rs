use catgrad_core::category::lang::*;
use catgrad_core::check::*;
use catgrad_core::stdlib::{nn::*, *};

////////////////////////////////////////////////////////////////////////////////
// Example program

pub struct LinearSigmoid;

impl Def<2, 1> for LinearSigmoid {
    fn ty(&self) -> ([Type; 2], [Type; 1]) {
        let t_x = Value::Tensor(TypeExpr::NdArrayType(NdArrayType {
            dtype: DtypeExpr::Constant(Dtype::F32),
            shape: ShapeExpr::Shape(vec![NatExpr::Var(0), NatExpr::Var(1)]),
        }));

        let t_p = Value::Tensor(TypeExpr::NdArrayType(NdArrayType {
            dtype: DtypeExpr::Constant(Dtype::F32),
            shape: ShapeExpr::Shape(vec![NatExpr::Var(1), NatExpr::Var(2)]),
        }));

        let t_y = Value::Tensor(TypeExpr::NdArrayType(NdArrayType {
            dtype: DtypeExpr::Constant(Dtype::F32),
            shape: ShapeExpr::Shape(vec![NatExpr::Mul(vec![NatExpr::Var(1), NatExpr::Var(2)])]),
        }));

        ([t_x, t_p], [t_y])
    }

    fn path(&self) -> Path {
        path(vec!["test", "linear_sigmoid"])
    }

    fn inline(
        &self,
        builder: &std::rc::Rc<
            std::cell::RefCell<open_hypergraphs::lax::OpenHypergraph<Object, Operation>>,
        >,
        [x, p]: [Var; 2],
    ) -> [Var; 1] {
        let x = matmul(builder, x, p);
        let x = Sigmoid.call(builder, [x]);

        // flatten result shape
        let [a, c] = unpack::<2>(builder, shape(builder, x.clone()));
        let t = pack::<1>(builder, [a * c]);

        [reshape(builder, t, x)]
    }
}
