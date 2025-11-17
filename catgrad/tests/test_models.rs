use catgrad::abstract_interpreter::Value;
use catgrad::category::lang::*;
use catgrad::stdlib::{nn::*, *};
use catgrad::typecheck::value_types::*;

////////////////////////////////////////////////////////////////////////////////
// Example program

pub struct LinearSigmoid;

impl Module<2, 1> for LinearSigmoid {
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
        path(vec!["test", "linear_sigmoid"]).unwrap()
    }

    fn def(&self, builder: &Builder, [x, p]: [Var; 2]) -> [Var; 1] {
        let x = matmul(builder, x, p);
        let x = Sigmoid.call(builder, [x]);

        // flatten result shape
        let [a, c] = unpack::<2>(builder, shape(builder, x.clone()));
        let t = pack::<1>(builder, [a * c]);

        [reshape(builder, t, x)]
    }
}

// You wouldn't normally do this- just for testing!
pub struct Add;
impl Module<2, 1> for Add {
    fn ty(&self) -> ([Type; 2], [Type; 1]) {
        let t_x0 = Value::Tensor(TypeExpr::NdArrayType(NdArrayType {
            dtype: DtypeExpr::Constant(Dtype::F32),
            shape: ShapeExpr::Shape(vec![NatExpr::Var(0), NatExpr::Var(1)]),
        }));

        let t_x1 = Value::Tensor(TypeExpr::NdArrayType(NdArrayType {
            dtype: DtypeExpr::Constant(Dtype::F32),
            shape: ShapeExpr::Shape(vec![NatExpr::Var(0), NatExpr::Var(1)]),
        }));

        let t_y = Value::Tensor(TypeExpr::NdArrayType(NdArrayType {
            dtype: DtypeExpr::Constant(Dtype::F32),
            shape: ShapeExpr::Shape(vec![NatExpr::Var(0), NatExpr::Var(1)]),
        }));

        ([t_x0, t_x1], [t_y])
    }

    fn path(&self) -> Path {
        path(vec!["test", "add"]).unwrap()
    }

    fn def(&self, _builder: &Builder, [x, y]: [Var; 2]) -> [Var; 1] {
        [x + y]
    }
}

pub struct BatchMatMul;
impl Module<2, 1> for BatchMatMul {
    fn ty(&self) -> ([Type; 2], [Type; 1]) {
        let t_x0 = Value::Tensor(TypeExpr::NdArrayType(NdArrayType {
            dtype: DtypeExpr::Constant(Dtype::F32),
            shape: ShapeExpr::Shape(vec![NatExpr::Var(0), NatExpr::Var(1), NatExpr::Var(2)]),
        }));

        let t_x1 = Value::Tensor(TypeExpr::NdArrayType(NdArrayType {
            dtype: DtypeExpr::Constant(Dtype::F32),
            shape: ShapeExpr::Shape(vec![NatExpr::Var(0), NatExpr::Var(2), NatExpr::Var(3)]),
        }));

        let t_y = Value::Tensor(TypeExpr::NdArrayType(NdArrayType {
            dtype: DtypeExpr::Constant(Dtype::F32),
            shape: ShapeExpr::Shape(vec![NatExpr::Var(0), NatExpr::Var(1), NatExpr::Var(3)]),
        }));

        ([t_x0, t_x1], [t_y])
    }

    fn path(&self) -> Path {
        path(vec!["test", "batch_matmul"]).unwrap()
    }

    fn def(&self, builder: &Builder, [x, y]: [Var; 2]) -> [Var; 1] {
        [matmul(builder, x, y)]
    }
}

pub struct TopK;
impl Module<1, 2> for TopK {
    fn ty(&self) -> ([Type; 1], [Type; 2]) {
        let input = Value::Tensor(TypeExpr::NdArrayType(NdArrayType {
            dtype: DtypeExpr::Constant(Dtype::F32),
            shape: ShapeExpr::Shape(vec![NatExpr::Constant(2), NatExpr::Constant(4)]),
        }));
        let values = Value::Tensor(TypeExpr::NdArrayType(NdArrayType {
            dtype: DtypeExpr::Constant(Dtype::F32),
            shape: ShapeExpr::Shape(vec![NatExpr::Constant(2), NatExpr::Constant(2)]),
        }));
        let indices = Value::Tensor(TypeExpr::NdArrayType(NdArrayType {
            dtype: DtypeExpr::Constant(Dtype::U32),
            shape: ShapeExpr::Shape(vec![NatExpr::Constant(2), NatExpr::Constant(2)]),
        }));

        ([input], [values, indices])
    }

    fn path(&self) -> Path {
        path(vec!["test", "topk"]).unwrap()
    }

    fn def(&self, builder: &Builder, [x]: [Var; 1]) -> [Var; 2] {
        let k = lit(builder, Literal::Nat(2));
        let (values, indices) = topk(builder, k, x);
        [values, indices]
    }
}
