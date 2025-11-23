use super::grammar;
use catgrad::prelude::Dtype;
use catgrad::typecheck::{DtypeExpr, NdArrayType, Type, TypeExpr};

/// Convert a [`typechecker::Type`] into an MLIR representation.
/// This maps everything except Nat to `Tensor`,
pub(crate) fn core_type_to_mlir(core_type: &Type) -> grammar::Type {
    match core_type {
        Type::Tensor(t) => grammar::Type::TensorType(type_expr_to_tensor_type(t)),
        // Meta types all go to Bool, but aren't used.
        Type::Shape(_) => grammar::Type::Bool,
        Type::Nat(_) => grammar::Type::Bool,
        Type::Type(_) => grammar::Type::Bool,
        Type::Dtype(_) => grammar::Type::Bool,
    }
}

fn type_expr_to_tensor_type(t: &TypeExpr) -> grammar::TensorType {
    match t {
        TypeExpr::NdArrayType(NdArrayType { dtype, shape }) => {
            let shape = match shape {
                catgrad::typecheck::ShapeExpr::Var(_) => grammar::Shape::Unknown,
                catgrad::typecheck::ShapeExpr::Shape(nat_exprs) => grammar::Shape::Shape(
                    nat_exprs
                        .iter()
                        .map(|dim| match dim {
                            catgrad::typecheck::NatExpr::Constant(c) => Some(*c),
                            catgrad::typecheck::NatExpr::Var(_) => None,
                            _ => todo!("unnormalized NatExpr"),
                        })
                        .collect(),
                ),

                // NOTE: this is invalid!
                catgrad::typecheck::ShapeExpr::OfType(_) => todo!(),
            };

            let dtype = match dtype {
                DtypeExpr::Var(_) => todo!(),
                DtypeExpr::OfType(_) => todo!(),
                DtypeExpr::Constant(dtype) => match dtype {
                    Dtype::F32 => "f32".to_string(),
                    Dtype::U32 => "i32".to_string(),
                },
            };
            grammar::TensorType { shape, dtype }
        }
        TypeExpr::Var(_) => todo!(),
    }
}

pub(crate) fn to_typed_identifier(
    (n, t): &(open_hypergraphs::lax::NodeId, Type),
) -> grammar::TypedIdentifier {
    grammar::TypedIdentifier {
        id: grammar::Identifier(n.0),
        ty: core_type_to_mlir(t),
    }
}
