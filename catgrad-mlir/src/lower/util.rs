use super::grammar;
use catgrad::prelude::Dtype;
use catgrad::typecheck::{DtypeExpr, NdArrayType, ShapeExpr, Type, TypeExpr};

/// Convert a [`typechecker::Type`] into an MLIR representation.
/// This maps everything except Nat to `Tensor`,
pub(crate) fn core_type_to_mlir(core_type: &Type) -> grammar::Type {
    match core_type {
        Type::Tensor(t) => grammar::Type::TensorType(type_expr_to_tensor_type(t)),
        // Shape types are represented as tensor<rank x index>
        Type::Shape(shape_expr) => {
            let rank = match shape_expr {
                ShapeExpr::Shape(dims) => dims.len(),
                ShapeExpr::Var(_) => todo!("Shape variables not supported"),
                ShapeExpr::OfType(_) => todo!("Shape OfType not supported"),
            };
            grammar::Type::TensorType(grammar::TensorType {
                shape: grammar::Shape::Shape(vec![Some(rank)]),
                dtype: "index".to_string(),
            })
        }
        // Meta types all go to Bool, but aren't used.
        Type::Nat(_) => grammar::Type::Index,
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
