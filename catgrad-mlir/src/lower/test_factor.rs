use super::factor::*;
use catgrad::category::lang::*;
use catgrad::path::path;
use catgrad::prelude::Type;
use catgrad::stdlib::*;
use std::collections::HashMap;

/// Test layer that creates (param() x id) >> matmul
struct ParamMatMulLayer;

impl Module<1, 1> for ParamMatMulLayer {
    fn ty(&self) -> ([Type; 1], [Type; 1]) {
        use catgrad::abstract_interpreter::Value;
        use catgrad::category::core::Dtype;
        use catgrad::typecheck::value_types::*;

        let input_type = Value::Tensor(TypeExpr::NdArrayType(NdArrayType {
            dtype: DtypeExpr::Constant(Dtype::F32),
            shape: ShapeExpr::Shape(vec![NatExpr::Var(0), NatExpr::Var(1)]),
        }));

        let output_type = Value::Tensor(TypeExpr::NdArrayType(NdArrayType {
            dtype: DtypeExpr::Constant(Dtype::F32),
            shape: ShapeExpr::Shape(vec![NatExpr::Var(0), NatExpr::Var(2)]),
        }));

        ([input_type], [output_type])
    }

    fn path(&self) -> catgrad::path::Path {
        path(vec!["test", "param_matmul"]).unwrap()
    }

    fn def(&self, builder: &Builder, [x]: [Var; 1]) -> [Var; 1] {
        // Create a parameter with path extending from module path
        let weight_path = self.path().extend(["weight"]).unwrap();
        let weight = param(builder, &weight_path);

        // Perform matrix multiplication: x @ weight
        let result = matmul(builder, x, weight);

        [result]
    }
}

#[test]
fn test_factor_param_matmul() {
    // Create the term using the module
    let layer = ParamMatMulLayer;
    let typed_term = layer.term().expect("Failed to create term");
    let term = typed_term.term;

    println!("Original term: {:?}", term);

    // Create HashMap of parameter operation names
    let mut parameter_ops = HashMap::new();
    let weight_path = layer.path().extend(["weight"]).unwrap();
    parameter_ops.insert(weight_path.to_string(), true);

    // Create predicate to identify parameter operations using HashMap lookup
    let is_parameter = |op: &Operation| -> bool {
        match op {
            Operation::Declaration(path) => *parameter_ops.get(&path.to_string()).unwrap_or(&false),
            _ => false,
        }
    };

    // Factor the term to separate parameters using lax interface
    let (p, f) = factor(&term, is_parameter);

    // Check we split 1 param out from the term
    assert_eq!(p.hypergraph.edges.len(), 1);
    assert_eq!(f.hypergraph.edges.len(), term.hypergraph.edges.len() - 1);
}
