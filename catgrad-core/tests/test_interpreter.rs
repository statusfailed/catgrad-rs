use catgrad_core::category::bidirectional::*;
use catgrad_core::category::core;
use catgrad_core::check::*;
use catgrad_core::nn::*;
use catgrad_core::util::build_typed;

use catgrad_core::interpreter::{Interpreter, backend::NdArrayBackend};

pub mod test_utils;

////////////////////////////////////////////////////////////////////////////////
// Example program

// (N, 1, A) × (N, A, B) → (N, B)
// matmul ; sigmoid ; reshape
pub fn linear_sigmoid() -> Term {
    let term = build_typed([Object::Tensor, Object::Tensor], |graph, [x, p]| {
        let x = matmul(graph, x, p);
        let x = sigmoid(graph, x);

        // flatten result shape
        let [a, c] = unpack::<2>(graph, shape(graph, x.clone()));
        let t = pack::<1>(graph, [a * c]);

        vec![reshape(graph, t, x)]
    });

    // TODO: construct *type* maps

    term.expect("invalid term")
}

#[test]
fn test_run_add() {
    let term = build_typed([Object::Tensor, Object::Tensor], |_, [x, y]| vec![x + y])
        .expect("invalid term");

    let t_f = Value::Tensor(TypeExpr::NdArrayType(NdArrayType {
        dtype: DtypeExpr::Constant(Dtype::F32),
        shape: ShapeExpr::Shape(vec![NatExpr::Var(0), NatExpr::Var(1)]),
    }));

    let t_g = Value::Tensor(TypeExpr::NdArrayType(NdArrayType {
        dtype: DtypeExpr::Constant(Dtype::F32),
        shape: ShapeExpr::Shape(vec![NatExpr::Var(0), NatExpr::Var(1)]),
    }));

    let ops = catgrad_core::category::bidirectional::op_decls();
    let env = catgrad_core::nn::stdlib();

    // Typecheck
    let _result = check_with(&ops, &env, term.clone(), vec![t_f, t_g]).unwrap();
    let backend = NdArrayBackend;
    let interpreter: Interpreter<NdArrayBackend> = Interpreter::new(ops, env);

    // Construct input values with shapes (N, 1, A) and (N, A, B)
    // Using N=2, A=3, B=4 for concrete dimensions
    let data: Vec<u32> = vec![1, 2, 3, 4, 5, 6]; // Shape (2, 1, 3)
    let input = catgrad_core::interpreter::Value::NdArray(
        catgrad_core::interpreter::TaggedNdArray::from_slice(
            &backend,
            &data,
            core::Shape(vec![2, 1, 3]),
        ),
    );

    let values = vec![input.clone(), input];
    let result = interpreter.run(term, values).unwrap();

    // Check the result
    println!("Interpreter result: {result:?}");

    // Create expected result (double the input data)
    let expected_data: Vec<u32> = data.iter().map(|&x| x * 2).collect();
    let expected = catgrad_core::interpreter::Value::NdArray(
        catgrad_core::interpreter::TaggedNdArray::from_slice(
            &backend,
            &expected_data,
            core::Shape(vec![2, 1, 3]),
        ),
    );

    // Compare the result with expected
    assert_eq!(
        result[0], expected,
        "Result should be double the input data"
    );
}
