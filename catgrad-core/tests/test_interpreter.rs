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

#[test]
fn test_run_batch_matmul() {
    let term = build_typed([Object::Tensor, Object::Tensor], |graph, [x, y]| {
        vec![matmul(graph, x, y)]
    })
    .expect("invalid term");

    let t_lhs = Value::Tensor(TypeExpr::NdArrayType(NdArrayType {
        dtype: DtypeExpr::Constant(Dtype::F32),
        shape: ShapeExpr::Shape(vec![NatExpr::Var(0), NatExpr::Var(1), NatExpr::Var(2)]),
    }));

    let t_rhs = Value::Tensor(TypeExpr::NdArrayType(NdArrayType {
        dtype: DtypeExpr::Constant(Dtype::F32),
        shape: ShapeExpr::Shape(vec![NatExpr::Var(0), NatExpr::Var(2), NatExpr::Var(3)]),
    }));

    let ops = catgrad_core::category::bidirectional::op_decls();
    let env = catgrad_core::nn::stdlib();

    // Typecheck
    let _result = check_with(&ops, &env, term.clone(), vec![t_lhs, t_rhs]).unwrap();
    let backend = NdArrayBackend;
    let interpreter: Interpreter<NdArrayBackend> = Interpreter::new(ops, env);

    // Construct batch matmul inputs with shapes [2, 2, 2] × [2, 2, 1] = [2, 2, 1]
    // Batch 0: [[1, 2], [3, 4]] × [[1], [2]] = [[5], [11]]
    // Batch 1: [[5, 6], [7, 8]] × [[3], [4]] = [[39], [53]]
    let lhs_data: Vec<f32> = vec![
        1.0, 2.0, 3.0, 4.0, // batch 0
        5.0, 6.0, 7.0, 8.0, // batch 1
    ];
    let lhs = catgrad_core::interpreter::Value::NdArray(
        catgrad_core::interpreter::TaggedNdArray::from_slice(
            &backend,
            &lhs_data,
            core::Shape(vec![2, 2, 2]),
        ),
    );

    let rhs_data: Vec<f32> = vec![
        1.0, 2.0, // batch 0
        3.0, 4.0, // batch 1
    ];
    let rhs = catgrad_core::interpreter::Value::NdArray(
        catgrad_core::interpreter::TaggedNdArray::from_slice(
            &backend,
            &rhs_data,
            core::Shape(vec![2, 2, 1]),
        ),
    );

    let values = vec![lhs, rhs];
    let result = interpreter.run(term, values).unwrap();

    // Check the result
    println!("Batch matmul result: {result:?}");

    // Create expected result
    let expected_data: Vec<f32> = vec![
        5.0, 11.0, // batch 0: [1*1+2*2, 3*1+4*2]
        39.0, 53.0, // batch 1: [5*3+6*4, 7*3+8*4]
    ];
    let expected = catgrad_core::interpreter::Value::NdArray(
        catgrad_core::interpreter::TaggedNdArray::from_slice(
            &backend,
            &expected_data,
            core::Shape(vec![2, 2, 1]),
        ),
    );

    // Compare the result with expected
    assert_eq!(
        result[0], expected,
        "Batch matmul result should match expected output"
    );
}
