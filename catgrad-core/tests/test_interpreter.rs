#![cfg(feature = "ndarray-backend")]

use catgrad_core::category::core;
use catgrad_core::category::lang::*;
use catgrad_core::check::*;

use catgrad_core::stdlib::*;

use catgrad_core::interpreter::Interpreter;
use catgrad_core::interpreter::backend::ndarray::NdArrayBackend;

pub mod test_models;
pub mod test_utils;
use test_models::{Add, BatchMatMul};

fn run_test_with_inputs<F>(
    TypedTerm {
        term, source_type, ..
    }: TypedTerm,
    build_inputs: F,
) -> Vec<catgrad_core::interpreter::Value<NdArrayBackend>>
where
    F: FnOnce(&NdArrayBackend) -> Vec<catgrad_core::interpreter::Value<NdArrayBackend>>,
{
    // Get stdlib / environment
    let ops = catgrad_core::stdlib::core_declarations();
    let env = catgrad_core::stdlib::stdlib();

    // Typecheck
    let _result = check_with(&ops, &env, term.clone(), source_type).unwrap();

    // Run interpreter
    let backend = NdArrayBackend;
    let interpreter: Interpreter<NdArrayBackend> = Interpreter::new(backend, ops, env);

    let values = build_inputs(&interpreter.backend);
    interpreter.run(term, values).unwrap()
}

#[test]
fn test_run_add() {
    let data: Vec<u32> = vec![1, 2, 3, 4, 5, 6]; // Shape (2, 1, 3)
    let result = run_test_with_inputs(Add.term().unwrap(), |backend| {
        let input = catgrad_core::interpreter::Value::NdArray(
            catgrad_core::interpreter::TaggedNdArray::from_slice(
                backend,
                &data,
                core::Shape(vec![2, 1, 3]),
            ),
        );
        vec![input.clone(), input]
    });

    println!("Interpreter result: {result:?}");

    // Create expected result (double the input data)
    let expected_data: Vec<u32> = data.iter().map(|&x| x * 2).collect();
    let backend = NdArrayBackend;
    let expected = catgrad_core::interpreter::Value::NdArray(
        catgrad_core::interpreter::TaggedNdArray::from_slice(
            &backend,
            &expected_data,
            core::Shape(vec![2, 1, 3]),
        ),
    );

    assert_eq!(
        result[0], expected,
        "Result should be double the input data"
    );
}

#[test]
fn test_run_batch_matmul() {
    // Construct batch matmul inputs with shapes [2, 2, 2] × [2, 2, 1] = [2, 2, 1]
    // Batch 0: [[1, 2], [3, 4]] × [[1], [2]] = [[5], [11]]
    // Batch 1: [[5, 6], [7, 8]] × [[3], [4]] = [[39], [53]]
    let x0_data: Vec<f32> = vec![
        1.0, 2.0, 3.0, 4.0, // batch 0
        5.0, 6.0, 7.0, 8.0, // batch 1
    ];
    let x1_data: Vec<f32> = vec![
        1.0, 2.0, // batch 0
        3.0, 4.0, // batch 1
    ];

    let result = run_test_with_inputs(BatchMatMul.term().unwrap(), |backend| {
        let x0 = catgrad_core::interpreter::Value::NdArray(
            catgrad_core::interpreter::TaggedNdArray::from_slice(
                backend,
                &x0_data,
                core::Shape(vec![2, 2, 2]),
            ),
        );

        let x1 = catgrad_core::interpreter::Value::NdArray(
            catgrad_core::interpreter::TaggedNdArray::from_slice(
                backend,
                &x1_data,
                core::Shape(vec![2, 2, 1]),
            ),
        );

        vec![x0, x1]
    });

    println!("Batch matmul result: {result:?}");

    // Create expected result
    let expected_data: Vec<f32> = vec![
        5.0, 11.0, // batch 0: [1*1+2*2, 3*1+4*2]
        39.0, 53.0, // batch 1: [5*3+6*4, 7*3+8*4]
    ];
    let backend = NdArrayBackend;
    let expected = catgrad_core::interpreter::Value::NdArray(
        catgrad_core::interpreter::TaggedNdArray::from_slice(
            &backend,
            &expected_data,
            core::Shape(vec![2, 2, 1]),
        ),
    );

    assert_eq!(
        result[0], expected,
        "Batch matmul result should match expected output"
    );
}
