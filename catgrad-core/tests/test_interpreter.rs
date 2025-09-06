#![cfg(feature = "ndarray-backend")]

use catgrad_core::category::core::Shape;
use catgrad_core::category::lang::*;
use catgrad_core::{check, check::*};

use catgrad_core::stdlib::*;

use catgrad_core::interpreter::backend::ndarray::NdArrayBackend;
use catgrad_core::interpreter::{Interpreter, Parameters, tensor};

pub mod test_models;
pub mod test_utils;
use catgrad_core::stdlib::nn::Exp;
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
    let env = catgrad_core::stdlib::stdlib();

    // Typecheck
    let _result = check_with(
        &env,
        &check::Parameters::default(),
        term.clone(),
        source_type,
    )
    .unwrap();

    // Run interpreter
    let backend = NdArrayBackend;
    let interpreter: Interpreter<NdArrayBackend> =
        Interpreter::new(backend, env, Parameters::default());

    let values = build_inputs(&interpreter.backend);
    interpreter.run(term, values).unwrap()
}

#[test]
fn test_run_add() {
    let data: Vec<u32> = vec![1, 2, 3, 4, 5, 6]; // Shape (2, 1, 3)
    let result = run_test_with_inputs(Add.term().unwrap(), |backend| {
        let input = tensor(backend, Shape(vec![2, 1, 3]), &data).unwrap();
        vec![input.clone(), input]
    });

    println!("Interpreter result: {result:?}");

    // Create expected result (double the input data)
    let expected_data: Vec<u32> = data.iter().map(|&x| x * 2).collect();
    let backend = NdArrayBackend;
    let expected = tensor(&backend, Shape(vec![2, 1, 3]), &expected_data).unwrap();

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
        let x0 = tensor(backend, Shape(vec![2, 2, 2]), &x0_data).unwrap();
        let x1 = tensor(backend, Shape(vec![2, 2, 1]), &x1_data).unwrap();
        vec![x0, x1]
    });

    let backend = NdArrayBackend;
    // Create expected result
    let expected_data: Vec<f32> = vec![
        5.0, 11.0, // batch 0: [1*1+2*2, 3*1+4*2]
        39.0, 53.0, // batch 1: [5*3+6*4, 7*3+8*4]
    ];
    let expected = tensor(&backend, Shape(vec![2, 2, 1]), &expected_data).unwrap();

    assert_eq!(
        result[0], expected,
        "Batch matmul result should match expected output"
    );
}

fn allclose_f32(a: &[f32], b: &[f32], rtol: f32, atol: f32) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter().zip(b.iter()).all(|(&x, &y)| {
        let diff = (x - y).abs();
        diff <= atol + rtol * y.abs()
    })
}

#[test]
fn test_run_exp() {
    let data: Vec<f32> = vec![0.0, 1.0, 2.0, -1.0]; // Shape (2, 2)
    let result = run_test_with_inputs(Exp.term().unwrap(), |backend| {
        vec![tensor(backend, Shape(vec![2, 2]), &data).unwrap()]
    });

    // make sure actual result is a single F32 array
    use catgrad_core::interpreter::{TaggedNdArray, Value};
    let actual = match &result[..] {
        [Value::NdArray(TaggedNdArray::F32([actual]))] => actual,
        xs => panic!("wrong output type: {xs:?}"),
    };

    // Create expected result (e^x for each element)
    let expected: Vec<f32> = data.iter().map(|&x| x.exp()).collect();

    assert!(
        allclose_f32(actual.as_slice().unwrap(), &expected, 1e-5, 1e-8),
        "actual should be close to expected!"
    );
}
