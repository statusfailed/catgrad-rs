#![cfg(feature = "ndarray-backend")]

use catgrad::category::core::Shape;
use catgrad::category::lang::*;
use catgrad::{typecheck, typecheck::*};

use catgrad::stdlib::*;

use catgrad::interpreter::backend::Backend;
use catgrad::interpreter::backend::ndarray::NdArrayBackend;
use catgrad::interpreter::{
    Interpreter, Parameters, TaggedTensor, TaggedTensorTuple, Value, tensor,
};

pub mod test_models;
pub mod test_utils;
use catgrad::stdlib::nn::Exp;
use test_models::{Add, BatchMatMul, TopK};

fn run_test_with_inputs<F>(
    TypedTerm {
        term, source_type, ..
    }: TypedTerm,
    build_inputs: F,
) -> Vec<catgrad::interpreter::Value<NdArrayBackend>>
where
    F: FnOnce(&NdArrayBackend) -> Vec<catgrad::interpreter::Value<NdArrayBackend>>,
{
    // Get stdlib / environment
    let env = catgrad::stdlib::stdlib();

    // Typecheck
    let _result = check_with(
        &env,
        &typecheck::Parameters::default(),
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

    let backend = NdArrayBackend;
    match (&result[0], &expected) {
        (Value::Tensor(TaggedTensor::U32([actual])), Value::Tensor(TaggedTensor::U32([exp]))) => {
            assert!(
                backend.compare(TaggedTensorTuple::U32([actual.clone(), exp.clone()])),
                "Result should be double the input data"
            );
        }
        _ => panic!("Expected U32 tensors"),
    }
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
    let backend = NdArrayBackend;
    match (&result[0], &expected) {
        (Value::Tensor(TaggedTensor::F32([actual])), Value::Tensor(TaggedTensor::F32([exp]))) => {
            assert!(
                backend.compare(TaggedTensorTuple::F32([actual.clone(), exp.clone()])),
                "Batch matmul result should match expected output"
            );
        }
        _ => panic!("Expected F32 tensors"),
    }
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
    use catgrad::interpreter::{TaggedVec, Value};

    let data: Vec<f32> = vec![0.0, 1.0, 2.0, -1.0]; // Shape (2, 2)
    let mut result = run_test_with_inputs(Exp.term().unwrap(), |backend| {
        vec![tensor(backend, Shape(vec![2, 2]), &data).unwrap()]
    });

    let actual = if let Some(Value::Tensor(tensor)) = result.pop() {
        assert!(result.is_empty()); // we only had one result
        tensor
    } else {
        panic!("Invalid result type");
    };

    // Create expected result (e^x for each element)
    let expected: Vec<f32> = data.iter().map(|&x| x.exp()).collect();
    let backend = NdArrayBackend;

    if let TaggedVec::F32(actual) = backend.to_vec(actual) {
        let is_close = allclose_f32(&actual, &expected, 1e-5, 1e-8);
        assert!(is_close, "actual should be close to expected!");
    } else {
        panic!("wrong tensor dtype");
    }
}

#[test]
fn test_run_topk() {
    let data: Vec<f32> = vec![
        0.1, 5.0, 3.0, 10.0, 5.0, //
        8.0, 9.0, 7.0, 6.0, 8.0,
    ];
    let result = run_test_with_inputs(TopK.term().unwrap(), |backend| {
        vec![tensor(backend, Shape(vec![2, 5]), &data).unwrap()]
    });

    assert_eq!(result.len(), 2);

    let backend = NdArrayBackend;
    let expected_values_data = vec![10.0f32, 5.0, 9.0, 8.0];
    let expected_indices_data = vec![3u32, 1, 1, 0];
    let expected_values = tensor(&backend, Shape(vec![2, 2]), &expected_values_data).unwrap();
    let expected_indices = tensor(&backend, Shape(vec![2, 2]), &expected_indices_data).unwrap();

    match (&result[0], &expected_values) {
        (Value::Tensor(TaggedTensor::F32([actual])), Value::Tensor(TaggedTensor::F32([exp]))) => {
            assert!(
                backend.compare(TaggedTensorTuple::F32([actual.clone(), exp.clone()])),
                "topk values should match expected output"
            );
        }
        _ => panic!("Expected F32 tensor for topk values"),
    }

    match (&result[1], &expected_indices) {
        (Value::Tensor(TaggedTensor::U32([actual])), Value::Tensor(TaggedTensor::U32([exp]))) => {
            assert!(
                backend.compare(TaggedTensorTuple::U32([actual.clone(), exp.clone()])),
                "topk indices should match expected output"
            );
        }
        _ => panic!("Expected U32 tensor for topk indices"),
    }
}
