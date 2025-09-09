#![cfg(feature = "candle-backend")]

use catgrad_core::category::core::Shape;
use catgrad_core::interpreter::backend::Backend;
use catgrad_core::interpreter::backend::candle::CandleBackend;
use catgrad_core::interpreter::{TaggedNdArray, TaggedNdArrayTuple};

#[test]
fn test_candle_backend_basic_operations() {
    let backend = CandleBackend::new();

    // Test scalar creation
    let scalar: <CandleBackend as Backend>::NdArray<f32> = backend.scalar(42.0f32);
    // Note: Candle creates a tensor with shape [1] for scalars, not []
    assert_eq!(scalar.0.shape().dims(), &[1]);

    // Test zeros creation
    let zeros: <CandleBackend as Backend>::NdArray<f32> = backend.zeros::<f32>(Shape(vec![2, 3]));
    assert_eq!(zeros.0.shape().dims(), &[2, 3]);

    // Test tensor creation from slice
    let data = vec![1.0f32, 2.0, 3.0, 4.0];
    let tensor: <CandleBackend as Backend>::NdArray<f32> = backend
        .ndarray_from_slice(&data, Shape(vec![2, 2]))
        .unwrap();
    assert_eq!(tensor.0.shape().dims(), &[2, 2]);
}

#[test]
fn test_candle_backend_arithmetic() {
    let backend = CandleBackend::new();

    // Create two tensors
    let data1 = vec![1.0f32, 2.0, 3.0, 4.0];
    let data2 = vec![2.0f32, 3.0, 4.0, 5.0];

    let tensor1: <CandleBackend as Backend>::NdArray<f32> = backend
        .ndarray_from_slice(&data1, Shape(vec![2, 2]))
        .unwrap();
    let tensor2: <CandleBackend as Backend>::NdArray<f32> = backend
        .ndarray_from_slice(&data2, Shape(vec![2, 2]))
        .unwrap();

    // Test addition
    let result = backend.add(TaggedNdArrayTuple::F32([tensor1, tensor2]));
    match result {
        TaggedNdArray::F32([arr]) => {
            assert_eq!(arr.0.shape().dims(), &[2, 2]);
        }
        _ => panic!("Expected F32 result"),
    }

    // Test multiplication
    let data3 = vec![1.0f32, 2.0, 3.0, 4.0];
    let data4 = vec![2.0f32, 3.0, 4.0, 5.0];
    let tensor3: <CandleBackend as Backend>::NdArray<f32> = backend
        .ndarray_from_slice(&data3, Shape(vec![2, 2]))
        .unwrap();
    let tensor4: <CandleBackend as Backend>::NdArray<f32> = backend
        .ndarray_from_slice(&data4, Shape(vec![2, 2]))
        .unwrap();

    let result = backend.mul(TaggedNdArrayTuple::F32([tensor3, tensor4]));
    match result {
        TaggedNdArray::F32([arr]) => {
            assert_eq!(arr.0.shape().dims(), &[2, 2]);
        }
        _ => panic!("Expected F32 result"),
    }
}

#[test]
fn test_candle_backend_matmul() {
    let backend = CandleBackend::new();

    // Create matrices for matmul: [2, 3] × [3, 2] = [2, 2]
    let data1 = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // [2, 3]
    let data2 = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // [3, 2]

    let tensor1: <CandleBackend as Backend>::NdArray<f32> = backend
        .ndarray_from_slice(&data1, Shape(vec![2, 3]))
        .unwrap();
    let tensor2: <CandleBackend as Backend>::NdArray<f32> = backend
        .ndarray_from_slice(&data2, Shape(vec![3, 2]))
        .unwrap();

    let result = backend.matmul(TaggedNdArrayTuple::F32([tensor1, tensor2]));
    match result {
        TaggedNdArray::F32([arr]) => {
            assert_eq!(arr.0.shape().dims(), &[2, 2]);
        }
        _ => panic!("Expected F32 result"),
    }
}

#[test]
fn test_candle_backend_reshape() {
    let backend = CandleBackend::new();

    let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let tensor: <CandleBackend as Backend>::NdArray<f32> = backend
        .ndarray_from_slice(&data, Shape(vec![2, 3]))
        .unwrap();

    let reshaped = backend.reshape(TaggedNdArray::F32([tensor]), Shape(vec![3, 2]));
    match reshaped {
        TaggedNdArray::F32([arr]) => {
            assert_eq!(arr.0.shape().dims(), &[3, 2]);
        }
        _ => panic!("Expected F32 result"),
    }
}

#[test]
fn test_candle_backend_cast() {
    let backend = CandleBackend::new();

    let data = vec![1.0f32, 2.0, 3.0, 4.0];
    let tensor: <CandleBackend as Backend>::NdArray<f32> = backend
        .ndarray_from_slice(&data, Shape(vec![2, 2]))
        .unwrap();

    // Cast F32 to U32
    let casted = backend.cast(
        TaggedNdArray::F32([tensor]),
        catgrad_core::category::core::Dtype::U32,
    );
    match casted {
        TaggedNdArray::U32([arr]) => {
            assert_eq!(arr.0.shape().dims(), &[2, 2]);
        }
        _ => panic!("Expected U32 result"),
    }
}
