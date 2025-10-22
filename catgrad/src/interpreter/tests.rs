//! Tests for the catgrad reference interpreter

use super::TaggedTensor;
use super::backend::ndarray::NdArrayBackend;
use crate::category::core::Shape;

#[test]
fn test_tagged_ndarray_constructors() {
    let backend = NdArrayBackend;

    // Test scalar constructor
    let scalar_f32: TaggedTensor<NdArrayBackend> =
        TaggedTensor::scalar(&NdArrayBackend, 2.5, crate::category::core::Dtype::F32);
    assert_eq!(scalar_f32.shape().0, vec![] as Vec<usize>);

    let scalar_u32: TaggedTensor<NdArrayBackend> =
        TaggedTensor::scalar(&NdArrayBackend, 100.0, crate::category::core::Dtype::U32);
    assert_eq!(scalar_u32.shape().0, vec![] as Vec<usize>);

    // Test from_slice constructor
    let matrix =
        TaggedTensor::from_slice(&backend, &[1.0f32, 2.0, 3.0, 4.0], Shape(vec![2, 2])).unwrap();
    assert_eq!(matrix.shape().0, vec![2, 2]);

    let vector = TaggedTensor::from_slice(&backend, &[10u32, 20, 30], Shape(vec![3])).unwrap();
    assert_eq!(vector.shape().0, vec![3]);
}
