//! Tests for the catgrad reference interpreter

use super::TaggedNdArray;
use super::backend::ndarray::NdArrayBackend;
use crate::category::core::Shape;

#[test]
fn test_tagged_ndarray_constructors() {
    let backend = NdArrayBackend;

    // Test scalar constructor
    let scalar_f32: TaggedNdArray<NdArrayBackend> = TaggedNdArray::scalar(&NdArrayBackend, 2.5f32);
    assert_eq!(scalar_f32.shape().0, vec![] as Vec<usize>);

    let scalar_u32: TaggedNdArray<NdArrayBackend> = TaggedNdArray::scalar(&NdArrayBackend, 100u32);
    assert_eq!(scalar_u32.shape().0, vec![] as Vec<usize>);

    // Test from_slice constructor
    let matrix =
        TaggedNdArray::from_slice(&backend, &[1.0f32, 2.0, 3.0, 4.0], Shape(vec![2, 2])).unwrap();
    assert_eq!(matrix.shape().0, vec![2, 2]);

    let vector = TaggedNdArray::from_slice(&backend, &[10u32, 20, 30], Shape(vec![3])).unwrap();
    assert_eq!(vector.shape().0, vec![3]);
}
