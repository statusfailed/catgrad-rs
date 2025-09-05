//! Tests for the catgrad reference interpreter

use super::backend::ndarray::NdArrayBackend;
use super::{TaggedNdArray, Value, lit_to_value};
use crate::category::core::Shape;
use crate::category::lang::Literal;

#[test]
fn test_literal_u32_scalar() {
    let literal = Literal::U32(42);
    let result: Value<NdArrayBackend> = lit_to_value(&NdArrayBackend, &literal);

    match result {
        Value::NdArray(arr) => {
            assert_eq!(arr.shape().0, vec![] as Vec<usize>);
            match arr {
                TaggedNdArray::U32([nd_arr]) => {
                    assert_eq!(nd_arr[[]], 42);
                }
                _ => panic!("Expected U32 TaggedNdArray"),
            }
        }
        _ => panic!("Expected NdArray value for U32 literal"),
    }
}

#[test]
fn test_literal_f32_scalar() {
    let literal = Literal::F32(3.15);
    let result: Value<NdArrayBackend> = lit_to_value(&NdArrayBackend, &literal);

    match result {
        Value::NdArray(arr) => {
            assert_eq!(arr.shape().0, vec![] as Vec<usize>);
            match arr {
                TaggedNdArray::F32([nd_arr]) => {
                    assert_eq!(nd_arr[[]], 3.15);
                }
                _ => panic!("Expected F32 TaggedNdArray"),
            }
        }
        _ => panic!("Expected NdArray value for F32 literal"),
    }
}

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
