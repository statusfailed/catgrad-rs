use super::{Interpreter, Value, lit_to_value};
use crate::category::bidirectional::{Environment, Literal, Object, Operation};
use open_hypergraphs::lax::OpenHypergraph;
use std::collections::HashMap;

#[test]
fn test_literal_u32() {
    let literal = Literal::U32(42);
    let result = lit_to_value(&literal);

    match result {
        Value::NdArray(tensor) => {
            assert_eq!(tensor.shape, vec![]);
            assert_eq!(tensor.strides, vec![]);
            assert_eq!(tensor.offset, 0);
            assert_eq!(tensor.buf, 42u32.to_ne_bytes().to_vec());
        }
        _ => panic!("Expected NdArray value for U32 literal"),
    }
}

#[test]
fn test_literal_f32() {
    let literal = Literal::F32(3.15);
    let result = lit_to_value(&literal);

    match result {
        Value::NdArray(tensor) => {
            assert_eq!(tensor.shape, vec![]);
            assert_eq!(tensor.strides, vec![]);
            assert_eq!(tensor.offset, 0);
            assert_eq!(tensor.buf, 3.15f32.to_ne_bytes().to_vec());
        }
        _ => panic!("Expected NdArray value for F32 literal"),
    }
}

#[test]
fn test_singleton_literal_evaluation() {
    let literal = Literal::U32(123);
    let op = Operation::Literal(literal.clone());

    // Create singleton OpenHypergraph: no sources, one target of type Object::Tensor
    let term = OpenHypergraph::singleton(op, vec![], vec![Object::Tensor]);

    let interpreter = Interpreter::new(
        HashMap::new(),
        Environment {
            operations: HashMap::new(),
        },
    );

    // Run with no input values
    let results = interpreter.run(term, vec![]).unwrap();

    // Should get one result
    assert_eq!(results.len(), 1);

    // Should match direct lit_to_value result
    let expected = lit_to_value(&literal);
    assert_eq!(results[0], expected);
}
