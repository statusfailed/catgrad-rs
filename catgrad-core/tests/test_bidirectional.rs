use catgrad_core::category::bidirectional::*;
use catgrad_core::nn::*;
use catgrad_core::util::build_typed;

////////////////////////////////////////////////////////////////////////////////
// Example program

// (N, 1, A) × (N, A, B) → (N, B)
// matmul ; sigmoid ; reshape
pub fn linear_sigmoid() -> Term {
    let term = build_typed([Object::Tensor, Object::Tensor], |graph, [x, p]| {
        let s = shape(graph, x.clone());
        let x = matmul(graph, x, p);
        let x = sigmoid(graph, x);

        let (dtype, [a, b]) = unpack::<2>(graph, s);
        let t = pack::<1>(graph, dtype, [a * b]);

        vec![reshape(graph, t, x)]
    });

    // TODO: construct *type* maps

    term.expect("invalid term")
}

#[test]
fn test_linear_sigmoid() {
    let term = linear_sigmoid();
    println!("{term:?}");
}
