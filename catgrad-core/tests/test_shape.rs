// a 2-layer neural network with shape-variables for:
//  - input size
//  - hidden size
//  - output size

use catgrad_core::{
    category::shape::{Builder, Object, Term, Var, matmul, reshape},
    util::build_typed,
};

////////////////////////////////////////////////////////////////////////////////
// Example layers

// linear : (1, m) × (m, n) → (1, n)
fn linear(builder: &Builder, x: Var, p: Var) -> Var {
    matmul(builder, x, p)
}

// A 2-fold composition of linear layers
fn linear2(builder: &Builder, p: Var, q: Var, x: Var) -> Var {
    let x = linear(builder, p, x);
    linear(builder, q, x)
}

////////////////////////////////////////////////////////////////////////////////
// Example terms

// flatmul is a matmul-and-then-reshape
// flatmul : (t: Shape) (p: s + (a, b)) (x : s + (b, c)) (t \cong s + (a, c)) : (y : t)
fn flatmul(builder: &Builder, t: Var, f: Var, g: Var) -> Var {
    reshape(builder, t, matmul(builder, f, g))
}

fn flatmul_term() -> Term {
    use Object::*;
    build_typed([Shape, Tensor, Tensor], |graph, [t, f, g]| {
        let result = vec![flatmul(graph, t, f, g)];
        result
    })
    .expect("valid term")
}

// Assemble the full term for `hidden`
fn hidden_term() -> Term {
    use Object::*;
    build_typed([Tensor, Tensor, Tensor], |graph, [p, q, x]| {
        let result = vec![linear2(graph, p, q, x)];
        result
    })
    .expect("valid term")
}

// Check we can construct the hidden layer network
#[test]
fn construct_example_terms() {
    hidden_term();
    flatmul_term();
}
