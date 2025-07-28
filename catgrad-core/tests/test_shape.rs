// a 2-layer neural network with shape-variables for:
//  - input size
//  - hidden size
//  - output size

use catgrad_core::{
    category::core,
    category::shape::{
        Builder, Object, Term, Var, annotate, coannotate, dtype_constant, matmul, reshape,
        shape_pack, shape_unpack,
    },
    check::check,
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

fn reshape_rank2(builder: &Builder, x: Var) -> Var {
    let (x, t) = coannotate(builder, x.clone());
    let (dtype, [a, b]) = shape_unpack::<2>(builder, t);
    let s = shape_pack::<1>(builder, dtype, [a * b]);

    reshape(builder, s, x)
}

fn reshape_rank2_term() -> Term {
    use Object::*;
    // TODO: how do we input the tensor type?
    build_typed([Tensor, Nat, Nat], |graph, [x, a, b]| {
        let dtype = dtype_constant(graph, core::Dtype::F32);
        let s = shape_pack::<2>(graph, dtype, [a, b]);
        let x = annotate(graph, x, s);
        let result = vec![reshape_rank2(graph, x)];
        result
    })
    .expect("valid term")
}

// flatmul is a matmul-and-then-reshape
// flatmul : (t: Shape) (p: s + (a, b)) (x : s + (b, c)) (t \cong s + (a, c)) : (y : t)
fn flatmul(builder: &Builder, t: Var, f: Var, g: Var) -> Var {
    reshape(builder, t, matmul(builder, f, g))
}

fn flatmul_term() -> Term {
    use Object::*;
    build_typed([NdArrayType, Tensor, Tensor], |graph, [t, f, g]| {
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

#[test]
fn test_reshape_rank2() {
    use catgrad_core::ssa::ssa;
    // reshape (A, B) → (A*B)
    let term = reshape_rank2_term();
    let ssa = ssa(term.clone().to_open_hypergraph());
    let debug = ssa
        .iter()
        .map(|ssa| format!("{ssa}"))
        .collect::<Vec<_>>()
        .join("\n");
    println!("{debug}");
    println!("{:?}", check(term));
}
