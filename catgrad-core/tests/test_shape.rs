// a 2-layer neural network with shape-variables for:
//  - input size
//  - hidden size
//  - output size

use catgrad_core::{
    category::core,
    category::shape::{
        Builder, Object, Operation, Term, Var, coannotate, matmul, reshape, shape_pack,
        shape_unpack,
    },
    ssa::{SSA, ssa},
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

////////////////////////////////////////////////////////////////////////////////
// Tests, utils

fn print_ssa(ssa: Vec<SSA<Object, Operation>>) {
    let str = ssa
        .iter()
        .map(|ssa| format!("{ssa}"))
        .collect::<Vec<_>>()
        .join("\n");
    println!("SSA:\n{str}");
}

// Check we can construct the hidden layer network
#[test]
fn construct_example_terms() {
    hidden_term();
    flatmul_term();
}

#[test]
fn test_reshape_rank2() {
    // reshape (A, B) → (A*B)
    use catgrad_core::check::*;
    use open_hypergraphs::lax::functor::*;

    let term = build_typed([Object::Tensor], |builder, [x]| {
        // Get shape of input (s)
        let (x, s) = coannotate(builder, x);

        // Assume it's rank-2, and unpack to two nats, a and b.
        let (dtype, [a, b]) = shape_unpack::<2>(builder, s);

        // Create output shape: a*b.
        let t = shape_pack::<1>(builder, dtype, [a * b]);

        // reshape input with shape argument.
        vec![reshape(builder, t, x)]
    })
    .expect("valid term");

    // Symbolic shape accepted by this term
    let ty = Value::Tensor(TypeExpr::NdArrayType(NdArrayType {
        dtype: DtypeExpr::Constant(core::Dtype::F32),
        shape: vec![NatExpr::Var(0), NatExpr::Var(1)],
    }));

    // Simplify
    let term = open_hypergraphs::lax::var::forget::Forget.map_arrow(&term);

    // debug output
    let ssa = ssa(term.clone().to_open_hypergraph());
    print_ssa(ssa);

    let result = check(term, vec![ty]).expect("checking failed");
    println!("Node values: ");
    for (i, value) in result.iter().enumerate() {
        println!("{i}: {value:?}");
    }
}
