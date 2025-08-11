use catgrad_core::category::bidirectional::*;
use catgrad_core::nn::*;
use catgrad_core::svg::to_svg;
use catgrad_core::util::build_typed;

use catgrad_core::check::*;

fn save_diagram_if_enabled(filename: &str, data: Vec<u8>) {
    if std::env::var("SAVE_DIAGRAMS").is_ok() {
        let output_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("images")
            .join(filename);
        std::fs::write(output_path, data).expect("write diagram file");
    }
}

////////////////////////////////////////////////////////////////////////////////
// Example program

// (N, 1, A) × (N, A, B) → (N, B)
// matmul ; sigmoid ; reshape
pub fn linear_sigmoid() -> Term {
    let term = build_typed([Object::Tensor, Object::Tensor], |graph, [x, p]| {
        let x = matmul(graph, x, p);
        let x = sigmoid(graph, x);

        // flatten result shape
        let [a, c] = unpack::<2>(graph, shape(graph, x.clone()));
        let t = pack::<1>(graph, [a * c]);

        vec![reshape(graph, t, x)]
    });

    // TODO: construct *type* maps

    term.expect("invalid term")
}

#[test]
fn test_construct_linear_sigmoid() {
    let sigmoid = sigmoid_term();
    println!("{sigmoid:?}");

    let term = linear_sigmoid();
    println!("{term:?}");
}

#[test]
fn test_graph_sigmoid() {
    let term = sigmoid_term();
    use open_hypergraphs::lax::functor::*;

    let term = open_hypergraphs::lax::var::forget::Forget.map_arrow(&term);
    let svg_bytes = to_svg(&term).expect("create svg");
    save_diagram_if_enabled("test_graph_sigmoid.svg", svg_bytes);
}

#[test]
fn test_graph_linear_sigmoid() {
    let term = linear_sigmoid();

    use open_hypergraphs::lax::functor::*;
    let term = open_hypergraphs::lax::var::forget::Forget.map_arrow(&term);

    let svg_bytes = to_svg(&term).expect("create svg");
    save_diagram_if_enabled("test_graph_linear_sigmoid.svg", svg_bytes);
}

// Shapecheck the linear-sigmoid term.
// This should allow us to generate a diagram similar to the one in test_graph_linear_sigmoid(),
// but where objects are "symbolic shapes".
#[test]
fn test_check_linear_sigmoid() {
    let term = linear_sigmoid();

    let t_f = Value::Tensor(TypeExpr::NdArrayType(NdArrayType {
        dtype: DtypeExpr::Constant(Dtype::F32),
        shape: ShapeExpr::Shape(vec![NatExpr::Var(0), NatExpr::Var(1)]),
    }));

    let t_g = Value::Tensor(TypeExpr::NdArrayType(NdArrayType {
        dtype: DtypeExpr::Constant(Dtype::F32),
        shape: ShapeExpr::Shape(vec![NatExpr::Var(1), NatExpr::Var(2)]),
    }));

    use open_hypergraphs::lax::functor::*;
    let term = open_hypergraphs::lax::var::forget::Forget.map_arrow(&term);

    let ops = catgrad_core::category::bidirectional::op_decls();
    let mut env = catgrad_core::nn::stdlib();
    for def in env.operations.values_mut() {
        def.term = open_hypergraphs::lax::var::forget::Forget.map_arrow(&def.term);
    }

    let result = check_with(&ops, &env, term.clone(), vec![t_f, t_g]).expect("valid");
    println!("result: {result:?}");

    // .... sigh.
    use open_hypergraphs::lax::{Hypergraph, OpenHypergraph};
    let term = OpenHypergraph {
        sources: term.sources,
        targets: term.targets,
        hypergraph: Hypergraph {
            nodes: result,
            edges: term.hypergraph.edges,
            adjacency: term.hypergraph.adjacency,
            quotient: term.hypergraph.quotient,
        },
    };

    let svg_bytes = to_svg(&term).expect("create svg");
    save_diagram_if_enabled("test_check_linear_sigmoid.svg", svg_bytes);
}

#[test]
fn test_check_sigmoid() {
    let term = sigmoid_term();
    let t = Value::Tensor(TypeExpr::NdArrayType(NdArrayType {
        dtype: DtypeExpr::Constant(Dtype::F32),
        shape: ShapeExpr::Var(0),
    }));

    use open_hypergraphs::lax::functor::*;
    let term = open_hypergraphs::lax::var::forget::Forget.map_arrow(&term);

    let ops = catgrad_core::category::bidirectional::op_decls();
    let mut env = catgrad_core::nn::stdlib();
    for def in env.operations.values_mut() {
        def.term = open_hypergraphs::lax::var::forget::Forget.map_arrow(&def.term);
    }

    let result = check_with(&ops, &env, term.clone(), vec![t]).expect("valid");
    println!("result: {result:?}");

    // .... sigh.
    use open_hypergraphs::lax::{Hypergraph, OpenHypergraph};
    let term = OpenHypergraph {
        sources: term.sources,
        targets: term.targets,
        hypergraph: Hypergraph {
            nodes: result,
            edges: term.hypergraph.edges,
            adjacency: term.hypergraph.adjacency,
            quotient: term.hypergraph.quotient,
        },
    };

    let svg_bytes = to_svg(&term).expect("create svg");
    save_diagram_if_enabled("test_check_sigmoid.svg", svg_bytes);
}

#[test]
fn test_check_exp() {
    let term = exp_term();
    let t = Value::Tensor(TypeExpr::NdArrayType(NdArrayType {
        dtype: DtypeExpr::Constant(Dtype::F32),
        shape: ShapeExpr::Var(0),
    }));

    use open_hypergraphs::lax::functor::*;
    let term = open_hypergraphs::lax::var::forget::Forget.map_arrow(&term);

    let ops = catgrad_core::category::bidirectional::op_decls();
    let mut env = catgrad_core::nn::stdlib();
    for def in env.operations.values_mut() {
        def.term = open_hypergraphs::lax::var::forget::Forget.map_arrow(&def.term);
    }

    let result = check_with(&ops, &env, term.clone(), vec![t]).expect("valid");
    println!("result: {result:?}");

    // .... sigh.
    use open_hypergraphs::lax::{Hypergraph, OpenHypergraph};
    let term = OpenHypergraph {
        sources: term.sources,
        targets: term.targets,
        hypergraph: Hypergraph {
            nodes: result,
            edges: term.hypergraph.edges,
            adjacency: term.hypergraph.adjacency,
            quotient: term.hypergraph.quotient,
        },
    };

    let svg_bytes = to_svg(&term).expect("create svg");
    save_diagram_if_enabled("test_check_exp.svg", svg_bytes);
}

/*
#[test]
fn test_cyclic_definition_fails() {
    todo!()
}
*/
