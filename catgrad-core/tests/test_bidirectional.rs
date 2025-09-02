use catgrad_core::category::lang::*;
use catgrad_core::check::*;
use catgrad_core::nn::*;
use catgrad_core::svg::to_svg;
use catgrad_core::util::build_typed;

pub mod test_utils;
use test_utils::{get_forget_op_decls, replace_nodes_in_hypergraph, save_diagram_if_enabled};

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

    run_check_test(term, vec![t_f, t_g], "test_check_linear_sigmoid.svg").expect("valid");
}

#[test]
fn test_check_sigmoid() {
    let term = sigmoid_term();
    let t = Value::Tensor(TypeExpr::NdArrayType(NdArrayType {
        dtype: DtypeExpr::Constant(Dtype::F32),
        shape: ShapeExpr::Var(0),
    }));

    run_check_test(term, vec![t], "test_check_sigmoid.svg").expect("valid");
}

#[test]
fn test_check_exp() {
    let term = exp_term();
    let t = Value::Tensor(TypeExpr::NdArrayType(NdArrayType {
        dtype: DtypeExpr::Constant(Dtype::F32),
        shape: ShapeExpr::Var(0),
    }));

    run_check_test(term, vec![t], "test_check_exp.svg").expect("valid");
}

#[allow(clippy::result_large_err)]
pub fn run_check_test(
    term: catgrad_core::category::lang::Term,
    input_types: Vec<Value>,
    svg_filename: &str,
) -> Result<(), ShapeCheckError> {
    use open_hypergraphs::lax::functor::*;

    let term = open_hypergraphs::lax::var::forget::Forget.map_arrow(&term);
    let (ops, env) = get_forget_op_decls();

    let result = check_with(&ops, &env, term.clone(), input_types)?;
    println!("result: {result:?}");

    let typed_term = replace_nodes_in_hypergraph(term, result);
    let svg_bytes = to_svg(&typed_term).expect("create svg");
    save_diagram_if_enabled(svg_filename, svg_bytes);

    Ok(())
}

/*
#[test]
fn test_cyclic_definition_fails() {
    todo!()
}
*/
