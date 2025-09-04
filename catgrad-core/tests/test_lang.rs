use catgrad_core::category::lang::*;
use catgrad_core::check::*;
use catgrad_core::stdlib::{nn::*, *};
use catgrad_core::svg::to_svg;
use catgrad_core::util::build_typed;

pub mod test_utils;
use test_utils::{
    get_forget_core_declarations, replace_nodes_in_hypergraph, save_diagram_if_enabled,
};

////////////////////////////////////////////////////////////////////////////////
// Example program

struct LinearSigmoid;
impl Def<2, 1> for LinearSigmoid {
    fn ty(&self) -> ([Type; 2], [Type; 1]) {
        let t_x = Value::Tensor(TypeExpr::NdArrayType(NdArrayType {
            dtype: DtypeExpr::Constant(Dtype::F32),
            shape: ShapeExpr::Shape(vec![NatExpr::Var(0), NatExpr::Var(1)]),
        }));

        let t_p = Value::Tensor(TypeExpr::NdArrayType(NdArrayType {
            dtype: DtypeExpr::Constant(Dtype::F32),
            shape: ShapeExpr::Shape(vec![NatExpr::Var(1), NatExpr::Var(2)]),
        }));

        let t_y = Value::Tensor(TypeExpr::NdArrayType(NdArrayType {
            dtype: DtypeExpr::Constant(Dtype::F32),
            shape: ShapeExpr::Shape(vec![NatExpr::Mul(vec![NatExpr::Var(1), NatExpr::Var(2)])]),
        }));

        ([t_x, t_p], [t_y])
    }

    fn path(&self) -> Path {
        path(vec!["test", "linear_sigmoid"])
    }

    fn inline(
        &self,
        builder: &std::rc::Rc<
            std::cell::RefCell<open_hypergraphs::lax::OpenHypergraph<Object, Operation>>,
        >,
        [x, p]: [Var; 2],
    ) -> [Var; 1] {
        let x = matmul(builder, x, p);
        let x = Sigmoid.call(builder, [x]);

        // flatten result shape
        let [a, c] = unpack::<2>(builder, shape(builder, x.clone()));
        let t = pack::<1>(builder, [a * c]);

        [reshape(builder, t, x)]
    }
}

#[test]
fn test_construct_linear_sigmoid() {
    let sigmoid = Sigmoid.term();
    println!("{sigmoid:?}");

    let term = LinearSigmoid.term();
    println!("{term:?}");
}

#[test]
fn test_graph_sigmoid() {
    let term = Sigmoid.term().term;
    use open_hypergraphs::lax::functor::*;

    let term = open_hypergraphs::lax::var::forget::Forget.map_arrow(&term);
    let svg_bytes = to_svg(&term).expect("create svg");
    save_diagram_if_enabled("test_graph_sigmoid.svg", svg_bytes);
}

#[test]
fn test_graph_linear_sigmoid() {
    let term = LinearSigmoid.term().term;

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
    let TypedTerm {
        term, source_type, ..
    } = LinearSigmoid.term();

    run_check_test(term, source_type, "test_check_linear_sigmoid.svg").expect("valid");
}

#[test]
fn test_check_sigmoid() {
    let TypedTerm {
        term, source_type, ..
    } = Sigmoid.term();

    run_check_test(term, source_type, "test_check_sigmoid.svg").expect("valid");
}

#[test]
fn test_check_exp() {
    let TypedTerm {
        term, source_type, ..
    } = Exp.term();
    run_check_test(term, source_type, "test_check_exp.svg").expect("valid");
}

#[allow(clippy::result_large_err)]
pub fn run_check_test(
    term: catgrad_core::category::lang::Term,
    input_types: Vec<Value>,
    svg_filename: &str,
) -> Result<(), ShapeCheckError> {
    use open_hypergraphs::lax::functor::*;

    let term = open_hypergraphs::lax::var::forget::Forget.map_arrow(&term);
    let (ops, env) = get_forget_core_declarations();

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
