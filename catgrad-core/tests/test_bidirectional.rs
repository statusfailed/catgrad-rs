use catgrad_core::category::bidirectional::*;
use catgrad_core::nn::*;
use catgrad_core::svg::to_svg;
use catgrad_core::util::build_typed;

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
fn test_construct_linear_sigmoid() {
    let sigmoid = sigmoid_term();
    println!("{sigmoid:?}");

    let term = linear_sigmoid();
    println!("{term:?}");
}

#[test]
fn test_graph_linear_sigmoid() {
    let term = linear_sigmoid();
    use open_hypergraphs::lax::functor::*;

    let term = open_hypergraphs::lax::var::forget::Forget.map_arrow(&term);
    let svg_bytes = to_svg(&term).expect("create svg");
    save_diagram_if_enabled("test_graph_linear_sigmoid.svg", svg_bytes);
}

/*
// Shapecheck the linear-sigmoid term.
// This should allow us to generate a diagram similar to the one in test_graph_linear_sigmoid(),
// but where objects are "symbolic shapes".
#[test]
fn test_check_linear_sigmoid() {
    todo!()
}
*/

/*
#[test]
fn test_cyclic_definition_fails() {
    todo!()
}
*/
