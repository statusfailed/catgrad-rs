use catgrad::prelude::{Environment, stdlib};

pub fn save_diagram_if_enabled<
    O: PartialEq + Clone + std::fmt::Display + std::fmt::Debug,
    A: PartialEq + Clone + std::fmt::Display + std::fmt::Debug,
>(
    filename: &str,
    term: &open_hypergraphs::lax::OpenHypergraph<O, A>,
) {
    #[cfg(feature = "svg")]
    {
        if std::env::var("SAVE_DIAGRAMS").is_ok() {
            let svg_bytes = catgrad::svg::to_svg(term).expect("create svg");
            let output_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("tests")
                .join("images")
                .join(filename);
            std::fs::write(output_path, svg_bytes).expect("write diagram file");
        }
    }

    #[cfg(not(feature = "svg"))]
    {
        // disable warnings
        println!("enable svg feature to save diagram to {filename} for term {term:?}");
    }
}

pub fn get_forget_core_declarations() -> Environment {
    let mut env = stdlib();
    for def in env.definitions.values_mut() {
        def.term = open_hypergraphs::lax::var::forget::forget_monogamous(&def.term);
    }
    env
}
