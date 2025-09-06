use catgrad_core::stdlib::{Environment, stdlib};

pub fn save_diagram_if_enabled(filename: &str, data: Vec<u8>) {
    if std::env::var("SAVE_DIAGRAMS").is_ok() {
        let output_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("images")
            .join(filename);
        std::fs::write(output_path, data).expect("write diagram file");
    }
}

pub fn get_forget_core_declarations() -> Environment {
    use open_hypergraphs::lax::functor::*;
    let mut env = stdlib();
    for def in env.definitions.values_mut() {
        def.term = open_hypergraphs::lax::var::forget::Forget.map_arrow(&def.term);
    }
    env
}
