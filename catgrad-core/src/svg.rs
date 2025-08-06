use crate::category::bidirectional::Term;

use open_hypergraphs_dot::{Orientation, generate_dot_with};
use std::path::PathBuf;

pub fn to_svg(term: &Term) -> Result<Vec<u8>, std::io::Error> {
    use graphviz_rust::{
        cmd::{CommandArg, Format},
        exec,
        printer::PrinterContext,
    };

    let opts = open_hypergraphs_dot::Options {
        node_label: Box::new(|n| format!("{n:?}")),
        edge_label: Box::new(|e| format!("{e}")),
        orientation: Orientation::LR,
        ..Default::default()
    };

    let dot_graph = generate_dot_with(term, &opts);

    exec(
        dot_graph,
        &mut PrinterContext::default(),
        vec![CommandArg::Format(Format::Svg)],
    )
}

pub fn save_svg<P: Into<PathBuf>>(term: &Term, path: P) -> Result<(), std::io::Error> {
    let svg_bytes = to_svg(term)?;
    std::fs::write(path.into(), svg_bytes)
}
