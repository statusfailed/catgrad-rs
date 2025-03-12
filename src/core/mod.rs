pub mod object;
pub mod operation;

pub use object::*;
pub use operation::*;

use open_hypergraphs::prelude::OpenHypergraph;

pub type Term = OpenHypergraph<GeneratingObject, operation::Operation>;
