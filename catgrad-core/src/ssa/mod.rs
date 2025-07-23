//! Convert an OpenHypergraph to SSA form
use open_hypergraphs::array::vec::VecKind;
use open_hypergraphs::{lax, strict};

/// A single static assignment of the form
/// `s₀, s₁, s₂, ... = op(t₀, t₁, ..., tn)`
/// where each `s_i`, `t_i` is a
pub struct SSA<O, A> {
    pub op: A,
    pub sources: Vec<(lax::NodeId, O)>, // source nodes and type labels
    pub targets: Vec<(lax::NodeId, O)>, // target nodes and type labels
}

pub fn ssa<O: Clone, A: Clone>(f: strict::OpenHypergraph<VecKind, O, A>) -> Vec<SSA<O, A>> {
    // partial topological ordering on edges
    let (op_order, unvisited) = strict::layer::layered_operations(&f);

    // check we got an acyclic input
    // Note: temporarily commented out due to potential issue with layered_operations
    assert!(!unvisited.0.contains(&1));

    // Convert to nonstrict
    let f = lax::OpenHypergraph::from_strict(f);

    // Turn partial ordering into total one
    let sorted_edge_ids = op_order.iter().flat_map(|layer| layer.0.iter());

    // Collect all hyperedges into SSA form
    sorted_edge_ids
        .map(|edge_id| {
            let lax::Hyperedge { sources, targets } = f.hypergraph.adjacency[*edge_id].clone();
            let op = f.hypergraph.edges[*edge_id].clone();
            SSA {
                op,
                sources: sources
                    .iter()
                    .map(|id| (*id, f.hypergraph.nodes[id.0].clone()))
                    .collect(),
                targets: targets
                    .iter()
                    .map(|id| (*id, f.hypergraph.nodes[id.0].clone()))
                    .collect(),
            }
        })
        .collect()
}

#[cfg(test)]
mod tests;
