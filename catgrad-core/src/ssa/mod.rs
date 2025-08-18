//! Convert an OpenHypergraph to SSA form
use open_hypergraphs::array::vec::VecKind;
use open_hypergraphs::{lax, strict};
use std::fmt::{self, Debug, Display};

/// A single static assignment of the form
/// `s₀, s₁, s₂, ... = op(t₀, t₁, ..., tn)`
/// where each `s_i`, `t_i` is a
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SSA<O, A> {
    pub op: A,
    pub edge_id: lax::EdgeId,
    pub sources: Vec<(lax::NodeId, O)>, // source nodes and type labels
    pub targets: Vec<(lax::NodeId, O)>, // target nodes and type labels
}

pub fn parallel_ssa<O: Clone, A: Clone>(
    f: strict::OpenHypergraph<VecKind, O, A>,
) -> Vec<Vec<SSA<O, A>>> {
    // partial topological ordering on edges
    let (op_order, unvisited) = strict::layer::layered_operations(&f);

    // check we got an acyclic input
    // Note: temporarily commented out due to potential issue with layered_operations
    assert!(!unvisited.0.contains(&1));

    // Convert to nonstrict
    let f = lax::OpenHypergraph::from_strict(f);

    // Keep as partial ordering - each layer is a Vec<SSA>
    op_order
        .iter()
        .map(|layer| {
            layer
                .0
                .iter()
                .map(|edge_id| {
                    let lax::Hyperedge { sources, targets } =
                        f.hypergraph.adjacency[*edge_id].clone();
                    let op = f.hypergraph.edges[*edge_id].clone();
                    SSA {
                        op,
                        edge_id: lax::EdgeId(*edge_id),
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
        })
        .collect()
}

pub fn ssa<O: Clone, A: Clone>(f: strict::OpenHypergraph<VecKind, O, A>) -> Vec<SSA<O, A>> {
    // Flatten the partial order into a total order
    parallel_ssa(f).into_iter().flatten().collect()
}

impl<O: Debug, A: Debug> Display for SSA<O, A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Print targets
        let target_strs: Vec<String> = self
            .targets
            .iter()
            .map(|(node_id, _type)| format!("v{}", node_id.0))
            .collect();

        // Print sources
        let source_strs: Vec<String> = self
            .sources
            .iter()
            .map(|(node_id, _type)| format!("v{}", node_id.0))
            .collect();

        write!(
            f,
            "{}:\t{} = {:?}({})",
            self.edge_id.0,
            target_strs.join(", "),
            self.op,
            source_strs.join(", ")
        )
    }
}

#[cfg(test)]
mod tests;
