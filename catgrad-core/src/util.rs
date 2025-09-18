use open_hypergraphs::lax::{OpenHypergraph, var::*};
use std::cell::RefCell;
use std::rc::Rc;

pub(crate) fn build_typed<const ARITY: usize, F, O: Clone, A: HasVar + Clone>(
    source_types: [O; ARITY],
    f: F,
) -> BuildResult<O, A>
where
    F: Fn(&Rc<RefCell<OpenHypergraph<O, A>>>, [Var<O, A>; ARITY]) -> Vec<Var<O, A>>,
{
    use std::array;
    build(move |state| {
        // use from_fn to avoid having to clone source_types
        let sources: [Var<_, _>; ARITY] =
            array::from_fn(|i| Var::new(state.clone(), source_types[i].clone()));

        let sources_vec = sources.iter().cloned().collect();
        let targets = f(state, sources);
        (sources_vec, targets)
    })
}

pub(crate) fn iter_to_array<T, const N: usize>(iter: impl Iterator<Item = T>) -> Option<[T; N]> {
    iter.collect::<Vec<T>>().try_into().ok()
}

pub fn replace_nodes_in_hypergraph<T, U, V>(
    term: OpenHypergraph<T, U>,
    new_nodes: Vec<V>,
) -> OpenHypergraph<V, U> {
    use open_hypergraphs::lax::Hypergraph;
    OpenHypergraph {
        hypergraph: Hypergraph {
            nodes: new_nodes,
            edges: term.hypergraph.edges,
            adjacency: term.hypergraph.adjacency,
            quotient: term.hypergraph.quotient,
        },
        sources: term.sources,
        targets: term.targets,
    }
}

// TODO: add to open-hypergraphs
pub(crate) fn map_nodes_and_edges<F: Fn(O1) -> O2, G: Fn(A1) -> A2, O1, O2, A1, A2>(
    term: OpenHypergraph<O1, A1>,
    f: F,
    g: G,
) -> OpenHypergraph<O2, A2> {
    use open_hypergraphs::lax::Hypergraph;
    let OpenHypergraph {
        sources,
        targets,
        hypergraph:
            Hypergraph {
                nodes,
                edges,
                adjacency,
                quotient,
            },
    } = term;

    OpenHypergraph {
        sources,
        targets,
        hypergraph: Hypergraph {
            nodes: nodes.into_iter().map(f).collect(),
            edges: edges.into_iter().map(g).collect(),
            adjacency,
            quotient,
        },
    }
}
