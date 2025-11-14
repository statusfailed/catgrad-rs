use open_hypergraphs::array::vec::VecArray;
use open_hypergraphs::array::vec::VecKind;
use open_hypergraphs::category::*;
use open_hypergraphs::finite_function::FiniteFunction;
use open_hypergraphs::lax;
use open_hypergraphs::operations::Operations;
use open_hypergraphs::strict;

/// Lax interface to factorization - takes a Rust predicate function
pub fn factor<O, A>(
    c: &lax::OpenHypergraph<O, A>,
    predicate: impl Fn(&A) -> bool,
) -> (lax::OpenHypergraph<O, A>, lax::OpenHypergraph<O, A>)
where
    O: Clone + PartialEq,
    A: Clone,
{
    let strict_c = c.clone().to_strict();

    // Create FiniteFunction from the predicate
    let is_selected = FiniteFunction::<VecKind>::new(
        VecArray(
            strict_c
                .h
                .x
                .0
                .iter()
                .map(|x| if predicate(x) { 1 } else { 0 })
                .collect(),
        ),
        2,
    )
    .unwrap();

    let (e_strict, d_strict) = factor_strict(&strict_c, &is_selected);
    let e_lax = lax::OpenHypergraph::from_strict(e_strict);
    let d_lax = lax::OpenHypergraph::from_strict(d_strict);
    (e_lax, d_lax)
}

/// Factor `c : A → B` into a pair of maps
/// `d : C × A → B` and `e : I → C`
/// such that `c ~= (e × id_A) ; d`
/// and `e` consists of operations selected by predicate `p`.
fn factor_strict<O, A>(
    c: &strict::OpenHypergraph<VecKind, O, A>,
    p: &FiniteFunction<VecKind>,
) -> (
    strict::OpenHypergraph<VecKind, O, A>,
    strict::OpenHypergraph<VecKind, O, A>,
)
where
    A: Clone,
    O: Clone + PartialEq,
{
    let k = select(p);
    assert_eq!(k.target(), c.h.x.len());
    assert!(k.source() <= k.target());

    // Find source/target wires of each of the selected operations
    let e_s = c.h.s.map_indexes(&k).unwrap();
    let e_t = c.h.t.map_indexes(&k).unwrap();

    // Compute the tensor of selected operations, then bend around the source
    // wires so that
    // `(e₀ ● e₁ ● ... ● en) : A → B`
    // becomes
    // `(e₀ ● e₁ ● ... ● en) : I → A × B`
    let selected_ops = Operations::new(
        (&k >> &c.h.x).unwrap(),
        e_s.map_semifinite(&c.h.w).unwrap(),
        e_t.map_semifinite(&c.h.w).unwrap(),
    )
    .unwrap();

    let mut e = strict::OpenHypergraph::tensor_operations(selected_ops);
    e = strict::OpenHypergraph {
        s: FiniteFunction::initial(e.h.w.len()),
        t: (&e.s | &e.t),
        h: e.h,
    };

    // Remove the selected operations from c (using twist to invert a predicate)
    // then add the source/target wires of those operations to the left boundary.
    let k_inv = p.compose(&FiniteFunction::twist(1, 1)).unwrap();
    let mut d = filter_operations(c, &k_inv);
    d = strict::OpenHypergraph {
        s: (&(&e_s.values | &e_t.values) | &d.s),
        t: d.t.clone(),
        h: d.h,
    };

    (e, d)
}

/// Given a predicate X → 2, return a function
/// `f : X' → X` such that `f >> p == 1`.
fn select(p: &FiniteFunction<VecKind>) -> FiniteFunction<VecKind> {
    if p.target() != 2 {
        panic!("p must be a predicate, but target = {}", p.target());
    }

    let indices: Vec<usize> = p
        .table
        .0
        .iter()
        .enumerate()
        .filter_map(|(i, &val)| if val != 0 { Some(i) } else { None })
        .collect();

    FiniteFunction::new(VecArray(indices), p.source()).unwrap()
}

/// Given an OpenHypergraph `f` with `X` operations,
/// and a predicate `p : X → 2`, remove those operations `x` from `d`
/// for which `p(x) == 0`.
fn filter_operations<O, A>(
    f: &strict::OpenHypergraph<VecKind, O, A>,
    p: &FiniteFunction<VecKind>,
) -> strict::OpenHypergraph<VecKind, O, A>
where
    A: Clone,
    O: Clone,
{
    assert_eq!(f.h.x.len(), p.source());

    // k : X' → X
    let k = select(p);
    let h = strict::Hypergraph {
        s: f.h.s.map_indexes(&k).unwrap(),
        t: f.h.t.map_indexes(&k).unwrap(),
        w: f.h.w.clone(),
        x: (&k >> &f.h.x).unwrap(),
    };

    strict::OpenHypergraph {
        s: f.s.clone(),
        t: f.t.clone(),
        h,
    }
}
