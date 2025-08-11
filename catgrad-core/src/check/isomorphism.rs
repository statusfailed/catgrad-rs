use super::types::NatExpr;
use std::collections::HashMap;

/// Check if shapes of known rank are isomorphic
pub fn isomorphic(x: Vec<NatExpr>, y: Vec<NatExpr>) -> bool {
    normalize(x) == normalize(y)
}

/// Normalize a product of NatExpr into a single NatExpr
pub fn normalize(exprs: Vec<NatExpr>) -> NatExpr {
    if exprs.is_empty() {
        return NatExpr::Constant(1);
    }

    let product = if exprs.len() == 1 {
        exprs.into_iter().next().unwrap()
    } else {
        NatExpr::Mul(exprs)
    };

    normalize_nat(product)
}

fn normalize_nat(expr: NatExpr) -> NatExpr {
    match expr {
        NatExpr::Var(v) => NatExpr::Var(v),
        NatExpr::Constant(c) => NatExpr::Constant(c),
        NatExpr::Mul(terms) => {
            let mut constants = 1;
            let mut vars: HashMap<usize, usize> = HashMap::new();
            let mut nested_muls = Vec::new();

            for term in terms {
                let normalized_term = normalize_nat(term);
                match normalized_term {
                    NatExpr::Constant(c) => constants *= c,
                    NatExpr::Var(v) => {
                        *vars.entry(v).or_insert(0) += 1;
                    }
                    NatExpr::Mul(inner_terms) => {
                        nested_muls.extend(inner_terms);
                    }
                }
            }

            if !nested_muls.is_empty() {
                return normalize_nat(NatExpr::Mul(
                    [
                        vec![NatExpr::Constant(constants)],
                        vars.into_iter()
                            .flat_map(|(v, count)| vec![NatExpr::Var(v); count])
                            .collect(),
                        nested_muls,
                    ]
                    .concat(),
                ));
            }

            let mut result = Vec::new();

            if constants != 1 {
                result.push(NatExpr::Constant(constants));
            }

            let mut var_terms: Vec<_> = vars.into_iter().collect();
            var_terms.sort_by_key(|(v, _)| *v);

            for (var, count) in var_terms {
                for _ in 0..count {
                    result.push(NatExpr::Var(var));
                }
            }

            match result.len() {
                0 => NatExpr::Constant(1),
                1 => result.into_iter().next().unwrap(),
                _ => NatExpr::Mul(result),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_isomorphic_basic() {
        let a = NatExpr::Var(0);
        let b = NatExpr::Var(1);

        // [a, b] should be isomorphic to [a*b] when considering shape isomorphism
        let x = vec![a.clone(), b.clone()];
        let y = vec![NatExpr::Mul(vec![a, b])];

        assert!(isomorphic(x, y));
    }

    #[test]
    fn test_isomorphic_constants() {
        // [2, 3] should be isomorphic to [6]
        let x = vec![NatExpr::Constant(2), NatExpr::Constant(3)];
        let y = vec![NatExpr::Constant(6)];

        assert!(isomorphic(x, y));
    }

    #[test]
    fn test_isomorphic_mixed() {
        let a = NatExpr::Var(0);

        // [a, 2] should be isomorphic to [2*a]
        let x = vec![a.clone(), NatExpr::Constant(2)];
        let y = vec![NatExpr::Mul(vec![NatExpr::Constant(2), a])];

        assert!(isomorphic(x, y));
    }

    #[test]
    fn test_not_isomorphic() {
        let a = NatExpr::Var(0);
        let b = NatExpr::Var(1);

        // [a, b] should NOT be isomorphic to [a, a, b]
        let x = vec![a.clone(), b.clone()];
        let y = vec![a.clone(), a, b];

        assert!(!isomorphic(x, y));
    }

    #[test]
    fn test_normalize_flattens_multiplication() {
        let a = NatExpr::Var(0);
        let nested = NatExpr::Mul(vec![
            a.clone(),
            NatExpr::Mul(vec![NatExpr::Constant(2), a.clone()]),
        ]);
        let flat = NatExpr::Mul(vec![NatExpr::Constant(2), a.clone(), a]);

        assert_eq!(normalize_nat(nested), normalize_nat(flat));
    }
}
