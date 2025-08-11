use super::types::NatExpr;
use std::collections::HashSet;

/// Check if shapes of known rank are isomorphic
pub fn isomorphic(x: Vec<NatExpr>, y: Vec<NatExpr>) -> bool {
    normalize_product(x) == normalize_product(y)
}

/// Normalize a product of NatExpr into a single NatExpr
fn normalize_product(exprs: Vec<NatExpr>) -> NatExpr {
    if exprs.is_empty() {
        return NatExpr::Constant(1);
    }

    let product = if exprs.len() == 1 {
        exprs.into_iter().next().unwrap()
    } else {
        NatExpr::Mul(exprs)
    };

    normalize(&product)
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct Mono(Vec<(usize, usize)>); // sorted (var, exp)

impl Mono {
    fn _unit() -> Self {
        Mono(Vec::new())
    }

    fn from_var(v: usize) -> Self {
        Mono(vec![(v, 1)])
    }

    fn mul(&self, other: &Self) -> Self {
        // merge two sorted vectors, summing exponents
        let (a, b) = (&self.0, &other.0);
        let mut i = 0usize;
        let mut j = 0usize;
        let mut out: Vec<(usize, usize)> = Vec::with_capacity(a.len() + b.len());
        while i < a.len() && j < b.len() {
            match a[i].0.cmp(&b[j].0) {
                std::cmp::Ordering::Less => {
                    out.push(a[i]);
                    i += 1;
                }
                std::cmp::Ordering::Greater => {
                    out.push(b[j]);
                    j += 1;
                }
                std::cmp::Ordering::Equal => {
                    let (v, ea) = a[i];
                    let (_, eb) = b[j];
                    out.push((v, ea + eb));
                    i += 1;
                    j += 1;
                }
            }
        }
        if i < a.len() {
            out.extend_from_slice(&a[i..]);
        }
        if j < b.len() {
            out.extend_from_slice(&b[j..]);
        }
        Mono(out)
    }
}

/// Boolean-coefficient polynomial + a separate natural constant term.
/// - `support` is the set of *non-constant* monomials present with coefficient > 0 (coefficients saturate to 1).
/// - `const_term` is the numeric constant (summed over all constants).
#[derive(Debug, Clone, Default)]
struct PolyBool {
    support: HashSet<Mono>,
    const_term: usize,
}

impl PolyBool {
    fn zero() -> Self {
        PolyBool {
            support: HashSet::new(),
            const_term: 0,
        }
    }
    fn from_var(v: usize) -> Self {
        let mut s = HashSet::new();
        s.insert(Mono::from_var(v));
        PolyBool {
            support: s,
            const_term: 0,
        }
    }
    fn from_const(c: usize) -> Self {
        PolyBool {
            support: HashSet::new(),
            const_term: c,
        }
    }

    fn is_zero(&self) -> bool {
        self.const_term == 0 && self.support.is_empty()
    }

    fn add_into(&mut self, other: &PolyBool) {
        // boolean coefficients for monomials: union of supports
        for m in &other.support {
            self.support.insert(m.clone());
        }
        // natural addition for constants
        self.const_term = self.const_term.saturating_add(other.const_term);
    }

    fn mul(&self, other: &PolyBool) -> PolyBool {
        if self.is_zero() || other.is_zero() {
            return PolyBool::zero();
        }

        // new constant
        let const_term = self.const_term.saturating_mul(other.const_term);

        // products of non-constant monomials
        let mut support: HashSet<Mono> = HashSet::new();
        for ma in &self.support {
            for mb in &other.support {
                support.insert(ma.mul(mb));
            }
        }

        // monomials * constant (if constant > 0, they stay present)
        if other.const_term > 0 {
            for ma in &self.support {
                support.insert(ma.clone());
            }
        }
        if self.const_term > 0 {
            for mb in &other.support {
                support.insert(mb.clone());
            }
        }

        PolyBool {
            support,
            const_term,
        }
    }
}

fn poly_of_bool(e: &NatExpr) -> PolyBool {
    match e {
        NatExpr::Var(v) => PolyBool::from_var(*v),
        NatExpr::Constant(c) => PolyBool::from_const(*c),
        NatExpr::Add(ts) => {
            let mut acc = PolyBool::zero();
            for t in ts {
                acc.add_into(&poly_of_bool(t));
            }
            acc
        }
        NatExpr::Mul(ts) => {
            // start with multiplicative unit: constant 1
            let mut acc = PolyBool::from_const(1);
            for t in ts {
                let pt = poly_of_bool(t);
                acc = acc.mul(&pt);
                if acc.is_zero() {
                    break;
                }
            }
            acc
        }
    }
}

fn mono_to_factors(m: &Mono) -> Vec<NatExpr> {
    let mut out = Vec::new();
    for &(v, e) in &m.0 {
        for _ in 0..e {
            out.push(NatExpr::Var(v));
        }
    }
    out
}

/// Normalize so that expressions equal under AC + distributivity + idempotent addition
/// (i.e., coefficients for non-constant monomials saturate to 1) map to the same tree.
pub fn normalize(expr: &NatExpr) -> NatExpr {
    let p = poly_of_bool(expr);

    // no terms at all
    if p.is_zero() {
        return NatExpr::Constant(0);
    }

    // collect monomials, sort canonically
    let mut monos: Vec<Mono> = p.support.into_iter().collect();
    monos.sort();

    let mut terms: Vec<NatExpr> = Vec::new();

    // include constant if present (>0)
    if p.const_term > 0 {
        terms.push(NatExpr::Constant(p.const_term));
    }

    // add each monomial once (boolean coeff)
    for m in monos {
        let factors = mono_to_factors(&m);
        let term = if factors.len() == 1 {
            factors.into_iter().next().unwrap()
        } else {
            NatExpr::Mul(factors)
        };
        terms.push(term);
    }

    if terms.len() == 1 {
        terms.pop().unwrap()
    } else {
        NatExpr::Add(terms)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn var(i: usize) -> NatExpr {
        NatExpr::Var(i)
    }
    fn cst(c: usize) -> NatExpr {
        NatExpr::Constant(c)
    }
    fn add(xs: Vec<NatExpr>) -> NatExpr {
        NatExpr::Add(xs)
    }
    fn mul(xs: Vec<NatExpr>) -> NatExpr {
        NatExpr::Mul(xs)
    }

    #[test]
    fn ac_iso_equal() {
        let e1 = mul(vec![var(0), add(vec![var(1), var(2)])]); // a*(b+c)
        let e2 = add(vec![mul(vec![var(0), var(1)]), mul(vec![var(0), var(2)])]); // ab+ac
        assert_eq!(normalize(&e1), normalize(&e2));
    }

    #[test]
    fn powers_and_coeffs() {
        // (a+b)(a+b) = aa + 2ab + bb
        let e1 = mul(vec![add(vec![var(0), var(1)]), add(vec![var(0), var(1)])]);
        let e2 = add(vec![
            mul(vec![var(0), var(0)]),
            mul(vec![cst(2), var(0), var(1)]),
            mul(vec![var(1), var(1)]),
        ]);
        assert_eq!(normalize(&e1), normalize(&e2));
    }

    #[test]
    fn zeros_and_ones() {
        let e = mul(vec![cst(0), add(vec![var(0), cst(5)])]);
        assert_eq!(normalize(&e), cst(0));

        let e = mul(vec![cst(1), var(3), cst(1)]);
        assert_eq!(normalize(&e), var(3));
    }

    #[test]
    fn constant_only() {
        let e = add(vec![cst(2), cst(3), cst(0)]);
        assert_eq!(normalize(&e), cst(5));
    }

    #[test]
    fn order_is_canonical() {
        let e1 = add(vec![mul(vec![var(2), var(1)]), mul(vec![cst(3), var(0)])]);
        let e2 = add(vec![mul(vec![cst(3), var(0)]), mul(vec![var(1), var(2)])]);
        assert_eq!(normalize(&e1), normalize(&e2));
    }
}
