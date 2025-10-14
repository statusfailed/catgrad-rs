use crate::category::core;
use crate::category::lang;

use super::module::*;

pub use crate::pass::to_core::Environment;
use crate::pass::to_core::core_declarations;

use std::collections::HashMap;

// helper to simplify stdlib defs list
fn to_pair<const A: usize, const B: usize, T: Module<A, B>>(
    def: T,
) -> (lang::Path, lang::TypedTerm) {
    (def.path(), def.term().unwrap())
}

/// Standard library of definitions
fn definitions() -> HashMap<lang::Path, lang::TypedTerm> {
    use super::nn::*;

    // NOTE: can't just map this since each invocation of to_pair is differently typed
    HashMap::from([
        to_pair(Sigmoid),
        to_pair(Exp),
        to_pair(Sqrt),
        to_pair(Gelu),
        //
    ])
}

/// Standard library declarations and definitions
pub fn stdlib() -> Environment {
    Environment {
        declarations: core_declarations(),
        definitions: definitions(),
    }
}

/// Convert parameter paths to Load operations with a given prefix
pub fn to_load_ops<'a, I>(
    prefix: lang::Path,
    paths: I,
) -> impl Iterator<Item = (lang::Path, core::Operation)>
where
    I: IntoIterator<Item = &'a lang::Path>,
{
    paths.into_iter().map(move |key| {
        let param_path = prefix.concat(key);
        (param_path, core::Operation::Load(key.clone()))
    })
}
