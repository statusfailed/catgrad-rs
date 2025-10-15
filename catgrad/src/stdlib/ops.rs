pub use crate::category::lang::ops::*;
use crate::prelude::{Builder, Var};

////////////////////////////////////////////////////////////////////////////////
// Helpers wrapping lang:: methods

/// Transpose a tensor using either symbolic (Var) or static (u32) dims
pub fn transpose(builder: &Builder, a: impl IntoNatVar, b: impl IntoNatVar, x: Var) -> Var {
    crate::category::lang::transpose(builder, a.to_var(builder), b.to_var(builder), x)
}

pub trait IntoNatVar {
    fn to_var(&self, builder: &Builder) -> Var;
}

impl IntoNatVar for Var {
    fn to_var(&self, _builder: &Builder) -> Var {
        self.clone()
    }
}

impl IntoNatVar for u32 {
    fn to_var(&self, builder: &Builder) -> Var {
        constant_nat(builder, *self)
    }
}

impl IntoNatVar for i32 {
    fn to_var(&self, builder: &Builder) -> Var {
        constant_nat(builder, (*self).try_into().unwrap())
    }
}

impl IntoNatVar for usize {
    fn to_var(&self, builder: &Builder) -> Var {
        constant_nat(builder, (*self).try_into().unwrap())
    }
}
