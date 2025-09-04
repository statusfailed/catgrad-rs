//! Standard library of declarations and definitions for catgrad's `lang` category.
pub mod ops;
pub use ops::*;

pub mod nn;

mod def;
pub use def::*;
