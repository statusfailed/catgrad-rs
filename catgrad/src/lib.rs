#![doc = include_str!("../../README.md")]
pub mod category;
pub mod stdlib;

// Compiler passes
pub mod pass;

// path::Path is a type of dot-separated strings used to name definitions and lang ops.
pub mod path;

// Shapechecking & Evaluation
pub mod interpreter;
pub mod ssa;
pub mod typecheck;

// general compiler tools
pub mod abstract_interpreter;
pub mod definition;

// Utilities
#[cfg(feature = "svg")]
pub mod svg;
pub mod util;

// entry point
pub mod prelude;

////////////////////////////////////////////////////////////////////////////////
// Macros

/// Syntax sugar for creating a shape
#[macro_export]
macro_rules! shape {
    ($b:expr, $($x:expr),+ $(,)?) => {{
        let dims = [ $( (&$x).to_var($b) ),+ ];
        pack::<{ shape!(@len $($x),+) }>($b, dims)
    }};
    (@len $($x:tt),+) => { <[()]>::len(&[ $( shape!(@u $x) ),+ ]) };
    (@u $x:tt) => { () };
}
