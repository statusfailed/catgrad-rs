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

/// Extend a path with new components, statically checked for validity
#[macro_export]
macro_rules! extend {
    ($prefix:expr $(, $lit:literal)+ $(,)?) => {{
        $(
            const _: () = {
                if $crate::path::is_valid_component($lit) { () }
                else { panic!(concat!("invalid PathComponent: `", $lit, "`")) }
            };
        )*
        $prefix.concat(&vec![ $( ($lit).try_into().unwrap() ),+ ].try_into().unwrap())
    }};
    ($prefix:expr $(,)?) => { $prefix };
}

/// Create a path from statically checked components
#[macro_export]
macro_rules! path {
    () => {
        $crate::extend!($crate::path::Path::empty())
    };
    ($($rest:tt)+) => {
        $crate::extend!($crate::path::Path::empty(), $($rest)+)
    };
}
