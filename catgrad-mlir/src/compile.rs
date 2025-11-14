use catgrad::prelude::*;
use open_hypergraphs::category::Arrow;
use std::ffi::CString;
use std::path::PathBuf;

use crate::codegen::codegen;
use crate::lower;
use crate::runtime::{Entrypoint, LlvmRuntime, MlirType};

pub fn compile(
    env: &Environment,
    params: &typecheck::Parameters,
    symbol: Path, // catgrad path
    output_so: PathBuf,
) -> LlvmRuntime {
    // Preprocess the term
    let (_param_paths, term) = lower::preprocess(env, params, symbol.clone());

    let source_types = term
        .source()
        .iter()
        .map(|t| catgrad_to_mlir_type(t.clone()))
        .collect();
    let target_types = term
        .target()
        .iter()
        .map(|t| catgrad_to_mlir_type(t.clone()))
        .collect();

    // Render the function using term_to_func
    let func = lower::term_to_func(&symbol.to_string(), term);
    let mlir_text = func.to_string();

    // Compile MLIR to shared library
    codegen(&mlir_text, &output_so).expect("Failed to compile MLIR");

    // Create entrypoint for the symbol
    let func_name = CString::new(symbol.to_string()).expect("Invalid function name");

    let entrypoint = Entrypoint {
        func_name,
        source_types,
        target_types,
    };

    LlvmRuntime::new(&output_so, vec![entrypoint]).expect("Failed to create runtime")
}

fn catgrad_to_mlir_type(catgrad_type: Type) -> MlirType {
    match catgrad_type {
        // TODO: better errors when this doesn't match!
        Type::Tensor(typecheck::TypeExpr::NdArrayType(typecheck::NdArrayType {
            shape: typecheck::ShapeExpr::Shape(shape),
            dtype,
        })) => {
            // TODO: handle other dtypes
            assert_eq!(dtype, typecheck::DtypeExpr::Constant(Dtype::F32));
            MlirType::Memref(shape.len())
        }
        _ => todo!("Unsupported catgrad type"),
        //Type::Nat(_) => todo!("unsupported"),
        //Type::Dtype(_) => todo!("unsupported"),
        //Type::Shape(_) => todo!("unsupported"),
        //Type::Type(_) => todo!("unsupported"),
    }
}
