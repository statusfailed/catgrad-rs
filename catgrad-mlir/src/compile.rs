use catgrad::prelude::*;
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
    // Render the function using lang_to_mlir
    let func = lower::lang_to_mlir(env, params, symbol.clone());
    let mlir_text = func.to_string();

    // Compile MLIR to shared library
    codegen(&mlir_text, &output_so).expect("Failed to compile MLIR");

    // Create entrypoint for the symbol
    let func_name = CString::new(symbol.to_string()).expect("Invalid function name");

    // Get types from the typed term in the environment
    let typed_term = env
        .definitions
        .get(&symbol)
        .expect("Symbol not found in environment");
    let source_types = typed_term
        .source_type
        .iter()
        .map(|t| catgrad_to_mlir_type(t.clone()))
        .collect();
    let target_types = typed_term
        .target_type
        .iter()
        .map(|t| catgrad_to_mlir_type(t.clone()))
        .collect();

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
