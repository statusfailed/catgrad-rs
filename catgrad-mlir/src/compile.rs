use catgrad::prelude::*;
use open_hypergraphs::category::Arrow;
use std::ffi::CString;
use std::path::PathBuf;

use crate::codegen::codegen;
use crate::lower;
use crate::runtime::{Entrypoint, LlvmRuntime, MlirType, MlirValue};

use std::collections::HashMap;

pub struct CompiledModel {
    pub runtime: LlvmRuntime,
    pub fn_param_args: HashMap<Path, Vec<Path>>,
    pub parameter_values: HashMap<Path, MlirValue>,
}

/// A safe wrapper around Runtime to compile and call models
///
/// # Parameters
///
/// - `env`: the set of definitions to compile to functions (may be inlined)
/// - `params`: types of additional constants added to the signature as nullary operations
/// - `entrypoint`: Entrypoint (currently just one, in future a Vec of potential entrypoints)
/// - `output_so`: Path to save generated shared object code.
impl CompiledModel {
    pub fn new(
        env: &Environment,
        params: &typecheck::Parameters,
        entrypoint: Path,
        output_so: PathBuf,
    ) -> Self {
        let (runtime, param_paths) = compile(env, params, entrypoint.clone(), output_so);
        let fn_param_args = HashMap::from([(entrypoint, param_paths)]);

        Self {
            runtime,
            fn_param_args,
            parameter_values: HashMap::new(),
        }
    }

    /// Set a parameter by name
    pub fn set_param(&mut self, name: Path, value: MlirValue) {
        // TODO: check that the type of `value` lines up with what's in `params.get(name)`.
        // override set in the hashmap
        self.parameter_values.insert(name, value);
    }

    /// Call a function from the Environment by name
    pub fn call(&self, fn_name: Path, args: Vec<MlirValue>) -> Vec<MlirValue> {
        // Collect parameter arguments
        let paths = self.fn_param_args.get(&fn_name).unwrap(); // TODO ERRORS

        // Clone MlirValue params; this is not a performance problem since each has an Rc ptr
        // to data.
        let params: Vec<MlirValue> = paths
            .iter()
            .map(|k| self.parameter_values.get(k).cloned().unwrap()) // TODO ERRORS
            .collect();

        // TODO: check param values match the actual term type!

        let mut all_args = params;
        all_args.extend(args);

        // Convert function name to CString
        let func_name = CString::new(fn_name.to_string()).expect("Invalid function name");

        self.runtime.call(&func_name, all_args).unwrap()
    }
}

fn compile(
    env: &Environment,
    params: &typecheck::Parameters,
    symbol: Path, // catgrad path
    output_so: PathBuf,
) -> (LlvmRuntime, Vec<Path>) {
    // Preprocess the term
    let (param_paths, term) = lower::preprocess(env, params, symbol.clone());

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

    let runtime = LlvmRuntime::new(&output_so, vec![entrypoint]).expect("Failed to create runtime");

    (runtime, param_paths)
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
