use catgrad::prelude::*;
use open_hypergraphs::category::Arrow;
use std::ffi::CString;
use std::path::PathBuf;

use crate::codegen::codegen;
use crate::lower;
use crate::runtime::{Entrypoint, LlvmRuntime, MlirType, MlirValue, RuntimeError};

use std::collections::HashMap;

#[derive(Debug)]
pub enum CompileError {
    Runtime(RuntimeError),
    FunctionNotFound(Path),
    ParameterNotFound(Path),
    InvalidFunctionName(String),
}

impl std::fmt::Display for CompileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CompileError::Runtime(err) => write!(f, "Runtime error: {}", err),
            CompileError::FunctionNotFound(path) => write!(f, "Function not found: {}", path),
            CompileError::ParameterNotFound(path) => write!(f, "Parameter not found: {}", path),
            CompileError::InvalidFunctionName(name) => write!(f, "Invalid function name: {}", name),
        }
    }
}

impl std::error::Error for CompileError {}

impl From<RuntimeError> for CompileError {
    fn from(err: RuntimeError) -> Self {
        CompileError::Runtime(err)
    }
}

pub struct CompiledModel {
    pub runtime: LlvmRuntime,
    pub fn_param_args: HashMap<Path, Vec<Path>>,
    _so_file: Option<tempfile::NamedTempFile>, // Keep temp file alive
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
    pub fn new(env: &Environment, params: &typecheck::Parameters, entrypoint: Path) -> Self {
        // Create secure temporary file for the shared library
        let temp_file =
            tempfile::NamedTempFile::with_suffix(".so").expect("Failed to create temporary file");
        let output_so = temp_file.path().to_path_buf();

        let (runtime, param_paths) = compile(env, params, entrypoint.clone(), output_so);
        let fn_param_args = HashMap::from([(entrypoint, param_paths)]);

        Self {
            runtime,
            fn_param_args,
            _so_file: Some(temp_file),
        }
    }

    /// Call a function from the Environment by name
    pub fn call(
        &self,
        fn_name: Path,
        parameter_values: &HashMap<Path, MlirValue>,
        args: Vec<MlirValue>,
    ) -> Result<Vec<MlirValue>, CompileError> {
        // Collect parameter arguments
        let paths = self
            .fn_param_args
            .get(&fn_name)
            .ok_or_else(|| CompileError::FunctionNotFound(fn_name.clone()))?;

        // Clone MlirValue params; this is not a performance problem since each has an Rc ptr
        // to data.
        let params: Vec<MlirValue> = paths
            .iter()
            .map(|k| {
                parameter_values
                    .get(k)
                    .cloned()
                    .ok_or_else(|| CompileError::ParameterNotFound(k.clone()))
            })
            .collect::<Result<Vec<_>, _>>()?;

        // TODO: check param values match the actual term type!

        let mut all_args = params;
        all_args.extend(args);

        // Convert function name to CString
        let func_name = CString::new(fn_name.to_string())
            .map_err(|_| CompileError::InvalidFunctionName(fn_name.to_string()))?;

        Ok(self.runtime.call(&func_name, all_args)?)
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
            dtype: _,
        })) => MlirType::Memref(shape.len()),
        // Shape types are represented as 1D tensors with rank elements
        Type::Shape(typecheck::ShapeExpr::Shape(_dims)) => MlirType::Memref(1),
        Type::Nat(_) => MlirType::I32,
        _ => todo!("Unsupported catgrad type: {:?}", catgrad_type),
        //Type::Nat(_) => todo!("unsupported"),
        //Type::Dtype(_) => todo!("unsupported"),
        //Type::Type(_) => todo!("unsupported"),
    }
}
