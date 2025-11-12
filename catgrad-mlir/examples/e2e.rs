use catgrad::prelude::*;
use catgrad::typecheck::*;
use catgrad_mlir::codegen::codegen;
use catgrad_mlir::pass::lang_to_mlir;
use catgrad_mlir::runtime::{Entrypoint, LlvmRuntime, MlirType};
use std::env;
use std::ffi::CString;

/// Construct, shapecheck, and lower an `Exp` function to MLIR, then codegen and run.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();

    if args.len() != 2 {
        eprintln!("Usage: {} <output.so>", args[0]);
        std::process::exit(1);
    }

    let output_so = std::path::Path::new(&args[1])
        .canonicalize()
        .unwrap_or_else(|_| std::env::current_dir().unwrap().join(&args[1]));

    // Step 1: Generate MLIR
    let model = ConcreteSigmoid;

    // Get the model as a typed term
    let typed_term = model.term().expect("Failed to create typed term");

    // Create parameters for the model
    let parameters = typecheck::Parameters::from([]);

    // Get stdlib environment and extend with parameter declarations
    let mut env = stdlib();
    env.declarations
        .extend(to_load_ops(model.path(), parameters.keys()));

    // Convert model to MLIR
    let mlir = lang_to_mlir(&env, &parameters, typed_term, &model.path().to_string());

    // Step 2: Codegen MLIR to shared library
    println!("\nCompiling to shared library {}...", &output_so.display());
    codegen(&mlir[0].to_string(), &output_so).unwrap();

    // Step 3: Run using runtime
    println!("\nExecuting with runtime...");
    let func_name = CString::new("test.sigmoid")?;

    // Create runtime with sigmoid entrypoint
    let entrypoint = Entrypoint {
        func_name: func_name.clone(),
        source_types: vec![MlirType::Memref(3)], // Single 3D memref input
        target_types: vec![MlirType::Memref(3)], // Single 3D memref output
    };

    // Initialize runtime for shared object file
    let runtime = LlvmRuntime::new(&output_so, vec![entrypoint])?;

    // Create input tensor using the runtime helper
    let input_data = vec![
        1.0f32, 2.0, 3.0, 4.0, // First slice
        5.0, 6.0, 7.0, 8.0, // Second slice
        9.0, 10.0, 11.0, 12.0, // Third slice
    ];
    let input_tensor = LlvmRuntime::tensor(input_data, vec![3, 1, 4], vec![4, 4, 1]);

    // Print the input tensor using Display
    println!("Input tensor: {}", input_tensor);

    // Call the function using the safe runtime API
    let results = runtime.call(&func_name, vec![input_tensor])?;

    // Print each result using Display
    for (i, result) in results.iter().enumerate() {
        println!("Output tensor {}: {}", i, result);
    }

    Ok(())
}

pub struct ConcreteSigmoid;

// Redo the sigmoid example with a concrete type
impl Module<1, 1> for ConcreteSigmoid {
    fn ty(&self) -> ([Type; 1], [Type; 1]) {
        // TODO: try replacing shape with ShapeExpr::Var(0)
        let ty = Type::Tensor(TypeExpr::NdArrayType(NdArrayType {
            dtype: DtypeExpr::Constant(Dtype::F32),
            shape: ShapeExpr::Shape(vec![3, 1, 4].into_iter().map(NatExpr::Constant).collect()),
        }));
        ([ty.clone()], [ty])
    }

    fn path(&self) -> Path {
        path(vec!["test", "sigmoid"]).unwrap()
    }

    fn def(&self, builder: &Builder, [x]: [Var; 1]) -> [Var; 1] {
        let sh = ops::shape(builder, x.clone());
        let one = ops::constant(builder, 1.0, &sh);
        let one = ops::cast(builder, one, ops::dtype(builder, x.clone()));

        [one.clone() / (one + nn::Exp.call(builder, [-x]))]
    }
}
