use catgrad::prelude::*;
use catgrad::typecheck::*;
use catgrad_mlir::{compile::CompiledModel, runtime::LlvmRuntime};
use std::env;

/// Construct, shapecheck, and lower an `Exp` function to MLIR, then codegen and run.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    ////////////////////////////////////////
    // Parse arguments
    let args: Vec<String> = env::args().collect();

    if args.len() != 2 {
        eprintln!("Usage: {} <output.so>", args[0]);
        std::process::exit(1);
    }

    let output_so = std::path::Path::new(&args[1])
        .canonicalize()
        .unwrap_or_else(|_| std::env::current_dir().unwrap().join(&args[1]));

    ////////////////////////////////////////
    // Setup model and environment
    let model = ConcreteSigmoid;
    let typed_term = model.term().expect("Failed to create typed term");
    let parameters = typecheck::Parameters::from([]);

    let mut env = stdlib();
    env.definitions.extend([(model.path(), typed_term)]);
    env.declarations
        .extend(to_load_ops(model.path(), parameters.keys()));

    ////////////////////////////////////////
    // Compile and set up runtime with compiled code
    println!("Compiling {} to {}...", model.path(), output_so.display());
    let compiled_model = CompiledModel::new(&env, &parameters, model.path(), output_so);

    ////////////////////////////////////////
    // Execute with example data
    let input_data = vec![
        1.0f32, 2.0, 3.0, 4.0, // 0
        5.0, 6.0, 7.0, 8.0, // 1
        9.0, 10.0, 11.0, 12.0, // 2
    ];
    let input_tensor = LlvmRuntime::tensor(input_data, vec![3, 1, 4], vec![4, 4, 1]);
    println!("Input tensor: {}", input_tensor);

    // Call the function using the CompiledModel API
    let results = compiled_model.call(model.path(), vec![input_tensor]);

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
