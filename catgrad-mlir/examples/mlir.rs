use catgrad::prelude::*;
use catgrad_mlir::pass::lang_to_mlir;

/// Construct, shapecheck, and lower an `Exp` function to MLIR
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = nn::Sigmoid;

    // Get the model as a typed term
    let typed_term = model.term().expect("Failed to create typed term");

    // Create parameters for the model
    let parameters = typecheck::Parameters::from([]);

    // Get stdlib environment and extend with parameter declarations
    let mut env = stdlib();
    env.declarations
        .extend(to_load_ops(model.path(), parameters.keys()));

    // Convert model to MLIR
    let mlir = lang_to_mlir(&env, &parameters, typed_term);

    // print MLIR
    println!("{}", mlir[0]);

    Ok(())
}
