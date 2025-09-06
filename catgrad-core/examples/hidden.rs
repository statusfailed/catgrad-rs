use catgrad_core::check::check_with;
use catgrad_core::prelude::*;

pub struct SimpleMNISTModel;

// Implement `Def`: this is like torch's `Module`.
impl Def<1, 1> for SimpleMNISTModel {
    // This should return the *detailed* type of the model
    // TODO: NOTE: API for writing types is still WIP.
    fn ty(&self) -> ([Type; 1], [Type; 1]) {
        use catgrad_core::check::*;

        let batch_size = NatExpr::Var(0);

        // Input shape B×28×28
        let t_x = Value::Tensor(TypeExpr::NdArrayType(NdArrayType {
            dtype: DtypeExpr::Constant(Dtype::F32),
            shape: ShapeExpr::Shape(vec![
                batch_size.clone(),
                NatExpr::Constant(28),
                NatExpr::Constant(28),
            ]),
        }));

        // Output shape B×10
        let t_y = Value::Tensor(TypeExpr::NdArrayType(NdArrayType {
            dtype: DtypeExpr::Constant(Dtype::F32),
            shape: ShapeExpr::Shape(vec![batch_size, NatExpr::Constant(10)]),
        }));

        ([t_x], [t_y])
    }

    // TODO: NOTE: it's not clear how user is supposed to know how to choose this name!
    fn path(&self) -> Path {
        path(vec!["model", "hidden"])
    }

    fn def(&self, builder: &Builder, [x]: [Var; 1]) -> [Var; 1] {
        // Flatten input from B×28×28 to B×784
        let [batch_size, h, w] = unpack::<3>(builder, shape(builder, x.clone()));
        let flat_size = h * w;
        let flat_shape = pack::<2>(builder, [batch_size, flat_size]);
        let x = reshape(builder, flat_shape, x);

        let p = param(builder, &path(vec!["0", "weights"]));

        // layer 1: B×784 @ 784×100 = B×100
        let x = matmul(builder, x, p);
        let x = nn::Sigmoid.call(builder, [x]);

        // layer 2: B×100 @ 100×10 = B×10
        let p = param(builder, &path(vec!["1", "weights"]));
        let x = matmul(builder, x, p);
        let x = nn::Sigmoid.call(builder, [x]);

        // result
        [x]
    }
}

// TODO: you would normally create this by reading the safetensors file!
// In user code, the param(<name>) op is just creating a declaration param.name...
use catgrad_core::check::Parameters;
use std::collections::HashMap;

pub fn params() -> Parameters {
    use catgrad_core::category::core::Dtype;
    use catgrad_core::check::{DtypeExpr, NatExpr, NdArrayType, ShapeExpr, TypeExpr, Value};

    let mut map = HashMap::new();

    // Layer 1: (28*28) → 100
    let layer1_type = Value::Tensor(TypeExpr::NdArrayType(NdArrayType {
        dtype: DtypeExpr::Constant(Dtype::F32),
        shape: ShapeExpr::Shape(vec![
            NatExpr::Mul(vec![NatExpr::Constant(28), NatExpr::Constant(28)]),
            NatExpr::Constant(100),
        ]),
    }));
    map.insert(path(vec!["0", "weights"]), layer1_type);

    // Layer 2: 100 → 10
    let layer2_type = Value::Tensor(TypeExpr::NdArrayType(NdArrayType {
        dtype: DtypeExpr::Constant(Dtype::F32),
        shape: ShapeExpr::Shape(vec![NatExpr::Constant(100), NatExpr::Constant(10)]),
    }));
    map.insert(path(vec!["1", "weights"]), layer2_type);

    Parameters::from(map)
}

fn param_declarations(
    params: &Parameters,
) -> impl Iterator<Item = (Path, catgrad_core::category::core::Operation)> + '_ {
    params.0.keys().map(|key| {
        let param_path = path(vec!["param"]).concat(key);
        (
            param_path,
            catgrad_core::category::core::Operation::Load(key.clone()),
        )
    })
}

fn main() {
    let model = SimpleMNISTModel;

    // Get the model as a typed term
    let typed_term = model.term().expect("Failed to create typed term");

    // Create parameters for the model
    let parameters = params();

    // Get stdlib environment and extend with parameter declarations
    let env = stdlib().extend_declarations(param_declarations(&parameters));

    // Shapecheck the model
    match check_with(
        &env,
        &parameters,
        typed_term.term.clone(),
        typed_term.source_type.clone(),
    ) {
        Ok(result_types) => {
            println!("✓ Model shapechecked successfully!");
            println!("Input types: {:?}", typed_term.source_type);
            println!("Output types: {:?}", result_types);
        }
        Err(error) => {
            println!("✗ Shapecheck failed: {:?}", error);
        }
    }

    // Note: To run the interpreter, you would typically:
    // 1. Create an NdArrayBackend
    // 2. Create an Interpreter with the backend, environment, and interpreter parameters
    // 3. Create actual tensor data that matches the input shape
    // 4. Call interpreter.run(term, input_values)

    println!(
        "Model definition complete. To run inference, create tensor data and use the interpreter."
    );
}
