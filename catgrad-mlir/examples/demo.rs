use catgrad_mlir::runtime::{Entrypoint, LlvmRuntime, MlirTensor, MlirType, MlirValue, call};
use std::ffi::CString;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create runtime with negate_f32 entrypoint
    let entrypoint = Entrypoint {
        func_name: CString::new("negate_f32")?,
        source_types: vec![], // TODO: integrate MlirType
        target_types: vec![], // TODO: integrate MlirType
    };

    // Initialize runtime for shared object file
    let runtime = LlvmRuntime::new(std::path::Path::new("./main.so"), vec![entrypoint])?;

    // Get the function pointer
    let func_name = CString::new("negate_f32")?;
    let (func_ptr, _entrypoint) = runtime
        .get_entrypoint(&func_name)
        .ok_or("Function not found in loaded library")?;

    // Create input tensor using the runtime helper
    let input_data = vec![
        1.0f32, 2.0, 3.0, 4.0, // First slice
        5.0, 6.0, 7.0, 8.0, // Second slice
        9.0, 10.0, 11.0, 12.0, // Third slice
    ];
    let input_tensor = LlvmRuntime::tensor(input_data, vec![3, 1, 4], vec![4, 4, 1]);

    // Print the input tensor (extract the MlirTensor from MlirValue)
    if let MlirValue::MlirTensor(ref tensor) = input_tensor {
        print_tensor_3d("input", tensor);
    }

    // Call the function dynamically
    let source_values = vec![input_tensor];
    let target_types = vec![MlirType::Memref(3), MlirType::Memref(3)];
    let results = call(func_ptr, source_values, target_types);

    // Print each result
    for (i, result) in results.iter().enumerate() {
        print_tensor_3d(&format!("output {}", i), result);
    }

    Ok(())
}

////////////////////////////////////////////////////////////////////////////////
// Utility functions

fn print_tensor_3d<T>(name: &str, tensor: &MlirTensor<T>)
where
    T: std::fmt::Display + Copy,
{
    unsafe {
        println!("{}...", name);

        // Use the actual sizes and strides for dynamic indexing (rank-3 only)
        if tensor.sizes.len() == 3 {
            for i in 0..tensor.sizes[0] {
                for j in 0..tensor.sizes[1] {
                    for k in 0..tensor.sizes[2] {
                        let idx =
                            i * tensor.strides[0] + j * tensor.strides[1] + k * tensor.strides[2];
                        print!(
                            "{:6.2} ",
                            *tensor.aligned.add((tensor.offset + idx) as usize)
                        );
                    }
                    println!();
                }
            }
        } else {
            println!(
                "  [Tensor printing only supports rank-3, got rank {}]",
                tensor.sizes.len()
            );
        }
    }
}
