use crate::backend::cpu::ndarray::{NdArray, TaggedNdArray};
use crate::core::Shape;
use safetensors;
use std::collections::HashMap;
// Read tensor data from safetensors file
pub fn read_safetensors(path: &str) -> HashMap<String, TaggedNdArray> {
    // Load file
    let data = std::fs::read(path).unwrap();
    let tensors = safetensors::SafeTensors::deserialize(&data).unwrap();

    // Initialize result map
    let mut result = HashMap::new();

    // Read each tensor
    for name in tensors.names() {
        let view = tensors.tensor(name).unwrap();
        let shape = Shape(view.shape().to_vec());

        // Convert dtype and load tensor data
        match view.dtype() {
            safetensors::Dtype::F32 => {
                let data: Vec<f32> = view
                    .data()
                    .chunks(4)
                    .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                    .collect();
                result.insert(
                    name.to_string(),
                    TaggedNdArray::F32(NdArray::new(data, shape)),
                );
            }
            // cast BF16 to F32 until we support BF16
            safetensors::Dtype::BF16 => {
                let data: Vec<f32> = view
                    .data()
                    .chunks(2)
                    .map(|b| half::bf16::from_le_bytes(b.try_into().unwrap()).to_f32())
                    .collect();
                result.insert(
                    name.to_string(),
                    TaggedNdArray::F32(NdArray::new(data, shape)),
                );
            }
            safetensors::Dtype::I64 => {
                println!("Ignoring I64 tensor: {}", name);
            }
            // Add other dtype conversions as needed
            _ => {
                panic!("Unsupported dtype {:?}", view.dtype())
            }
        }
    }

    result
}
