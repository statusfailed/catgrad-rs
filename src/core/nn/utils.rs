use crate::backend::cpu::ndarray::{NdArray, TaggedNdArray};
use crate::core::Shape;
use safetensors;
use std::collections::HashMap;
use std::path::PathBuf;

// Read tensor data from safetensors file
pub fn read_safetensors(path: &str) -> HashMap<String, TaggedNdArray> {
    // Load file
    // let data = std::fs::read(path).unwrap();
    // Memory map the file for faster access
    let file = std::fs::File::open(path).unwrap();
    let data = unsafe { memmap2::Mmap::map(&file).unwrap() };
    let tensors = safetensors::SafeTensors::deserialize(&data).unwrap();

    // Initialize result map
    let mut result = HashMap::new();

    // Read each tensor
    for (name, view) in tensors.tensors() {
        let shape = Shape(view.shape().to_vec());
        let tensor_data = view.data();

        // Convert dtype and load tensor data
        match view.dtype() {
            safetensors::Dtype::F32 => {
                let data: Vec<f32> = tensor_data
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                    .collect();
                result.insert(
                    name.to_string(),
                    TaggedNdArray::F32(NdArray::new(data, shape)),
                );
            }
            // cast BF16 to F32 until we support BF16
            safetensors::Dtype::BF16 => {
                let data: Vec<f32> = tensor_data
                    .chunks_exact(2)
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

// Given a model.safetensors or a symlink to it stored in Huggingface local cache
// return the paths to its config and tokenizer files.
pub fn get_model_files(model_path: &str) -> (PathBuf, PathBuf) {
    let mut model_path = PathBuf::from(model_path);
    if let Ok(link) = model_path.read_link() {
        model_path = link;
    }

    let model_dir = model_path.parent().unwrap();
    (
        model_dir.join("config.json"),
        model_dir.join("tokenizer.json"),
    )
}

pub fn argmax(v: &[f32]) -> i32 {
    v.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(idx, _)| idx)
        .unwrap() as i32
}
