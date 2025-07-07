use catgrad::backend::cpu::ndarray::{NdArray, TaggedNdArray};
use catgrad::core::Shape;
use hf_hub::api::sync::Api;
use std::collections::HashMap;
use std::path::PathBuf;

fn read_safetensors_file(path: PathBuf, map: &mut HashMap<String, TaggedNdArray>) {
    let file = std::fs::File::open(path).unwrap();
    let data = unsafe { memmap2::Mmap::map(&file).unwrap() };
    let tensors = safetensors::SafeTensors::deserialize(&data).unwrap();

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
                map.insert(
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
                map.insert(
                    name.to_string(),
                    TaggedNdArray::F32(NdArray::new(data, shape)),
                );
            }
            safetensors::Dtype::I64 => {
                log::warn!("Ignoring I64 tensor: {name}");
            }
            // Add other dtype conversions as needed
            _ => {
                panic!("Unsupported dtype {:?}", view.dtype())
            }
        }
    }
}

pub fn read_safetensors(path: &str) -> HashMap<String, TaggedNdArray> {
    let mut map = HashMap::new();
    read_safetensors_file(PathBuf::from(path), &mut map);
    map
}

pub fn read_safetensors_multiple(path: Vec<PathBuf>) -> HashMap<String, TaggedNdArray> {
    let mut map = HashMap::new();
    for path in path {
        read_safetensors_file(path, &mut map);
    }
    map
}

pub fn get_model_files(model: &str) -> (Vec<PathBuf>, PathBuf, PathBuf, PathBuf) {
    let api = Api::new().unwrap();

    let repo = api.model(model.to_string());

    // Get the model.safetensor file(s)
    let m = if let Ok(index) = repo.get("model.safetensors.index.json") {
        let index = std::fs::File::open(index).unwrap();
        let json: serde_json::Value = serde_json::from_reader(&index).unwrap();
        let mut set = std::collections::HashSet::new();
        if let Some(weight_map) = json.get("weight_map").unwrap().as_object() {
            for v in weight_map.values() {
                set.insert(v.as_str().unwrap().to_string());
            }
        }
        set.iter().map(|p| repo.get(p).unwrap()).collect()
    } else {
        vec![repo.get("model.safetensors").unwrap()]
    };

    let c = repo.get("config.json").unwrap();
    let t = repo.get("tokenizer.json").unwrap();
    let tc = repo.get("tokenizer_config.json").unwrap();

    (m, c, t, tc)
}
