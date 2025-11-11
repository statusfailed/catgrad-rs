use libffi::middle::*;
use libffi::raw;
use std::ffi::{CString, c_void};

////////////////////////////////////////////////////////////////////////////////
// Known types

/// Struct format for a Memref3d given to and returned from the .so's entrypoint
#[repr(C)]
#[derive(Debug)]
struct Memref3d {
    allocated: *mut f32,
    aligned: *mut f32,
    offset: i64,
    sizes: [i64; 3],
    strides: [i64; 3],
}

#[derive(Debug)]
enum MlirValue {
    Memref3d(Memref3d),
    I64(i64),
}

impl MlirValue {
    fn to_type(&self) -> MlirType {
        match self {
            MlirValue::Memref3d(_) => MlirType::Memref3d,
            MlirValue::I64(_) => MlirType::I64,
        }
    }

    fn to_args<'a>(&'a self) -> Vec<Arg<'a>> {
        match self {
            MlirValue::Memref3d(memref) => {
                // Expand memref into individual args matching C signature
                vec![
                    Arg::new(&memref.allocated),
                    Arg::new(&memref.aligned),
                    Arg::new(&memref.offset),
                    Arg::new(&memref.sizes[0]),
                    Arg::new(&memref.sizes[1]),
                    Arg::new(&memref.sizes[2]),
                    Arg::new(&memref.strides[0]),
                    Arg::new(&memref.strides[1]),
                    Arg::new(&memref.strides[2]),
                ]
            }
            MlirValue::I64(val) => vec![Arg::new(val)],
        }
    }
}

#[derive(Debug, Clone)]
enum MlirType {
    Memref3d,
    I64,
}

impl MlirType {
    // Get individual field types
    fn to_fields(&self) -> Vec<Type> {
        match self {
            MlirType::Memref3d => vec![
                Type::pointer(), // allocated
                Type::pointer(), // aligned
                Type::i64(),     // offset
                Type::i64(),     // sizes[0]
                Type::i64(),     // sizes[1]
                Type::i64(),     // sizes[2]
                Type::i64(),     // strides[0]
                Type::i64(),     // strides[1]
                Type::i64(),     // strides[2]
            ],
            MlirType::I64 => vec![Type::i64()],
        }
    }

    // Get the struct type (wraps fields in a struct)
    fn to_type(&self) -> Type {
        let fields = self.to_fields();
        if fields.len() == 1 {
            fields[0].clone()
        } else {
            Type::structure(fields)
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// Utility to call the function pointer with specified source/target types

// TODO: MEMORY MANAGEMENT
// Returned `Value`s internally have ptrs to array data which needs to be freed manually.
// FIX: take ownership when wrapping in a more accessible catgrad type.
fn call(
    ptr: CodePtr,
    source_values: Vec<MlirValue>,
    target_types: Vec<MlirType>,
) -> Vec<MlirValue> {
    // Derive source types from the source values (flat map to get individual arg types)
    let source_types: Vec<Type> = source_values
        .iter()
        .flat_map(|v| v.to_type().to_fields())
        .collect();

    // Build args from source values (flatten all args)
    let args: Vec<Arg> = source_values.iter().flat_map(|v| v.to_args()).collect();

    // Build return type from target types
    let result_fields: Vec<Type> = target_types.iter().map(|vt| vt.to_type()).collect();
    let return_type = Type::structure(result_fields);

    unsafe {
        // Make Cif from source/return types
        let cif = Cif::new(source_types, return_type);

        // Calculate result buffer size dynamically
        let result_size = calculate_result_size(&target_types);
        let mut result = vec![0u8; result_size];

        // Call using raw ffi_call directly
        raw::ffi_call(
            cif.as_raw_ptr(),
            Some(*ptr.as_safe_fun()),
            result.as_mut_ptr().cast::<c_void>(),
            args.as_ptr() as *mut *mut c_void,
        );

        // Parse results dynamically from the buffer
        parse_results_from_buffer(&result, &target_types)
    }
}

fn calculate_result_size(target_types: &[MlirType]) -> usize {
    target_types
        .iter()
        .map(|vt| match vt {
            MlirType::Memref3d => std::mem::size_of::<Memref3d>(),
            MlirType::I64 => std::mem::size_of::<i64>(),
        })
        .sum()
}

fn parse_results_from_buffer(buffer: &[u8], target_types: &[MlirType]) -> Vec<MlirValue> {
    unsafe {
        let mut offset = 0;
        let mut results = Vec::new();

        for target_type in target_types {
            let value = match target_type {
                MlirType::Memref3d => {
                    let memref: Memref3d =
                        std::ptr::read(buffer.as_ptr().add(offset) as *const Memref3d);
                    offset += std::mem::size_of::<Memref3d>();
                    MlirValue::Memref3d(memref)
                }
                MlirType::I64 => {
                    let val: i64 = std::ptr::read(buffer.as_ptr().add(offset) as *const i64);
                    offset += std::mem::size_of::<i64>();
                    MlirValue::I64(val)
                }
            };
            results.push(value);
        }

        results
    }
}

////////////////////////////////////////////////////////////////////////////////
// Main example

fn main() -> Result<(), Box<dyn std::error::Error>> {
    unsafe {
        // Dynamically load the compiled shared object
        let lib_path = CString::new("./main.so")?;
        let handle = libc::dlopen(lib_path.as_ptr(), libc::RTLD_NOW);
        if handle.is_null() {
            let error = libc::dlerror();
            if !error.is_null() {
                let error_str = std::ffi::CStr::from_ptr(error).to_string_lossy();
                return Err(format!("Failed to load library: {}", error_str).into());
            }
            return Err("Failed to load library".into());
        }

        // Look up the MLIR-exported function by symbol name
        let func_name = CString::new("negate_f32")?;
        let func_ptr = libc::dlsym(handle, func_name.as_ptr());
        if func_ptr.is_null() {
            let error = libc::dlerror();
            if !error.is_null() {
                let error_str = std::ffi::CStr::from_ptr(error).to_string_lossy();
                libc::dlclose(handle);
                return Err(format!("Failed to find symbol: {}", error_str).into());
            }
            libc::dlclose(handle);
            return Err("Failed to find symbol".into());
        }

        // Define a static 3x1x4 tensor as the input buffer
        let mut input = [
            [[1.0f32, 2.0, 3.0, 4.0]],
            [[5.0, 6.0, 7.0, 8.0]],
            [[9.0, 10.0, 11.0, 12.0]],
        ];

        // Create input value
        let input_ptr = input.as_mut_ptr() as *mut f32;
        let input_memref = Memref3d {
            allocated: input_ptr,
            aligned: input_ptr,
            offset: 0,
            sizes: [3, 1, 4],
            strides: [4, 4, 1],
        };

        // Print the input tensor using the memref
        print_memref3d("input", &input_memref);

        // Call the function dynamically
        let source_values = vec![MlirValue::Memref3d(input_memref)];
        let target_types = vec![MlirType::Memref3d, MlirType::Memref3d];
        let results = call(CodePtr(func_ptr), source_values, target_types);

        // Print each result
        for (i, result) in results.iter().enumerate() {
            match result {
                MlirValue::Memref3d(memref) => {
                    print_memref3d(&format!("output {}", i), memref);
                }
                MlirValue::I64(val) => {
                    println!("output {}: {}", i, val);
                }
            }
        }

        // Free heap memory for any Memref3d results
        // Only free if it's not pointing to our stack input
        for result in &results {
            if let MlirValue::Memref3d(memref) = result {
                if memref.allocated != input_ptr {
                    libc::free(memref.allocated as *mut c_void);
                }
            }
        }

        // Unload the shared object
        libc::dlclose(handle);
    }

    Ok(())
}

////////////////////////////////////////////////////////////////////////////////
// Utility functions

fn print_memref3d(name: &str, memref: &Memref3d) {
    unsafe {
        println!("{}...", name);
        for i in 0..3 {
            for j in 0..1 {
                for k in 0..4 {
                    let idx = i * 4 + j * 4 + k;
                    print!("{:6.2} ", *memref.aligned.add(idx as usize));
                }
                println!();
            }
        }
    }
}
