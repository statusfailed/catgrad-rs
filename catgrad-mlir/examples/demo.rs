use libffi::middle::*;
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
enum Value {
    Memref3d(Memref3d),
    I64(i64),
}

impl Value {
    fn to_type(&self) -> ValueType {
        match self {
            Value::Memref3d(_) => ValueType::Memref3d,
            Value::I64(_) => ValueType::I64,
        }
    }

    fn to_args(&self) -> Vec<Arg> {
        match self {
            Value::Memref3d(memref) => {
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
            Value::I64(val) => vec![Arg::new(val)],
        }
    }
}

#[derive(Debug, Clone)]
enum ValueType {
    Memref3d,
    I64,
}

impl ValueType {
    // Get individual field types
    fn to_fields(&self) -> Vec<Type> {
        match self {
            ValueType::Memref3d => vec![
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
            ValueType::I64 => vec![Type::i64()],
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

////////////////////////////////////////////////////////////////////////////////
// Utility to call the function pointer with specified source/target types

fn call(ptr: CodePtr, source_values: Vec<Value>, target_types: Vec<ValueType>) -> Vec<Value> {
    unsafe {
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

        let cif = Cif::new(source_types, return_type);

        // For now, use the concrete struct approach since raw bytes aren't working
        #[repr(C)]
        struct DualMemrefResult {
            negated: Memref3d,
            unchanged: Memref3d,
        }

        let result: DualMemrefResult = cif.call(ptr, &args);

        // Convert to our Value enum
        vec![
            Value::Memref3d(result.negated),
            Value::Memref3d(result.unchanged),
        ]
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
        let source_values = vec![Value::Memref3d(input_memref)];
        let target_types = vec![ValueType::Memref3d, ValueType::Memref3d];
        let results = call(CodePtr(func_ptr), source_values, target_types);

        // Print each result
        for (i, result) in results.iter().enumerate() {
            match result {
                Value::Memref3d(memref) => {
                    print_memref3d(&format!("output {}", i), memref);
                }
                Value::I64(val) => {
                    println!("output {}: {}", i, val);
                }
            }
        }

        // Free heap memory for any Memref3d results
        // Only free if it's not pointing to our stack input
        for result in &results {
            if let Value::Memref3d(memref) = result {
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
