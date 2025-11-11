use libffi::middle::*;
use libffi::raw;
use std::ffi::{CString, c_void};

////////////////////////////////////////////////////////////////////////////////
// Known types

// included temporarily so we can handle dynamic args before dynamic returns.
#[repr(C)]
#[derive(Debug)]
struct Memref3d {
    allocated: *mut f32,
    aligned: *mut f32,
    offset: i64,
    sizes: [i64; 3],
    strides: [i64; 3],
}

/// NOTE: this is *NOT* the memref
#[derive(Debug)]
struct MlirTensor<T> {
    aligned_ptr: *mut T,
    offset: i64, // offset into ptr
    sizes: Vec<i64>,
    strides: Vec<i64>,
}

/// Struct format for a Memref3d given to and returned from the .so's entrypoint
impl<T> MlirTensor<T> {
    /// Construct Args for each of the fields, as if for the below C struct.
    /// Assumes allocated == aligned.
    /// ```c
    /// #[repr(C)]
    /// struct Memref {
    ///     allocated: *mut f32,
    ///     aligned: *mut f32,
    ///     offset: i64,
    ///     sizes: [i64; N],
    ///     strides: [i64; N],
    /// }
    /// ```
    //
    // TODO: fix the below- this isn't pushing args, but ptrs
    fn to_args<'a>(&'a self) -> Vec<Arg<'a>> {
        let mut result = Vec::with_capacity(3 + 2 * self.sizes.len());
        result.push(Arg::new(&self.aligned_ptr)); // assume allocated == aligned
        result.push(Arg::new(&self.aligned_ptr));
        result.push(Arg::new(&self.offset));
        for size in &self.sizes {
            result.push(Arg::new(size));
        }
        for stride in &self.strides {
            result.push(Arg::new(stride));
        }
        result
    }
}

#[derive(Debug)]
enum MlirValue {
    MlirTensor(MlirTensor<f32>), // TODO: f32 specialisation
    I64(i64),
}

impl MlirValue {
    fn to_type(&self) -> MlirType {
        match self {
            MlirValue::MlirTensor(_) => MlirType::Memref3d,
            MlirValue::I64(_) => MlirType::I64,
        }
    }

    fn to_args<'a>(&'a self) -> Vec<Arg<'a>> {
        match self {
            MlirValue::MlirTensor(tensor) => tensor.to_args(),
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
) -> Vec<MlirTensor<f32>> {
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

fn parse_results_from_buffer(buffer: &[u8], target_types: &[MlirType]) -> Vec<MlirTensor<f32>> {
    unsafe {
        let mut offset = 0;
        let mut results = Vec::new();

        for target_type in target_types {
            let tensor = match target_type {
                MlirType::Memref3d => {
                    // TODO: This is hardcoded for rank-3. Need to make dynamic based on actual rank.
                    // Parse the memref struct fields manually
                    let allocated = std::ptr::read(buffer.as_ptr().add(offset) as *const *mut f32);
                    offset += std::mem::size_of::<*mut f32>();

                    let aligned = std::ptr::read(buffer.as_ptr().add(offset) as *const *mut f32);
                    offset += std::mem::size_of::<*mut f32>();

                    let memref_offset = std::ptr::read(buffer.as_ptr().add(offset) as *const i64);
                    offset += std::mem::size_of::<i64>();

                    // Read the 3 sizes (hardcoded for rank-3)
                    let mut sizes = Vec::with_capacity(3);
                    for _ in 0..3 {
                        let size = std::ptr::read(buffer.as_ptr().add(offset) as *const i64);
                        sizes.push(size);
                        offset += std::mem::size_of::<i64>();
                    }

                    // Read the 3 strides (hardcoded for rank-3)
                    let mut strides = Vec::with_capacity(3);
                    for _ in 0..3 {
                        let stride = std::ptr::read(buffer.as_ptr().add(offset) as *const i64);
                        strides.push(stride);
                        offset += std::mem::size_of::<i64>();
                    }

                    MlirTensor {
                        aligned_ptr: aligned, // Shallow copy of pointer
                        offset: memref_offset,
                        sizes,
                        strides,
                    }
                }
                MlirType::I64 => {
                    panic!("I64 not handled yet")
                }
            };
            results.push(tensor);
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

        // Create input tensor
        let input_ptr = input.as_mut_ptr() as *mut f32;
        let input_tensor = MlirTensor {
            aligned_ptr: input_ptr,
            offset: 0,
            sizes: vec![3, 1, 4],
            strides: vec![4, 4, 1],
        };

        // Print the input tensor
        print_tensor("input", &input_tensor);

        // Call the function dynamically
        let source_values = vec![MlirValue::MlirTensor(input_tensor)];
        let target_types = vec![MlirType::Memref3d, MlirType::Memref3d];
        let results = call(CodePtr(func_ptr), source_values, target_types);

        // Print each result
        for (i, result) in results.iter().enumerate() {
            print_tensor(&format!("output {}", i), result);
        }

        // Free heap memory for any MlirTensor results
        // Only free if it's not pointing to our stack input
        for result in &results {
            if result.aligned_ptr != input_ptr {
                libc::free(result.aligned_ptr as *mut c_void);
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

fn print_tensor<T>(name: &str, tensor: &MlirTensor<T>)
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
                            *tensor.aligned_ptr.add((tensor.offset + idx) as usize)
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
