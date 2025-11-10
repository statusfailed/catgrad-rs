use libffi::middle::*;
use std::ffi::{CString, c_void};

#[repr(C)]
struct Memref3d {
    allocated: *mut f32,
    aligned: *mut f32,
    offset: i64,
    sizes: [i64; 3],
    strides: [i64; 3],
}

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

        // Print the input tensor contents
        println!("input...");
        for i in 0..3 {
            for j in 0..1 {
                for k in 0..4 {
                    print!("{:6.2} ", input[i][j][k]);
                }
                println!();
            }
        }

        // Construct a memref descriptor describing the input buffer
        let input_ptr = input.as_mut_ptr() as *mut f32;
        let offset = 0i64;
        let sizes = [3i64, 1i64, 4i64];
        let strides = [4i64, 4i64, 1i64];

        // Set up libffi for dynamic function call
        // Function signature: fn(*mut f32, *mut f32, i64, i64, i64, i64, i64, i64, i64) -> Memref3d
        let arg_types = vec![
            Type::pointer(), // allocated
            Type::pointer(), // aligned
            Type::i64(),     // offset
            Type::i64(),     // sizes[0]
            Type::i64(),     // sizes[1]
            Type::i64(),     // sizes[2]
            Type::i64(),     // strides[0]
            Type::i64(),     // strides[1]
            Type::i64(),     // strides[2]
        ];

        // Return type is a struct - expand arrays as individual fields
        let struct_fields = vec![
            Type::pointer(), // allocated
            Type::pointer(), // aligned
            Type::i64(),     // offset
            Type::i64(),     // sizes[0]
            Type::i64(),     // sizes[1]
            Type::i64(),     // sizes[2]
            Type::i64(),     // strides[0]
            Type::i64(),     // strides[1]
            Type::i64(),     // strides[2]
        ];
        let return_type = Type::structure(struct_fields);

        let cif = Cif::new(arg_types, return_type);

        // Prepare arguments with proper lifetime management
        let ptr_arg1 = input_ptr as *mut c_void;
        let ptr_arg2 = input_ptr as *mut c_void;

        let mut args: Vec<Arg> = Vec::new();
        args.push(Arg::new(&ptr_arg1));
        args.push(Arg::new(&ptr_arg2));
        args.push(Arg::new(&offset));
        args.push(Arg::new(&sizes[0]));
        args.push(Arg::new(&sizes[1]));
        args.push(Arg::new(&sizes[2]));
        args.push(Arg::new(&strides[0]));
        args.push(Arg::new(&strides[1]));
        args.push(Arg::new(&strides[2]));

        // Call the function dynamically
        let result: Memref3d = cif.call(CodePtr(func_ptr), &args);

        // Print the negated output tensor from the returned memref
        println!("output...");
        for i in 0..3 {
            for j in 0..1 {
                for k in 0..4 {
                    let idx = i * 4 + j * 4 + k;
                    print!("{:6.2} ", *result.aligned.add(idx as usize));
                }
                println!();
            }
        }

        // Free the heap memory allocated inside the MLIR function
        libc::free(result.allocated as *mut c_void);

        // Unload the shared object
        libc::dlclose(handle);
    }

    Ok(())
}

