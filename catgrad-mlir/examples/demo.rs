use std::ffi::{CString, c_void};
use std::ptr;

#[repr(C)]
struct Memref3d {
    allocated: *mut f32,
    aligned: *mut f32,
    offset: i64,
    sizes: [i64; 3],
    strides: [i64; 3],
}

type NegateFn =
    unsafe extern "C" fn(*mut f32, *mut f32, i64, i64, i64, i64, i64, i64, i64) -> Memref3d;

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

        let negate_f32: NegateFn = std::mem::transmute(func_ptr);

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
        let input_memref = Memref3d {
            allocated: input.as_mut_ptr() as *mut f32,
            aligned: input.as_mut_ptr() as *mut f32,
            offset: 0,
            sizes: [3, 1, 4],
            strides: [4, 4, 1],
        };

        // Invoke the MLIR function
        let result = negate_f32(
            input_memref.allocated,
            input_memref.aligned,
            input_memref.offset,
            input_memref.sizes[0],
            input_memref.sizes[1],
            input_memref.sizes[2],
            input_memref.strides[0],
            input_memref.strides[1],
            input_memref.strides[2],
        );

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

